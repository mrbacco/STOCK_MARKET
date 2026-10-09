#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: call_tracker.py
#############################

"""Live record of the plain-language calls shown on the Today page.

Each call says that a stock will beat (or lag) its market over the holding
period. It is saved once per market, horizon, as-of date, and stock, together
with the closing prices of every stock in the market on that date. When the
price history reaches `horizon` sessions past the as-of date, the call is
checked: the stock's return is compared with the equal-weighted average return
of the same market, which is exactly what the model predicts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from app_logging import bac_log_kv
from database import database_connection, ensure_schema

DEFAULT_CALLS_DB = Path(__file__).resolve().parent / "data" / "prediction_calls.db"
BEATS = "beats"
LAGS = "lags"


@dataclass(frozen=True)
class Call:
    ticker: str
    direction: str  # BEATS or LAGS
    predicted_excess: float  # in %


@dataclass(frozen=True)
class LiveRecord:
    made: int
    checked: int
    right: int
    # Date of the oldest call still waiting for its outcome, if any.
    next_check_after: pd.Timestamp | None


def _connect(db_path: str | Path | None = None):
    return database_connection(DEFAULT_CALLS_DB, db_path)


def _initialize(db_path: str | Path | None = None) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_calls (
                market TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                as_of TEXT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                predicted_excess REAL NOT NULL,
                made_at TEXT NOT NULL,
                realized_excess REAL,
                checked_at TEXT,
                PRIMARY KEY (market, horizon, as_of, ticker)
            );
            """
        )


def _ensure_store(db_path: str | Path | None = None) -> None:
    ensure_schema("prediction_calls", DEFAULT_CALLS_DB, db_path, lambda: _initialize(db_path))


def record_calls(
    market: str,
    horizon: int,
    as_of: pd.Timestamp,
    calls: Iterable[Call],
    *,
    db_path: str | Path | None = None,
) -> None:
    """Save today's calls; a call already saved for this date is kept as first made."""
    rows = [
        (
            str(market),
            int(horizon),
            pd.Timestamp(as_of).date().isoformat(),
            call.ticker,
            call.direction,
            float(call.predicted_excess),
            pd.Timestamp.now(tz="UTC").isoformat(),
        )
        for call in calls
    ]
    if not rows:
        return
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO prediction_calls (
                market, horizon, as_of, ticker, direction, predicted_excess, made_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, horizon, as_of, ticker) DO NOTHING
            """,
            rows,
        )


def _date(text: str) -> pd.Timestamp:
    """Parse a stored ISO date; stored dates are never missing."""
    return cast(pd.Timestamp, pd.Timestamp(text))


def _log_return_after(history: pd.DataFrame, as_of: pd.Timestamp, sessions: int) -> float | None:
    """Log return from the close on `as_of` to the close `sessions` bars later."""
    frame = history.dropna(subset=["Close"])
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    position = np.flatnonzero(np.asarray(dates == as_of.normalize()))
    if position.size == 0:
        return None
    start = int(position[0])
    if start + sessions >= len(frame):
        return None
    close = np.asarray(frame["Close"], dtype=float)
    if close[start] <= 0 or close[start + sessions] <= 0:
        return None
    return float(np.log(close[start + sessions] / close[start]))


def check_calls(
    market: str,
    horizon: int,
    price_data: Mapping[str, pd.DataFrame],
    *,
    db_path: str | Path | None = None,
) -> int:
    """Settle every open call whose holding period has ended; return how many."""
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        open_calls = connection.execute(
            """
            SELECT as_of, ticker FROM prediction_calls
            WHERE market = ? AND horizon = ? AND checked_at IS NULL
            """,
            (str(market), int(horizon)),
        ).fetchall()
    if not open_calls:
        return 0
    settled = []
    market_returns: dict[str, float | None] = {}
    for row in open_calls:
        as_of_text, ticker = str(row["as_of"]), str(row["ticker"])
        as_of = _date(as_of_text)
        if as_of_text not in market_returns:
            returns = [
                value
                for value in (
                    _log_return_after(history, as_of, horizon) for history in price_data.values()
                )
                if value is not None
            ]
            # Need most of the market to have reported, like the model's target.
            enough = len(returns) >= max(2, int(0.8 * len(price_data)))
            market_returns[as_of_text] = float(np.mean(returns)) if enough else None
        market_return = market_returns[as_of_text]
        stock_return = (
            _log_return_after(price_data[ticker], as_of, horizon) if ticker in price_data else None
        )
        if market_return is None or stock_return is None:
            continue
        settled.append(
            (
                (stock_return - market_return) * 100.0,
                pd.Timestamp.now(tz="UTC").isoformat(),
                str(market),
                int(horizon),
                as_of_text,
                ticker,
            )
        )
    if settled:
        with _connect(db_path) as connection:
            connection.executemany(
                """
                UPDATE prediction_calls SET realized_excess = ?, checked_at = ?
                WHERE market = ? AND horizon = ? AND as_of = ? AND ticker = ?
                """,
                settled,
            )
        bac_log_kv("call_tracker.check", market=market, horizon=horizon, settled=len(settled))
    return len(settled)


def live_record(
    market: str,
    horizon: int,
    *,
    db_path: str | Path | None = None,
) -> LiveRecord:
    """Count calls made and how many of the checked ones came true."""
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT as_of, direction, realized_excess, checked_at FROM prediction_calls
            WHERE market = ? AND horizon = ?
            """,
            (str(market), int(horizon)),
        ).fetchall()
    checked = [row for row in rows if row["checked_at"] is not None]
    # A tie with the market makes neither call right.
    right = sum(
        1
        for row in checked
        if (
            float(row["realized_excess"]) > 0
            if row["direction"] == BEATS
            else float(row["realized_excess"]) < 0
        )
    )
    waiting = [_date(str(row["as_of"])) for row in rows if row["checked_at"] is None]
    return LiveRecord(
        made=len(rows),
        checked=len(checked),
        right=right,
        next_check_after=min(waiting) if waiting else None,
    )
