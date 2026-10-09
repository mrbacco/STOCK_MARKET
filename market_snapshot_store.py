#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: market_snapshot_store.py
#############################

"""Durable last-known-good OHLCV snapshots for provider-outage recovery.

The public market-data provider is an external dependency and will occasionally
rate-limit, time out, or return a partial batch. Forecasting should degrade
honestly during those incidents instead of becoming completely blank.

SQLite is used by the lightweight local app. The existing database adapter
automatically uses PostgreSQL when ``DATABASE_URL`` is configured, so the same
store also works across production replicas without changing this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app_logging import bac_debug_kv
from database import database_connection, ensure_schema


DEFAULT_MARKET_SNAPSHOT_DB = (
    Path(__file__).resolve().parent / "data" / "market_snapshots.db"
)
SNAPSHOT_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _connect(db_path: str | Path | None = None):
    """Open PostgreSQL in production or the explicit/local SQLite database."""
    return database_connection(DEFAULT_MARKET_SNAPSHOT_DB, db_path)


def _utc_iso(value: object | None = None) -> str:
    """Normalize timestamps so SQLite and PostgreSQL store the same text form."""
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _bar_timestamp_text(value: object) -> str:
    """Store market bars as timezone-naive ISO timestamps, matching the app."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def initialize_market_snapshot_store(
    db_path: str | Path | None = None,
) -> Path:
    """Create the snapshot table idempotently."""
    path = Path(db_path) if db_path is not None else DEFAULT_MARKET_SNAPSHOT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_price_snapshots (
                ticker TEXT NOT NULL,
                period_name TEXT NOT NULL,
                interval_name TEXT NOT NULL,
                bar_at TEXT NOT NULL,
                price_open REAL NOT NULL,
                price_high REAL NOT NULL,
                price_low REAL NOT NULL,
                price_close REAL NOT NULL,
                volume REAL NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (ticker, period_name, interval_name, bar_at)
            );

            -- The primary key already indexes these columns in this order; an
            -- older separate index duplicated it and doubled the index size.
            DROP INDEX IF EXISTS idx_market_price_snapshot_lookup;
            """
        )
    return path


def _ensure_store(db_path: str | Path | None = None) -> None:
    """Create the schema on first use per process and database target."""
    ensure_schema(
        "market_snapshots",
        DEFAULT_MARKET_SNAPSHOT_DB,
        db_path,
        lambda: initialize_market_snapshot_store(db_path),
    )


@dataclass(frozen=True)
class PriceSnapshot:
    """One ticker/period/interval history to persist as last-known-good."""

    ticker: str
    period: str
    interval: str
    history: pd.DataFrame
    fetched_at: object | None = None


def _snapshot_rows(snapshot: PriceSnapshot) -> list[tuple]:
    """Normalize one history into database rows; empty when it is unusable."""
    required = set(SNAPSHOT_REQUIRED_COLUMNS)
    history = snapshot.history
    if history.empty or not required.issubset(history.columns):
        bac_debug_kv(
            "market_snapshot.save",
            ticker=snapshot.ticker,
            period=snapshot.period,
            interval=snapshot.interval,
            status="skipped_invalid_history",
            rows=len(history),
        )
        return []

    frame = history.loc[:, SNAPSHOT_REQUIRED_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    frame["Volume"] = frame["Volume"].fillna(0.0).clip(lower=0.0)
    fetched_text = _utc_iso(snapshot.fetched_at)
    return [
        (
            str(snapshot.ticker).upper(),
            str(snapshot.period),
            str(snapshot.interval),
            _bar_timestamp_text(row.Date),
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            float(row.Volume),
            fetched_text,
        )
        for row in frame.itertuples(index=False)
    ]


def save_price_history_snapshots(
    snapshots: Iterable[PriceSnapshot],
    *,
    db_path: str | Path | None = None,
) -> int:
    """Atomically replace several snapshots in one database transaction.

    One commit for a whole batch avoids a connection and disk sync per
    ticker. Readers see either every earlier snapshot or every new one.
    """
    rows_by_key: dict[tuple[str, str, str], list[tuple]] = {}
    for snapshot in snapshots:
        rows = _snapshot_rows(snapshot)
        if rows:
            rows_by_key[rows[0][:3]] = rows
    if not rows_by_key:
        return 0

    _ensure_store(db_path)
    with _connect(db_path) as connection:
        for key, rows in rows_by_key.items():
            connection.execute(
                """
                DELETE FROM market_price_snapshots
                WHERE ticker = ? AND period_name = ? AND interval_name = ?
                """,
                key,
            )
            connection.executemany(
                """
                INSERT INTO market_price_snapshots (
                    ticker, period_name, interval_name, bar_at,
                    price_open, price_high, price_low, price_close,
                    volume, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    saved_rows = sum(len(rows) for rows in rows_by_key.values())
    bac_debug_kv(
        "market_snapshot.save",
        snapshots=len(rows_by_key),
        rows=saved_rows,
        status="saved",
    )
    return saved_rows


def save_price_history_snapshot(
    ticker: str,
    period: str,
    interval: str,
    history: pd.DataFrame,
    *,
    fetched_at: object | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Atomically replace one ticker/period/interval last-known-good snapshot."""
    return save_price_history_snapshots(
        [PriceSnapshot(ticker, period, interval, history, fetched_at)],
        db_path=db_path,
    )


def _snapshot_frame(rows: list[Any]) -> pd.DataFrame:
    """Convert stored rows into the app's OHLCV frame with provenance attrs."""
    frame = pd.DataFrame([dict(row) for row in rows]).rename(
        columns={
            "bar_at": "Date",
            "price_open": "Open",
            "price_high": "High",
            "price_low": "Low",
            "price_close": "Close",
            "volume": "Volume",
        }
    )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    fetched_at = str(frame["fetched_at"].iloc[-1])
    frame = frame.loc[:, SNAPSHOT_REQUIRED_COLUMNS]

    # pandas attrs travel with cache serialization and allow the view to label
    # stale inputs without changing every forecasting function's API.
    frame.attrs["bac_data_status"] = "last_known_good"
    frame.attrs["bac_fetched_at"] = fetched_at
    frame.attrs["bac_latest_bar"] = str(frame["Date"].iloc[-1])
    return frame


def load_price_history_snapshots(
    tickers: Iterable[str],
    period: str,
    interval: str,
    *,
    db_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load several snapshots with one query; missing tickers are omitted."""
    normalized_tickers = sorted({str(ticker).upper() for ticker in tickers})
    if not normalized_tickers:
        return {}
    _ensure_store(db_path)
    placeholders = ",".join("?" for _ in normalized_tickers)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT ticker, bar_at, price_open, price_high, price_low, price_close,
                   volume, fetched_at
            FROM market_price_snapshots
            WHERE period_name = ? AND interval_name = ? AND ticker IN ({placeholders})
            ORDER BY ticker, bar_at
            """,
            (str(period), str(interval), *normalized_tickers),
        ).fetchall()

    rows_by_ticker: dict[str, list[Any]] = {}
    for row in rows:
        rows_by_ticker.setdefault(str(row["ticker"]), []).append(row)
    snapshots = {
        ticker: _snapshot_frame(ticker_rows)
        for ticker, ticker_rows in rows_by_ticker.items()
    }
    bac_debug_kv(
        "market_snapshot.load",
        period=period,
        interval=interval,
        requested=len(normalized_tickers),
        found=len(snapshots),
    )
    return snapshots


def load_price_history_snapshot(
    ticker: str,
    period: str,
    interval: str,
    *,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a stale-but-usable snapshot and attach transparent provenance."""
    snapshots = load_price_history_snapshots([ticker], period, interval, db_path=db_path)
    return snapshots.get(str(ticker).upper(), pd.DataFrame())
