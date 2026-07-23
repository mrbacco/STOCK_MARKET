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

from pathlib import Path

import pandas as pd

from app_logging import bac_log_kv
from database import database_connection


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

            CREATE INDEX IF NOT EXISTS idx_market_price_snapshot_lookup
                ON market_price_snapshots (
                    ticker, period_name, interval_name, bar_at
                );
            """
        )
    return path


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
    required = set(SNAPSHOT_REQUIRED_COLUMNS)
    if history.empty or not required.issubset(history.columns):
        bac_log_kv(
            "market_snapshot.save",
            ticker=ticker,
            period=period,
            interval=interval,
            status="skipped_invalid_history",
            rows=len(history),
        )
        return 0

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
    if frame.empty:
        return 0

    normalized_ticker = str(ticker).upper()
    fetched_text = _utc_iso(fetched_at)
    rows = [
        (
            normalized_ticker,
            str(period),
            str(interval),
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

    initialize_market_snapshot_store(db_path)
    with _connect(db_path) as connection:
        # Delete and insert occur in one database transaction. Readers therefore
        # see either the earlier complete snapshot or the new complete snapshot.
        connection.execute(
            """
            DELETE FROM market_price_snapshots
            WHERE ticker = ? AND period_name = ? AND interval_name = ?
            """,
            (normalized_ticker, str(period), str(interval)),
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

    bac_log_kv(
        "market_snapshot.save",
        ticker=normalized_ticker,
        period=period,
        interval=interval,
        rows=len(rows),
        latest_bar=rows[-1][3],
        fetched_at=fetched_text,
        status="saved",
    )
    return len(rows)


def load_price_history_snapshot(
    ticker: str,
    period: str,
    interval: str,
    *,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a stale-but-usable snapshot and attach transparent provenance."""
    initialize_market_snapshot_store(db_path)
    normalized_ticker = str(ticker).upper()
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT bar_at, price_open, price_high, price_low, price_close,
                   volume, fetched_at
            FROM market_price_snapshots
            WHERE ticker = ? AND period_name = ? AND interval_name = ?
            ORDER BY bar_at
            """,
            (normalized_ticker, str(period), str(interval)),
        ).fetchall()

    if not rows:
        bac_log_kv(
            "market_snapshot.load",
            ticker=normalized_ticker,
            period=period,
            interval=interval,
            status="miss",
        )
        return pd.DataFrame()

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
    fetched_at = str(frame.pop("fetched_at").iloc[-1])

    # pandas attrs travel with Streamlit/Redis serialization and allow the view
    # to label stale inputs without changing every forecasting function's API.
    frame.attrs["bac_data_status"] = "last_known_good"
    frame.attrs["bac_fetched_at"] = fetched_at
    frame.attrs["bac_latest_bar"] = str(frame["Date"].iloc[-1])
    bac_log_kv(
        "market_snapshot.load",
        ticker=normalized_ticker,
        period=period,
        interval=interval,
        rows=len(frame),
        latest_bar=frame.attrs["bac_latest_bar"],
        fetched_at=fetched_at,
        status="hit",
    )
    return frame.loc[:, SNAPSHOT_REQUIRED_COLUMNS]
