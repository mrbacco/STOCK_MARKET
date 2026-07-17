#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_store.py
#############################

"""SQLite persistence for point-in-time news sentiment and the watchlist."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from app_config import SENTIMENT_MAX_WATCHLIST
from app_logging import bac_log_kv

DEFAULT_SENTIMENT_DB = Path(__file__).resolve().parent / "data" / "sentiment.db"


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else DEFAULT_SENTIMENT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(db_path: str | Path | None = None):
    connection = sqlite3.connect(_resolve_db_path(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _utc_iso(value: Any | None = None) -> str:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def initialize_sentiment_store(db_path: str | Path | None = None) -> Path:
    """Create the database and indexes idempotently."""
    path = _resolve_db_path(db_path)
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS news_sentiment (
                ticker TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                company_name TEXT NOT NULL DEFAULT '',
                published_at TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                link TEXT NOT NULL DEFAULT '',
                sentiment REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                positive_probability REAL NOT NULL,
                neutral_probability REAL NOT NULL,
                negative_probability REAL NOT NULL,
                model_name TEXT NOT NULL,
                PRIMARY KEY (ticker, content_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_time
                ON news_sentiment (ticker, published_at, first_seen_at);

            CREATE TABLE IF NOT EXISTS sentiment_watchlist (
                ticker TEXT PRIMARY KEY,
                company_name TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sentiment_collector_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
    return path


def update_watchlist(
    company_by_ticker: Mapping[str, str],
    db_path: str | Path | None = None,
    max_tickers: int = SENTIMENT_MAX_WATCHLIST,
) -> None:
    """Add or refresh tickers and retain the most recently used bounded set."""
    initialize_sentiment_store(db_path)
    updated_at = _utc_iso()
    clean_items = [
        (str(ticker).upper().strip(), str(company).strip(), updated_at)
        for ticker, company in company_by_ticker.items()
        if str(ticker).strip()
    ]
    if not clean_items:
        return

    with _connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO sentiment_watchlist (ticker, company_name, active, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                active = 1,
                updated_at = excluded.updated_at
            """,
            clean_items,
        )
        connection.execute(
            """
            UPDATE sentiment_watchlist
            SET active = 0
            WHERE ticker NOT IN (
                SELECT ticker
                FROM sentiment_watchlist
                ORDER BY active DESC, updated_at DESC
                LIMIT ?
            )
            """,
            (max_tickers,),
        )
    bac_log_kv("sentiment.store.watchlist", refreshed=len(clean_items), max_tickers=max_tickers)


def load_active_watchlist(db_path: str | Path | None = None) -> dict[str, str]:
    """Return active tickers ordered from most recently refreshed."""
    initialize_sentiment_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT ticker, company_name
            FROM sentiment_watchlist
            WHERE active = 1
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return {str(row["ticker"]): str(row["company_name"]) for row in rows}


def existing_content_hashes(
    ticker: str,
    content_hashes: Iterable[str],
    db_path: str | Path | None = None,
) -> set[str]:
    """Return hashes already persisted for one ticker."""
    hashes = [str(value) for value in content_hashes]
    if not hashes:
        return set()
    initialize_sentiment_store(db_path)
    placeholders = ",".join("?" for _ in hashes)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT content_hash
            FROM news_sentiment
            WHERE ticker = ? AND content_hash IN ({placeholders})
            """,
            [ticker, *hashes],
        ).fetchall()
    return {str(row["content_hash"]) for row in rows}


def save_news_sentiment(
    rows: Iterable[Mapping[str, Any]],
    db_path: str | Path | None = None,
) -> int:
    """Upsert scored headlines while preserving their original first-seen time."""
    prepared_rows = []
    for row in rows:
        prepared_rows.append(
            (
                str(row["ticker"]),
                str(row["content_hash"]),
                str(row.get("company_name", "")),
                _utc_iso(row["published_at"]),
                _utc_iso(row["first_seen_at"]),
                _utc_iso(row.get("last_seen_at", row["first_seen_at"])),
                str(row.get("source", "")),
                str(row["title"]),
                str(row.get("summary", "")),
                str(row.get("link", "")),
                float(row["sentiment"]),
                str(row["sentiment_label"]),
                float(row["positive_probability"]),
                float(row["neutral_probability"]),
                float(row["negative_probability"]),
                str(row["model_name"]),
            )
        )
    if not prepared_rows:
        return 0

    initialize_sentiment_store(db_path)
    with _connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO news_sentiment (
                ticker, content_hash, company_name, published_at, first_seen_at,
                last_seen_at, source, title, summary, link, sentiment,
                sentiment_label, positive_probability, neutral_probability,
                negative_probability, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, content_hash) DO UPDATE SET
                last_seen_at = excluded.last_seen_at,
                source = excluded.source,
                link = excluded.link,
                sentiment = excluded.sentiment,
                sentiment_label = excluded.sentiment_label,
                positive_probability = excluded.positive_probability,
                neutral_probability = excluded.neutral_probability,
                negative_probability = excluded.negative_probability,
                model_name = excluded.model_name
            """,
            prepared_rows,
        )
    bac_log_kv("sentiment.store.save", rows=len(prepared_rows))
    return len(prepared_rows)


def load_sentiment_history(
    ticker: str,
    db_path: str | Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load persisted point-in-time sentiment for features or display."""
    initialize_sentiment_store(db_path)
    limit_clause = "LIMIT ?" if limit is not None else ""
    parameters: list[Any] = [ticker]
    if limit is not None:
        parameters.append(int(limit))
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT ticker, company_name, published_at, first_seen_at, last_seen_at,
                   source, title, summary, link, sentiment, sentiment_label,
                   positive_probability, neutral_probability,
                   negative_probability, model_name, content_hash
            FROM news_sentiment
            WHERE ticker = ?
            ORDER BY published_at DESC
            {limit_clause}
            """,
            parameters,
        ).fetchall()

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    for column in ("published_at", "first_seen_at", "last_seen_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
    return frame.sort_values(["published_at", "first_seen_at"]).reset_index(drop=True)


def set_collector_state(
    state_key: str,
    state_value: Any,
    db_path: str | Path | None = None,
) -> None:
    """Persist lightweight collector health information for the UI."""
    initialize_sentiment_store(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO sentiment_collector_state (state_key, state_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = excluded.updated_at
            """,
            (state_key, str(state_value), _utc_iso()),
        )


def get_collector_status(db_path: str | Path | None = None) -> dict[str, Any]:
    """Return collection health plus persisted article and watchlist counts."""
    initialize_sentiment_store(db_path)
    with _connect(db_path) as connection:
        state_rows = connection.execute(
            "SELECT state_key, state_value, updated_at FROM sentiment_collector_state"
        ).fetchall()
        article_count = int(connection.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0])
        watchlist_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM sentiment_watchlist WHERE active = 1"
            ).fetchone()[0]
        )
    status = {str(row["state_key"]): str(row["state_value"]) for row in state_rows}
    status["article_count"] = article_count
    status["watchlist_count"] = watchlist_count
    return status
