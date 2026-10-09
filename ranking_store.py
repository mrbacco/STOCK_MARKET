#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: ranking_store.py
#############################

"""Day-long persistence of the pooled ranking.

Training the pooled ensemble takes 15-20 seconds per market, yet daily data
only changes when a new bar appears. A ranking is therefore saved under a key
that includes the market, horizon, feature set, ticker set, and the first and
last price dates, and is reused until one of those changes (a new session
starts, the history window changes, or the model changes). Refresh data
clears a market's saved rankings. Tables are stored as Parquet and the
diagnostics as JSON, so nothing is unpickled.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app_logging import bac_log_kv
from database import database_connection, ensure_schema

DEFAULT_RANKING_DB = Path(__file__).resolve().parent / "data" / "ranking_cache.db"
# Older entries are pruned on save; a ranking is only reused on its own day.
KEEP_DAYS = 7


def _connect(db_path: str | Path | None = None):
    return database_connection(DEFAULT_RANKING_DB, db_path)


def _initialize(db_path: str | Path | None = None) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ranking_cache (
                cache_key TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                created_at TEXT NOT NULL,
                ranking_parquet TEXT NOT NULL,
                evaluation_parquet TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL
            );
            """
        )


def _ensure_store(db_path: str | Path | None = None) -> None:
    ensure_schema("ranking_cache", DEFAULT_RANKING_DB, db_path, lambda: _initialize(db_path))


def ranking_cache_key(
    market: str,
    horizon: int,
    model_label: str,
    price_data: Mapping[str, pd.DataFrame],
) -> str:
    """Identify a ranking by its market, horizon, model, tickers, and price span."""
    tickers = sorted(price_data)
    dates = [pd.to_datetime(frame["Date"]) for frame in price_data.values() if not frame.empty]
    first = min(series.min() for series in dates) if dates else pd.NaT
    last = max(series.max() for series in dates) if dates else pd.NaT
    fingerprint = hashlib.sha256(
        "|".join([*tickers, str(first), str(last)]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{market}|{int(horizon)}|{model_label}|{fingerprint}"


def _to_text(frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _from_text(text: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(base64.b64decode(text)))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def save_ranking(
    cache_key: str,
    market: str,
    *,
    ranking: pd.DataFrame,
    evaluation: pd.DataFrame,
    diagnostics: Mapping[str, object],
    db_path: str | Path | None = None,
) -> None:
    """Store one ranking and drop entries older than a week."""
    now = pd.Timestamp.now(tz="UTC")
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO ranking_cache (
                cache_key, market, created_at, ranking_parquet, evaluation_parquet,
                diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                created_at = excluded.created_at,
                ranking_parquet = excluded.ranking_parquet,
                evaluation_parquet = excluded.evaluation_parquet,
                diagnostics_json = excluded.diagnostics_json
            """,
            (
                cache_key,
                str(market),
                now.isoformat(),
                _to_text(ranking),
                _to_text(evaluation),
                json.dumps(dict(diagnostics), default=_json_default),
            ),
        )
        connection.execute(
            "DELETE FROM ranking_cache WHERE created_at < ?",
            ((now - pd.Timedelta(days=KEEP_DAYS)).isoformat(),),
        )
    bac_log_kv("ranking_store.save", market=market, rows=len(ranking))


def load_ranking(
    cache_key: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    """Return (ranking, evaluation, diagnostics) for a saved key, or None."""
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT ranking_parquet, evaluation_parquet, diagnostics_json
            FROM ranking_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    return (
        _from_text(row["ranking_parquet"]),
        _from_text(row["evaluation_parquet"]),
        json.loads(row["diagnostics_json"]),
    )


def clear_rankings(market: str, *, db_path: str | Path | None = None) -> None:
    """Forget a market's saved rankings, so the next view retrains."""
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM ranking_cache WHERE market = ?", (str(market),))
