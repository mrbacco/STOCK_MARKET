#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: walk_forward_store.py
#############################

"""Persistence and background execution of walk-forward tests.

A walk-forward run takes tens of seconds to minutes, so it runs in a
background thread and its latest result per universe and horizon is stored in
PostgreSQL or local SQLite. Predictions are stored as compressed Parquet (no
pickled objects), the summary and fold table as JSON.
"""

from __future__ import annotations

import base64
import contextvars
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app_logging import bac_log_kv
from database import database_connection, ensure_schema

DEFAULT_WALK_FORWARD_DB = Path(__file__).resolve().parent / "data" / "walk_forward.db"
WALK_FORWARD_HISTORY = "5y"


def _connect(db_path: str | Path | None = None):
    return database_connection(DEFAULT_WALK_FORWARD_DB, db_path)


def initialize_walk_forward_store(db_path: str | Path | None = None) -> None:
    with _connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS walk_forward_runs (
                universe TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                folds_json TEXT NOT NULL,
                predictions_parquet TEXT NOT NULL,
                PRIMARY KEY (universe, horizon)
            );
            """
        )


def _ensure_store(db_path: str | Path | None = None) -> None:
    ensure_schema(
        "walk_forward",
        DEFAULT_WALK_FORWARD_DB,
        db_path,
        lambda: initialize_walk_forward_store(db_path),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class StoredWalkForward:
    universe: str
    horizon: int
    created_at: str
    summary: dict[str, Any]
    folds: pd.DataFrame
    predictions: pd.DataFrame


def save_walk_forward(
    universe: str,
    horizon: int,
    *,
    summary: dict[str, Any],
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    db_path: str | Path | None = None,
) -> None:
    """Replace the stored result for one universe and horizon."""
    buffer = io.BytesIO()
    predictions.to_parquet(buffer, index=False, compression="zstd")
    values = (
        str(universe),
        int(horizon),
        pd.Timestamp.now(tz="UTC").isoformat(),
        json.dumps(summary, default=_json_default),
        folds.to_json(orient="records", date_format="iso", default_handler=str),
        base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    _ensure_store(db_path)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO walk_forward_runs (
                universe, horizon, created_at, summary_json, folds_json, predictions_parquet
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(universe, horizon) DO UPDATE SET
                created_at = excluded.created_at,
                summary_json = excluded.summary_json,
                folds_json = excluded.folds_json,
                predictions_parquet = excluded.predictions_parquet
            """,
            values,
        )
    bac_log_kv("walk_forward.store", universe=universe, horizon=horizon, rows=len(predictions))


def load_walk_forward(
    universe: str,
    horizon: int,
    *,
    with_predictions: bool = True,
    db_path: str | Path | None = None,
) -> StoredWalkForward | None:
    """Return the latest stored result, or None if this pair was never tested."""
    _ensure_store(db_path)
    columns = "created_at, summary_json, folds_json" + (
        ", predictions_parquet" if with_predictions else ""
    )
    with _connect(db_path) as connection:
        row = connection.execute(
            f"SELECT {columns} FROM walk_forward_runs WHERE universe = ? AND horizon = ?",
            (str(universe), int(horizon)),
        ).fetchone()
    if row is None:
        return None
    predictions = pd.DataFrame()
    if with_predictions:
        predictions = pd.read_parquet(io.BytesIO(base64.b64decode(row["predictions_parquet"])))
    return StoredWalkForward(
        universe=str(universe),
        horizon=int(horizon),
        created_at=str(row["created_at"]),
        summary=json.loads(row["summary_json"]),
        folds=pd.DataFrame(json.loads(row["folds_json"])),
        predictions=predictions,
    )


# --- Background execution ---------------------------------------------------------


@dataclass
class WalkForwardStatus:
    running: bool = False
    progress: float = 0.0
    message: str = ""
    error: str = ""


_STATUS: dict[tuple[str, int], WalkForwardStatus] = {}
_STATUS_LOCK = threading.Lock()


def walk_forward_status(universe: str, horizon: int) -> WalkForwardStatus:
    with _STATUS_LOCK:
        status = _STATUS.get((universe, int(horizon)), WalkForwardStatus())
        return WalkForwardStatus(status.running, status.progress, status.message, status.error)


def _update(key: tuple[str, int], **changes: Any) -> None:
    with _STATUS_LOCK:
        status = _STATUS.setdefault(key, WalkForwardStatus())
        for name, value in changes.items():
            setattr(status, name, value)


def _execute(universe: str, horizon: int) -> None:
    from market_data import classify_price_histories, get_price_history_batch
    from market_sources import MARKET_SOURCE_REGISTRY
    from sentiment_store import load_sentiment_history
    from walk_forward import run_walk_forward

    key = (universe, int(horizon))
    try:
        _update(key, message="Loading five years of prices", progress=0.0)
        source = MARKET_SOURCE_REGISTRY[universe]
        prices = get_price_history_batch(source.tickers, period=WALK_FORWARD_HISTORY, interval="1d")
        health = classify_price_histories(prices, realtime_mode=False)
        live = {ticker: prices[ticker] for ticker in health.live_tickers}
        sentiment = {ticker: load_sentiment_history(ticker) for ticker in live}
        result = run_walk_forward(
            live,
            sentiment,
            horizon=horizon,
            progress=lambda fraction, message: _update(key, progress=fraction, message=message),
        )
        if result.predictions.empty:
            raise RuntimeError("Not enough history for a walk-forward test.")
        save_walk_forward(
            universe,
            horizon,
            summary=result.summary,
            folds=result.folds,
            predictions=result.predictions,
        )
        _update(key, running=False, progress=1.0, message="Done", error="")
    except Exception as ex:
        bac_log_kv("walk_forward.run", universe=universe, horizon=horizon, error=str(ex))
        _update(key, running=False, message="Failed", error=str(ex))


def start_walk_forward(universe: str, horizon: int) -> bool:
    """Start a background run unless one is already running for this pair."""
    key = (universe, int(horizon))
    with _STATUS_LOCK:
        current = _STATUS.get(key)
        if current is not None and current.running:
            return False
        _STATUS[key] = WalkForwardStatus(running=True, message="Starting")
    context = contextvars.copy_context()
    threading.Thread(
        target=context.run,
        args=(_execute, universe, int(horizon)),
        name=f"walk-forward-{universe}-{horizon}",
        daemon=True,
    ).start()
    return True
