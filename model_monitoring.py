#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: model_monitoring.py
#############################

"""Persistent forecast resolution, rolling quality metrics, and drift history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app_logging import bac_log_kv, bac_log_section


DEFAULT_MONITORING_DB = Path(__file__).resolve().parent / "data" / "model_monitoring.db"


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else DEFAULT_MONITORING_DB
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


def _utc_iso(value: object | None = None) -> str:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _naive_timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def initialize_monitoring_store(db_path: str | Path | None = None) -> Path:
    """Create forecast and model-run tables idempotently."""
    path = _resolve_db_path(db_path)
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS forecast_observations (
                market_source TEXT NOT NULL,
                ticker TEXT NOT NULL,
                forecast_origin TEXT NOT NULL,
                target_at TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                regime TEXT NOT NULL DEFAULT 'Unknown',
                origin_close REAL NOT NULL,
                predicted_close REAL NOT NULL,
                predicted_return REAL NOT NULL,
                lower_80 REAL,
                upper_80 REAL,
                sentiment_score REAL,
                created_at TEXT NOT NULL,
                actual_close REAL,
                actual_return REAL,
                absolute_error REAL,
                return_absolute_error REAL,
                direction_correct INTEGER,
                interval_hit_80 INTEGER,
                resolved_at TEXT,
                PRIMARY KEY (
                    market_source, ticker, forecast_origin, target_at, model_name
                )
            );

            CREATE INDEX IF NOT EXISTS idx_forecast_observations_pending
                ON forecast_observations (market_source, ticker, resolved_at, target_at);

            CREATE TABLE IF NOT EXISTS market_model_runs (
                market_source TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                as_of TEXT NOT NULL,
                candidate_tickers INTEGER,
                evaluation_dates INTEGER,
                evaluation_mae REAL,
                baseline_mae REAL,
                directional_accuracy REAL,
                probability_brier REAL,
                interval_coverage_80 REAL,
                selection_mean_excess REAL,
                selection_hit_rate REAL,
                sentiment_observed_rows INTEGER,
                model_weights_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                PRIMARY KEY (market_source, horizon, as_of)
            );
            """
        )
    return path


def record_forecast(
    *,
    market_source: str,
    ticker: str,
    forecast_origin: object,
    target_at: object,
    horizon: int,
    model_name: str,
    regime: str,
    origin_close: float,
    predicted_close: float,
    predicted_return: float,
    lower_80: float | None = None,
    upper_80: float | None = None,
    sentiment_score: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Persist the first displayed forecast for an origin without rewriting it."""
    initialize_monitoring_store(db_path)
    values = (
        str(market_source),
        str(ticker).upper(),
        _naive_timestamp_text(forecast_origin),
        _naive_timestamp_text(target_at),
        int(horizon),
        str(model_name),
        str(regime),
        float(origin_close),
        float(predicted_close),
        float(predicted_return),
        None if lower_80 is None or pd.isna(lower_80) else float(lower_80),
        None if upper_80 is None or pd.isna(upper_80) else float(upper_80),
        None if sentiment_score is None or pd.isna(sentiment_score) else float(sentiment_score),
        _utc_iso(),
    )
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO forecast_observations (
                market_source, ticker, forecast_origin, target_at, horizon,
                model_name, regime, origin_close, predicted_close,
                predicted_return, lower_80, upper_80, sentiment_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                market_source, ticker, forecast_origin, target_at, model_name
            ) DO NOTHING
            """,
            values,
        )
    bac_log_kv(
        "monitoring.record_forecast",
        market_source=market_source,
        ticker=ticker,
        horizon=horizon,
        model_name=model_name,
        target_at=_naive_timestamp_text(target_at),
    )


def _history_close_at_or_after(history: pd.DataFrame, target_at: pd.Timestamp) -> float | None:
    """Return the first realized bar at the target timestamp/session."""
    if history.empty or not {"Date", "Close"}.issubset(history.columns):
        return None
    dates = pd.to_datetime(history["Date"], errors="coerce")
    if target_at == target_at.normalize():
        mask = dates.dt.normalize() >= target_at.normalize()
    else:
        mask = dates >= target_at
    available = history.loc[mask, ["Date", "Close"]].dropna().sort_values("Date")
    if available.empty:
        return None
    return float(available["Close"].iloc[0])


def resolve_pending_forecasts(
    price_data: Mapping[str, pd.DataFrame],
    market_source: str,
    db_path: str | Path | None = None,
) -> int:
    """Resolve all pending forecasts whose target bar is now present."""
    initialize_monitoring_store(db_path)
    if not price_data:
        return 0
    tickers = [str(ticker).upper() for ticker in price_data]
    placeholders = ",".join("?" for _ in tickers)
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM forecast_observations
            WHERE market_source = ?
              AND resolved_at IS NULL
              AND ticker IN ({placeholders})
            ORDER BY target_at
            """,
            [str(market_source), *tickers],
        ).fetchall()

        resolved = 0
        for row in rows:
            ticker = str(row["ticker"])
            target_at = pd.Timestamp(row["target_at"])
            actual_close = _history_close_at_or_after(price_data[ticker], target_at)
            if actual_close is None:
                continue
            origin_close = float(row["origin_close"])
            predicted_close = float(row["predicted_close"])
            predicted_return = float(row["predicted_return"])
            actual_return = actual_close / origin_close - 1.0
            direction_correct = int(
                np.sign(predicted_return) == np.sign(actual_return)
            )
            lower_80 = row["lower_80"]
            upper_80 = row["upper_80"]
            interval_hit = (
                None
                if lower_80 is None or upper_80 is None
                else int(float(lower_80) <= actual_close <= float(upper_80))
            )
            connection.execute(
                """
                UPDATE forecast_observations
                SET actual_close = ?, actual_return = ?, absolute_error = ?,
                    return_absolute_error = ?, direction_correct = ?,
                    interval_hit_80 = ?, resolved_at = ?
                WHERE market_source = ? AND ticker = ? AND forecast_origin = ?
                  AND target_at = ? AND model_name = ?
                """,
                (
                    actual_close,
                    actual_return,
                    abs(actual_close - predicted_close),
                    abs(actual_return - predicted_return),
                    direction_correct,
                    interval_hit,
                    _utc_iso(),
                    row["market_source"],
                    ticker,
                    row["forecast_origin"],
                    row["target_at"],
                    row["model_name"],
                ),
            )
            resolved += 1
    bac_log_kv(
        "monitoring.resolve_pending",
        market_source=market_source,
        pending_rows=len(rows),
        resolved_rows=resolved,
    )
    return resolved


def record_market_model_run(
    market_source: str,
    horizon: int,
    as_of: object,
    diagnostics: Mapping[str, Any],
    db_path: str | Path | None = None,
) -> None:
    """Persist one pooled-model validation snapshot for drift inspection."""
    if not diagnostics:
        return
    initialize_monitoring_store(db_path)
    values = (
        str(market_source),
        int(horizon),
        _naive_timestamp_text(pd.Timestamp(as_of).normalize()),
        int(diagnostics.get("Candidate tickers", 0)),
        int(diagnostics.get("Evaluation dates", 0)),
        float(diagnostics.get("Evaluation MAE", np.nan)),
        float(diagnostics.get("Zero-excess baseline MAE", np.nan)),
        float(diagnostics.get("Directional accuracy", np.nan)),
        float(diagnostics.get("Probability Brier score", np.nan)),
        float(diagnostics.get("80% interval coverage", np.nan)),
        float(diagnostics.get("Top-10 realized mean excess", np.nan)),
        float(diagnostics.get("Top-10 realized hit rate", np.nan)),
        int(diagnostics.get("Sentiment-observed rows", 0)),
        json.dumps(diagnostics.get("Model weights", {}), sort_keys=True),
        _utc_iso(),
    )
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO market_model_runs (
                market_source, horizon, as_of, candidate_tickers,
                evaluation_dates, evaluation_mae, baseline_mae,
                directional_accuracy, probability_brier,
                interval_coverage_80, selection_mean_excess,
                selection_hit_rate, sentiment_observed_rows,
                model_weights_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_source, horizon, as_of) DO UPDATE SET
                candidate_tickers = excluded.candidate_tickers,
                evaluation_dates = excluded.evaluation_dates,
                evaluation_mae = excluded.evaluation_mae,
                baseline_mae = excluded.baseline_mae,
                directional_accuracy = excluded.directional_accuracy,
                probability_brier = excluded.probability_brier,
                interval_coverage_80 = excluded.interval_coverage_80,
                selection_mean_excess = excluded.selection_mean_excess,
                selection_hit_rate = excluded.selection_hit_rate,
                sentiment_observed_rows = excluded.sentiment_observed_rows,
                model_weights_json = excluded.model_weights_json,
                created_at = excluded.created_at
            """,
            values,
        )
    bac_log_kv(
        "monitoring.record_market_run",
        market_source=market_source,
        horizon=horizon,
        as_of=values[2],
        sentiment_observed_rows=values[12],
    )


def load_forecast_quality(
    market_source: str,
    horizon: int | None = None,
    limit: int = 250,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate resolved production forecasts by model, horizon, and regime."""
    initialize_monitoring_store(db_path)
    horizon_filter = "AND horizon = ?" if horizon is not None else ""
    parameters: list[Any] = [str(market_source)]
    if horizon is not None:
        parameters.append(int(horizon))
    parameters.append(int(limit))
    with _connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM forecast_observations
            WHERE market_source = ? AND resolved_at IS NOT NULL
            {horizon_filter}
            ORDER BY resolved_at DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    quality = (
        frame.groupby(["model_name", "horizon", "regime"], as_index=False)
        .agg(
            Forecasts=("ticker", "size"),
            MAE=("absolute_error", "mean"),
            Return_MAE=("return_absolute_error", "mean"),
            Directional_accuracy=("direction_correct", "mean"),
            Interval_coverage_80=("interval_hit_80", "mean"),
            Last_resolved=("resolved_at", "max"),
        )
    )
    quality["Return MAE"] = quality.pop("Return_MAE") * 100.0
    quality["Directional accuracy"] = quality.pop("Directional_accuracy") * 100.0
    quality["80% interval coverage"] = quality.pop("Interval_coverage_80") * 100.0
    quality = quality.rename(
        columns={
            "model_name": "Model",
            "horizon": "Horizon",
            "regime": "Regime",
            "Last_resolved": "Last resolved",
        }
    )
    bac_log_kv(
        "monitoring.load_forecast_quality",
        market_source=market_source,
        resolved_rows=len(frame),
        summary_rows=len(quality),
    )
    return quality


def load_market_model_history(
    market_source: str,
    horizon: int,
    limit: int = 60,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load recent pooled validation snapshots used to identify drift."""
    initialize_monitoring_store(db_path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT as_of, candidate_tickers, evaluation_dates, evaluation_mae,
                   baseline_mae, directional_accuracy, probability_brier,
                   interval_coverage_80, selection_mean_excess,
                   selection_hit_rate, sentiment_observed_rows,
                   model_weights_json
            FROM market_model_runs
            WHERE market_source = ? AND horizon = ?
            ORDER BY as_of DESC
            LIMIT ?
            """,
            (str(market_source), int(horizon), int(limit)),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    return frame.sort_values("as_of").reset_index(drop=True)


def latest_drift_summary(model_history: pd.DataFrame) -> dict[str, float]:
    """Compare the latest run with the median of earlier stored snapshots."""
    if model_history.empty or len(model_history) < 2:
        return {}
    latest = model_history.iloc[-1]
    reference = model_history.iloc[:-1].tail(20).median(numeric_only=True)
    drift = {
        "MAE drift": float(latest["evaluation_mae"] - reference["evaluation_mae"]),
        "Direction drift": float(
            latest["directional_accuracy"] - reference["directional_accuracy"]
        ),
        "Brier drift": float(
            latest["probability_brier"] - reference["probability_brier"]
        ),
        "Coverage drift": float(
            latest["interval_coverage_80"] - reference["interval_coverage_80"]
        ),
    }
    bac_log_kv("monitoring.latest_drift_summary", **drift)
    return drift
