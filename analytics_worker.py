#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: analytics_worker.py
#############################

"""Standalone worker that precomputes pooled rankings and ticker backtests.

The Streamlit web tier should set ANALYTICS_READ_ONLY=true in a scaled
deployment.  This worker runs with it disabled, fills the shared Redis cache,
and records pooled-model diagnostics in PostgreSQL.  Local development does not
need the worker because cache misses are computed synchronously by default.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping

import pandas as pd

from app_config import MARKET_SOURCES, MAX_CHARTED_PERFORMERS
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from cache_control import (
    dequeue_analytics_jobs,
    finish_analytics_job,
    set_cache_scope,
)
from forecasting import backtest_forecast_model, forecast_feature_model
from market_data import classify_price_histories, get_price_history_batch
from market_model import rank_market_candidates
from market_sources import MARKET_SOURCE_REGISTRY, resolve_market_calendar
from model_monitoring import record_market_model_run
from runtime_config import (
    ANALYTICS_HORIZONS,
    ANALYTICS_INTERVAL_SECONDS,
    ANALYTICS_PERIODS,
)
from sentiment_store import load_sentiment_history


def precompute_market(source: str, period: str, horizon: int) -> dict[str, int]:
    """Warm one market/period/horizon variant and its top ticker analytics."""
    # Match the web tier's automatic-market scope exactly so scheduled warming
    # and on-demand jobs address the same Redis generations and result keys.
    set_cache_scope(f"{source}:{period}:1d")
    bac_log_kv(
        "analytics.worker.market",
        source=source,
        period=period,
        horizon=horizon,
        status="started",
    )
    performers = MARKET_SOURCE_REGISTRY[source].load_performers()
    tickers = performers.get("Ticker", pd.Series(dtype=str)).astype(str).tolist()
    price_data = get_price_history_batch(tickers, period=period, interval="1d")
    # Rank exactly the candidate pool the web tier ranks (fresh histories only)
    # so both processes address the same shared-cache entry.
    health = classify_price_histories(price_data, realtime_mode=False)
    valid_price_data = {ticker: price_data[ticker] for ticker in health.live_tickers}
    sentiment_by_ticker = {
        ticker: load_sentiment_history(ticker)
        for ticker in valid_price_data
    }
    result = rank_market_candidates(
        valid_price_data,
        forecast_horizon=horizon,
        sentiment_by_ticker=sentiment_by_ticker,
        top_n=MAX_CHARTED_PERFORMERS,
    )
    ranking = result.get("ranking", pd.DataFrame())
    diagnostics = result.get("diagnostics", {})
    top_tickers = (
        ranking["Ticker"].astype(str).tolist()
        if not ranking.empty
        else list(valid_price_data)[:MAX_CHARTED_PERFORMERS]
    )
    bac_log_list_preview("analytics.worker.market", "top_tickers", top_tickers)

    for ticker in top_tickers:
        history = valid_price_data[ticker]
        calendar_name = resolve_market_calendar(source, ticker)
        # Warm both model variants. The UI will choose the sentiment model only
        # after the same paired, out-of-sample promotion rule it already uses.
        forecast_feature_model(history, horizon, market_calendar=calendar_name)
        backtest_forecast_model(history, horizon, market_calendar=calendar_name)
        sentiment_history = sentiment_by_ticker.get(ticker, pd.DataFrame())
        if not sentiment_history.empty:
            forecast_feature_model(
                history,
                horizon,
                sentiment_history=sentiment_history,
                include_sentiment=True,
                market_calendar=calendar_name,
            )
            backtest_forecast_model(
                history,
                horizon,
                sentiment_history=sentiment_history,
                include_sentiment=True,
                market_calendar=calendar_name,
            )

    if diagnostics and valid_price_data:
        ranking_as_of = max(
            pd.Timestamp(frame["Date"].max())
            for frame in valid_price_data.values()
        )
        record_market_model_run(source, horizon, ranking_as_of, diagnostics)

    summary = {
        "candidates": len(valid_price_data),
        "ranked": len(ranking),
        "warmed_tickers": len(top_tickers),
    }
    bac_log_kv("analytics.worker.market", source=source, period=period, horizon=horizon, **summary)
    return summary


def process_requested_jobs(limit: int = 10) -> int:
    """Compute user-requested cache misses outside the Streamlit web process."""
    handlers = {
        "market-ranking": rank_market_candidates,
        "forecast-curve": forecast_feature_model,
        "forecast-backtest": backtest_forecast_model,
    }
    jobs = dequeue_analytics_jobs(limit=limit)
    for job in jobs:
        job_type = str(job.get("job_type", ""))
        scope = str(job.get("scope", "default"))
        arguments = job.get("arguments", {})
        set_cache_scope(scope)
        try:
            handler = handlers.get(job_type)
            if handler is None:
                raise ValueError(f"Unsupported analytics job type: {job_type}")
            # Jobs carry the cached function's bound arguments by name.
            # Positional tuples are still accepted from older queue entries.
            if isinstance(arguments, Mapping):
                handler(**arguments)
            else:
                handler(*arguments)
            bac_log_kv(
                "analytics.worker.request",
                job_type=job_type,
                scope=scope,
                status="completed",
            )
        except Exception as ex:
            bac_log_kv(
                "analytics.worker.request",
                job_type=job_type,
                scope=scope,
                status="failed",
                error=str(ex),
            )
        finally:
            # Failure removes the marker as well, allowing a later rerun to
            # enqueue a fresh attempt after provider/model conditions improve.
            finish_analytics_job(job)
    return len(jobs)


def run_cycle() -> None:
    """Warm every configured production variant, continuing after isolated failures."""
    for source in MARKET_SOURCES:
        for period in ANALYTICS_PERIODS:
            for horizon in ANALYTICS_HORIZONS:
                try:
                    precompute_market(source, period, horizon)
                except Exception as ex:
                    bac_log_kv(
                        "analytics.worker.market",
                        source=source,
                        period=period,
                        horizon=horizon,
                        status="failed",
                        error=str(ex),
                    )
                # User-requested variants take priority between scheduled warm
                # jobs instead of waiting for the entire market matrix to finish.
                process_requested_jobs(limit=2)


def run_worker(once: bool = False, interval: int = ANALYTICS_INTERVAL_SECONDS) -> None:
    """Run one warm-up cycle or continue at a bounded production cadence."""
    poll_seconds = max(int(interval), 60)
    next_scheduled_cycle = 0.0
    while True:
        processed = process_requested_jobs(limit=10)
        now = time.monotonic()
        if now >= next_scheduled_cycle:
            started = now
            bac_log_section("analytics.worker", "Precomputation cycle started.")
            run_cycle()
            elapsed = time.monotonic() - started
            next_scheduled_cycle = started + poll_seconds
            bac_log_kv("analytics.worker", status="cycle_complete", elapsed_seconds=elapsed)
        if once:
            return
        # Polling the lightweight Redis list frequently keeps cold UI variants
        # responsive; scheduled full-market work retains the slower cadence.
        if processed == 0:
            time.sleep(5.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute market rankings and backtests.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--interval",
        type=int,
        default=ANALYTICS_INTERVAL_SECONDS,
        help="Seconds between cycle starts (minimum 60).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_worker(once=arguments.once, interval=arguments.interval)
