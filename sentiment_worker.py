#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_worker.py
#############################

"""Standalone 24/7 sentiment collector for use outside the Streamlit process."""

from __future__ import annotations

import argparse
import time

from app_config import SENTIMENT_COLLECTION_INTERVAL_SECONDS
from app_logging import bac_log_kv
from sentiment_service import collect_active_watchlist_once


def run_worker(once: bool = False, poll_seconds: int = SENTIMENT_COLLECTION_INTERVAL_SECONDS) -> None:
    """Run one collection or continue until the process is stopped."""
    interval = max(int(poll_seconds), 60)
    while True:
        try:
            result = collect_active_watchlist_once()
            bac_log_kv("sentiment.worker", **result)
        except Exception as ex:
            bac_log_kv("sentiment.worker", error=str(ex))
        if once:
            return
        time.sleep(interval)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously collect financial-news sentiment.")
    parser.add_argument("--once", action="store_true", help="Run one collection cycle and exit.")
    parser.add_argument(
        "--interval",
        type=int,
        default=SENTIMENT_COLLECTION_INTERVAL_SECONDS,
        help="Polling interval in seconds (minimum 60).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_worker(once=arguments.once, poll_seconds=arguments.interval)
