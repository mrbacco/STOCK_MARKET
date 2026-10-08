#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_startup_performance.py
#############################

"""Tests for the startup-latency optimizations."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import market_data
import sentiment_analysis
import sentiment_service
from market_snapshot_store import (
    PriceSnapshot,
    load_price_history_snapshots,
    save_price_history_snapshots,
)


def _history(close: float, fetched_at: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-10-06", "2026-10-07"]),
            "Open": [close, close],
            "High": [close, close],
            "Low": [close, close],
            "Close": [close * 0.98, close],
            "Volume": [1_000.0, 1_000.0],
        }
    )
    if fetched_at is not None:
        frame.attrs["bac_fetched_at"] = fetched_at
    return frame


class StartupPerformanceTest(unittest.TestCase):
    def test_snapshot_batches_round_trip_in_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "snapshots.db"
            saved = save_price_history_snapshots(
                [
                    PriceSnapshot("aaa", "5d", "1d", _history(10.0)),
                    PriceSnapshot("BBB", "5d", "1d", _history(20.0)),
                ],
                db_path=database,
            )
            loaded = load_price_history_snapshots(
                ["AAA", "BBB", "MISSING"], "5d", "1d", db_path=database
            )

        self.assertEqual(4, saved)
        self.assertEqual({"AAA", "BBB"}, set(loaded))
        self.assertEqual(20.0, float(loaded["BBB"]["Close"].iloc[-1]))
        self.assertEqual("last_known_good", loaded["AAA"].attrs["bac_data_status"])

    def test_collector_waits_for_its_startup_delay(self) -> None:
        collected = threading.Event()
        with patch.object(
            sentiment_service,
            "collect_active_watchlist_once",
            side_effect=lambda db_path=None: collected.set(),
        ):
            collector = sentiment_service.BackgroundSentimentCollector(
                startup_delay_seconds=0.5
            )
            collector.request_collection()
            try:
                self.assertFalse(collected.wait(0.2))
                self.assertTrue(collected.wait(2.0))
            finally:
                collector.stop()

    def test_cached_finbert_loads_without_the_hub_and_retries_online(self) -> None:
        calls = []

        def factory(task, **kwargs):
            calls.append(kwargs.get("local_files_only", False))
            if kwargs.get("local_files_only"):
                raise OSError("incomplete cache")
            return object()

        with patch.object(sentiment_analysis, "_model_is_cached", return_value=True):
            analyzer = sentiment_analysis.FinancialSentimentAnalyzer(pipeline_factory=factory)

        self.assertEqual([True, False], calls)
        self.assertEqual(sentiment_analysis.SENTIMENT_MODEL_NAME, analyzer.active_model_name)

    def test_first_leaderboard_uses_snapshots_then_live_prices(self) -> None:
        listings = {"AAA": "A plc", "BBB": "B plc"}
        recent = pd.Timestamp.now(tz="UTC").isoformat()
        snapshots = {ticker: _history(10.0, recent) for ticker in listings}
        live = {ticker: _history(50.0) for ticker in listings}
        live_loads = []

        def live_loader(tickers, period, interval):
            time.sleep(0.1)
            live_loads.append(tuple(tickers))
            return live

        market_data._PREVIEW_REFRESH_STATE.clear()
        with (
            patch.object(market_data, "_load_price_snapshots_safely", return_value=snapshots),
            patch.object(market_data, "get_price_history_batch", side_effect=live_loader),
        ):
            preview = market_data._rank_latest_daily_performers(listings, 10, "tests.preview")
            self.assertEqual(
                market_data.SNAPSHOT_PREVIEW_STATUS, preview.attrs["bac_data_status"]
            )
            self.assertEqual(10.0, float(preview["Last price"].iloc[0]))

            deadline = time.monotonic() + 5
            while market_data.snapshot_refresh_pending() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(market_data.snapshot_refresh_pending())

            refreshed = market_data._rank_latest_daily_performers(listings, 10, "tests.preview")
        market_data._PREVIEW_REFRESH_STATE.clear()

        self.assertNotIn("bac_data_status", refreshed.attrs)
        self.assertEqual(50.0, float(refreshed["Last price"].iloc[0]))
        self.assertEqual(2, len(live_loads))
        self.assertFalse(market_data._is_live_leaderboard(preview))

    def test_old_snapshots_are_not_used_as_a_preview(self) -> None:
        stale = {"AAA": _history(10.0, "2026-01-01T00:00:00+00:00")}
        with patch.object(market_data, "_load_price_snapshots_safely", return_value=stale):
            self.assertIsNone(market_data._snapshot_preview(["AAA"], "5d", "1d"))

    def test_app_shell_import_does_not_load_the_model_stack(self) -> None:
        script = (
            "import sys, ui_state, ui_components, global_markets, market_sources; "
            "print('LOADED=' + ','.join(m for m in ('sklearn', 'chart_pipeline', "
            "'pandas_market_calendars') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            timeout=120,
        )
        self.assertEqual(0, result.returncode, result.stderr[-2000:])
        loaded = [line for line in result.stdout.splitlines() if line.startswith("LOADED=")]
        self.assertEqual(["LOADED="], loaded)


if __name__ == "__main__":
    unittest.main()
