#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_market_snapshot_store.py
#############################

"""Offline regression tests for last-known-good market-data recovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import market_data
from market_snapshot_store import (
    load_price_history_snapshot,
    save_price_history_snapshot,
)


def _sample_history(periods: int = 60) -> pd.DataFrame:
    """Return deterministic OHLCV history with enough rows for forecasts."""
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    close = pd.Series(range(100, 100 + periods), dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


class MarketSnapshotStoreTest(unittest.TestCase):
    def test_snapshot_round_trip_preserves_history_and_marks_it_stale(self):
        """A complete provider response should be recoverable after an outage."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "market-snapshots.db"
            history = _sample_history()

            saved = save_price_history_snapshot(
                "AAPL",
                "1y",
                "1d",
                history,
                fetched_at="2026-07-23T12:00:00Z",
                db_path=database,
            )
            restored = load_price_history_snapshot(
                "AAPL",
                "1y",
                "1d",
                db_path=database,
            )

            self.assertEqual(len(history), saved)
            self.assertEqual(len(history), len(restored))
            self.assertEqual("last_known_good", restored.attrs["bac_data_status"])
            self.assertEqual(
                "2026-07-23T12:00:00+00:00",
                restored.attrs["bac_fetched_at"],
            )
            pd.testing.assert_series_equal(
                history["Close"],
                restored["Close"],
                check_names=False,
            )

    def test_batch_loader_uses_snapshot_when_batch_and_single_retry_fail(self):
        """Forecast inputs should survive a total provider outage."""
        stale = _sample_history()
        stale.attrs["bac_data_status"] = "last_known_good"
        stale.attrs["bac_fetched_at"] = "2026-07-23T12:00:00+00:00"

        with (
            patch.object(
                market_data,
                "call_provider",
                side_effect=TimeoutError("provider unavailable"),
            ),
            patch.object(
                market_data,
                "_load_price_snapshot_safely",
                return_value=stale,
            ) as load_snapshot,
        ):
            result = market_data._compute_price_history_batch(
                ["AAPL"],
                period="1y",
                interval="1d",
            )

        self.assertEqual(60, len(result["AAPL"]))
        self.assertEqual(
            "last_known_good",
            result["AAPL"].attrs["bac_data_status"],
        )
        load_snapshot.assert_called_once_with("AAPL", "1y", "1d")

    def test_freshness_detects_a_frozen_provider_payload(self):
        """Very old bars must not be presented as a current forecast origin."""
        history = _sample_history(periods=5)
        diagnosis = market_data.assess_price_history_freshness(
            history,
            realtime_mode=False,
            now="2026-07-23T12:00:00Z",
        )

        self.assertEqual("stale", diagnosis["status"])
        self.assertGreater(
            diagnosis["age_hours"],
            diagnosis["maximum_age_hours"],
        )

    def test_realtime_forecast_input_stays_fixed_inside_five_minute_bucket(self):
        """Minute-by-minute chart redraws should not refit an active model bar."""
        from views import prepare_realtime_forecast_history

        base = pd.Timestamp("2026-07-23 14:00:00")
        first = _sample_history(periods=65)
        first["Date"] = pd.date_range(base, periods=65, freq="min")
        second = pd.concat(
            [
                first,
                pd.DataFrame(
                    {
                        "Date": [base + pd.Timedelta(minutes=65)],
                        "Open": [165.0],
                        "High": [166.0],
                        "Low": [164.0],
                        "Close": [165.5],
                        "Volume": [1_000_000.0],
                    }
                ),
            ],
            ignore_index=True,
        )

        first_model = prepare_realtime_forecast_history(
            first,
            realtime_mode=True,
        )
        second_model = prepare_realtime_forecast_history(
            second,
            realtime_mode=True,
        )

        # 65 minutes ends at 15:04 and the appended bar is 15:05. The first
        # model excludes the active 15:00 bucket; the second admits it only
        # when the 15:05 bucket begins.
        self.assertEqual(base + pd.Timedelta(minutes=59), first_model["Date"].iloc[-1])
        self.assertEqual(base + pd.Timedelta(minutes=64), second_model["Date"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
