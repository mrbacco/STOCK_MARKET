#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_scalability_runtime.py
#############################

"""Offline regression tests for the scalable runtime adapters."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cache_control
import database
import forecasting
import market_data
import pandas as pd
from database import database_connection
from provider_runtime import call_provider


def _dated_history(periods: int = 30) -> pd.DataFrame:
    """Return a small daily OHLCV frame dated well in the past."""
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    closes = [100.0 + index for index in range(periods)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": 1_000_000.0,
        }
    )


class ScalabilityRuntimeTest(unittest.TestCase):
    def test_cache_generation_invalidation_is_scoped(self):
        """Refreshing one market must not invalidate an unrelated market."""
        with patch.object(cache_control, "get_redis_client", return_value=None):
            ireland_scope = "market:Ireland: ISEQ 20 leaders"
            italy_scope = "market:Italy: FTSE MIB leaders"
            ireland_before = cache_control.get_cache_generation(ireland_scope)
            italy_before = cache_control.get_cache_generation(italy_scope)

            ireland_after = cache_control.bump_cache_generation(ireland_scope)

            self.assertEqual(ireland_after, ireland_before + 1)
            self.assertEqual(
                cache_control.get_cache_generation(italy_scope),
                italy_before,
            )

    def test_explicit_database_path_keeps_sqlite_test_isolation(self):
        """An explicit path must never be redirected to a production database."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runtime.db"
            with database_connection(path, path) as connection:
                self.assertEqual(connection.backend, "sqlite")
                connection.execute("CREATE TABLE probe (value TEXT NOT NULL)")
                connection.execute("INSERT INTO probe (value) VALUES (?)", ["ok"])
            with database_connection(path, path) as connection:
                row = connection.execute("SELECT value FROM probe").fetchone()
                self.assertEqual(row["value"], "ok")

    def test_provider_call_retries_a_transient_failure(self):
        """A bounded retry should recover without changing the provider result."""
        attempts = []

        def flaky_operation():
            attempts.append(1)
            if len(attempts) == 1:
                raise TimeoutError("temporary")
            return "ready"

        with (
            patch("provider_runtime.get_redis_client", return_value=None),
            patch("provider_runtime.time.sleep", return_value=None),
            patch("provider_runtime.random.uniform", return_value=0.0),
        ):
            result = call_provider(
                "test-provider-scalability",
                "test-operation",
                flaky_operation,
                attempts=2,
            )

        self.assertEqual(result, "ready")
        self.assertEqual(len(attempts), 2)

    def test_forecast_curve_can_compute_on_the_first_web_render(self):
        """A cold curve must not leave Streamlit waiting for a nonexistent rerun."""
        history = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01")], "Close": [100.0]})
        expected = pd.DataFrame({"pred_close": [101.0]})
        forecasting.forecast_feature_model.cache_clear()
        with patch.object(
            cache_control,
            "shared_cache_get_or_compute",
            return_value=expected,
        ) as shared_cache:
            result = forecasting.forecast_feature_model(history, 3)
        forecasting.forecast_feature_model.cache_clear()

        pd.testing.assert_frame_equal(result, expected)
        self.assertTrue(shared_cache.call_args.kwargs["allow_compute"])

    def test_yfinance_cache_uses_a_verified_writable_directory(self):
        """Provider metadata must not depend on an unwritable user cache."""
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(market_data.yf, "set_tz_cache_location") as set_location,
        ):
            cache_path = Path(temporary_directory) / "provider-cache"
            configured = market_data.configure_yfinance_cache(cache_path)

            self.assertEqual(cache_path, configured)
            self.assertTrue(cache_path.is_dir())
            self.assertFalse((cache_path / ".bac-yfinance-write-probe").exists())
            set_location.assert_called_once_with(str(cache_path))

    def test_forecast_execution_failure_degrades_to_an_empty_result(self):
        """One estimator/cache failure must not crash the entire charts page."""
        history = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-01-01")],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [1_000_000.0],
            }
        )
        forecasting.forecast_feature_model.cache_clear()
        with patch.object(
            cache_control,
            "shared_cache_get_or_compute",
            side_effect=RuntimeError("broken model artifact"),
        ):
            result = forecasting.forecast_feature_model(history, points_ahead=3)

        self.assertTrue(result.empty)

    def test_empty_provider_history_has_an_actionable_diagnosis(self):
        """An empty forecast should say that market data, not AI, is missing."""
        diagnosis = forecasting.diagnose_forecast_readiness(pd.DataFrame())

        self.assertEqual("invalid_history", diagnosis["status"])
        self.assertIn("market-data provider", diagnosis["message"])


    def test_shared_cache_key_ignores_provenance_and_forming_bar(self):
        """A re-download of the same history must address the same shared entry."""
        first = _dated_history()
        first.attrs["bac_fetched_at"] = "2026-07-23T12:00:00+00:00"
        redownload = first.copy()
        redownload.attrs["bac_fetched_at"] = "2026-07-23T12:01:00+00:00"
        redownload.loc[redownload.index[-1], "Close"] += 0.5

        stable = lambda frame: cache_control._material_digest(frame, stable=True)
        exact = lambda frame: cache_control._material_digest(frame, stable=False)
        self.assertEqual(stable(first), stable(redownload))
        self.assertNotEqual(exact(first), exact(redownload))

        revised_history = first.copy()
        revised_history.loc[revised_history.index[0], "Close"] += 1.0
        next_session = pd.concat([first, first.tail(1).assign(Date=pd.Timestamp("2026-02-01"))])
        self.assertNotEqual(stable(first), stable(revised_history))
        self.assertNotEqual(stable(first), stable(next_session))

    def test_read_only_miss_queues_named_arguments_and_degrades(self):
        """A cold read-only entry is queued once by name and returns the fallback."""
        computed = []

        @cache_control.cached_result(
            "test-read-only",
            ttl_seconds=60,
            max_entries=4,
            allow_compute=False,
            on_failure=lambda: "pending",
        )
        def expensive(history, horizon=3):
            computed.append(horizon)
            return "ready"

        with (
            patch.object(cache_control, "get_redis_client", return_value=None),
            patch.object(cache_control, "enqueue_analytics_job") as enqueue,
        ):
            result = expensive(_dated_history())

        self.assertEqual("pending", result)
        self.assertEqual([], computed)
        job_type, _scope, arguments = enqueue.call_args.args
        self.assertEqual("test-read-only", job_type)
        self.assertEqual({"history", "horizon"}, set(arguments))

    def test_local_cache_returns_independent_copies(self):
        """Mutating a cached result must not leak into the next caller."""
        calls = []

        @cache_control.cached_result(
            "test-local-copy",
            ttl_seconds=60,
            max_entries=4,
            shared=False,
        )
        def load(ticker):
            calls.append(ticker)
            return {ticker: _dated_history()}

        first = load("AAA")
        first["AAA"].attrs["bac_data_status"] = "provider_stale"
        second = load("AAA")

        self.assertEqual(["AAA"], calls)
        self.assertNotIn("bac_data_status", second["AAA"].attrs)

    def test_stale_histories_are_excluded_from_the_rankable_pool(self):
        """Web and worker must rank the same fresh-only candidate pool."""
        fresh = _dated_history()
        recovered = _dated_history()
        recovered.attrs["bac_data_status"] = "last_known_good"
        frozen = _dated_history()
        frozen.attrs["bac_data_status"] = "live"

        health = market_data.classify_price_histories(
            {"FRESH": fresh, "RECOVERED": recovered, "FROZEN": frozen, "EMPTY": pd.DataFrame()},
            realtime_mode=False,
        )

        self.assertEqual(["FRESH"], health.live_tickers)
        self.assertEqual(["RECOVERED", "FROZEN"], health.stale_tickers)
        self.assertEqual("provider_stale", frozen.attrs["bac_data_status"])


    def test_schema_runs_once_per_target_and_again_if_the_file_vanishes(self):
        """Store calls must not re-run DDL, but a deleted SQLite file is rebuilt."""
        calls = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "schema.db"

            def initialize():
                calls.append(path)
                path.touch()

            for _ in range(3):
                database.ensure_schema("test-store", path, path, initialize)
            self.assertEqual(1, len(calls))

            path.unlink()
            database.ensure_schema("test-store", path, path, initialize)
            self.assertEqual(2, len(calls))

    def test_postgres_connections_are_borrowed_from_the_pool(self):
        """Each unit of work reuses a pooled connection and never closes it."""
        events = []

        class FakeConnection:
            def execute(self, statement, parameters):
                events.append(("execute", statement, parameters))

            def close(self):
                events.append(("close",))

        class FakePool:
            @contextmanager
            def connection(self):
                events.append(("checkout",))
                yield FakeConnection()
                events.append(("return",))

        with (
            patch.object(database, "DATABASE_URL", "postgresql://example/db"),
            patch.object(database, "_postgres_pool", return_value=FakePool()),
        ):
            with database.database_connection("unused.db") as connection:
                self.assertEqual("postgresql", connection.backend)
                connection.execute("SELECT ?", [1])

        self.assertEqual(
            [("checkout",), ("execute", "SELECT %s", (1,)), ("return",)],
            events,
        )


if __name__ == "__main__":
    unittest.main()
