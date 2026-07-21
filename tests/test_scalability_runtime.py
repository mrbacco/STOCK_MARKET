#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_scalability_runtime.py
#############################

"""Offline regression tests for the scalable runtime adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cache_control
import forecasting
import pandas as pd
from database import database_connection
from provider_runtime import call_provider


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

    def test_read_only_forecast_miss_is_queued(self):
        """A web cache miss should enqueue work and return without model fitting."""
        history = pd.DataFrame({"Date": [pd.Timestamp("2026-01-01")], "Close": [100.0]})
        with (
            patch.object(
                forecasting,
                "_forecast_feature_model_cached",
                side_effect=cache_control.SharedCacheMiss("cold"),
            ),
            patch.object(forecasting, "enqueue_analytics_job", return_value=True) as enqueue,
        ):
            result = forecasting.forecast_feature_model(history, points_ahead=3)

        self.assertTrue(result.empty)
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[0], "forecast-curve")


if __name__ == "__main__":
    unittest.main()
