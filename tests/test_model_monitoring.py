#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_model_monitoring.py
#############################

"""Offline tests for persistent forecast resolution and drift metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_monitoring import (
    latest_drift_summary,
    load_forecast_quality,
    load_market_model_history,
    record_forecast,
    record_market_model_run,
    resolve_pending_forecasts,
)


class ModelMonitoringTest(unittest.TestCase):
    def test_forecast_is_resolved_into_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "monitoring.db"
            record_forecast(
                market_source="Test market",
                ticker="TEST",
                forecast_origin="2026-01-02",
                target_at="2026-01-05",
                horizon=1,
                model_name="Price + sentiment",
                regime="Normal volatility",
                origin_close=100.0,
                predicted_close=105.0,
                predicted_return=0.05,
                lower_80=102.0,
                upper_80=108.0,
                sentiment_score=0.6,
                db_path=database,
            )
            price_data = {
                "TEST": pd.DataFrame(
                    {
                        "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                        "Close": [100.0, 106.0],
                    }
                )
            }

            resolved = resolve_pending_forecasts(
                price_data,
                "Test market",
                db_path=database,
            )
            quality = load_forecast_quality(
                "Test market",
                horizon=1,
                db_path=database,
            )

            self.assertEqual(1, resolved)
            self.assertEqual(1, int(quality["Forecasts"].iloc[0]))
            self.assertAlmostEqual(1.0, float(quality["MAE"].iloc[0]))
            self.assertAlmostEqual(100.0, float(quality["Directional accuracy"].iloc[0]))
            self.assertAlmostEqual(100.0, float(quality["80% interval coverage"].iloc[0]))

    def test_market_run_history_reports_latest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "monitoring.db"
            baseline = {
                "Candidate tickers": 20,
                "Evaluation dates": 30,
                "Evaluation MAE": 0.8,
                "Zero-excess baseline MAE": 1.0,
                "Directional accuracy": 55.0,
                "Probability Brier score": 0.24,
                "80% interval coverage": 79.0,
                "Top-10 realized mean excess": 0.1,
                "Top-10 realized hit rate": 54.0,
                "Sentiment-observed rows": 100,
                "Model weights": {"Ridge": 1.0},
            }
            latest = dict(baseline)
            latest.update(
                {
                    "Evaluation MAE": 1.0,
                    "Directional accuracy": 52.0,
                    "Probability Brier score": 0.26,
                    "80% interval coverage": 75.0,
                }
            )
            record_market_model_run(
                "Test market", 3, "2026-01-02", baseline, db_path=database
            )
            record_market_model_run(
                "Test market", 3, "2026-01-03", latest, db_path=database
            )
            history = load_market_model_history(
                "Test market", 3, db_path=database
            )
            drift = latest_drift_summary(history)

            self.assertAlmostEqual(0.2, drift["MAE drift"])
            self.assertAlmostEqual(-3.0, drift["Direction drift"])
            self.assertAlmostEqual(0.02, drift["Brier drift"])
            self.assertAlmostEqual(-4.0, drift["Coverage drift"])


if __name__ == "__main__":
    unittest.main()
