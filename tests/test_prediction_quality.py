#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_prediction_quality.py
#############################

"""Tests for rank IC, the LightGBM ranker, GARCH volatility, and conformal bands."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

import market_model
from app_config import PANEL_FEATURE_COLUMNS
from conformal import adaptive_conformal_bands
from model_monitoring import (
    initialize_monitoring_store,
    load_market_model_history,
    record_market_model_run,
)
from volatility import horizon_volatility


def _clustered_returns(periods: int = 260, shock_at: int = 200, seed: int = 1) -> pd.Series:
    """GARCH(1,1)-style daily log returns with one large shock."""
    generator = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.10, 0.85
    returns = np.zeros(periods)
    variance = np.full(periods, omega / (1 - alpha - beta))
    for t in range(1, periods):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * generator.standard_normal()
    returns[shock_at] = -6.0
    return pd.Series(returns / 100.0)


class PredictionQualityTest(unittest.TestCase):
    def test_rank_ic_measures_daily_ordering(self) -> None:
        dates = pd.Series(np.repeat(pd.date_range("2026-01-01", periods=4), 5))
        realized = np.tile(np.arange(5.0), 4)

        perfect, _ = market_model.mean_rank_ic(dates, realized * 2, realized)
        inverse, _ = market_model.mean_rank_ic(dates, -realized, realized)
        constant, _ = market_model.mean_rank_ic(dates, np.zeros(20), realized)

        self.assertAlmostEqual(1.0, perfect)
        self.assertAlmostEqual(-1.0, inverse)
        self.assertTrue(np.isnan(constant))

        # Missing outcomes drop only their own pair, like pandas' Spearman.
        with_gap = realized.copy()
        with_gap[0] = np.nan
        gapped, _ = market_model.mean_rank_ic(dates, realized, with_gap)
        self.assertAlmostEqual(1.0, gapped)

    def test_garch_reacts_to_a_shock_and_decays(self) -> None:
        returns = _clustered_returns()
        dates = pd.Series(pd.date_range("2025-01-01", periods=len(returns), freq="B"))
        forecast, method = horizon_volatility(
            returns, dates, fit_end=dates.iloc[179], horizon=3
        )

        self.assertEqual("garch", method)
        self.assertGreater(forecast.iloc[200], 2 * forecast.iloc[199])
        self.assertLess(forecast.iloc[215], forecast.iloc[200])

    def test_short_history_falls_back_to_ewma(self) -> None:
        returns = _clustered_returns(periods=60, shock_at=50)
        dates = pd.Series(pd.date_range("2025-01-01", periods=60, freq="B"))
        forecast, method = horizon_volatility(returns, dates, fit_end=dates.iloc[40], horizon=3)

        self.assertEqual("ewma", method)
        self.assertGreater(forecast.iloc[50], forecast.iloc[49])

    def test_conformal_bands_hit_their_target_coverage(self) -> None:
        generator = np.random.default_rng(3)
        dates = pd.Series(np.repeat(pd.date_range("2026-01-01", periods=40, freq="B"), 20))
        scores = np.abs(generator.standard_normal(len(dates)))
        bands = adaptive_conformal_bands(
            calibration_scores=np.abs(generator.standard_normal(600)),
            evaluation_dates=dates,
            evaluation_scores=scores,
            target_coverage=0.8,
            horizon=3,
        )
        self.assertAlmostEqual(0.8, bands.evaluation_coverage, delta=0.05)

    def test_bands_only_learn_from_realized_outcomes(self) -> None:
        """Day 4's 2-day outcomes are known on day 6, never earlier."""
        dates = pd.Series(np.repeat(pd.date_range("2026-01-01", periods=10, freq="B"), 5))
        calm = np.full(len(dates), 0.5)
        shocked = calm.copy()
        shocked[dates == dates.unique()[4]] = 100.0
        def quantiles(scores: np.ndarray) -> np.ndarray:
            return adaptive_conformal_bands(
                calibration_scores=np.linspace(0.1, 1.0, 100),
                evaluation_dates=dates,
                evaluation_scores=scores,
                target_coverage=0.8,
                horizon=2,
            ).evaluation_quantile

        before = quantiles(calm)
        after = quantiles(shocked)
        unrealized = (dates < dates.unique()[4 + 2]).to_numpy()
        np.testing.assert_allclose(before[unrealized], after[unrealized])
        self.assertTrue(np.any(after[~unrealized] > before[~unrealized]))

    def test_volatility_exponent_is_learned_from_errors(self) -> None:
        generator = np.random.default_rng(4)
        sigma = np.exp(generator.uniform(-5, -3, 2_000))
        proportional = sigma * generator.standard_normal(2_000)
        independent = 0.02 * generator.standard_normal(2_000)

        self.assertAlmostEqual(
            1.0, market_model.volatility_scaling_exponent(proportional, sigma), delta=0.1
        )
        self.assertAlmostEqual(
            0.0, market_model.volatility_scaling_exponent(independent, sigma), delta=0.1
        )

    def test_ranker_learns_a_planted_cross_sectional_signal(self) -> None:
        generator = np.random.default_rng(5)
        dates = np.repeat(pd.date_range("2025-01-01", periods=120, freq="B"), 20)
        frame = pd.DataFrame(
            generator.standard_normal((len(dates), len(PANEL_FEATURE_COLUMNS))),
            columns=list(PANEL_FEATURE_COLUMNS),
        )
        frame["Date"] = dates
        frame["Ticker"] = np.tile([f"T{i:02d}" for i in range(20)], 120)
        signal_feature = PANEL_FEATURE_COLUMNS[0]
        frame["target_excess_log_return"] = (
            0.01 * frame[signal_feature] + 0.005 * generator.standard_normal(len(frame))
        )
        training = pd.DataFrame(frame.loc[frame["Date"] < dates[-20 * 20]])
        prediction = pd.DataFrame(frame.loc[frame["Date"] >= dates[-20 * 20]])

        predicted = market_model._fit_predict_ranker(training, prediction, forecast_horizon=1)

        assert predicted is not None
        ic, _ = market_model.mean_rank_ic(
            prediction["Date"], predicted, prediction["target_excess_log_return"]
        )
        self.assertGreater(ic, 0.5)

    def test_existing_monitoring_store_gains_rank_ic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "monitoring.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    """
                    CREATE TABLE market_model_runs (
                        market_source TEXT NOT NULL, horizon INTEGER NOT NULL,
                        as_of TEXT NOT NULL, candidate_tickers INTEGER,
                        evaluation_dates INTEGER, evaluation_mae REAL,
                        baseline_mae REAL, directional_accuracy REAL,
                        probability_brier REAL, interval_coverage_80 REAL,
                        selection_mean_excess REAL, selection_hit_rate REAL,
                        sentiment_observed_rows INTEGER,
                        model_weights_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (market_source, horizon, as_of)
                    )
                    """
                )
                connection.commit()
            initialize_monitoring_store(database)
            record_market_model_run(
                "Test market", 3, "2026-01-02", {"Rank IC": 0.04, "Evaluation MAE": 1.0},
                db_path=database,
            )
            history = load_market_model_history("Test market", 3, db_path=database)

        self.assertAlmostEqual(0.04, float(history["rank_ic"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
