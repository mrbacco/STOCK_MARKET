#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_market_model.py
#############################

"""Deterministic tests for the pooled ensemble and chronological split."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_model import _weighted_prediction, rank_market_candidates, split_panel_dates


def _synthetic_market(ticker_count: int = 6, periods: int = 420) -> dict[str, pd.DataFrame]:
    """Return a small correlated market with stable ticker-specific drift.

    The 12-month factors need a year of prices before the first training row.
    """
    generator = np.random.default_rng(1234)
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    common_return = generator.normal(0.0002, 0.007, periods)
    histories: dict[str, pd.DataFrame] = {}
    for ticker_index in range(ticker_count):
        ticker_return = (
            common_return
            + generator.normal(0.0, 0.004, periods)
            + (ticker_index - 2) * 0.0001
        )
        close = 100.0 * np.exp(np.cumsum(ticker_return))
        open_price = close * (1.0 + generator.normal(0.0, 0.001, periods))
        histories[f"TEST{ticker_index}"] = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_price,
                "High": np.maximum(open_price, close) * 1.004,
                "Low": np.minimum(open_price, close) * 0.996,
                "Close": close,
                "Volume": generator.integers(800_000, 1_200_000, periods),
            }
        )
    return histories


class MarketModelTest(unittest.TestCase):
    def test_date_splits_have_horizon_embargoes(self) -> None:
        dates = pd.date_range("2025-01-02", periods=150, freq="B")
        split = split_panel_dates(pd.DatetimeIndex(dates), forecast_horizon=3)

        self.assertTrue(split)
        tuning_start_position = dates.get_loc(split["tuning"][0])
        base_end_position = dates.get_loc(split["base"][-1])
        evaluation_start_position = dates.get_loc(split["evaluation"][0])
        pre_evaluation_end_position = dates.get_loc(split["pre_evaluation"][-1])
        self.assertGreaterEqual(tuning_start_position - base_end_position - 1, 3)
        self.assertGreaterEqual(
            evaluation_start_position - pre_evaluation_end_position - 1,
            3,
        )

    def test_pooled_ensemble_returns_rank_probabilities_and_intervals(self) -> None:
        result = rank_market_candidates(
            _synthetic_market(),
            forecast_horizon=3,
            sentiment_by_ticker={},
            top_n=4,
        )

        ranking = result["ranking"]
        assert isinstance(ranking, pd.DataFrame)
        diagnostics = result["diagnostics"]
        # Every candidate is ranked; top_n only scopes the selection backtest.
        self.assertEqual(6, len(ranking))
        self.assertEqual(list(range(1, 7)), ranking["Rank"].tolist())
        self.assertTrue(
            {
                "Expected excess return",
                "Probability outperform",
                "Lower 80",
                "Upper 80",
                "Model disagreement",
                "Signal",
            }.issubset(ranking.columns)
        )
        self.assertTrue(ranking["Probability outperform"].between(0, 100).all())
        self.assertAlmostEqual(1.0, sum(diagnostics["Model weights"].values()), places=6)
        self.assertGreaterEqual(diagnostics["Evaluation dates"], 20)


    def test_blend_rescales_weights_when_a_member_is_missing(self) -> None:
        """A model that fails after tuning must not shrink the blend towards zero."""
        weights = {"Ridge": 0.5, "Elastic Net": 0.3, "Histogram gradient boosting": 0.2}
        predictions = {
            "Ridge": np.array([0.02, -0.01]),
            "Elastic Net": np.array([0.04, 0.01]),
        }

        blended = _weighted_prediction(predictions, weights)

        expected = (0.5 * predictions["Ridge"] + 0.3 * predictions["Elastic Net"]) / 0.8
        np.testing.assert_allclose(expected, blended)
        self.assertEqual(0, _weighted_prediction({}, weights).size)


if __name__ == "__main__":
    unittest.main()
