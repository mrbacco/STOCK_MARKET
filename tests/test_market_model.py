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

from market_model import rank_market_candidates, split_panel_dates


def _synthetic_market(ticker_count: int = 6, periods: int = 150) -> dict[str, pd.DataFrame]:
    """Return a small correlated market with stable ticker-specific drift."""
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
        diagnostics = result["diagnostics"]
        self.assertEqual(4, len(ranking))
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


if __name__ == "__main__":
    unittest.main()
