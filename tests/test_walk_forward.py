#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_walk_forward.py
#############################

"""Tests for the walk-forward test, its store, evidence, and portfolio backtest."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import walk_forward_store
from model_evidence import assess_ranking_evidence, assess_walk_forward_evidence
from portfolio_backtest import top_n_backtest
from walk_forward import run_walk_forward


def _drifting_market(tickers: int = 12, periods: int = 520) -> dict[str, pd.DataFrame]:
    """Stocks with persistent, different drifts, so relative strength predicts."""
    generator = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    market = generator.normal(0.0002, 0.008, periods)
    histories = {}
    for index in range(tickers):
        drift = (index - tickers / 2) * 0.0006
        close = 100 * np.exp(np.cumsum(market + drift + generator.normal(0, 0.006, periods)))
        histories[f"T{index:02d}"] = pd.DataFrame(
            {"Date": dates, "Open": close, "High": close * 1.005, "Low": close * 0.995,
             "Close": close, "Volume": 1_000_000.0}
        )
    return histories


def _number(summary: dict, key: str) -> float:
    value = summary[key]
    assert isinstance(value, (int, float))
    return float(value)


class WalkForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.horizon = 3
        cls.result = run_walk_forward(
            _drifting_market(), {}, horizon=cls.horizon, retrain_every=63,
            min_train_dates=200, max_train_dates=300, tuning_dates=40, top_n=3,
        )

    def test_every_fold_trains_only_on_realized_targets(self) -> None:
        folds = self.result.folds
        self.assertGreater(len(folds), 1)
        for _, fold in folds.iterrows():
            trained_through = np.datetime64(str(fold["Trained through"])[:10], "D")
            retrained_on = np.datetime64(str(fold["Retrained on"])[:10], "D")
            gap = int(np.busday_count(trained_through, retrained_on))
            # The last training target ends `horizon` bars later, before the test starts.
            self.assertGreaterEqual(gap, self.horizon)

    def test_predictions_are_out_of_sample_and_summarized(self) -> None:
        predictions = self.result.predictions
        first_test = pd.Timestamp(str(self.result.folds["Retrained on"].min()))
        self.assertTrue((pd.to_datetime(predictions["Date"]) >= first_test).all())
        summary = self.result.summary
        # Persistent drift makes relative strength informative.
        self.assertGreater(_number(summary, "Rank IC"), 0)
        self.assertGreater(_number(summary, "Momentum rank IC"), 0)
        for key in ("Rank IC t-stat", "Positive quarters", "Portfolio Cumulative net excess"):
            self.assertIn(key, summary)

    def test_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "walk_forward.db"
            walk_forward_store.save_walk_forward(
                "test_universe", 3, summary=self.result.summary, folds=self.result.folds,
                predictions=self.result.predictions, db_path=database,
            )
            stored = walk_forward_store.load_walk_forward("test_universe", 3, db_path=database)
            light = walk_forward_store.load_walk_forward(
                "test_universe", 3, with_predictions=False, db_path=database
            )
            missing = walk_forward_store.load_walk_forward("test_universe", 5, db_path=database)
        assert stored is not None and light is not None
        self.assertIsNone(missing)
        self.assertEqual(len(self.result.predictions), len(stored.predictions))
        self.assertAlmostEqual(self.result.summary["Rank IC"], stored.summary["Rank IC"])
        self.assertTrue(light.predictions.empty)

    def test_background_run_reports_progress_and_saves(self) -> None:
        saved = {}
        market = _drifting_market()
        with (
            patch("market_data.get_price_history_batch",
                  side_effect=lambda tickers, period, interval: market),
            patch("sentiment_store.load_sentiment_history", return_value=pd.DataFrame()),
            patch.object(walk_forward_store, "save_walk_forward",
                         side_effect=lambda universe, horizon, **kwargs: saved.update(kwargs)),
        ):
            self.assertTrue(walk_forward_store.start_walk_forward("dax40", 3))
            self.assertFalse(walk_forward_store.start_walk_forward("dax40", 3))
            deadline = time.monotonic() + 120
            while walk_forward_store.walk_forward_status("dax40", 3).running:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.1)
        status = walk_forward_store.walk_forward_status("dax40", 3)
        self.assertEqual("", status.error)
        self.assertIn("summary", saved)


class EvidenceAndPortfolioTest(unittest.TestCase):
    def test_walk_forward_evidence_levels(self) -> None:
        base = {"Test years": 4.0, "Positive quarters": 70.0, "Top-N": 10, "Reversal rank IC": 0.01}
        supported = assess_walk_forward_evidence(
            {**base, "Rank IC": 0.04, "Rank IC t-stat": 2.5, "Portfolio Cumulative net excess": 12.0}
        )
        tentative = assess_walk_forward_evidence(
            {**base, "Rank IC": 0.01, "Rank IC t-stat": 1.2, "Portfolio Cumulative net excess": 2.0}
        )
        none = assess_walk_forward_evidence(
            {**base, "Rank IC": -0.01, "Rank IC t-stat": -1.6, "Portfolio Cumulative net excess": -30.0}
        )
        self.assertEqual("supported", supported.level)
        self.assertTrue(supported.show_signals)
        self.assertEqual("tentative", tentative.level)
        self.assertEqual("none", none.level)
        self.assertIn("reversal", none.detail)

    def test_walk_forward_takes_precedence_over_the_short_window(self) -> None:
        short_window = {"Rank IC": 0.2, "Rank IC t-stat": 5.0,
                        "Top-10 realized mean excess": 3.0, "Evaluation dates": 30}
        long_test = {"Rank IC": -0.02, "Rank IC t-stat": -2.0, "Test years": 4.0,
                     "Positive quarters": 30.0, "Top-N": 10,
                     "Portfolio Cumulative net excess": -20.0, "Reversal rank IC": 0.02}
        self.assertEqual("supported", assess_ranking_evidence(short_window).level)
        self.assertEqual("none", assess_ranking_evidence(short_window, long_test).level)

    def test_top_n_backtest_rebalances_without_overlap_and_charges_costs(self) -> None:
        dates = np.repeat(pd.bdate_range("2026-01-01", periods=12), 6)
        tickers = np.tile([f"S{i}" for i in range(6)], 12)
        predicted = np.tile(np.arange(6, 0, -1, dtype=float), 12)
        realized = np.tile([0.02, 0.01, 0.0, 0.0, -0.01, -0.02], 12)
        evaluation = pd.DataFrame({"Date": dates, "Ticker": tickers,
                                   "predicted_excess_return": predicted,
                                   "target_excess_log_return": realized})
        free = top_n_backtest(evaluation, horizon=3, top_n=2, cost_bps=0)
        costly = top_n_backtest(evaluation, horizon=3, top_n=2, cost_bps=25)

        self.assertEqual(4, len(free.periods))  # 12 dates, every 3rd one
        self.assertGreater(free.summary["Cumulative net excess"], 0)
        self.assertLess(free.summary["Cumulative bottom-N excess"], 0)
        # Same holdings every period: only the entry trade is charged.
        self.assertAlmostEqual(0.0, free.summary["Average turnover"])
        self.assertAlmostEqual(0.25, costly.summary["Total cost"])


if __name__ == "__main__":
    unittest.main()
