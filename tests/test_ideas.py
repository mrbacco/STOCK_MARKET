#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_ideas.py
#############################

"""Tests of the plain-language highlights shown on the Today page."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ideas import describe_stock, market_mood, pick_ideas, risk_level, stock_traits

PERIODS = 300


def _history(drift: float, noise: float = 0.01, seed: int = 0) -> pd.DataFrame:
    returns = np.random.default_rng(seed).normal(drift, noise, PERIODS)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Date": pd.bdate_range("2025-01-01", periods=PERIODS), "Close": close})


def _ranking(expected: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Expected excess return": value,
                "Lower 80": value - 3.0,
                "Upper 80": value + 3.0,
                "Probability outperform": 50.0 + value * 5,
                "Sentiment score": 0.0,
            }
            for ticker, value in expected.items()
        ]
    )


class IdeasTest(unittest.TestCase):
    def test_ideas_are_split_by_the_sign_of_the_expected_return(self) -> None:
        expected = {"A": 2.0, "B": 1.0, "C": 0.5, "D": 0.2, "E": -0.3, "F": -1.5}
        prices = {ticker: _history(0.0005, seed=index) for index, ticker in enumerate(expected)}
        best, worst = pick_ideas(_ranking(expected), prices, {"A": "Alpha"})
        self.assertEqual(["A", "B", "C"], [idea.ticker for idea in best])
        self.assertEqual("Alpha", best[0].company)
        # The weakest stock comes first, and no positive stock is called a risk.
        self.assertEqual(["F", "E"], [idea.ticker for idea in worst])

    def test_no_stock_is_highlighted_when_none_is_expected_to_beat_the_market(self) -> None:
        expected = {"A": -0.1, "B": -0.4}
        prices = {ticker: _history(0.0, seed=index) for index, ticker in enumerate(expected)}
        best, worst = pick_ideas(_ranking(expected), prices, {})
        self.assertEqual([], best)
        self.assertEqual(2, len(worst))

    def test_reasons_describe_trend_and_news(self) -> None:
        prices = {
            "UP": _history(0.003, noise=0.005, seed=1),
            "DOWN": _history(-0.002, seed=3),
            **{f"FLAT{index}": _history(0.0, noise=0.002, seed=10 + index) for index in range(3)},
        }
        traits = stock_traits(prices)
        reasons = describe_stock(traits.loc["UP"], sentiment=0.5)
        self.assertIn("Strong uptrend over the past year", reasons)
        self.assertLessEqual(len(reasons), 3)
        self.assertIn("Weak trend over the past year", describe_stock(traits.loc["DOWN"], None))

    def test_risk_tag_marks_the_jumpiest_and_steadiest_stocks(self) -> None:
        prices = {
            f"S{index}": _history(0.0, noise=0.005 * (index + 1), seed=index) for index in range(5)
        }
        traits = stock_traits(prices)
        self.assertEqual("low", risk_level(traits.loc["S0"]))
        self.assertEqual("", risk_level(traits.loc["S2"]))
        self.assertEqual("high", risk_level(traits.loc["S4"]))

    def test_market_mood_reads_breadth_and_volatility(self) -> None:
        rising = {f"S{index}": _history(0.002, noise=0.004, seed=index) for index in range(6)}
        self.assertEqual("positive", market_mood(rising).level)
        falling = {f"S{index}": _history(-0.002, noise=0.004, seed=index) for index in range(6)}
        self.assertEqual("weak", market_mood(falling).level)
        shocked = {}
        for index in range(6):
            frame = _history(0.0, noise=0.004, seed=index)
            close = frame["Close"].to_numpy(copy=True)
            # The last 20 sessions swing ten times harder, all stocks together.
            shock = np.random.default_rng(99).normal(0, 0.04, 20)
            close[-20:] = close[-21] * np.exp(np.cumsum(shock))
            shocked[f"S{index}"] = frame.assign(Close=close)
        self.assertEqual("nervous", market_mood(shocked).level)
        self.assertEqual("unknown", market_mood({}).level)


if __name__ == "__main__":
    unittest.main()
