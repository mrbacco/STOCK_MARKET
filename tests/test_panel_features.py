#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_panel_features.py
#############################

"""Point-in-time and transform tests for the configurable panel features."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app_config import PANEL_FEATURE_COLUMNS
from market_model import (
    FACTOR_FEATURE_COLUMNS,
    MARKET_LEVEL_COLUMNS,
    TURBULENCE_FEATURE_COLUMNS,
    PanelConfig,
    build_market_panel,
    informative_feature_columns,
)

PERIODS = 420


def _market(seed: int = 3, shock_at: int | None = None) -> dict[str, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=PERIODS)
    common = generator.normal(0.0003, 0.008, PERIODS)
    if shock_at is not None:
        common[shock_at] = -0.08
    histories = {}
    for index in range(8):
        returns = common * (0.6 + 0.1 * index) + generator.normal(0, 0.01, PERIODS)
        close = 100 * np.exp(np.cumsum(returns))
        histories[f"S{index}"] = pd.DataFrame(
            {"Date": dates, "Open": close, "High": close * 1.01, "Low": close * 0.99,
             "Close": close, "Volume": 1_000_000.0}
        )
    return histories


def _with_future_changed(market: dict[str, pd.DataFrame], after: int) -> dict[str, pd.DataFrame]:
    changed = {}
    for ticker, frame in market.items():
        frame = frame.copy()
        frame.loc[frame.index[after:], ["Open", "High", "Low", "Close"]] *= 1.5
        changed[ticker] = frame
    return changed


class PanelFeatureTest(unittest.TestCase):
    def test_feature_columns_follow_the_config(self) -> None:
        self.assertEqual(PANEL_FEATURE_COLUMNS, PanelConfig().feature_columns)
        full = PanelConfig(factor_features=True, turbulence=True).feature_columns
        for column in (*FACTOR_FEATURE_COLUMNS, *TURBULENCE_FEATURE_COLUMNS):
            self.assertIn(column, full)

    def test_factors_and_turbulence_ignore_future_prices(self) -> None:
        config = PanelConfig(factor_features=True, turbulence=True)
        cutoff = 350
        market = _market()
        original = build_market_panel(market, 5, {}, config).set_index(["Date", "Ticker"])
        altered = build_market_panel(_with_future_changed(market, cutoff), 5, {}, config).set_index(
            ["Date", "Ticker"]
        )
        last_safe = pd.bdate_range("2024-01-01", periods=PERIODS)[cutoff - 1]
        columns = [*FACTOR_FEATURE_COLUMNS, *TURBULENCE_FEATURE_COLUMNS]
        before = original.loc[original.index.get_level_values("Date") <= last_safe, columns]
        after = altered.loc[altered.index.get_level_values("Date") <= last_safe, columns]
        pd.testing.assert_frame_equal(before, after)
        # The factors are populated once enough history exists.
        self.assertTrue(original["momentum_12_1"].notna().any())

    def test_turbulence_spikes_on_a_market_shock(self) -> None:
        panel = build_market_panel(_market(shock_at=380), 5, {}, PanelConfig(turbulence=True))
        by_date = panel.groupby("Date")["turbulence"].first()
        shock_date = pd.bdate_range("2024-01-01", periods=PERIODS)[380]
        calm = by_date.iloc[300:370]
        self.assertGreater(by_date.loc[shock_date], calm.quantile(0.95))

    def test_cross_sectional_ranks_transform_only_stock_level_columns(self) -> None:
        raw = build_market_panel(_market(), 5, {}, PanelConfig(factor_features=True))
        ranked = build_market_panel(
            _market(), 5, {}, PanelConfig(factor_features=True, cross_sectional_ranks=True)
        )
        stock_columns = [column for column in ("ret_5", "vol_20", "momentum_6_1")]
        values = ranked[stock_columns].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        self.assertTrue((finite >= -0.5).all() and (finite <= 0.5).all())
        for column in ("market_ret_5", "market_breadth_1"):
            self.assertIn(column, MARKET_LEVEL_COLUMNS)
            pd.testing.assert_series_equal(raw[column], ranked[column])
        # Raw volatility stays available for the uncertainty bands.
        pd.testing.assert_series_equal(raw["vol_20_raw"], ranked["vol_20_raw"])
        # Ranking preserves each date's order.
        day = ranked["Date"].unique()[300]
        def order(frame: pd.DataFrame) -> list[str]:
            day_rows = pd.DataFrame(frame.loc[frame["Date"] == day])
            return [str(ticker) for ticker in day_rows.sort_values(by="ret_5")["Ticker"]]

        self.assertEqual(order(raw), order(ranked))

    def test_features_without_training_variation_are_left_out(self) -> None:
        panel = build_market_panel(_market(), 5, {}, PanelConfig(cross_sectional_ranks=True))
        columns = PanelConfig(cross_sectional_ranks=True).feature_columns
        # No news reached the training rows, so sentiment never varies there.
        kept = informative_feature_columns(panel, columns)
        self.assertNotIn("sentiment_24h", kept)
        self.assertIn("ret_5", kept)
        self.assertIn("market_ret_5", kept)
        # Sentiment that varies on enough dates is kept.
        varied = panel.copy()
        varied["sentiment_24h"] = np.random.default_rng(0).normal(size=len(varied))
        self.assertIn("sentiment_24h", informative_feature_columns(varied, columns))


if __name__ == "__main__":
    unittest.main()
