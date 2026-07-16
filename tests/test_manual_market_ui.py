#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_manual_market_ui.py
#############################

"""Regression test for the geographical manual-ticker sidebar workflow."""

from __future__ import annotations

import unittest

import pandas as pd

import market_data
from app_config import MANUAL_SOURCE
from streamlit.testing.v1 import AppTest


def _sample_price_frame() -> pd.DataFrame:
    """Return enough deterministic history for the manual overview metrics."""
    dates = pd.date_range("2026-01-01", periods=65, freq="B")
    closes = pd.Series(range(100, 165), dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes - 0.5,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": 1_000_000.0,
        }
    )


def _sample_leaderboard(ticker: str, company: str) -> pd.DataFrame:
    """Return a small automatic-source frame so the first app run is offline."""
    return pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": 1.25,
                "Last price": 100.0,
                "Last session": pd.Timestamp("2026-07-15").date(),
            }
        ]
    )


class ManualMarketUiTest(unittest.TestCase):
    """Exercise the full widget path from source selection to normalized ticker."""

    @classmethod
    def setUpClass(cls) -> None:
        """Replace public-data calls so this UI test is fast and deterministic."""
        cls.original_ireland_loader = market_data.get_iseq20_top_performers
        cls.original_ftse_loader = market_data.get_ftse_mib_index
        cls.original_us_loader = market_data.get_us_top_performers
        cls.original_history_loader = market_data.get_price_history_batch

        market_data.get_iseq20_top_performers = lambda: _sample_leaderboard(
            "A5G.IR",
            "AIB Group",
        )
        market_data.get_ftse_mib_index = lambda: _sample_leaderboard(
            "FTSEMIB.MI",
            "FTSE MIB",
        )
        market_data.get_us_top_performers = lambda: _sample_leaderboard(
            "AAPL",
            "Apple",
        )
        market_data.get_price_history_batch = (
            lambda tickers, period, interval: {
                ticker: _sample_price_frame() for ticker in tickers
            }
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the real loaders so this module has no global side effects."""
        market_data.get_iseq20_top_performers = cls.original_ireland_loader
        market_data.get_ftse_mib_index = cls.original_ftse_loader
        market_data.get_us_top_performers = cls.original_us_loader
        market_data.get_price_history_batch = cls.original_history_loader

    def test_manual_market_selection_and_custom_suffix(self) -> None:
        """A typed German symbol should reach the overview as `BMW.DE`."""
        app = AppTest.from_file("app.py")
        app.run(timeout=60)
        self.assertEqual([], list(app.exception))

        ticker_source = next(
            widget
            for widget in app.segmented_control
            if widget.label == "Ticker source"
        )
        ticker_source.set_value(MANUAL_SOURCE).run(timeout=60)
        self.assertEqual([], list(app.exception))

        market_selector = next(
            widget
            for widget in app.selectbox
            if widget.label == "Geographical area / stock market"
        )
        market_selector.set_value("Germany - Xetra").run(timeout=60)
        self.assertEqual([], list(app.exception))

        ticker_selector = next(
            widget
            for widget in app.multiselect
            if widget.label == "Available tickers"
        )
        ticker_selector.set_value(["BMW"]).run(timeout=60)
        self.assertEqual([], list(app.exception))

        top_mover = next(
            metric for metric in app.metric if metric.label == "Top mover"
        )
        self.assertEqual("BMW.DE", top_mover.value)


if __name__ == "__main__":
    unittest.main()
