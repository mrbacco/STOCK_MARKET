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
import sentiment_service
import sentiment_store
import views
from app_config import FTSE_MIB_SOURCE, MANUAL_SOURCE
from app_logging import bac_log_kv, bac_log_section
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
        bac_log_section("tests.manual_market_ui", "Applying deterministic loader patches.")
        cls.original_ireland_loader = market_data.get_iseq20_top_performers
        cls.original_ftse_loader = market_data.get_ftse_mib_top_performers
        cls.original_us_loader = market_data.get_us_top_performers
        cls.original_history_loader = market_data.get_price_history_batch
        cls.original_views_history_loader = views.get_price_history_batch
        cls.original_background_collector = sentiment_service.ensure_background_sentiment_collector
        cls.original_watchlist_updater = sentiment_store.update_watchlist
        cls.original_collector_status = sentiment_store.get_collector_status

        market_data.get_iseq20_top_performers = lambda: _sample_leaderboard(
            "A5G.IR",
            "AIB Group",
        )
        market_data.get_ftse_mib_top_performers = lambda: _sample_leaderboard(
            "ENEL.MI",
            "Enel",
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
        # views.py imports the loader directly, so patch its bound reference as
        # well. This keeps fragment/UI tests deterministic regardless of module
        # import order in the full suite.
        views.get_price_history_batch = market_data.get_price_history_batch
        sentiment_service.ensure_background_sentiment_collector = lambda: None
        sentiment_store.update_watchlist = lambda company_by_ticker: None
        sentiment_store.get_collector_status = lambda: {
            "article_count": 0,
            "watchlist_count": 0,
        }
        bac_log_section("tests.manual_market_ui", "Deterministic loader patches applied.")

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the real loaders so this module has no global side effects."""
        bac_log_section("tests.manual_market_ui", "Restoring original data loaders.")
        market_data.get_iseq20_top_performers = cls.original_ireland_loader
        market_data.get_ftse_mib_top_performers = cls.original_ftse_loader
        market_data.get_us_top_performers = cls.original_us_loader
        market_data.get_price_history_batch = cls.original_history_loader
        views.get_price_history_batch = cls.original_views_history_loader
        sentiment_service.ensure_background_sentiment_collector = cls.original_background_collector
        sentiment_store.update_watchlist = cls.original_watchlist_updater
        sentiment_store.get_collector_status = cls.original_collector_status

    def test_manual_market_selection_and_custom_suffix(self) -> None:
        """A typed German symbol should reach the overview as `BMW.DE`."""
        bac_log_section("tests.manual_market_ui", "Starting manual market suffix regression test.")
        app = AppTest.from_file("app.py")
        app.run(timeout=60)
        self.assertEqual([], list(app.exception))
        bac_log_kv("tests.manual_market_ui", stage="initial_run", exception_count=len(list(app.exception)))

        ticker_source = next(
            widget
            for widget in app.segmented_control
            if widget.label == "Ticker source"
        )
        ticker_source.set_value(MANUAL_SOURCE).run(timeout=60)
        self.assertEqual([], list(app.exception))
        bac_log_kv("tests.manual_market_ui", stage="manual_source_selected", exception_count=len(list(app.exception)))

        market_selector = next(
            widget
            for widget in app.selectbox
            if widget.label == "Geographical area / stock market"
        )
        market_selector.set_value("Germany - Xetra").run(timeout=60)
        self.assertEqual([], list(app.exception))
        bac_log_kv("tests.manual_market_ui", stage="germany_selected", exception_count=len(list(app.exception)))

        ticker_selector = next(
            widget
            for widget in app.multiselect
            if widget.label == "Available tickers"
        )
        ticker_selector.set_value(["BMW"]).run(timeout=60)
        self.assertEqual([], list(app.exception))
        bac_log_kv("tests.manual_market_ui", stage="bmw_selected", exception_count=len(list(app.exception)))

        top_mover = next(
            metric for metric in app.metric if metric.label == "Top mover"
        )
        bac_log_kv("tests.manual_market_ui", top_mover_value=top_mover.value)
        self.assertEqual("BMW.DE", top_mover.value)
        bac_log_section("tests.manual_market_ui", "Manual market suffix regression test passed.")

    def test_charts_view_renders_with_sentiment_pipeline_enabled(self) -> None:
        """The forecast view should remain usable while sentiment history warms up."""
        app = AppTest.from_file("app.py")
        app.run(timeout=60)
        active_view = next(
            widget for widget in app.segmented_control if widget.label == "View"
        )
        active_view.set_value("Charts").run(timeout=60)
        self.assertEqual([], list(app.exception))
        self.assertTrue(any(subheader.value == "Forecast backtest" for subheader in app.subheader))

    def test_realtime_mode_enables_free_live_chart_fragment(self) -> None:
        """Free polling should be visible and enabled without an API key."""
        app = AppTest.from_file("app.py")
        app.run(timeout=60)

        realtime_toggle = next(
            widget for widget in app.toggle if widget.label == "Real-time Mode"
        )
        realtime_toggle.set_value(True).run(timeout=60)
        live_toggle = next(
            widget for widget in app.toggle if widget.label == "Live chart updates"
        )
        self.assertTrue(live_toggle.value)

        active_view = next(
            widget for widget in app.segmented_control if widget.label == "View"
        )
        active_view.set_value("Charts").run(timeout=60)

        self.assertEqual([], list(app.exception))
        self.assertTrue(
            any(
                "Live updates on" in caption.value
                for caption in app.caption
            )
        )

    def test_italy_source_renders_ranked_leader_charts(self) -> None:
        """Italy should use constituent-leader labels instead of the old index view."""
        app = AppTest.from_file("app.py")
        app.run(timeout=60)
        ticker_source = next(
            widget for widget in app.segmented_control if widget.label == "Ticker source"
        )
        ticker_source.set_value(FTSE_MIB_SOURCE).run(timeout=60)
        active_view = next(
            widget for widget in app.segmented_control if widget.label == "View"
        )
        active_view.set_value("Charts").run(timeout=60)

        self.assertEqual([], list(app.exception))
        self.assertTrue(
            any(
                subheader.value == "FTSE MIB top 10 daily performers"
                for subheader in app.subheader
            )
        )
        leader = next(
            metric for metric in app.metric if metric.label == "Top FTSE MIB daily mover"
        )
        self.assertEqual("ENEL.MI", leader.value)


if __name__ == "__main__":
    unittest.main()
