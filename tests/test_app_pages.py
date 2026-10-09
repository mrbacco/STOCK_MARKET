#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_app_pages.py
#############################

"""Offline end-to-end tests of every page through Streamlit's AppTest."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
import zlib
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import chart_pipeline
import global_markets
import market_data
import ranking_store
import sentiment_service
import sentiment_store
import ui_components

# The 12-month factors need a year of prices before the first training row.
PERIODS = 520


def _synthetic_history(ticker: str) -> pd.DataFrame:
    """Deterministic daily OHLCV per ticker with a shared market factor."""
    market = np.random.default_rng(7).normal(0.0003, 0.008, PERIODS)
    own = np.random.default_rng(zlib.crc32(ticker.encode())).normal(0.0, 0.01, PERIODS)
    close = 100.0 * np.exp(np.cumsum(market + own))
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=PERIODS)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


def _price_batch(tickers, period, interval):
    return {ticker: _synthetic_history(ticker) for ticker in tickers}


def _leaderboard(universe_key):
    from market_sources import MARKET_SOURCE_REGISTRY

    source = MARKET_SOURCE_REGISTRY[universe_key]
    return pd.DataFrame(
        [
            {"Ticker": ticker, "Company": company, "Daily change": float(index % 7) - 3.0,
             "Last price": 100.0, "Last session": pd.Timestamp.now().date()}
            for index, (ticker, company) in enumerate(source.listings.items())
        ]
    )


class AppPagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Synthetic rankings must never land in the real day-long ranking store.
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.patchers = [
            patch.object(ranking_store, "DEFAULT_RANKING_DB", Path(cls.temp_dir.name) / "rankings.db"),
            patch.object(market_data, "get_price_history_batch", side_effect=_price_batch),
            patch.object(global_markets, "get_price_history_batch", side_effect=_price_batch),
            patch.object(chart_pipeline, "get_price_history_batch", side_effect=_price_batch),
            patch.object(market_data, "get_universe_leaderboard", side_effect=_leaderboard),
            patch.object(sentiment_service, "ensure_background_sentiment_collector", return_value=None),
            patch.object(sentiment_store, "update_watchlist", return_value=False),
            patch.object(sentiment_store, "get_collector_status",
                         return_value={"article_count": 0, "watchlist_count": 0}),
            patch.object(sentiment_store, "load_sentiment_history", return_value=pd.DataFrame()),
            patch.object(chart_pipeline, "load_sentiment_history", return_value=pd.DataFrame()),
            patch.object(chart_pipeline, "resolve_pending_forecasts", return_value=0),
            patch.object(chart_pipeline, "record_forecast", return_value=None),
            patch.object(chart_pipeline, "record_market_model_run", return_value=None),
            patch.object(ui_components, "load_forecast_quality", return_value=pd.DataFrame()),
            patch.object(ui_components, "load_market_model_history", return_value=pd.DataFrame()),
        ]
        for patcher in cls.patchers:
            patcher.start()
        # Start from an empty overview so the test exercises the loader.
        global_markets.reset_overview()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in cls.patchers:
            patcher.stop()
        cls.temp_dir.cleanup()

    def _app(self, universe: str = "ftsemib") -> AppTest:
        app = AppTest.from_file("app.py", default_timeout=180)
        app.session_state["universe"] = universe
        app.session_state["watchlist_tickers"] = ["AAPL", "SAP.DE", "7203.T"]
        app.run()
        self.assertEqual([], [error.value for error in app.exception])
        return app

    def _open(self, app: AppTest, page: str) -> AppTest:
        app.switch_page(page).run()
        self.assertEqual([], [error.value for error in app.exception], page)
        return app

    def test_today_is_the_default_page_and_never_shows_untested_ideas(self) -> None:
        app = self._app()
        self.assertTrue(any(header.value.startswith("Today in") for header in app.subheader))
        banners = [*app.success, *app.info, *app.warning, *app.error]
        self.assertTrue(any("Market mood" in banner.value for banner in banners))
        # Synthetic prices have no stored walk-forward test, so no ideas are highlighted.
        with patch("ui_state.load_walk_forward", return_value=None):
            app.run()
        self.assertTrue(any("No reliable ideas" in banner.value for banner in app.warning))
        self.assertNotIn("See details", [button.label for button in app.button])

    def test_today_highlights_ideas_when_the_model_has_evidence(self) -> None:
        summary = {
            "Rank IC": 0.06, "Rank IC t-stat": 1.6, "Positive quarters": 70.0,
            "Portfolio Cumulative net excess": 40.0, "Top-N": 10, "Test years": 3.0,
            "Reversal rank IC": 0.0, "Model features": "base+factors+ranks",
        }
        with patch("ui_state.walk_forward_summary", return_value=summary):
            app = self._app()
        self.assertEqual([], [error.value for error in app.exception])
        markdown = " ".join(block.value for block in app.markdown)
        self.assertIn("Stocks to look at", markdown)
        self.assertIn("Early evidence", markdown)

    def test_global_markets_page_renders(self) -> None:
        app = self._open(self._app(), "app_pages/world_markets.py")
        # The overview loads in a background thread; wait for it, then rerender.
        deadline = time.monotonic() + 120
        while global_markets.overview_loading() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(global_markets.overview_loading())
        app.run()
        self.assertEqual([], [error.value for error in app.exception])
        labels = [metric.label for metric in app.metric]
        self.assertIn("S&P 500", labels)
        self.assertTrue(any(header.value == "Stock markets" for header in app.subheader))

    def test_market_ranking_shows_evidence_before_the_ranking(self) -> None:
        app = self._open(self._app(), "app_pages/market_ranking.py")
        banners = [*app.success, *app.info, *app.warning]
        # The two-line caveat (or a validated-edge note) precedes the ranking.
        self.assertTrue(any("Caution" in banner.value or "edge" in banner.value for banner in banners))
        self.assertIn("Rank IC", [metric.label for metric in app.metric])
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_stock_page_shows_the_model_view_and_an_opt_in_projection(self) -> None:
        app = self._app()
        self._open(app, "app_pages/market_ranking.py")
        self._open(app, "app_pages/stock.py")
        labels = [metric.label for metric in app.metric]
        self.assertIn("Last close", labels)
        self.assertIn("Expected vs the market", labels)
        # The single-stock projection is computed only when asked for.
        self.assertEqual(0, len(app.get("plotly_chart")))
        app.toggle(key="stock_projection").set_value(True).run()
        self.assertEqual([], [error.value for error in app.exception])
        self.assertGreaterEqual(len(app.get("plotly_chart")), 1)

    def test_portfolio_news_and_model_health_render(self) -> None:
        app = self._app()
        self._open(app, "app_pages/market_ranking.py")
        self._open(app, "app_pages/portfolio.py")
        self.assertIn("Net excess return", [metric.label for metric in app.metric])
        self._open(app, "app_pages/news.py")
        self._open(app, "app_pages/model_health.py")

    def test_watchlist_uses_momentum_instead_of_the_pooled_model(self) -> None:
        app = self._open(self._app("watchlist"), "app_pages/market_ranking.py")
        self.assertTrue(any("momentum" in banner.value for banner in app.info))


if __name__ == "__main__":
    unittest.main()
