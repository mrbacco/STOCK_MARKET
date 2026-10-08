#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_prediction_ranking_ui.py
#############################

"""Streamlit regression test for automatic prediction-ranked top-ten charts."""

from __future__ import annotations

import importlib
import unittest

import numpy as np
import pandas as pd

import chart_pipeline
import market_data
import sentiment_service
import sentiment_store
from streamlit.testing.v1 import AppTest


def _ranking_histories() -> dict[str, pd.DataFrame]:
    generator = np.random.default_rng(77)
    dates = pd.date_range("2025-01-02", periods=150, freq="B")
    common_return = generator.normal(0.0002, 0.006, len(dates))
    histories: dict[str, pd.DataFrame] = {}
    for ticker_index in range(12):
        returns = (
            common_return
            + generator.normal(0.0, 0.003, len(dates))
            + ticker_index * 0.00005
        )
        close = 100.0 * np.exp(np.cumsum(returns))
        histories[f"AUTO{ticker_index:02d}"] = pd.DataFrame(
            {
                "Date": dates,
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": generator.integers(900_000, 1_100_000, len(dates)),
            }
        )
    return histories


class PredictionRankingUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Import lazily so test discovery does not bind ``views`` to the real
        # market loader before the older Streamlit regression tests patch it.
        cls.views_module = importlib.import_module("views")
        views = cls.views_module
        cls.histories = _ranking_histories()
        cls.originals = {
            "ireland": market_data.get_iseq20_top_performers,
            "ftse": market_data.get_ftse_mib_top_performers,
            "us": market_data.get_us_top_performers,
            "history": market_data.get_price_history_batch,
            "views_history": views.get_price_history_batch,
            "pipeline_history": chart_pipeline.get_price_history_batch,
            "collector": sentiment_service.ensure_background_sentiment_collector,
            "watchlist": sentiment_store.update_watchlist,
            "status": sentiment_store.get_collector_status,
            "sentiment": chart_pipeline.load_sentiment_history,
            "resolve": chart_pipeline.resolve_pending_forecasts,
            "record_forecast": chart_pipeline.record_forecast,
            "record_run": chart_pipeline.record_market_model_run,
            "quality": views.load_forecast_quality,
            "history_runs": views.load_market_model_history,
        }
        leaderboard = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Company": f"Automatic Company {index}",
                    "Daily change": 12.0 - index,
                    "Last price": float(history["Close"].iloc[-1]),
                    "Last session": history["Date"].iloc[-1].date(),
                }
                for index, (ticker, history) in enumerate(cls.histories.items())
            ]
        )
        market_data.get_iseq20_top_performers = lambda: leaderboard.copy()
        market_data.get_ftse_mib_top_performers = lambda: leaderboard.copy()
        market_data.get_us_top_performers = lambda: leaderboard.copy()
        market_data.get_price_history_batch = lambda tickers, period, interval: {
            ticker: cls.histories[ticker].copy() for ticker in tickers
        }
        # ``views`` imports the loader directly, so patch that bound reference too.
        views.get_price_history_batch = market_data.get_price_history_batch
        chart_pipeline.get_price_history_batch = market_data.get_price_history_batch
        sentiment_service.ensure_background_sentiment_collector = lambda: None
        sentiment_store.update_watchlist = lambda company_by_ticker: None
        sentiment_store.get_collector_status = lambda: {
            "article_count": 0,
            "watchlist_count": 0,
        }
        chart_pipeline.load_sentiment_history = lambda ticker: pd.DataFrame()
        chart_pipeline.resolve_pending_forecasts = lambda price_data, market_source: 0
        chart_pipeline.record_forecast = lambda **kwargs: None
        chart_pipeline.record_market_model_run = lambda *args, **kwargs: None
        views.load_forecast_quality = lambda *args, **kwargs: pd.DataFrame()
        views.load_market_model_history = lambda *args, **kwargs: pd.DataFrame()

    @classmethod
    def tearDownClass(cls) -> None:
        views = cls.views_module
        market_data.get_iseq20_top_performers = cls.originals["ireland"]
        market_data.get_ftse_mib_top_performers = cls.originals["ftse"]
        market_data.get_us_top_performers = cls.originals["us"]
        market_data.get_price_history_batch = cls.originals["history"]
        views.get_price_history_batch = cls.originals["views_history"]
        chart_pipeline.get_price_history_batch = cls.originals["pipeline_history"]
        sentiment_service.ensure_background_sentiment_collector = cls.originals["collector"]
        sentiment_store.update_watchlist = cls.originals["watchlist"]
        sentiment_store.get_collector_status = cls.originals["status"]
        chart_pipeline.load_sentiment_history = cls.originals["sentiment"]
        chart_pipeline.resolve_pending_forecasts = cls.originals["resolve"]
        chart_pipeline.record_forecast = cls.originals["record_forecast"]
        chart_pipeline.record_market_model_run = cls.originals["record_run"]
        views.load_forecast_quality = cls.originals["quality"]
        views.load_market_model_history = cls.originals["history_runs"]

    def test_charts_use_the_model_ranked_top_ten(self) -> None:
        app = AppTest.from_file("app.py")
        app.run(timeout=120)
        view_control = next(
            widget for widget in app.segmented_control if widget.label == "View"
        )
        view_control.set_value("Charts").run(timeout=120)

        self.assertEqual([], list(app.exception))
        self.assertTrue(
            any(
                subheader.value == "Model-ranked top 10 forward opportunities"
                for subheader in app.subheader
            )
        )
        predicted_leader = next(
            metric for metric in app.metric if metric.label == "Top predicted ticker"
        )
        self.assertIn(predicted_leader.value, self.histories)
        self.assertEqual(10, len(app.get("plotly_chart")))


if __name__ == "__main__":
    unittest.main()
