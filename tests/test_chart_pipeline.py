#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_chart_pipeline.py
#############################

"""Streamlit-free tests for the Charts view data and model pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import chart_pipeline
from market_sources import WATCHLIST_KEY


def _daily_history(periods: int = 260, seed: int = 11) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(generator.normal(0.0003, 0.01, periods)))
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-02", periods=periods, freq="B"),
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": generator.integers(900_000, 1_100_000, periods).astype(float),
        }
    )


class ChartPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        chart_pipeline.forecast_feature_model.cache_clear()
        chart_pipeline.backtest_forecast_model.cache_clear()

    def test_empty_intraday_load_falls_back_to_daily_settings(self) -> None:
        responses = {
            "1m": {"AAA": pd.DataFrame()},
            "1d": {"AAA": _daily_history(60)},
        }
        with patch.object(
            chart_pipeline,
            "get_price_history_batch",
            side_effect=lambda tickers, period, interval: responses[interval],
        ):
            prices = chart_pipeline.load_chart_prices(
                ["AAA"], period="1d", interval="1m", realtime_mode=True, forecast_points=30
            )

        self.assertTrue(prices.daily_fallback)
        self.assertFalse(prices.realtime_mode)
        self.assertEqual("1d", prices.interval)
        self.assertEqual(5, prices.forecast_points)
        self.assertEqual(["AAA"], prices.valid_tickers)

    def test_ticker_forecast_is_ready_with_bands_and_backtest_row(self) -> None:
        forecast = chart_pipeline.build_ticker_forecast(
            "AAA",
            _daily_history(),
            ticker_source=WATCHLIST_KEY,
            realtime_mode=False,
            interval="1d",
            forecast_points=3,
            preloaded_sentiment=pd.DataFrame(),
        )

        self.assertEqual("ready", forecast.status)
        self.assertEqual(chart_pipeline.PRICE_ONLY_MODEL, forecast.active_model)
        self.assertEqual(3, len(forecast.forecast))
        self.assertEqual(3, len(forecast.future_dates))
        self.assertTrue({"lower_80", "upper_80"}.issubset(forecast.forecast.columns))
        summary = forecast.backtest_summary()
        assert summary is not None
        self.assertIn("Projected return", summary)
        self.assertEqual("AAA", summary["Ticker"])

    def test_short_history_reports_a_diagnosis(self) -> None:
        forecast = chart_pipeline.build_ticker_forecast(
            "AAA",
            _daily_history(25),
            ticker_source=WATCHLIST_KEY,
            realtime_mode=False,
            interval="1d",
            forecast_points=3,
            preloaded_sentiment=pd.DataFrame(),
        )

        self.assertEqual("unavailable", forecast.status)
        assert forecast.diagnosis is not None
        self.assertIn("message", forecast.diagnosis)
        self.assertIsNone(forecast.backtest_summary())

    def test_stale_forecasts_are_shown_but_not_recorded(self) -> None:
        history = _daily_history()
        forecast = chart_pipeline.build_ticker_forecast(
            "AAA",
            history,
            ticker_source=WATCHLIST_KEY,
            realtime_mode=False,
            interval="1d",
            forecast_points=3,
            preloaded_sentiment=pd.DataFrame(),
        )
        stale_history = history.copy()
        stale_history.attrs["bac_data_status"] = "last_known_good"
        stale = chart_pipeline.build_ticker_forecast(
            "AAA",
            stale_history,
            ticker_source=WATCHLIST_KEY,
            realtime_mode=False,
            interval="1d",
            forecast_points=3,
            preloaded_sentiment=pd.DataFrame(),
        )

        with patch.object(chart_pipeline, "record_forecast") as record:
            chart_pipeline.record_displayed_forecast(
                stale, monitoring_market="Manual tickers", ranking=pd.DataFrame()
            )
            record.assert_not_called()
            chart_pipeline.record_displayed_forecast(
                forecast, monitoring_market="Manual tickers", ranking=pd.DataFrame()
            )
            record.assert_called_once()
        self.assertEqual("AAA", record.call_args.kwargs["ticker"])
        self.assertEqual(3, record.call_args.kwargs["horizon"])


if __name__ == "__main__":
    unittest.main()
