#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_forecast_calendar.py
#############################

"""Offline regression tests for exchange-aware projection timestamps."""

from __future__ import annotations

import unittest

import pandas as pd

from forecasting import future_projection_dates


class ForecastCalendarTest(unittest.TestCase):
    def test_nyse_daily_projection_skips_independence_day(self) -> None:
        projected = future_projection_dates(
            pd.Timestamp("2025-07-03"),
            points_ahead=2,
            realtime_mode=False,
            interval="1d",
            market_calendar="NYSE",
        )

        self.assertEqual(pd.Timestamp("2025-07-07"), projected[0])
        self.assertEqual(pd.Timestamp("2025-07-08"), projected[1])

    def test_nyse_intraday_projection_rolls_to_next_session(self) -> None:
        projected = future_projection_dates(
            pd.Timestamp("2025-07-03 13:00"),
            points_ahead=2,
            realtime_mode=True,
            interval="5m",
            market_calendar="NYSE",
        )

        # July 3, 2025 was an early close and July 4 was closed.  A bar after
        # 13:00 must therefore begin on the Monday session.
        self.assertEqual(pd.Timestamp("2025-07-07 09:35"), projected[0])
        self.assertEqual(pd.Timestamp("2025-07-07 09:40"), projected[1])


if __name__ == "__main__":
    unittest.main()
