#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_price_formatting.py
#############################

"""Normalization of raw Yahoo Finance histories."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_data import format_price_history


class PriceFormattingTest(unittest.TestCase):
    def test_rows_without_a_close_are_dropped(self) -> None:
        # A batch download gives a Helsinki stock a row for a session only
        # other exchanges traded: no close, but a zero volume.
        raw = pd.DataFrame(
            {
                "Open": [9.5, np.nan],
                "High": [9.6, np.nan],
                "Low": [9.0, np.nan],
                "Close": [9.1, np.nan],
                "Volume": [1_000.0, 0.0],
            },
            index=pd.DatetimeIndex(["2026-10-08", "2026-10-09"], name="Date"),
        )
        formatted = format_price_history(raw)
        self.assertEqual([pd.Timestamp("2026-10-08")], formatted["Date"].tolist())
        self.assertFalse(formatted["Close"].isna().any())


if __name__ == "__main__":
    unittest.main()
