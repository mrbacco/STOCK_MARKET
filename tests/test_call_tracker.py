#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_call_tracker.py
#############################

"""Live call log and the walk-forward track record."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from call_tracker import BEATS, LAGS, Call, check_calls, live_record, record_calls
from walk_forward import call_track_record

DATES = pd.bdate_range("2026-01-01", periods=40)


def _history(daily_return: float) -> pd.DataFrame:
    return pd.DataFrame({"Date": DATES, "Close": 100 * np.exp(daily_return * np.arange(len(DATES)))})


class CallTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Path(self.temp_dir.name) / "calls.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_calls_are_checked_against_the_market_after_the_horizon(self) -> None:
        # The market (equal-weight average) falls, so FLAT beats it.
        prices = {"UP": _history(0.004), "FLAT": _history(0.0), "DOWN": _history(-0.008)}
        as_of = DATES[10]
        calls = [Call("UP", BEATS, 2.0), Call("DOWN", BEATS, 1.0), Call("FLAT", LAGS, -1.0)]
        record_calls("m", 5, as_of, calls, db_path=self.db)
        # Saving the same day again keeps the first call.
        record_calls("m", 5, as_of, [Call("UP", LAGS, -9.0)], db_path=self.db)

        early = {ticker: frame.iloc[:13] for ticker, frame in prices.items()}
        self.assertEqual(0, check_calls("m", 5, early, db_path=self.db))
        self.assertEqual(LiveRecordTuple(3, 0, 0), _counts(live_record("m", 5, db_path=self.db)))

        self.assertEqual(3, check_calls("m", 5, prices, db_path=self.db))
        # UP beat the market (right), DOWN did not (wrong), and FLAT beat the
        # falling market, so calling it a laggard was wrong.
        record = live_record("m", 5, db_path=self.db)
        self.assertEqual(LiveRecordTuple(3, 3, 1), _counts(record))
        self.assertIsNone(record.next_check_after)

    def test_track_record_counts_independent_periods(self) -> None:
        rows = []
        for index, date in enumerate(pd.bdate_range("2026-01-01", periods=10)):
            for stock in range(8):
                rows.append(
                    {
                        "Date": date,
                        "Ticker": f"S{stock}",
                        "predicted_excess_return": -stock,
                        # The top-ranked stock wins on even dates only.
                        "target_excess_log_return": (1.0 if index % 2 == 0 else -1.0)
                        if stock == 0
                        else -0.1 * stock + 0.35,
                    }
                )
        record = call_track_record(pd.DataFrame(rows), horizon=2, basket_size=3)
        # Every second date is used: dates 0, 2, 4, 6, 8, all even.
        self.assertEqual(5, record["Periods"])
        self.assertEqual(5, record["Top pick right"])
        self.assertEqual(5, record["Basket right"])


class LiveRecordTuple(tuple):
    def __new__(cls, made: int, checked: int, right: int):
        return super().__new__(cls, (made, checked, right))


def _counts(record) -> LiveRecordTuple:
    return LiveRecordTuple(record.made, record.checked, record.right)


if __name__ == "__main__":
    unittest.main()
