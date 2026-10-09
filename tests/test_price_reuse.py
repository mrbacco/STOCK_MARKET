#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_price_reuse.py
#############################

"""Reuse of recently saved long daily histories instead of downloading."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import market_data


def _saved(hours_ago: float) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": pd.bdate_range("2026-01-01", periods=5), "Close": 1.0})
    frame.attrs["bac_data_status"] = "last_known_good"
    frame.attrs["bac_fetched_at"] = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours_ago)
    ).isoformat()
    return frame


class PriceReuseTest(unittest.TestCase):
    def setUp(self) -> None:
        market_data._LIVE_REQUIRED_AFTER.clear()

    def _reuse(self, saved: dict, period: str = "5y", interval: str = "1d"):
        with patch.object(market_data, "_load_price_snapshots_safely", return_value=saved):
            return market_data._recent_saved_histories(["A", "B"], period, interval)

    def test_recent_copies_of_every_ticker_are_reused_as_live(self) -> None:
        reused = self._reuse({"A": _saved(1), "B": _saved(2)})
        assert reused is not None
        self.assertEqual({"A", "B"}, set(reused))
        self.assertEqual("live", reused["A"].attrs["bac_data_status"])

    def test_old_missing_or_short_period_copies_download_instead(self) -> None:
        self.assertIsNone(self._reuse({"A": _saved(1), "B": _saved(30)}))
        self.assertIsNone(self._reuse({"A": _saved(1)}))
        self.assertIsNone(self._reuse({"A": _saved(1), "B": _saved(1)}, period="5d"))
        self.assertIsNone(self._reuse({"A": _saved(1), "B": _saved(1)}, interval="5m"))

    def test_refresh_requires_a_newer_download(self) -> None:
        market_data.require_live_prices(["B"])
        self.assertIsNone(self._reuse({"A": _saved(1), "B": _saved(1)}))


if __name__ == "__main__":
    unittest.main()
