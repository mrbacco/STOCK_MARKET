#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_market_leader_rankings.py
#############################

"""Deterministic tests for automatic market-leader selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import market_data
from app_config import FTSE_MIB_MILAN_LISTINGS


def _two_session_history(daily_change: float) -> pd.DataFrame:
    """Build a minimal normalized price frame with a known percentage move."""
    previous_close = 100.0
    last_close = previous_close * (1 + daily_change / 100)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-16", "2026-07-17"]),
            "Close": [previous_close, last_close],
        }
    )


class MarketLeaderRankingTest(unittest.TestCase):
    """Verify automatic sources rank and cap their leaderboards consistently."""

    def test_ranker_returns_best_ten_in_descending_order(self) -> None:
        listings = {f"T{index:02d}": f"Company {index:02d}" for index in range(12)}
        price_data = {
            ticker: _two_session_history(float(index))
            for index, ticker in enumerate(listings)
        }

        with patch.object(market_data, "get_price_history_batch", return_value=price_data):
            result = market_data._rank_latest_daily_performers(
                listings,
                limit=10,
                log_context="tests.market_leader_rankings",
            )

        self.assertEqual(10, len(result))
        self.assertEqual(
            ["T11", "T10", "T09", "T08", "T07", "T06", "T05", "T04", "T03", "T02"],
            result["Ticker"].tolist(),
        )
        self.assertTrue(result["Daily change"].is_monotonic_decreasing)

    def test_ftse_mib_tracks_all_yahoo_supported_milan_constituents(self) -> None:
        self.assertEqual(39, len(FTSE_MIB_MILAN_LISTINGS))
        self.assertNotIn("STMMI.MI", FTSE_MIB_MILAN_LISTINGS)


if __name__ == "__main__":
    unittest.main()
