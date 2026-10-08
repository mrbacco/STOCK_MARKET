#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_market_sources.py
#############################

"""Consistency tests for the automatic market-source registry."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import market_data
from app_config import FTSE_MIB_SOURCE, MANUAL_SOURCE, MARKET_SOURCES
from market_sources import (
    MARKET_SOURCE_REGISTRY,
    get_market_source,
    resolve_market_calendar,
)


class MarketSourceRegistryTest(unittest.TestCase):
    def test_every_sidebar_source_is_registered_with_a_real_loader(self) -> None:
        self.assertEqual(MARKET_SOURCES, tuple(MARKET_SOURCE_REGISTRY))
        for source in MARKET_SOURCE_REGISTRY.values():
            self.assertTrue(callable(getattr(market_data, source.loader_name)), source.label)
            self.assertEqual(3, len(source.price_display()))

    def test_manual_mode_has_no_registry_entry(self) -> None:
        self.assertIsNone(get_market_source(MANUAL_SOURCE))
        self.assertIsNone(get_market_source(None))

    def test_loader_is_resolved_at_call_time(self) -> None:
        """Replacing the market_data attribute must change what the source loads."""
        leaderboard = pd.DataFrame([{"Ticker": "ENEL.MI", "Company": "Enel"}])
        with patch.object(market_data, "get_ftse_mib_top_performers", return_value=leaderboard):
            loaded = get_market_source(FTSE_MIB_SOURCE).load_performers()
        self.assertEqual(["ENEL.MI"], loaded["Ticker"].tolist())

    def test_calendar_uses_source_then_manual_suffix(self) -> None:
        self.assertEqual("XMIL", resolve_market_calendar(FTSE_MIB_SOURCE, "AAPL"))
        self.assertEqual("XETR", resolve_market_calendar(MANUAL_SOURCE, "BMW.DE"))
        self.assertEqual("NYSE", resolve_market_calendar(None, "MSFT"))


if __name__ == "__main__":
    unittest.main()
