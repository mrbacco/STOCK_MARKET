#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_market_sources.py
#############################

"""Consistency tests for the stock-universe registry."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

import market_data
from market_sources import (
    DEFAULT_UNIVERSE,
    MARKET_SOURCE_REGISTRY,
    MARKET_SOURCES,
    WATCHLIST_KEY,
    all_listed_tickers,
    company_name,
    get_market_source,
    resolve_market_calendar,
    source_for_ticker,
)


class MarketSourceRegistryTest(unittest.TestCase):
    def test_every_universe_is_well_formed(self) -> None:
        self.assertIn(DEFAULT_UNIVERSE, MARKET_SOURCES)
        self.assertEqual(MARKET_SOURCES, tuple(MARKET_SOURCE_REGISTRY))
        for key, source in MARKET_SOURCE_REGISTRY.items():
            self.assertEqual(key, source.key)
            self.assertGreaterEqual(len(source.tickers), 20, key)
            self.assertEqual(len(source.tickers), len(set(source.tickers)), key)
            self.assertEqual(3, len(source.price_display()))

    def test_watchlist_has_no_registry_entry(self) -> None:
        self.assertIsNone(get_market_source(WATCHLIST_KEY))
        self.assertIsNone(get_market_source(None))

    def test_loader_is_resolved_at_call_time(self) -> None:
        """Replacing the market_data loader must change what a universe loads."""
        leaderboard = pd.DataFrame([{"Ticker": "ENEL.MI", "Company": "Enel"}])
        with patch.object(market_data, "get_universe_leaderboard", return_value=leaderboard) as loader:
            source = get_market_source("ftsemib")
            assert source is not None
            loaded = source.load_performers()
        loader.assert_called_once_with("ftsemib")
        self.assertEqual(["ENEL.MI"], loaded["Ticker"].tolist())

    def test_calendar_uses_universe_then_ticker_suffix(self) -> None:
        self.assertEqual("XMIL", resolve_market_calendar("ftsemib", "AAPL"))
        # Multi-exchange universes and the watchlist resolve by suffix.
        self.assertEqual("XETR", resolve_market_calendar("eurostoxx50", "SAP.DE"))
        self.assertEqual("XBRU", resolve_market_calendar("eurostoxx50", "ABI.BR"))
        self.assertEqual("XETR", resolve_market_calendar(WATCHLIST_KEY, "BMW.DE"))
        self.assertEqual("NYSE", resolve_market_calendar(None, "MSFT"))

    def test_ticker_lookups_span_every_universe(self) -> None:
        listed = all_listed_tickers()
        self.assertEqual("Toyota Motor", company_name("7203.T"))
        self.assertEqual("UNKNOWN", company_name("UNKNOWN"))
        source = source_for_ticker("0700.HK")
        assert source is not None
        self.assertEqual("hangseng_leaders", source.key)
        self.assertGreater(len(listed), 300)


if __name__ == "__main__":
    unittest.main()
