#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: market_sources.py
#############################

"""Registry of the stock universes the app ranks, plus the user's watchlist.

Each `MarketSource` is a fixed, named universe from `universe_catalog`, with
its exchange calendar and currency display. Adding a market is one registry
entry. Keys are stable identifiers used in cache scopes, monitoring records,
and session state; labels are what the sidebar shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import universe_catalog as catalog
from app_config import MARKET_CALENDAR_BY_SUFFIX
from app_logging import bac_debug_kv

EURO_SYMBOL = "€"
WATCHLIST_KEY = "watchlist"
WATCHLIST_LABEL = "My watchlist"
DEFAULT_UNIVERSE = "eurostoxx50"


@dataclass(frozen=True)
class MarketSource:
    """One fixed stock universe and how to display it."""

    # Stable identifier for cache scopes, monitoring, and session state.
    key: str
    # Sidebar label.
    label: str
    # Region heading used to group universes.
    region: str
    # Yahoo Finance ticker -> company name.
    listings: dict[str, str] = field(repr=False)
    # `pandas_market_calendars` identifier, or None for a multi-exchange
    # universe whose tickers resolve their own calendar by Yahoo suffix.
    calendar: str | None
    currency_prefix: str
    price_format: str
    price_axis_label: str
    description: str

    @property
    def tickers(self) -> list[str]:
        return list(self.listings)

    def load_performers(self) -> pd.DataFrame:
        """Return this universe's latest daily-move leaderboard."""
        import market_data

        return market_data.get_universe_leaderboard(self.key)

    def price_display(self) -> tuple[str, str, str]:
        """Return the prefix, table format, and chart-axis label."""
        return self.currency_prefix, self.price_format, self.price_axis_label


def _euro(key: str, label: str, region: str, listings: dict[str, str], calendar: str | None,
          description: str) -> MarketSource:
    return MarketSource(
        key=key,
        label=label,
        region=region,
        listings=listings,
        calendar=calendar,
        currency_prefix=EURO_SYMBOL,
        price_format=f"{EURO_SYMBOL}%.2f",
        price_axis_label="Price (EUR)",
        description=description,
    )


_UNIVERSES = (
    MarketSource(
        key="dow30",
        label="United States - Dow Jones 30",
        region="Americas",
        listings=catalog.DOW_30_LISTINGS,
        calendar="NYSE",
        currency_prefix="$",
        price_format="$%.2f",
        price_axis_label="Price (USD)",
        description="The 30 Dow Jones Industrial Average members.",
    ),
    MarketSource(
        key="nasdaq_leaders",
        label="United States - Nasdaq-100 leaders",
        region="Americas",
        listings=catalog.NASDAQ_LEADERS_LISTINGS,
        calendar="NYSE",
        currency_prefix="$",
        price_format="$%.2f",
        price_axis_label="Price (USD)",
        description="Forty of the largest Nasdaq-100 members.",
    ),
    _euro(
        "eurostoxx50",
        "Euro area - Euro Stoxx 50",
        "Europe",
        catalog.EURO_STOXX_50_LISTINGS,
        None,
        "The 50 Euro Stoxx 50 blue chips across euro-area exchanges.",
    ),
    MarketSource(
        key="ftse100_leaders",
        label="United Kingdom - FTSE 100 leaders",
        region="Europe",
        listings=catalog.FTSE_100_LEADERS_LISTINGS,
        calendar="LSE",
        currency_prefix="",
        price_format="%.1f GBp",
        price_axis_label="Price (GBp)",
        description="Forty of the largest FTSE 100 members, quoted in pence.",
    ),
    _euro("dax40", "Germany - DAX 40", "Europe", catalog.DAX_40_LISTINGS, "XETR",
          "The 40 DAX members on Xetra."),
    _euro("cac40", "France - CAC 40", "Europe", catalog.CAC_40_LISTINGS, "XPAR",
          "The 40 CAC 40 members on Euronext Paris."),
    _euro("ftsemib", "Italy - FTSE MIB", "Europe", catalog.FTSE_MIB_MILAN_LISTINGS, "XMIL",
          "The Yahoo-supported FTSE MIB members on Borsa Italiana."),
    _euro("iseq20", "Ireland - ISEQ 20", "Europe", catalog.ISEQ_20_DUBLIN_LISTINGS, "XDUB",
          "The tracked ISEQ 20 members on Euronext Dublin."),
    MarketSource(
        key="nikkei_leaders",
        label="Japan - Nikkei 225 leaders",
        region="Asia-Pacific",
        listings=catalog.NIKKEI_LEADERS_LISTINGS,
        calendar="JPX",
        currency_prefix="¥",
        price_format="¥%.0f",
        price_axis_label="Price (JPY)",
        description="Thirty of the largest Nikkei 225 members on the Tokyo exchange.",
    ),
    MarketSource(
        key="hangseng_leaders",
        label="Hong Kong - Hang Seng leaders",
        region="Asia-Pacific",
        listings=catalog.HANG_SENG_LEADERS_LISTINGS,
        calendar="HKEX",
        currency_prefix="HK$",
        price_format="HK$%.2f",
        price_axis_label="Price (HKD)",
        description="Thirty of the largest Hang Seng members.",
    ),
)

MARKET_SOURCE_REGISTRY: dict[str, MarketSource] = {source.key: source for source in _UNIVERSES}
# Ordered keys of every rankable universe (the watchlist is not one of them).
MARKET_SOURCES: tuple[str, ...] = tuple(MARKET_SOURCE_REGISTRY)
assert DEFAULT_UNIVERSE in MARKET_SOURCE_REGISTRY


def get_market_source(universe_key: str | None) -> MarketSource | None:
    """Return a registered universe, or None for the watchlist."""
    return MARKET_SOURCE_REGISTRY.get(str(universe_key))


def universe_label(universe_key: str) -> str:
    """Return the sidebar label for a universe key or the watchlist."""
    if universe_key == WATCHLIST_KEY:
        return WATCHLIST_LABEL
    source = get_market_source(universe_key)
    return source.label if source is not None else str(universe_key)


def company_name(ticker: str) -> str:
    """Return a known company name for a ticker in any universe."""
    for source in _UNIVERSES:
        if ticker in source.listings:
            return source.listings[ticker]
    return ticker


def source_for_ticker(ticker: str) -> MarketSource | None:
    """Return the first registered universe that lists a ticker."""
    return next((source for source in _UNIVERSES if ticker in source.listings), None)


def all_listed_tickers() -> dict[str, str]:
    """Return every catalogued ticker with its company name, de-duplicated."""
    listed: dict[str, str] = {}
    for source in _UNIVERSES:
        for ticker, name in source.listings.items():
            listed.setdefault(ticker, name)
    return listed


def resolve_market_calendar(universe_key: str | None, ticker: str = "") -> str:
    """Return the best exchange-calendar identifier for a ticker.

    Single-exchange universes use their registered calendar. Multi-exchange
    universes and the watchlist resolve each ticker by its Yahoo suffix,
    defaulting to NYSE for unsuffixed (U.S.) symbols.
    """
    source = get_market_source(universe_key)
    if source is not None and source.calendar is not None:
        calendar_name = source.calendar
    else:
        ticker_upper = str(ticker).upper()
        calendar_name = next(
            (
                calendar
                for suffix, calendar in MARKET_CALENDAR_BY_SUFFIX.items()
                if ticker_upper.endswith(suffix)
            ),
            "NYSE",
        )

    bac_debug_kv(
        "market_sources.resolve_market_calendar",
        universe=universe_key,
        ticker=ticker,
        calendar_name=calendar_name,
    )
    return calendar_name
