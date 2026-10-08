#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: market_sources.py
#############################

"""Registry of the automatic market sources shown in the sidebar.

Everything that differs between the Ireland, Italy, and U.S. sources lives in
one `MarketSource` entry: its performer loader, exchange calendar, currency
display, and user-facing copy. Adding a market is a single registry entry
instead of another branch in the app, views, and worker. Manual tickers use
`ticker_catalog.ManualMarketPreset`, which plays the same role per exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app_config import (
    FTSE_MIB_SOURCE,
    IRELAND_SOURCE,
    MARKET_CALENDAR_BY_SUFFIX,
    MARKET_SOURCES,
    US_SOURCE,
)
from app_logging import bac_debug_kv

EURO_SYMBOL = "€"
MANUAL_CHART_HEADING = "Top momentum stocks - history and feature-based forecast"


@dataclass(frozen=True)
class MarketSource:
    """Describe one automatic market source and its presentation."""

    # The sidebar option text; also the monitoring and cache-scope identifier.
    label: str
    # Name of the `market_data` loader. It is resolved at call time so tests
    # and alternative providers can replace the module attribute.
    loader_name: str
    # `pandas_market_calendars` identifier for session-aware projections.
    calendar: str
    currency_prefix: str
    price_format: str
    price_axis_label: str
    sidebar_caption: str
    overview_caption: str
    empty_error: str
    # Labels used when the pooled model ranking is unavailable and the view
    # falls back to the source's current daily ordering.
    fallback_leader_label: str
    fallback_performance_label: str
    fallback_heading: str
    chart_heading: str
    # European listings show a reminder that broker access is not implied.
    show_broker_access_note: bool = False

    def load_performers(self) -> pd.DataFrame:
        """Return this source's current candidate leaderboard."""
        import market_data

        return getattr(market_data, self.loader_name)()

    def price_display(self) -> tuple[str, str, str]:
        """Return the prefix, table format, and chart-axis label."""
        return self.currency_prefix, self.price_format, self.price_axis_label


MARKET_SOURCE_REGISTRY: dict[str, MarketSource] = {
    source.label: source
    for source in (
        MarketSource(
            label=IRELAND_SOURCE,
            loader_name="get_iseq20_top_performers",
            calendar="XDUB",
            currency_prefix=EURO_SYMBOL,
            price_format=f"{EURO_SYMBOL}%.2f",
            price_axis_label="Price (EUR)",
            sidebar_caption=(
                "Ranks the tracked ISEQ 20 Euronext Dublin listings by their "
                "latest available daily close."
            ),
            overview_caption=(
                "The leaderboard ranks the latest available daily close from the "
                "tracked ISEQ 20 Euronext Dublin universe."
            ),
            empty_error="No ISEQ 20 price data was returned. Try refreshing in a moment.",
            fallback_leader_label="Top ISEQ 20 daily mover",
            fallback_performance_label="Best ISEQ 20 daily change",
            fallback_heading="ISEQ 20 top daily performers",
            chart_heading="ISEQ 20 candidates - history and feature-based forecast",
            show_broker_access_note=True,
        ),
        MarketSource(
            label=FTSE_MIB_SOURCE,
            loader_name="get_ftse_mib_top_performers",
            calendar="XMIL",
            currency_prefix=EURO_SYMBOL,
            price_format=f"{EURO_SYMBOL}%.2f",
            price_axis_label="Price (EUR)",
            sidebar_caption=(
                "Ranks the Yahoo-supported FTSE MIB constituents by their latest "
                "available daily close and charts the top 10."
            ),
            overview_caption=(
                "The leaderboard ranks the latest available daily close across "
                "Yahoo-supported FTSE MIB constituents."
            ),
            empty_error="No FTSE MIB constituent data was returned. Try refreshing in a moment.",
            fallback_leader_label="Top FTSE MIB daily mover",
            fallback_performance_label="Best FTSE MIB daily change",
            fallback_heading="FTSE MIB top 10 daily performers",
            chart_heading="FTSE MIB candidates - history and feature-based forecast",
            show_broker_access_note=True,
        ),
        MarketSource(
            label=US_SOURCE,
            loader_name="get_us_top_performers",
            calendar="NYSE",
            currency_prefix="$",
            price_format="$%.2f",
            price_axis_label="Price (USD)",
            sidebar_caption=(
                "Uses Yahoo Finance's U.S. large-cap daily-gainers screen and "
                "charts the top 10 equities."
            ),
            overview_caption=(
                "The leaderboard uses Yahoo Finance's predefined U.S. equity "
                "filter for liquid daily gainers."
            ),
            empty_error=(
                "No top performers were returned by the market screener. "
                "Try refreshing in a moment."
            ),
            fallback_leader_label="Top detected daily gainer",
            fallback_performance_label="Best detected daily change",
            fallback_heading="Detected top 10 daily gainers",
            chart_heading="Detected U.S. candidates - history and feature-based forecast",
        ),
    )
}

# The registry and the sidebar option list must never drift apart.
assert tuple(MARKET_SOURCE_REGISTRY) == MARKET_SOURCES


def get_market_source(ticker_source: str | None) -> MarketSource | None:
    """Return the automatic source for a sidebar value, or None for manual mode."""
    return MARKET_SOURCE_REGISTRY.get(str(ticker_source))


def resolve_market_calendar(ticker_source: str | None, ticker: str = "") -> str:
    """Return the best exchange-calendar identifier for the active context.

    Automatic sources use their registered exchange. Manual symbols do not
    carry a source, so their Yahoo suffix selects the most likely exchange.
    """
    source = get_market_source(ticker_source)
    if source is not None:
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
        ticker_source=ticker_source,
        ticker=ticker,
        calendar_name=calendar_name,
    )
    return calendar_name
