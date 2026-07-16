#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app_config.py
#############################

"""Central constants and tiny configuration helpers for the app.

Keeping constants in one module makes the rest of the code easier to scan and
reduces the chance of drift between the data, modeling, and UI layers.
"""

from __future__ import annotations

from typing import Any

US_SCREENER_QUERY = "day_gainers"
AUTO_DETECTED_PERFORMERS = 10
MAX_CHARTED_PERFORMERS = 10

IRELAND_SOURCE = "Ireland: ISEQ 20 leaders"
FTSE_MIB_SOURCE = "Italy: FTSE MIB index"
US_SOURCE = "U.S. daily gainers"
MANUAL_SOURCE = "Manual tickers"
MARKET_SOURCES = (IRELAND_SOURCE, FTSE_MIB_SOURCE, US_SOURCE)

VIEW_OPTIONS = ("Overview", "Charts", "News")
DEFAULT_TICKER_SOURCE = IRELAND_SOURCE
DEFAULT_VIEW = "Overview"

MOMENTUM_PERIODS = 30
BACKTEST_TRAINING_POINTS = 60
MAX_BACKTEST_POINTS = 30
MIN_BACKTEST_POINTS = 5
MODEL_LOOKBACK_POINTS = 180
MIN_MODEL_TRAINING_ROWS = 30
RSI_PERIOD = 14

FTSE_MIB_TICKER = "FTSEMIB.MI"
FTSE_MIB_NAME = "FTSE MIB index"

INTRADAY_FREQUENCIES = {"1m": "1min", "2m": "2min", "5m": "5min"}

# The feature list is centralized here so both the training and inference paths
# always operate on the same ordered set of model inputs.
MODEL_FEATURE_COLUMNS = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "sma_gap_5",
    "sma_gap_10",
    "sma_gap_20",
    "vol_5",
    "vol_20",
    "trend_spread_5_20",
    "drawdown_20",
    "rsi_14",
    "volume_change_1",
    "volume_ratio_5",
    "intraday_return",
    "range_pct",
)

# The Ireland mode is intentionally explicit and finite, rather than using a
# dynamic screener, because the app wants a stable, named market universe.
ISEQ_20_DUBLIN_LISTINGS = {
    "A5G.IR": "AIB Group",
    "BIRG.IR": "Bank of Ireland Group",
    "C5H.IR": "Cairn Homes",
    "DQ7A.IR": "Donegal Investment Group",
    "EG7.IR": "FBD Holdings",
    "GL9.IR": "Glanbia",
    "GVR.IR": "Glenveagh Properties",
    "GRP.IR": "Greencoat Renewables",
    "HMSO.IR": "Hammerson",
    "IR5B.IR": "Irish Continental Group",
    "IRES.IR": "Irish Residential Properties REIT",
    "KMR.IR": "Kenmare Resources",
    "KRZ.IR": "Kerry Group",
    "KRX.IR": "Kingspan Group",
    "MLC.IR": "Malin",
    "MIO.IR": "Mincon Group",
    "OIZ.IR": "Origin Enterprises",
    "PTSB.IR": "Permanent TSB",
    "RYA.IR": "Ryanair Holdings",
    "UPR.IR": "Uniphar",
}


def initialize_session_defaults(session_state: Any) -> None:
    """Seed Streamlit session state with stable defaults on each rerun.

    Streamlit exposes a session-state proxy rather than a plain mutable mapping,
    so this helper intentionally accepts `Any` and uses only the small surface
    area that the app needs: `.get(...)` and item assignment.
    """
    if session_state is None or not hasattr(session_state, "get"):
        return

    try:
        if session_state.get("ticker_source") not in {*MARKET_SOURCES, MANUAL_SOURCE}:
            session_state["ticker_source"] = DEFAULT_TICKER_SOURCE
        if session_state.get("active_view") not in VIEW_OPTIONS:
            session_state["active_view"] = DEFAULT_VIEW
    except Exception:
        # In bare Python execution or other non-Streamlit contexts, session
        # state may not behave like the normal runtime proxy. Silently skip so
        # the app can still be imported or statically checked.
        return


def resolve_price_display(ticker_source: str) -> tuple[str, str, str]:
    """Return the symbol, format string, and axis label for the active market."""
    euro_symbol = "\u20ac"
    if ticker_source in {IRELAND_SOURCE, FTSE_MIB_SOURCE}:
        return euro_symbol, f"{euro_symbol}%.2f", "Price (EUR)"
    if ticker_source == US_SOURCE:
        return "$", "$%.2f", "Price (USD)"
    return "", "%.2f", "Price (listing currency)"


def selected_horizon_label(realtime_mode: bool, interval: str, forecast_points: int) -> str:
    """Translate the numeric horizon into a label that reads naturally in the UI."""
    if realtime_mode:
        return f"{forecast_points} {interval} bars"
    return f"{forecast_points} business days"
