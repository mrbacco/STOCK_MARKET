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

from app_logging import bac_debug_kv

MAX_CHARTED_PERFORMERS = 10

MOMENTUM_PERIODS = 30
BACKTEST_TRAINING_POINTS = 120
MAX_BACKTEST_POINTS = 30
# Model promotion decisions must be based on a meaningful run of genuinely
# paired, out-of-sample forecasts.  Twenty is intentionally conservative while
# remaining practical with the one-year price history fetched by the app.
MIN_BACKTEST_POINTS = 20
MODEL_LOOKBACK_POINTS = 180
MIN_MODEL_TRAINING_ROWS = 30
RSI_PERIOD = 14

# Sentiment collection is intentionally frequent enough to capture short-lived
# news changes without repeatedly hammering the RSS source. The standalone
# worker and the in-process Streamlit collector share these settings.
SENTIMENT_COLLECTION_INTERVAL_SECONDS = 300
SENTIMENT_MAX_NEWS_ITEMS = 20
SENTIMENT_MAX_WATCHLIST = 50
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_FEATURE_WINDOW_HOURS = 24
MIN_SENTIMENT_TRAINING_BARS = 10

INTRADAY_FREQUENCIES = {"1m": "1min", "2m": "2min", "5m": "5min"}

# Live prices can change each minute, but refitting and walk-forward testing
# every model on each poll would create unnecessary CPU load. Intraday models
# use only bars before the currently active bucket of this size.
REALTIME_MODEL_REFRESH_FREQUENCY = "5min"

# `pandas_market_calendars` identifiers for manual symbols, chosen by Yahoo
# suffix. Automatic sources register their exchange in `market_sources.py`.
MARKET_CALENDAR_BY_SUFFIX = {
    ".IR": "XDUB",
    ".MI": "XMIL",
    ".L": "LSE",
    ".DE": "XETR",
    ".PA": "XPAR",
    ".AS": "XAMS",
    ".MC": "XMAD",
    ".TO": "TSX",
    ".AX": "ASX",
    ".T": "JPX",
    ".HK": "HKEX",
    ".SW": "SIX",
    ".BR": "XBRU",
    ".HE": "XHEL",
}

# These times are only a resilience fallback when the optional calendar package
# cannot be imported.  Normal app execution uses the full exchange schedule.
FALLBACK_MARKET_SESSION_HOURS = {
    "NYSE": ("09:30", "16:00"),
    "XDUB": ("08:00", "16:30"),
    "XMIL": ("09:00", "17:30"),
    "LSE": ("08:00", "16:30"),
    "XETR": ("09:00", "17:30"),
    "XPAR": ("09:00", "17:30"),
    "XAMS": ("09:00", "17:30"),
    "XMAD": ("09:00", "17:30"),
    "TSX": ("09:30", "16:00"),
    "ASX": ("10:00", "16:00"),
    "JPX": ("09:00", "15:30"),
    "HKEX": ("09:30", "16:00"),
    "SIX": ("09:00", "17:30"),
    "XBRU": ("09:00", "17:30"),
    "XHEL": ("10:00", "18:30"),
}

# The feature list is centralized here so both the training and inference paths
# always operate on the same ordered set of model inputs.
PRICE_FEATURE_COLUMNS = (
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

# These aggregates are calculated strictly from headlines that were both
# published and first observed before each forecast timestamp.
SENTIMENT_FEATURE_COLUMNS = (
    "sentiment_24h",
    "sentiment_change_24h",
    "news_count_24h",
    "sentiment_disagreement_24h",
    "recency_weighted_sentiment_24h",
    "negative_share_24h",
    "news_volume_shock_24h",
    "source_quality_24h",
    "source_diversity_24h",
    "sentiment_relevance_24h",
    "event_intensity_24h",
)

# Cross-sectional context lets the market-wide model distinguish a stock's own
# momentum from a move shared by the whole selected exchange.  These names are
# centralized because training, inference, diagnostics, and tests all rely on
# the exact same feature order.
MARKET_CONTEXT_FEATURE_COLUMNS = (
    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "relative_strength_5",
    "relative_strength_20",
    "market_volatility_20",
    "market_breadth_1",
    "rolling_beta_20",
    "rolling_correlation_20",
    "log_dollar_volume",
)
PANEL_FEATURE_COLUMNS = (
    *PRICE_FEATURE_COLUMNS,
    *SENTIMENT_FEATURE_COLUMNS,
    *MARKET_CONTEXT_FEATURE_COLUMNS,
)

# Three chronological partitions keep model weighting and final performance
# evaluation separate.  A forecast-horizon gap is inserted between partitions.
PANEL_MIN_BASE_TRAINING_DATES = 60
PANEL_TUNING_DATES = 30
PANEL_EVALUATION_DATES = 30
PANEL_RANDOM_STATE = 42


def selected_horizon_label(realtime_mode: bool, interval: str, forecast_points: int) -> str:
    """Translate the numeric horizon into a label that reads naturally in the UI."""
    if realtime_mode:
        label = f"{forecast_points} {interval} bars"
        bac_debug_kv(
            "app_config.selected_horizon_label",
            realtime_mode=realtime_mode,
            interval=interval,
            forecast_points=forecast_points,
            label=label,
        )
        return label
    label = f"{forecast_points} business days"
    bac_debug_kv(
        "app_config.selected_horizon_label",
        realtime_mode=realtime_mode,
        interval=interval,
        forecast_points=forecast_points,
        label=label,
    )
    return label
