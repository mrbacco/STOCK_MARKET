#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app.py
#############################

"""Main Streamlit entry point for the Stock Market Intelligence Dashboard.

The app script is intentionally slimmer now:
- configuration lives in `app_config.py`
- terminal logging lives in `app_logging.py`
- market and news loading lives in `market_data.py`
- forecasting logic lives in `forecasting.py`
- Streamlit page rendering lives in `views.py`

This layout keeps the top-level file easier to read during daily development,
while still letting Streamlit rerun the script normally on every interaction.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_config import (
    DEFAULT_TICKER_SOURCE,
    FTSE_MIB_SOURCE,
    IRELAND_SOURCE,
    MANUAL_SOURCE,
    MAX_CHARTED_PERFORMERS,
    US_SOURCE,
    VIEW_OPTIONS,
    initialize_session_defaults,
    resolve_price_display,
)
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from cache_control import invalidate_market_scope, set_cache_scope
from market_data import (
    get_ftse_mib_top_performers,
    get_iseq20_top_performers,
    get_us_top_performers,
)
from sentiment_service import ensure_background_sentiment_collector
from sentiment_store import get_collector_status, update_watchlist
from runtime_config import LIVE_CHART_REFRESH_SECONDS, RUN_IN_PROCESS_SENTIMENT
from ticker_catalog import (
    DEFAULT_MANUAL_MARKET,
    format_manual_ticker_option,
    get_manual_market_preset,
    initialize_manual_market_state,
    manual_market_labels,
    normalize_manual_tickers,
)
from views import (
    render_charts_view,
    render_live_charts_view,
    render_news_view,
    render_overview_view,
)

st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)

# This first log line makes it easy to spot the start of a brand-new rerun.
bac_log_section("app", "Streamlit script booting.")

st.title("Stock Market Intelligence Dashboard")
st.caption("Multi-market data, investing news, sentiment trends, and feature-based forecast charts")

# Session defaults are re-applied on each rerun so stale widget state does not
# leave the app in an invalid branch after a code or widget change.
initialize_session_defaults(st.session_state)
initialize_manual_market_state(st.session_state)
# Local mode starts the daemon before any market-data branch can call `st.stop()`.
# Production replicas leave this disabled because the supervised worker owns
# collection independently of browser sessions and web-process restarts.
sentiment_collector = (
    ensure_background_sentiment_collector()
    if RUN_IN_PROCESS_SENTIMENT
    else None
)
bac_log_kv(
    "app.sentiment_runtime",
    in_process_collector=RUN_IN_PROCESS_SENTIMENT,
    worker_mode=not RUN_IN_PROCESS_SENTIMENT,
)
bac_log_kv(
    "app.session_defaults",
    active_view=st.session_state.get("active_view"),
    ticker_source=st.session_state.get("ticker_source"),
    manual_market=st.session_state.get("manual_market"),
)

# These defaults keep variables available after the sidebar closes, even when
# the automatic Ireland, Italy, or U.S. sources are currently selected.
manual_market_label = st.session_state.get("manual_market", DEFAULT_MANUAL_MARKET)
manual_market_preset = get_manual_market_preset(manual_market_label)
manual_ticker_choices: list[str] = []

# The sidebar owns all user-controlled parameters for market source, history
# mode, and which page view should render below.
with st.sidebar:
    st.header("Configuration")

    ticker_source = st.segmented_control(
        "Ticker source",
        [IRELAND_SOURCE, FTSE_MIB_SOURCE, US_SOURCE, MANUAL_SOURCE],
        required=True,
        key="ticker_source",
        width="stretch",
    )
    cache_scope = str(ticker_source or DEFAULT_TICKER_SOURCE)
    bac_log_kv("app.sidebar", ticker_source=ticker_source)

    if ticker_source == MANUAL_SOURCE:
        # Manual mode first narrows the context to a geographical exchange. This
        # makes ticker discovery easier and gives charts the correct currency.
        manual_market_label = st.selectbox(
            "Geographical area / stock market",
            manual_market_labels(),
            key="manual_market",
        )
        manual_market_preset = get_manual_market_preset(manual_market_label)
        bac_log_kv(
            "app.sidebar.manual_market",
            market=manual_market_label,
            exchange=manual_market_preset.exchange,
            yahoo_suffix=manual_market_preset.yahoo_suffix,
        )

        # A searchable multiselect is more approachable than a blank text box.
        # `accept_new_options=True` preserves the original manual-entry freedom.
        manual_ticker_choices = st.multiselect(
            "Available tickers",
            options=manual_market_preset.ticker_symbols,
            format_func=lambda ticker: format_manual_ticker_option(
                str(ticker),
                manual_market_preset,
            ),
            max_selections=MAX_CHARTED_PERFORMERS,
            accept_new_options=True,
            placeholder="Choose shares or type a Yahoo Finance symbol",
            help=(
                "Select from the examples or type another ticker. "
                "Press Enter to add a custom symbol."
            ),
            key=f"manual_tickers_{manual_market_preset.key}",
        )
        suffix_guidance = (
            f"Unsuffixed custom symbols automatically receive `{manual_market_preset.yahoo_suffix}`."
            if manual_market_preset.yahoo_suffix
            else "Enter custom symbols using Yahoo Finance ticker notation."
        )
        st.caption(
            f"{manual_market_preset.description} {suffix_guidance} "
            "The selected market controls the currency labels."
        )
        bac_log_list_preview(
            "app.sidebar.manual_market",
            "manual_ticker_choices",
            [str(choice) for choice in manual_ticker_choices],
        )
        # Manual portfolios receive their own generation namespace. Refreshing
        # one user's selection does not evict analytics for every manual user.
        normalized_scope_tickers = sorted(
            str(choice).upper().strip()
            for choice in manual_ticker_choices
            if str(choice).strip()
        )
        cache_scope = ":".join(
            [
                str(ticker_source),
                manual_market_preset.key,
                "|".join(normalized_scope_tickers) or "empty",
            ]
        )
    elif ticker_source == IRELAND_SOURCE:
        st.caption(
            "Ranks the tracked ISEQ 20 Euronext Dublin listings by their latest available daily close."
        )
    elif ticker_source == FTSE_MIB_SOURCE:
        st.caption(
            "Ranks the Yahoo-supported FTSE MIB constituents by their latest available daily close and charts the top 10."
        )
    else:
        st.caption(
            "Uses Yahoo Finance's U.S. large-cap daily-gainers screen and charts the top 10 equities."
        )

    realtime_mode = st.toggle("Real-time Mode", value=False)
    bac_log_kv("app.sidebar", realtime_mode=realtime_mode)

    # The history controls are mode-specific so the user sees only relevant options.
    if realtime_mode:
        live_updates_enabled = st.toggle(
            "Live chart updates",
            value=True,
            help=(
                "Refresh the Charts view automatically using free Yahoo Finance "
                f"polling every {LIVE_CHART_REFRESH_SECONDS} seconds."
            ),
        )
        period = st.selectbox("Intraday Window", ["1d", "5d"], index=0)
        interval = st.selectbox("Intraday Interval", ["1m", "2m", "5m"], index=0)
        forecast_points = st.slider("Forecast horizon bars", min_value=7, max_value=60, value=30)
    else:
        live_updates_enabled = False
        period = st.selectbox("History Window", ["6mo", "1y", "2y"], index=1)
        interval = "1d"
        forecast_points = st.slider(
            "Forecast horizon business days",
            min_value=1,
            max_value=5,
            value=3,
            help="Sentiment is initially validated on short, news-sensitive horizons.",
        )
    bac_log_kv(
        "app.sidebar",
        period=period,
        interval=interval,
        forecast_points=forecast_points,
        live_updates_enabled=live_updates_enabled,
    )

    # Period and interval complete the namespace, so refreshing a 1-minute view
    # does not evict another session's daily or long-history cache entries.
    cache_scope = f"{cache_scope}:{period}:{interval}"

    # ContextVar state is isolated to this script thread and is inherited by
    # downstream data/model calls made during the current rerun.
    set_cache_scope(cache_scope)

    if st.button("Refresh now", type="primary"):
        refresh_scope = cache_scope
        market_generation, model_generation = invalidate_market_scope(refresh_scope)
        bac_log_kv(
            "app.sidebar.refresh",
            scope=refresh_scope,
            market_generation=market_generation,
            model_generation=model_generation,
        )
        st.rerun()

    active_view = st.segmented_control(
        "View",
        VIEW_OPTIONS,
        required=True,
        key="active_view",
        width="stretch",
    )
    active_view = active_view or "Overview"
    bac_log_kv("app.sidebar", active_view=active_view)

ticker_source = ticker_source or DEFAULT_TICKER_SOURCE
if ticker_source == MANUAL_SOURCE:
    # Manual mode can represent several currencies, so its market preset owns
    # display formatting instead of the automatic-source configuration helper.
    price_prefix, price_format, price_axis_label = manual_market_preset.price_display()
else:
    price_prefix, price_format, price_axis_label = resolve_price_display(ticker_source)
bac_log_kv(
    "app.display",
    ticker_source=ticker_source,
    manual_market=manual_market_label if ticker_source == MANUAL_SOURCE else None,
    price_prefix=price_prefix,
    price_format=price_format,
    price_axis_label=price_axis_label,
)

# Ticker resolution happens after the sidebar so each rerun uses the latest UI state.
if ticker_source == IRELAND_SOURCE:
    detected_performers = get_iseq20_top_performers()
    tickers = detected_performers["Ticker"].tolist()
elif ticker_source == FTSE_MIB_SOURCE:
    detected_performers = get_ftse_mib_top_performers()
    tickers = detected_performers["Ticker"].tolist()
elif ticker_source == US_SOURCE:
    detected_performers = get_us_top_performers()
    tickers = detected_performers["Ticker"].tolist()
else:
    tickers = normalize_manual_tickers(
        manual_ticker_choices,
        manual_market_preset,
        max_tickers=MAX_CHARTED_PERFORMERS,
    )
    # Known company names improve Google News searches. Custom symbols fall back
    # to their ticker, which preserves the previous manual-mode behavior.
    detected_performers = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Company": manual_market_preset.company_name(ticker),
            }
            for ticker in tickers
        ]
    )

bac_log_kv(
    "app.tickers",
    detected_rows=len(detected_performers),
    ticker_count=len(tickers),
    manual_market=manual_market_label if ticker_source == MANUAL_SOURCE else None,
)
bac_log_list_preview("app.tickers", "resolved_tickers", tickers)

if realtime_mode:
    if live_updates_enabled:
        st.info(
            f"Live chart polling is on. The Charts view refreshes every "
            f"{LIVE_CHART_REFRESH_SECONDS} seconds while this browser tab is active."
        )
    else:
        st.info("Live chart polling is off. Click 'Refresh now' to update values.")

if ticker_source in {IRELAND_SOURCE, FTSE_MIB_SOURCE}:
    st.caption(
        "This is a European market view, not a broker eligibility check. "
        "Confirm that your broker gives your account access to the relevant exchange."
    )

bac_log_kv(
    "app.run_context",
    ticker_source=ticker_source,
    tickers=tickers,
    period=period,
    interval=interval,
    realtime_mode=realtime_mode,
    forecast_points=forecast_points,
    active_view=active_view,
)

# Stop early if no tickers exist for the chosen mode, and make the reason clear
# both in the UI and in the terminal logs.
if not tickers:
    bac_log_section("app", "No tickers were available for the chosen source.")
    if ticker_source == IRELAND_SOURCE:
        st.error("No ISEQ 20 price data was returned. Try refreshing in a moment.")
    elif ticker_source == FTSE_MIB_SOURCE:
        st.error("No FTSE MIB constituent data was returned. Try refreshing in a moment.")
    elif ticker_source == US_SOURCE:
        st.error("No top performers were returned by the market screener. Try refreshing in a moment.")
    else:
        st.warning("Choose at least one ticker from the selected market, or type a custom symbol.")
    st.stop()

# Persist the current universe before the shared worker starts. The collector
# continues polling this bounded watchlist independently of Streamlit reruns.
company_by_ticker = (
    {
        str(ticker): str(company)
        for ticker, company in detected_performers.set_index("Ticker")["Company"].to_dict().items()
    }
    if not detected_performers.empty and "Company" in detected_performers.columns
    else {ticker: ticker for ticker in tickers}
)
update_watchlist({ticker: company_by_ticker.get(ticker, ticker) for ticker in tickers})
if sentiment_collector is not None and hasattr(sentiment_collector, "request_collection"):
    sentiment_collector.request_collection()
else:
    # Streamlit's offline AppTest harness replaces the cached background
    # resource with `None`.  The app remains renderable in that deterministic
    # environment while normal runtime processes still wake the real daemon.
    bac_log_section(
        "app.sentiment_collector",
        "Immediate wake-up unavailable in this runtime.",
    )
collector_status = get_collector_status()
collector_mode = (
    "in-process every 5 minutes"
    if RUN_IN_PROCESS_SENTIMENT
    else "dedicated worker"
)
st.caption(
    f"Sentiment collector: {collector_mode} · "
    f"{collector_status.get('article_count', 0)} stored articles · "
    f"{collector_status.get('watchlist_count', 0)} tracked tickers."
)

# Delegate the heavy rendering work to the dedicated view modules.
if active_view == "Overview":
    bac_log_section("app", "Rendering overview view.")
    render_overview_view(
        ticker_source=ticker_source,
        tickers=tickers,
        detected_performers=detected_performers,
        realtime_mode=realtime_mode,
        period=period,
        interval=interval,
        price_prefix=price_prefix,
        price_format=price_format,
    )
elif active_view == "Charts":
    bac_log_section("app", "Rendering charts view.")
    if realtime_mode and live_updates_enabled:
        bac_log_kv(
            "app.live_charts",
            status="enabled",
            refresh_seconds=LIVE_CHART_REFRESH_SECONDS,
            cache_scope=cache_scope,
        )
        render_live_charts_view(
            ticker_source=ticker_source,
            tickers=tickers,
            detected_performers=detected_performers,
            period=period,
            interval=interval,
            forecast_points=forecast_points,
            price_prefix=price_prefix,
            price_axis_label=price_axis_label,
            price_format=price_format,
            cache_scope=cache_scope,
        )
    else:
        render_charts_view(
            ticker_source=ticker_source,
            tickers=tickers,
            detected_performers=detected_performers,
            realtime_mode=realtime_mode,
            period=period,
            interval=interval,
            forecast_points=forecast_points,
            price_prefix=price_prefix,
            price_axis_label=price_axis_label,
            price_format=price_format,
        )
else:
    bac_log_section("app", "Rendering news view.")
    render_news_view(
        ticker_source=ticker_source,
        tickers=tickers,
        detected_performers=detected_performers,
    )

st.caption(
    "Data sources: Yahoo Finance (prices) and Google News RSS (headlines). "
    "Ireland mode ranks a tracked ISEQ 20 Euronext Dublin universe by the latest available daily close. "
    "Forecast quality is measured with a horizon-matched walk-forward backtest and no-change baseline. "
    "The sentiment candidate is promoted only after it beats the price-only model in recent walk-forward MAE. "
    "This dashboard provides directional insight only, not investment advice."
)

bac_log_section("app", "Render cycle completed.")
