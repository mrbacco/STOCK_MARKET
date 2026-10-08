#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app.py
#############################

"""Entry point of the Stock Market Intelligence app.

This script runs before every page: it configures the app, starts the
sentiment collector, renders the shared sidebar, and hands over to the page
selected in the top navigation. Page bodies live in `app_pages/`; shared state
lives in `ui_state.py` and shared renderers in `ui_components.py`.
"""

from __future__ import annotations

import streamlit as st

from app_logging import bac_debug_kv, bac_log_section
from global_markets import ensure_overview_warmup
from runtime_config import RUN_IN_PROCESS_SENTIMENT
from sentiment_service import ensure_background_sentiment_collector
from sentiment_store import update_watchlist
from ui_state import initialize_session_state, render_sidebar

st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon=":material/query_stats:",
    layout="wide",
)
bac_log_section("app", "Streamlit script booting.")

initialize_session_state()
# Local mode runs the news collector inside this process. Production replicas
# leave it to the supervised sentiment worker.
sentiment_collector = (
    ensure_background_sentiment_collector() if RUN_IN_PROCESS_SENTIMENT else None
)

selection = render_sidebar()
# Start loading the Global markets overview in the background right away.
ensure_overview_warmup()
bac_debug_kv(
    "app.selection",
    universe=selection.universe_key,
    tickers=len(selection.tickers),
    period=selection.period,
    horizon=selection.horizon,
)

# Keep the collector's bounded watchlist on the stocks being viewed. It only
# wakes for newly tracked tickers, so ordinary reruns do no news work.
if selection.tickers and update_watchlist(selection.companies):
    if sentiment_collector is not None and hasattr(sentiment_collector, "request_collection"):
        sentiment_collector.request_collection()

page = st.navigation(
    [
        st.Page("app_pages/world_markets.py", title="Global markets", icon=":material/public:", default=True),
        st.Page("app_pages/market_ranking.py", title="Market ranking", icon=":material/leaderboard:"),
        st.Page("app_pages/stock.py", title="Stock", icon=":material/candlestick_chart:"),
        st.Page("app_pages/portfolio.py", title="Portfolio", icon=":material/account_balance_wallet:"),
        st.Page("app_pages/news.py", title="News & sentiment", icon=":material/newspaper:"),
        st.Page("app_pages/model_health.py", title="Model health", icon=":material/monitor_heart:"),
    ],
    position="top",
)
page.run()

st.caption(
    "Prices from Yahoo Finance, headlines from Google News RSS, scored with FinBERT. "
    "Model output is research, not investment advice."
)
bac_log_section("app", "Render cycle completed.")
