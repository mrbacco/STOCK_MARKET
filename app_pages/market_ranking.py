#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/market_ranking.py
#############################

"""Market ranking: the pooled model's view of one universe, gated by evidence."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from market_data import SNAPSHOT_PREVIEW_STATUS, growth_score, snapshot_refresh_pending
from model_evidence import assess_ranking_evidence
from runtime_config import ANALYTICS_READ_ONLY
from ui_components import (
    leaderboard_column_config,
    ranking_column_config,
    render_data_health,
    render_evidence,
    render_validation_strip,
    selectable_table,
)
from ui_state import current_selection, leaderboard_frame, market_analysis

selection = current_selection()
price_prefix, price_format, _axis = selection.price_display()
st.subheader(selection.label)
if not selection.tickers:
    st.info("Add stocks to your watchlist in the sidebar.", icon=":material/playlist_add:")
    st.stop()


@st.fragment(run_every="2s")
def await_live_leaderboard(saved_at: str | None) -> None:
    """Rerun the page once background live prices replace the saved preview."""
    if not snapshot_refresh_pending():
        st.rerun()
    saved_time = pd.Timestamp(saved_at).strftime("%H:%M UTC") if saved_at else "an earlier session"
    st.caption(f":orange[Showing prices saved at {saved_time} while live prices load.]")


leaderboard = leaderboard_frame(selection)
if leaderboard.attrs.get("bac_data_status") == SNAPSHOT_PREVIEW_STATUS:
    await_live_leaderboard(leaderboard.attrs.get("bac_fetched_at"))

analysis = market_analysis(selection)
if analysis is None or not analysis.prices.valid_tickers:
    st.error(
        "No price history is available for this selection. Check the tickers and your "
        "connection, then refresh.",
        icon=":material/error:",
    )
    st.stop()

render_data_health(analysis.health, analysis.prices.price_data)
if analysis.resolved_forecasts:
    st.toast(f"Scored {analysis.resolved_forecasts} earlier projections.")

if selection.is_watchlist:
    # A watchlist is too small and mixed for the pooled market model.
    st.info(
        "The pooled ranking model needs a full universe. Watchlist stocks are ordered by "
        "30-session momentum; open a stock for its projection.",
        icon=":material/info:",
    )
    momentum = pd.DataFrame(
        [{
            "Ticker": ticker,
            "Company": selection.companies.get(ticker, ticker),
            "30-session momentum": growth_score(analysis.prices.price_data[ticker]),
            "Last close": float(analysis.prices.price_data[ticker]["Close"].iloc[-1]),
        }
        for ticker in analysis.prices.valid_tickers]
    ).sort_values("30-session momentum", ascending=False)
    selectable_table(
        momentum,
        key="watchlist_momentum",
        column_config={
            "30-session momentum": st.column_config.NumberColumn(format="%+.2f%%"),
            "Last close": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.stop()

ranking = analysis.ranking
evidence = assess_ranking_evidence(ranking.diagnostics)
render_evidence(evidence)

if ranking.ranking.empty:
    if ANALYTICS_READ_ONLY:
        st.info(
            "The analytics worker is preparing this universe, period, and horizon. The daily "
            "moves below are shown until it is ready.",
            icon=":material/hourglass:",
        )
    st.markdown("**Latest daily moves**")
    selectable_table(
        leaderboard,
        key="leaderboard_fallback",
        column_config=leaderboard_column_config(leaderboard, price_format),
    )
    st.stop()

render_validation_strip(ranking.diagnostics)

table = ranking.ranking.copy()
# Company names come from the registry, so they never depend on the leaderboard.
table["Company"] = [selection.companies.get(str(ticker), str(ticker)) for ticker in table["Ticker"]]
columns = [
    "Rank", "Ticker", "Company",
    *(["Signal"] if evidence.show_signals else []),
    "Expected excess return", "Probability outperform", "Lower 80", "Upper 80",
    "Predicted volatility", "Model disagreement", "Sentiment score",
]
st.markdown(
    f"**Model ranking over the next {selection.horizon} sessions**"
    + ("" if evidence.show_signals else " (unproven: scores only, no signals)")
)
selectable_table(
    table,
    key="ranking_table",
    column_order=columns,
    column_config=ranking_column_config(),
)
st.caption(
    "Expected excess is the predicted return relative to this universe's average. The 80% "
    "band is scaled by each stock's GARCH volatility and recalibrated as outcomes arrive. "
    "Select a row to open the stock."
)

with st.expander("Latest daily moves"):
    selectable_table(
        leaderboard,
        key="leaderboard_daily",
        column_config=leaderboard_column_config(leaderboard, price_format),
    )
