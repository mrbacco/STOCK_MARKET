#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/portfolio.py
#############################

"""Portfolio: what holding the model's top picks would have earned after costs."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model_evidence import assess_ranking_evidence
from portfolio_backtest import top_n_backtest
from ui_components import render_evidence
from ui_state import current_selection, market_analysis, walk_forward_summary

selection = current_selection()
st.subheader(f"Top-N strategy - {selection.label}")
st.caption(
    "Buys the model's highest-ranked stocks in equal weights, holds them for one forecast "
    "horizon, then rebalances. Returns are relative to this universe's equal-weight average, "
    "so a strategy with no skill earns about zero before costs. The bottom-N line holds the "
    "model's lowest-ranked stocks as a control."
)
if selection.is_watchlist:
    st.info("Choose a stock universe in the sidebar; the strategy needs the pooled ranking.",
            icon=":material/info:")
    st.stop()

analysis = market_analysis(selection)
if analysis is None or analysis.ranking.evaluation.empty:
    st.info("Not enough history to evaluate the ranking yet. Try a longer history window.",
            icon=":material/hourglass:")
    st.stop()

render_evidence(
    assess_ranking_evidence(analysis.ranking.diagnostics, walk_forward_summary(selection)),
    compact=True,
)

universe_size = int(np.asarray(analysis.ranking.evaluation.groupby("Date")["Ticker"].nunique()).min())
with st.container(horizontal=True):
    top_n = st.slider("Stocks held", min_value=1, max_value=max(1, universe_size // 2),
                      value=min(5, max(1, universe_size // 2)), key="portfolio_top_n")
    cost_bps = st.slider("Trading cost per trade (basis points)", min_value=0, max_value=50,
                         value=10, key="portfolio_cost_bps",
                         help="Commission plus spread for each purchase or sale, as a share of the amount traded.")

result = top_n_backtest(
    analysis.ranking.evaluation,
    horizon=selection.horizon,
    top_n=top_n,
    cost_bps=cost_bps,
)
if result.periods.empty:
    st.info("The evaluation period is too short for this setting.", icon=":material/info:")
    st.stop()

summary = result.summary
with st.container(horizontal=True):
    st.metric("Net excess return", f"{summary['Cumulative net excess']:+.2f}%", border=True,
              help="Compounded excess return after trading costs over the evaluation period.")
    st.metric("Bottom-N control", f"{summary['Cumulative bottom-N excess']:+.2f}%", border=True)
    st.metric("Information ratio", f"{summary['Information ratio']:.2f}", border=True,
              help="Annualized mean excess return divided by its volatility.")
    st.metric("Hit rate", f"{summary['Hit rate']:.0f}%", border=True,
              help="Share of holding periods with a positive net excess return.")
    st.metric("Max drawdown", f"{summary['Max drawdown']:.2f}%", border=True)
    st.metric("Average turnover", f"{summary['Average turnover']:.0f}%", border=True,
              help="Share of the portfolio replaced at each rebalance.")

periods = result.periods
figure = go.Figure()
for column, name, dash in (
    ("Cumulative net", "Top-N after costs", "solid"),
    ("Cumulative gross", "Top-N before costs", "dot"),
    ("Cumulative bottom-N", "Bottom-N (control)", "dash"),
):
    figure.add_trace(
        go.Scatter(x=periods["Date"], y=periods[column] * 100, name=name, mode="lines+markers",
                   line={"dash": dash})
    )
figure.add_hline(y=0, line_color="gray", line_width=1)
figure.update_layout(
    template="plotly_white", height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10},
    yaxis_title="Cumulative excess return (%)", legend={"orientation": "h", "y": 1.1},
)
st.plotly_chart(figure, key="portfolio_curve")
st.caption(
    f"{int(summary['Periods'])} non-overlapping {selection.horizon}-session holding periods from "
    "the untouched evaluation window. A short window makes these figures noisy; a longer "
    "history window gives a longer test."
)

with st.expander("Holdings by period"):
    st.dataframe(
        periods.assign(**{
            column: periods[column] * 100
            for column in ("Gross excess", "Cost", "Net excess", "Bottom-N excess", "Turnover")
        }),
        column_order=["Date", "Holdings", "Gross excess", "Cost", "Net excess", "Bottom-N excess", "Turnover"],
        column_config={
            "Date": st.column_config.DateColumn("Rebalance date"),
            "Gross excess": st.column_config.NumberColumn(format="%+.2f%%"),
            "Cost": st.column_config.NumberColumn(format="%.3f%%"),
            "Net excess": st.column_config.NumberColumn(format="%+.2f%%"),
            "Bottom-N excess": st.column_config.NumberColumn(format="%+.2f%%"),
            "Turnover": st.column_config.NumberColumn(format="%.0f%%"),
        },
        hide_index=True,
    )
