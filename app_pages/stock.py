#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/stock.py
#############################

"""Stock: one company's key numbers, the tested model's view, price, and news."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from chart_pipeline import build_ticker_forecast, record_displayed_forecast
from global_markets import performance_metrics
from ideas import forecast_cone
from market_data import get_news, get_price_history_batch
from market_sources import WATCHLIST_KEY, company_name, source_for_ticker
from model_evidence import assess_walk_forward_evidence
from runtime_config import RUN_IN_PROCESS_SENTIMENT
from sentiment_store import load_sentiment_history
from ui_components import (
    backtest_caption,
    backtest_column_config,
    forecast_figure,
    price_cone_chart,
    render_forecast_caption,
    render_forecast_status,
)
from ui_state import (
    HOLDING_PERIODS,
    current_selection,
    market_analysis,
    track_record,
    walk_forward_summary,
)

# Six months keeps a one-month forecast cone readable next to the history.
CHART_SESSIONS = 126
EVIDENCE_BADGES = {
    "supported": ":green-badge[:material/verified: Tested edge]",
    "tentative": ":orange-badge[:material/science: Early evidence, not proven]",
    "none": ":gray-badge[No proven edge here]",
    "untested": ":gray-badge[Not tested yet]",
}

selection = current_selection()
# The model view and the forecast cone need the market ranking; it is saved for
# the day, so this is usually instant.
analysis = market_analysis(selection, compute=not selection.is_watchlist)

# Stocks opened from the Global markets page may sit outside this universe.
requested = st.session_state.get("stock_ticker")
options = list(dict.fromkeys([*selection.tickers, *([requested] if requested else [])]))
if not options:
    st.info("Choose a universe or add stocks to your watchlist in the sidebar.", icon=":material/info:")
    st.stop()
if analysis is not None and analysis.has_ranking:
    default_ticker = str(analysis.ranking.ranking["Ticker"].iloc[0])
else:
    default_ticker = options[0]
if st.session_state.get("stock_ticker") not in options:
    st.session_state["stock_ticker"] = default_ticker

ticker = str(
    st.selectbox(
        "Stock",
        options,
        format_func=lambda symbol: f"{symbol} - {company_name(str(symbol))}",
        key="stock_ticker",
    )
)
company = company_name(ticker)
home_source = selection.source if ticker in selection.tickers and selection.source else source_for_ticker(ticker)
price_prefix, price_format, price_axis_label = (
    home_source.price_display() if home_source is not None else ("", "%.2f", "Price (listing currency)")
)
universe_key = home_source.key if home_source is not None else WATCHLIST_KEY

if analysis is not None and ticker in analysis.prices.price_data:
    history = analysis.prices.price_data[ticker]
else:
    with st.spinner(f"Loading {ticker}..."):
        history = get_price_history_batch([ticker], period=selection.period, interval="1d")[ticker]
if history.empty:
    st.error(f"No price history is available for {ticker}.", icon=":material/error:")
    st.stop()

metrics = performance_metrics(history)
st.subheader(f"{company} ({ticker})")
with st.container(horizontal=True):
    st.metric(
        "Last close",
        f"{price_prefix}{float(metrics['Last']):,.2f}",
        f"{float(metrics['1D']):+.2f}%" if pd.notna(metrics["1D"]) else None,
        border=True,
    )
    for label in ("1M", "3M", "YTD"):
        value = metrics[label]
        st.metric(label, f"{float(value):+.2f}%" if pd.notna(value) else "n/a", border=True)
    volatility = metrics["Volatility 20D"]
    st.metric(
        "Volatility (20D)",
        f"{float(volatility):.1f}%" if pd.notna(volatility) else "n/a",
        border=True,
        help="Annualized standard deviation of the last 20 daily returns.",
    )

# --- Model view ----------------------------------------------------------------
period_label = HOLDING_PERIODS.get(selection.horizon, f"{selection.horizon} sessions")
ranking_frame = analysis.ranking.ranking if analysis is not None else pd.DataFrame()
ranking_row = (
    ranking_frame[ranking_frame["Ticker"] == ticker]
    if not ranking_frame.empty and "Ticker" in ranking_frame.columns
    else pd.DataFrame()
)
with st.container(border=True):
    if ranking_row.empty:
        st.markdown(f"**Prediction, next {period_label}**")
        st.caption(
            "The model ranks a whole market at a time. Open Today for this stock's market "
            "to see where it stands."
        )
    else:
        row = ranking_row.iloc[0]
        tested = walk_forward_summary(selection)
        evidence = assess_walk_forward_evidence(tested) if tested else None
        level = evidence.level if evidence is not None else "untested"
        badge = EVIDENCE_BADGES.get(level, "")
        st.markdown(f"**Prediction, next {period_label}** &nbsp; {badge}")
        rank, size = int(row["Rank"]), len(ranking_frame)
        quarter = max(size // 4, 1)
        market_name = selection.label.split(" - ")[-1]
        record = track_record(selection)
        if rank <= quarter:
            sentence = f"{company} should **beat** the {market_name}."
            share = record.get("Top quarter right %")
            group = "top quarter"
        elif rank > size - quarter:
            sentence = f"{company} should **lag** the {market_name}."
            share = record.get("Bottom quarter right %")
            group = "bottom quarter"
        else:
            sentence = f"{company} should move roughly **in line with** the {market_name}."
            share, group = None, ""
        st.markdown(f"##### {sentence}")
        if share is not None:
            st.markdown(
                f":material/history: In testing, stocks the model put in the {group} of this "
                f"market did as called **{share:.0f}%** of the time. One stock is a close call; "
                "a basket of the top picks is more reliable (see Today)."
            )
        with st.container(horizontal=True):
            st.metric("Rank in this market", f"{int(row['Rank'])} of {len(ranking_frame)}", border=True)
            st.metric(
                "Prediction vs the market",
                f"{float(row['Expected excess return']):+.1f}%",
                border=True,
                help="Predicted return minus the average stock in this market.",
            )
            st.metric(
                "Likely range",
                f"{float(row['Lower 80']):+.1f}% to {float(row['Upper 80']):+.1f}%",
                border=True,
                help="8 outcomes in 10 fall in this range, relative to the market.",
            )
        if level not in ("supported", "tentative"):
            st.caption(
                "In this market the model has not shown it can beat chance, so treat these "
                "numbers as unproven."
            )

# --- Price ---------------------------------------------------------------------
recent = history.dropna(subset=["Close"]).tail(CHART_SESSIONS)
cone = None
if not ranking_row.empty:
    forecast_row = ranking_row.iloc[0]
    cone = forecast_cone(
        pd.Timestamp(recent["Date"].iloc[-1]),
        float(recent["Close"].iloc[-1]),
        selection.horizon,
        expected_pct=float(forecast_row["Expected excess return"]),
        low_pct=float(forecast_row["Lower 80"]),
        high_pct=float(forecast_row["Upper 80"]),
    )
st.markdown(
    f"**Price, past 6 months and next {period_label}**"
    if cone is not None
    else "**Price, past 6 months**"
)
st.altair_chart(price_cone_chart(recent, cone, y_label=price_axis_label))
if cone is not None:
    final = cone.iloc[-1]
    st.caption(
        f"Dashed line: expected path to {price_prefix}{final['Expected']:,.2f}. Shaded: likely "
        f"range, {price_prefix}{final['Low']:,.2f} to {price_prefix}{final['High']:,.2f} "
        "(8 outcomes in 10). The model predicts moves against the market, so this assumes "
        "the market itself stays flat."
    )

# --- News ----------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**News**")
    articles = load_sentiment_history(ticker)
    if articles.empty:
        st.caption("No headlines have been collected for this stock yet.")
        if RUN_IN_PROCESS_SENTIMENT and st.button("Fetch headlines now", icon=":material/download:"):
            with st.spinner("Fetching and scoring headlines..."):
                get_news(ticker, company)
            st.rerun()
    else:
        published = pd.to_datetime(articles["published_at"], utc=True)
        recent_week = articles[published >= published.max() - pd.Timedelta(days=7)]
        tone = float(np.nanmean(np.asarray(recent_week["sentiment"], dtype=float)))
        mood = "positive" if tone >= 0.2 else "negative" if tone <= -0.2 else "mixed"
        st.caption(f"{len(recent_week)} headlines in the past week; overall tone is {mood}.")
        st.dataframe(
            articles.sort_values("published_at", ascending=False).head(8),
            column_order=["published_at", "title", "sentiment_label", "link"],
            column_config={
                "published_at": st.column_config.DatetimeColumn("Published", format="MMM DD, HH:mm"),
                "title": "Headline",
                "sentiment_label": "Tone",
                "link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
            hide_index=True,
        )

# --- Advanced ------------------------------------------------------------------
if st.toggle(
    "Show the experimental single-stock price projection",
    key="stock_projection",
    help="A separate model that projects this stock's price on its own. Its backtest is often "
    "no better than assuming no change, so it is hidden by default.",
):
    with st.container(border=True):
        with st.spinner("Fitting the projection and its walk-forward backtest..."):
            forecast = build_ticker_forecast(
                ticker,
                history,
                ticker_source=universe_key,
                realtime_mode=False,
                interval="1d",
                forecast_points=selection.horizon,
            )
        record_displayed_forecast(forecast, monitoring_market=universe_key, ranking=ranking_frame)
        render_forecast_status(forecast)
        st.plotly_chart(
            forecast_figure(forecast, price_axis_label=price_axis_label), key=f"projection_{ticker}"
        )
        render_forecast_caption(forecast, horizon=selection.horizon, price_prefix=price_prefix)
        summary = forecast.backtest_summary()
        if summary is not None:
            st.dataframe(
                pd.DataFrame([summary]),
                column_config=backtest_column_config(price_format),
                hide_index=True,
            )
            st.caption(backtest_caption(selection.horizon))

# --- Intraday ------------------------------------------------------------------
if st.toggle("Show intraday prices (5-minute bars, last 5 sessions)", key="stock_intraday"):
    intraday = get_price_history_batch([ticker], period="5d", interval="5m")[ticker]
    if intraday.empty:
        st.caption("No intraday data is available right now.")
    else:
        st.line_chart(
            intraday[["Date", "Close"]], x="Date", y="Close", y_label=price_axis_label,
            x_label="", height=280,
        )
        st.caption("Yahoo Finance intraday bars can be delayed by up to 15-20 minutes.")
