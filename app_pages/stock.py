#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/stock.py
#############################

"""Stock: one company's price, projection, model view, and news sentiment."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chart_pipeline import build_ticker_forecast, record_displayed_forecast
from global_markets import performance_metrics
from market_data import get_news, get_price_history_batch
from market_sources import WATCHLIST_KEY, company_name, source_for_ticker
from model_evidence import assess_ranking_evidence
from runtime_config import RUN_IN_PROCESS_SENTIMENT
from sentiment_store import load_sentiment_history
from ui_components import (
    backtest_caption,
    backtest_column_config,
    forecast_figure,
    render_forecast_caption,
    render_forecast_status,
)
from ui_state import current_selection, market_analysis

selection = current_selection()
analysis = market_analysis(selection, compute=False)

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

# --- Projection ----------------------------------------------------------------
with st.container(border=True):
    st.markdown(f"**{selection.horizon}-session projection**")
    with st.spinner("Fitting the projection and its walk-forward backtest..."):
        forecast = build_ticker_forecast(
            ticker,
            history,
            ticker_source=universe_key,
            realtime_mode=False,
            interval="1d",
            forecast_points=selection.horizon,
        )
    ranking_frame = analysis.ranking.ranking if analysis is not None else pd.DataFrame()
    record_displayed_forecast(forecast, monitoring_market=universe_key, ranking=ranking_frame)
    render_forecast_status(forecast)
    st.plotly_chart(forecast_figure(forecast, price_axis_label=price_axis_label), key=f"projection_{ticker}")
    render_forecast_caption(forecast, horizon=selection.horizon, price_prefix=price_prefix)
    summary = forecast.backtest_summary()
    if summary is not None:
        with st.expander("Projection backtest"):
            st.dataframe(
                pd.DataFrame([summary]),
                column_config=backtest_column_config(price_format),
                hide_index=True,
            )
            st.caption(backtest_caption(selection.horizon))

# --- Model view ----------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Market model view**")
    ranking_row = (
        ranking_frame[ranking_frame["Ticker"] == ticker]
        if not ranking_frame.empty and "Ticker" in ranking_frame.columns
        else pd.DataFrame()
    )
    if ranking_row.empty:
        st.caption(
            "Open Market ranking for this stock's universe to see where the pooled model "
            "places it."
        )
    else:
        row = ranking_row.iloc[0]
        evidence = assess_ranking_evidence(analysis.ranking.diagnostics if analysis else {})
        with st.container(horizontal=True):
            st.metric("Rank", f"{int(row['Rank'])} of {len(ranking_frame)}", border=True)
            st.metric("Expected excess", f"{float(row['Expected excess return']):+.2f}%", border=True)
            st.metric("Probability outperform", f"{float(row['Probability outperform']):.0f}%", border=True)
            st.metric(
                "80% band",
                f"{float(row['Lower 80']):+.1f}% to {float(row['Upper 80']):+.1f}%",
                border=True,
            )
        st.caption(f"Evidence for this universe: {evidence.headline.lower()}.")

# --- Sentiment -----------------------------------------------------------------
with st.container(border=True):
    st.markdown("**News sentiment**")
    articles = load_sentiment_history(ticker)
    if articles.empty:
        st.caption("No headlines have been collected for this stock yet.")
        if RUN_IN_PROCESS_SENTIMENT and st.button("Fetch headlines now", icon=":material/download:"):
            with st.spinner("Fetching and scoring headlines..."):
                get_news(ticker, company)
            st.rerun()
    else:
        daily = (
            articles.assign(day=pd.to_datetime(articles["published_at"]).dt.normalize())
            .groupby("day")
            .agg(sentiment=("sentiment", "mean"), articles=("title", "size"))
            .tail(60)
        )
        figure = go.Figure()
        figure.add_trace(
            go.Bar(x=daily.index, y=daily["articles"], name="Articles", yaxis="y2",
                   marker_color="rgba(150, 150, 160, 0.35)")
        )
        figure.add_trace(
            go.Scatter(x=daily.index, y=daily["sentiment"], name="Average sentiment",
                       mode="lines+markers")
        )
        figure.update_layout(
            template="plotly_white",
            height=300,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis={"title": "Sentiment (-1 to +1)", "range": [-1, 1]},
            yaxis2={"title": "Articles", "overlaying": "y", "side": "right", "showgrid": False},
            legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(figure, key=f"sentiment_{ticker}")
        recent = articles.sort_values("published_at", ascending=False).head(15)
        st.dataframe(
            recent,
            column_order=["published_at", "source", "title", "sentiment_label", "sentiment", "link"],
            column_config={
                "published_at": st.column_config.DatetimeColumn("Published", format="MMM DD, HH:mm"),
                "source": "Source",
                "title": "Headline",
                "sentiment_label": "Label",
                "sentiment": st.column_config.NumberColumn("Score", format="%+.2f"),
                "link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
            hide_index=True,
        )

# --- Intraday ------------------------------------------------------------------
if st.toggle("Show intraday prices (5-minute bars, last 5 sessions)", key="stock_intraday"):
    intraday = get_price_history_batch([ticker], period="5d", interval="5m")[ticker]
    if intraday.empty:
        st.caption("No intraday data is available right now.")
    else:
        figure = go.Figure(
            go.Scatter(x=intraday["Date"], y=intraday["Close"], mode="lines", name="5-minute close")
        )
        figure.update_layout(
            template="plotly_white", height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10},
            yaxis_title=price_axis_label,
        )
        st.plotly_chart(figure, key=f"intraday_{ticker}")
        st.caption("Yahoo Finance intraday bars can be delayed by up to 15-20 minutes.")
