#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app.py
#############################

import datetime as dt
from typing import List

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
analyzer = SentimentIntensityAnalyzer()


def bac_log(message: str) -> None:
    # Print debug-friendly logs so they are visible in the terminal window.
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BAC_LOG] {timestamp} | {message}")


def parse_tickers(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    # Remove duplicates while preserving order.
    seen = set()
    clean = []
    for t in parts:
        if t not in seen:
            seen.add(t)
            clean.append(t)
    return clean


@st.cache_data(ttl=60)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=period, interval=interval)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist


@st.cache_data(ttl=30)
def get_price_history_batch(tickers: List[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    # Batch fetch is much faster and more reliable for intraday refreshes.
    result = {t: pd.DataFrame() for t in tickers}
    if not tickers:
        return result

    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
            timeout=4,
        )
    except Exception as ex:
        bac_log(f"Batch download error: {ex}")
        return result

    if data is None or data.empty:
        return result

    # Handle the shape difference between single-ticker and multi-ticker responses.
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            tdf = data[ticker].copy().dropna(how="all")
            if tdf.empty:
                continue
            tdf = tdf.reset_index()[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
            tdf = tdf.rename(columns={"Datetime": "Date"})
            tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.tz_localize(None)
            result[ticker] = tdf
    else:
        single = data.copy().dropna(how="all")
        if not single.empty:
            single = single.reset_index()[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
            single = single.rename(columns={"Datetime": "Date"})
            single["Date"] = pd.to_datetime(single["Date"]).dt.tz_localize(None)
            result[tickers[0]] = single

    return result


@st.cache_data(ttl=900)
def get_news(ticker: str, max_items: int = 20) -> pd.DataFrame:
    url = f"https://news.google.com/rss/search?q={ticker}+stock+investing&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        published = getattr(entry, "published", "")
        summary = getattr(entry, "summary", "")
        title = getattr(entry, "title", "")
        link = getattr(entry, "link", "")
        score = analyzer.polarity_scores(f"{title}. {summary}")["compound"]
        rows.append(
            {
                "ticker": ticker,
                "published": published,
                "title": title,
                "summary": summary,
                "link": link,
                "sentiment": score,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["sentiment_label"] = pd.cut(
        df["sentiment"],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


def forecast_trend(close_series: pd.Series, days_ahead: int = 30) -> pd.DataFrame:
    prices = close_series.dropna().to_numpy(dtype=float)
    if len(prices) < 30:
        return pd.DataFrame()

    # Fit a lightweight trend model for directional forecasting.
    x = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    model = LinearRegression()
    model.fit(x, y)

    future_x = np.arange(len(prices), len(prices) + days_ahead).reshape(-1, 1)
    preds = model.predict(future_x)

    return pd.DataFrame({"day": future_x.flatten(), "pred_close": preds})


def growth_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 30:
        return float("-inf")
    start = df["Close"].iloc[-30]
    end = df["Close"].iloc[-1]
    if start == 0:
        return float("-inf")
    return ((end - start) / start) * 100.0


st.title("Stock Market Intelligence Dashboard")
st.caption("Public data, investing news, sentiment trends, and simple predictive charts")

with st.sidebar:
    st.header("Configuration")
    raw_tickers = st.text_input(
        "Tickers (comma-separated)",
        ", ".join(DEFAULT_TICKERS),
        help="Example: AAPL, MSFT, NVDA",
    )
    tickers = parse_tickers(raw_tickers)
    realtime_mode = st.toggle("Real-time Mode", value=False)

    if realtime_mode:
        period = st.selectbox("Intraday Window", ["1d", "5d"], index=0)
        interval = st.selectbox("Intraday Interval", ["1m", "2m", "5m"], index=0)
        forecast_days = st.slider("Forecast Points", min_value=7, max_value=60, value=30)
    else:
        period = st.selectbox("History Window", ["6mo", "1y", "2y"], index=1)
        interval = "1d"
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=60, value=30)

    if st.button("Refresh now", type="primary"):
        bac_log("Manual refresh requested by user")
        st.cache_data.clear()
        st.rerun()

if realtime_mode:
    st.info("Real-time mode is using manual refresh. Click 'Refresh now' to update values.")

# Search terminal for [BAC_LOG] to track what the app is processing.
bac_log(
    f"Input tickers={tickers}, period={period}, interval={interval}, realtime_mode={realtime_mode}, forecast_points={forecast_days}"
)

if not tickers:
    st.warning("Add at least one ticker symbol.")
    st.stop()

if realtime_mode:
    realtime_tickers = tickers[:3]
    if len(tickers) > 3:
        st.warning("Real-time mode currently tracks the first 3 tickers to keep updates responsive.")
        bac_log(f"Realtime ticker cap applied. Original={len(tickers)} using={realtime_tickers}")
    price_data = get_price_history_batch(realtime_tickers, period=period, interval=interval)
    active_tickers = realtime_tickers
else:
    price_data = {t: get_price_history(t, period=period, interval=interval) for t in tickers}
    active_tickers = tickers

valid_tickers = [t for t in active_tickers if not price_data[t].empty]
bac_log(f"Valid tickers with price data={valid_tickers}")

if realtime_mode and not valid_tickers:
    bac_log("Realtime fetch returned empty. Falling back to daily history for display stability")
    st.warning("Real-time data is temporarily unavailable. Showing recent daily history instead.")
    fallback_tickers = active_tickers[:3]
    price_data = {t: get_price_history(t, period="6mo", interval="1d") for t in fallback_tickers}
    valid_tickers = [t for t in fallback_tickers if not price_data[t].empty]

if not valid_tickers:
    st.error("No price data was returned. Check ticker symbols and try again.")
    st.stop()

scores = {t: growth_score(price_data[t]) for t in valid_tickers}
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top_growers = [t for t, _ in ranked[: min(3, len(ranked))]]
bac_log(f"Top growers (30-day score)={ranked[: min(3, len(ranked))]}")

col1, col2, col3 = st.columns(3)
col1.metric("Tracked Tickers", len(valid_tickers))
col2.metric("Top Grower", top_growers[0])
col3.metric("Best 30-Day Growth", f"{ranked[0][1]:.2f}%")

if realtime_mode:
    quote_cols = st.columns(min(3, len(top_growers)))
    for i, ticker in enumerate(top_growers[:3]):
        price_series = price_data[ticker]["Close"].dropna()
        if len(price_series) >= 2:
            current_price = float(price_series.iloc[-1])
            previous_price = float(price_series.iloc[-2])
            delta_value = current_price - previous_price
            quote_cols[i].metric(f"{ticker} Live", f"${current_price:.2f}", f"{delta_value:+.2f}")

st.subheader("Top Growing Stocks - Historical + Predicted Trend")
for ticker in top_growers:
    df = price_data[ticker]
    fc = forecast_trend(df["Close"], days_ahead=forecast_days)
    bac_log(
        f"Charting ticker={ticker}, close_points={len(df)}, forecast_points={len(fc) if not fc.empty else 0}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name=f"{ticker} Close",
            line={"width": 2},
        )
    )

    # Highlight the most recent value so live updates are obvious.
    fig.add_trace(
        go.Scatter(
            x=[df["Date"].iloc[-1]],
            y=[df["Close"].iloc[-1]],
            mode="markers",
            name=f"{ticker} Latest",
            marker={"size": 10},
        )
    )

    if not fc.empty:
        last_date = df["Date"].iloc[-1]
        future_dates = [last_date + dt.timedelta(days=i + 1) for i in range(len(fc))]
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=fc["pred_close"],
                mode="lines",
                name=f"{ticker} Forecast",
                line={"dash": "dash", "width": 2},
            )
        )

    fig.update_layout(
        title=f"{ticker}: Price Trend & Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Investing News and Sentiment")
news_frames = []
for t in valid_tickers:
    news_df = get_news(t)
    bac_log(f"Fetched news for {t}: rows={len(news_df)}")
    if not news_df.empty:
        news_frames.append(news_df)

if news_frames:
    all_news = pd.concat(news_frames, ignore_index=True)
    bac_log(f"Total combined news rows={len(all_news)}")

    sentiment_by_ticker = (
        all_news.groupby("ticker", as_index=False)["sentiment"].mean().sort_values("sentiment", ascending=False)
    )

    bar = go.Figure(
        data=[
            go.Bar(
                x=sentiment_by_ticker["ticker"],
                y=sentiment_by_ticker["sentiment"],
                marker_color=[
                    "#2ca02c" if s > 0.05 else "#d62728" if s < -0.05 else "#7f7f7f"
                    for s in sentiment_by_ticker["sentiment"]
                ],
                name="Average Sentiment",
            )
        ]
    )
    bar.update_layout(
        title="Average News Sentiment by Ticker",
        xaxis_title="Ticker",
        yaxis_title="Compound Sentiment Score",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(bar, use_container_width=True)

    st.dataframe(
        all_news[["ticker", "published", "title", "sentiment_label", "sentiment", "link"]]
        .sort_values(by="published", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )
else:
    bac_log("No news rows available from RSS fetch at this run")
    st.info("No news items were fetched right now. Try again in a moment.")

st.caption(
    "Data sources: Yahoo Finance (prices) and Google News RSS (headlines). "
    "This dashboard provides directional insight only, not investment advice."
)
