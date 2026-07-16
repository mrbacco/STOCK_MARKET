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
MOMENTUM_PERIODS = 30
MIN_FORECAST_TRAINING_POINTS = 30
BACKTEST_TRAINING_POINTS = 60
MAX_BACKTEST_POINTS = 30
MIN_BACKTEST_POINTS = 5
INTRADAY_FREQUENCIES = {"1m": "1min", "2m": "2min", "5m": "5min"}
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


def linear_trend_projection(prices: np.ndarray, points_ahead: int) -> np.ndarray:
    x_values = np.arange(len(prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_values, prices)

    future_x_values = np.arange(len(prices), len(prices) + points_ahead).reshape(-1, 1)
    return model.predict(future_x_values)


def forecast_trend(close_series: pd.Series, points_ahead: int = 30) -> pd.DataFrame:
    prices = close_series.dropna().to_numpy(dtype=float)
    if len(prices) < MIN_FORECAST_TRAINING_POINTS:
        return pd.DataFrame()

    # Fit a lightweight trend model for directional forecasting.
    predictions = linear_trend_projection(prices, points_ahead)

    return pd.DataFrame(
        {"projection_point": np.arange(1, points_ahead + 1), "pred_close": predictions}
    )


@st.cache_data(max_entries=100)
def backtest_linear_trend(
    price_history: pd.DataFrame,
    training_points: int = BACKTEST_TRAINING_POINTS,
    max_test_points: int = MAX_BACKTEST_POINTS,
) -> pd.DataFrame:
    required_columns = {"Date", "Close"}
    if not required_columns.issubset(price_history.columns):
        return pd.DataFrame()

    history = price_history[["Date", "Close"]].copy()
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history.dropna().sort_values("Date").reset_index(drop=True)

    available_test_points = len(history) - training_points
    test_points = min(max_test_points, available_test_points)
    if test_points < MIN_BACKTEST_POINTS:
        return pd.DataFrame()

    test_start_index = len(history) - test_points
    rows = []
    for target_index in range(test_start_index, len(history)):
        training_prices = history["Close"].iloc[target_index - training_points : target_index].to_numpy(
            dtype=float
        )
        last_training_close = float(training_prices[-1])
        predicted_close = float(linear_trend_projection(training_prices, points_ahead=1)[0])
        actual_close = float(history["Close"].iloc[target_index])

        predicted_direction = np.sign(predicted_close - last_training_close)
        actual_direction = np.sign(actual_close - last_training_close)
        rows.append(
            {
                "date": history["Date"].iloc[target_index],
                "actual_close": actual_close,
                "predicted_close": predicted_close,
                "baseline_close": last_training_close,
                "absolute_error": abs(actual_close - predicted_close),
                "baseline_absolute_error": abs(actual_close - last_training_close),
                "direction_correct": bool(predicted_direction == actual_direction),
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(ticker: str, backtest: pd.DataFrame) -> dict:
    model_mae = float(backtest["absolute_error"].mean())
    baseline_mae = float(backtest["baseline_absolute_error"].mean())
    mape = float(
        (
            backtest["absolute_error"] / backtest["actual_close"].abs().replace(0, np.nan)
        ).mean()
        * 100
    )
    directional_accuracy = float(backtest["direction_correct"].mean() * 100)
    mae_improvement = (
        ((baseline_mae - model_mae) / baseline_mae) * 100 if baseline_mae > 0 else np.nan
    )

    return {
        "Ticker": ticker,
        "Forecasts": len(backtest),
        "Model MAE": model_mae,
        "MAPE": mape,
        "Directional accuracy": directional_accuracy,
        "No-change MAE": baseline_mae,
        "MAE improvement vs. no-change": mae_improvement,
    }


def future_projection_dates(
    last_date: pd.Timestamp, points_ahead: int, realtime_mode: bool, interval: str
) -> pd.DatetimeIndex:
    last_timestamp = pd.Timestamp(last_date)
    if realtime_mode:
        frequency = INTRADAY_FREQUENCIES[interval]
        return pd.date_range(
            start=last_timestamp + pd.Timedelta(frequency),
            periods=points_ahead,
            freq=frequency,
        )
    return pd.bdate_range(start=last_timestamp + pd.offsets.BDay(1), periods=points_ahead)


def growth_score(df: pd.DataFrame, periods: int = MOMENTUM_PERIODS) -> float:
    if df.empty or len(df) < periods + 1:
        return float("-inf")
    start = df["Close"].iloc[-(periods + 1)]
    end = df["Close"].iloc[-1]
    if start == 0:
        return float("-inf")
    return ((end - start) / start) * 100.0


def momentum_label(realtime_mode: bool, interval: str) -> str:
    if realtime_mode:
        return f"{MOMENTUM_PERIODS}-bar ({interval})"
    return f"{MOMENTUM_PERIODS}-session"


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
        forecast_points = st.slider("Trend projection bars", min_value=7, max_value=60, value=30)
    else:
        period = st.selectbox("History Window", ["6mo", "1y", "2y"], index=1)
        interval = "1d"
        forecast_points = st.slider("Trend projection business days", min_value=7, max_value=60, value=30)

    if st.button("Refresh now", type="primary"):
        bac_log("Manual refresh requested by user")
        st.cache_data.clear()
        st.rerun()

if realtime_mode:
    st.info("Real-time mode is using manual refresh. Click 'Refresh now' to update values.")

# Search terminal for [BAC_LOG] to track what the app is processing.
bac_log(
    f"Input tickers={tickers}, period={period}, interval={interval}, realtime_mode={realtime_mode}, forecast_points={forecast_points}"
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
current_momentum_label = momentum_label(realtime_mode, interval)
bac_log(f"Top growers (30-day score)={ranked[: min(3, len(ranked))]}")

col1, col2, col3 = st.columns(3)
col1.metric("Tracked Tickers", len(valid_tickers))
col2.metric(f"Top {current_momentum_label} mover", top_growers[0])
col3.metric(f"Best {current_momentum_label} growth", f"{ranked[0][1]:.2f}%")

if realtime_mode:
    quote_cols = st.columns(min(3, len(top_growers)))
    for i, ticker in enumerate(top_growers[:3]):
        price_series = price_data[ticker]["Close"].dropna()
        if len(price_series) >= 2:
            current_price = float(price_series.iloc[-1])
            previous_price = float(price_series.iloc[-2])
            delta_value = current_price - previous_price
            quote_cols[i].metric(
                f"{ticker} latest {interval} close",
                f"${current_price:.2f}",
                f"{delta_value:+.2f} vs. prior bar",
            )
    st.caption(
        "Intraday figures use the latest returned bar close. The delta is versus the prior bar, "
        "not a live tick or daily change."
    )

st.subheader("Top momentum stocks - history and linear trend projection")
st.caption(
    "The projection extrapolates the price trend only; it is not a price target. "
    "Use the walk-forward backtest below to judge how it performed on recent unseen data."
)
for ticker in top_growers:
    df = price_data[ticker]
    fc = forecast_trend(df["Close"], points_ahead=forecast_points)
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
        future_dates = future_projection_dates(last_date, len(fc), realtime_mode, interval)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=fc["pred_close"],
                mode="lines",
                name=f"{ticker} Linear trend projection",
                line={"dash": "dash", "width": 2},
            )
        )

    fig.update_layout(
        title=f"{ticker}: Price Trend & Linear Projection",
        xaxis_title="Timestamp" if realtime_mode else "Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig)

st.subheader("Forecast backtest")
forecast_horizon_label = f"next {interval} bar" if realtime_mode else "next daily session"
st.caption(
    f"Walk-forward test of up to {MAX_BACKTEST_POINTS} unseen {forecast_horizon_label} forecasts. "
    f"Each forecast is trained only on the preceding {BACKTEST_TRAINING_POINTS} observations and "
    "compared with a no-change baseline."
)

backtest_rows = []
for ticker in top_growers:
    backtest = backtest_linear_trend(price_data[ticker])
    if not backtest.empty:
        backtest_rows.append(summarize_backtest(ticker, backtest))

if backtest_rows:
    st.dataframe(
        pd.DataFrame(backtest_rows),
        column_config={
            "Model MAE": st.column_config.NumberColumn("Model MAE", format="$%.2f"),
            "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
            "Directional accuracy": st.column_config.NumberColumn(
                "Directional accuracy", format="%.1f%%"
            ),
            "No-change MAE": st.column_config.NumberColumn("No-change MAE", format="$%.2f"),
            "MAE improvement vs. no-change": st.column_config.NumberColumn(
                "MAE improvement vs. no-change", format="%.1f%%"
            ),
        },
        hide_index=True,
    )
    st.caption(
        "Positive MAE improvement means the linear trend model beat the no-change baseline; "
        "negative values mean it performed worse."
    )
else:
    st.info(
        "Not enough price observations to backtest this trend model. "
        f"At least {BACKTEST_TRAINING_POINTS + MIN_BACKTEST_POINTS} observations are required."
    )

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
    st.plotly_chart(bar)

    st.dataframe(
        all_news[["ticker", "published", "title", "sentiment_label", "sentiment", "link"]]
        .sort_values(by="published", ascending=False)
        .reset_index(drop=True),
    )
else:
    bac_log("No news rows available from RSS fetch at this run")
    st.info("No news items were fetched right now. Try again in a moment.")

st.caption(
    "Data sources: Yahoo Finance (prices) and Google News RSS (headlines). "
    "Forecast quality is measured with a walk-forward backtest and no-change baseline. "
    "This dashboard provides directional insight only, not investment advice."
)
