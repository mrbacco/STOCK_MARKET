#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app.py
#############################

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

US_SCREENER_QUERY = "day_gainers"
AUTO_DETECTED_PERFORMERS = 10
MAX_CHARTED_PERFORMERS = 10
IRELAND_SOURCE = "Ireland: ISEQ 20 leaders"
US_SOURCE = "U.S. daily gainers"
MANUAL_SOURCE = "Manual tickers"
MARKET_SOURCES = (IRELAND_SOURCE, US_SOURCE)
VIEW_OPTIONS = ("Overview", "Charts", "News")
MOMENTUM_PERIODS = 30
BACKTEST_TRAINING_POINTS = 60
MAX_BACKTEST_POINTS = 30
MIN_BACKTEST_POINTS = 5
MODEL_LOOKBACK_POINTS = 180
MIN_MODEL_TRAINING_ROWS = 30
RSI_PERIOD = 14
INTRADAY_FREQUENCIES = {"1m": "1min", "2m": "2min", "5m": "5min"}
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


@st.cache_data(ttl=60, max_entries=2)
def get_us_top_performers(limit: int = AUTO_DETECTED_PERFORMERS) -> pd.DataFrame:
    columns = ["Ticker", "Company", "Daily change", "Last price"]
    try:
        response = yf.screen(US_SCREENER_QUERY, count=max(limit * 3, 30))
    except Exception as ex:
        bac_log(f"Top-performer screener error: {ex}")
        return pd.DataFrame(columns=columns)

    rows = []
    seen_tickers = set()
    for quote in response.get("quotes", []):
        if quote.get("quoteType") != "EQUITY":
            continue

        ticker = str(quote.get("symbol", "")).upper()
        if not ticker or ticker in seen_tickers:
            continue

        try:
            daily_change = float(quote.get("regularMarketChangePercent"))
        except (TypeError, ValueError):
            continue

        if not np.isfinite(daily_change):
            continue

        try:
            last_price = float(quote.get("regularMarketPrice"))
        except (TypeError, ValueError):
            last_price = np.nan

        company = (
            quote.get("longName")
            or quote.get("shortName")
            or quote.get("displayName")
            or ticker
        )
        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": daily_change,
                "Last price": last_price,
            }
        )
        seen_tickers.add(ticker)

        if len(rows) >= limit:
            break

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values("Daily change", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def format_price_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    history = history.reset_index()
    date_column = "Datetime" if "Datetime" in history.columns else "Date"
    required_columns = {date_column, "Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(history.columns):
        return pd.DataFrame()

    history = history[[date_column, "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={date_column: "Date"}
    )
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    return history


@st.cache_data(ttl=60)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    return format_price_history(history)


@st.cache_data(ttl=30, max_entries=100)
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

    # Handle both yfinance column orientations and date-index names.
    if isinstance(data.columns, pd.MultiIndex):
        first_level = set(data.columns.get_level_values(0))
        second_level = set(data.columns.get_level_values(1))
        for ticker in tickers:
            if ticker in first_level:
                ticker_data = data[ticker].copy()
            elif ticker in second_level:
                ticker_data = data.xs(ticker, axis=1, level=1).copy()
            else:
                continue
            ticker_frame: pd.DataFrame
            if isinstance(ticker_data, pd.Series):
                ticker_frame = ticker_data.to_frame().T
            else:
                ticker_frame = ticker_data
            result[ticker] = format_price_history(ticker_frame.dropna(how="all"))
    else:
        result[tickers[0]] = format_price_history(data.copy().dropna(how="all"))

    return result


@st.cache_data(ttl=300, max_entries=2)
def get_iseq20_top_performers(limit: int = AUTO_DETECTED_PERFORMERS) -> pd.DataFrame:
    columns = ["Ticker", "Company", "Daily change", "Last price", "Last session"]
    price_data = get_price_history_batch(
        list(ISEQ_20_DUBLIN_LISTINGS),
        period="5d",
        interval="1d",
    )
    rows = []
    for ticker, company in ISEQ_20_DUBLIN_LISTINGS.items():
        history = price_data.get(ticker, pd.DataFrame())
        if history.empty:
            continue

        closes = history[["Date", "Close"]].dropna().sort_values("Date")
        if len(closes) < 2:
            continue

        previous_close = float(closes["Close"].iloc[-2])
        last_price = float(closes["Close"].iloc[-1])
        if previous_close == 0:
            continue

        daily_change = ((last_price - previous_close) / previous_close) * 100
        if not np.isfinite(daily_change):
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": daily_change,
                "Last price": last_price,
                "Last session": pd.Timestamp(closes["Date"].iloc[-1]).date(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values("Daily change", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


@st.cache_data(ttl=900, max_entries=200)
def get_news(ticker: str, company_name: str = "", max_items: int = 20) -> pd.DataFrame:
    query = quote_plus(f"{company_name or ticker} stock investing")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
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


def prepare_model_history(price_history: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(price_history.columns):
        return pd.DataFrame()

    history = price_history[list(required_columns)].copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        history[column] = pd.to_numeric(history[column], errors="coerce")

    history = history.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    history = history.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(
        drop=True
    )
    history = history[(history["Open"] > 0) & (history["High"] > 0) & (history["Low"] > 0)]
    history = history[history["Close"] > 0].copy()
    history["Volume"] = history["Volume"].fillna(0.0).clip(lower=0.0)
    return history.reset_index(drop=True)


def compute_rsi(close_series: pd.Series, window: int = RSI_PERIOD) -> pd.Series:
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss.ne(0), 100.0)
    rsi = rsi.where(average_gain.ne(0), 0.0)
    rsi = rsi.mask(average_gain.eq(0) & average_loss.eq(0), 50.0)
    return rsi.fillna(50.0)


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)

    close = history["Close"]
    volume = history["Volume"].replace(0, np.nan)
    one_bar_log_return = np.log(close).diff()
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    volume_mean_5 = volume.rolling(5).mean()

    features = pd.DataFrame(
        {
            "ret_1": close.pct_change(1),
            "ret_3": close.pct_change(3),
            "ret_5": close.pct_change(5),
            "ret_10": close.pct_change(10),
            "ret_20": close.pct_change(20),
            "sma_gap_5": (close / sma_5) - 1.0,
            "sma_gap_10": (close / sma_10) - 1.0,
            "sma_gap_20": (close / sma_20) - 1.0,
            "vol_5": one_bar_log_return.rolling(5).std(),
            "vol_20": one_bar_log_return.rolling(20).std(),
            "trend_spread_5_20": (sma_5 / sma_20) - 1.0,
            "drawdown_20": (close / close.rolling(20).max()) - 1.0,
            "rsi_14": (compute_rsi(close) - 50.0) / 50.0,
            "volume_change_1": history["Volume"].pct_change(1),
            "volume_ratio_5": (history["Volume"] / volume_mean_5) - 1.0,
            "intraday_return": (history["Close"] - history["Open"]) / history["Open"],
            "range_pct": (history["High"] - history["Low"]) / history["Close"],
        },
        index=history.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


def build_forecast_training_frame(price_history: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    history = prepare_model_history(price_history)
    if history.empty or forecast_horizon < 1:
        return pd.DataFrame()

    feature_frame = build_feature_frame(history)
    target_log_return = np.log(history["Close"].shift(-forecast_horizon) / history["Close"])
    training_frame = pd.concat([history[["Date", "Close"]], feature_frame], axis=1)
    training_frame["target_log_return"] = target_log_return
    training_frame = training_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*MODEL_FEATURE_COLUMNS, "target_log_return"]
    )
    if len(training_frame) > MODEL_LOOKBACK_POINTS:
        training_frame = training_frame.iloc[-MODEL_LOOKBACK_POINTS:]
    return training_frame.reset_index(drop=True)


def fit_forecast_model(training_frame: pd.DataFrame) -> Pipeline | None:
    if len(training_frame) < MIN_MODEL_TRAINING_ROWS:
        return None

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(training_frame.loc[:, MODEL_FEATURE_COLUMNS], training_frame["target_log_return"])
    return model


def predict_horizon_close(price_history: pd.DataFrame, forecast_horizon: int) -> dict:
    history = prepare_model_history(price_history)
    if history.empty:
        return {}

    feature_frame = build_feature_frame(history)
    if feature_frame.empty:
        return {}

    latest_features = feature_frame.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if latest_features.loc[:, MODEL_FEATURE_COLUMNS].isna().any(axis=None):
        return {}

    training_frame = build_forecast_training_frame(history, forecast_horizon)
    model = fit_forecast_model(training_frame)
    if model is None:
        return {}

    predicted_log_return = float(model.predict(latest_features.loc[:, MODEL_FEATURE_COLUMNS])[0])
    last_close = float(history["Close"].iloc[-1])
    predicted_close = last_close * float(np.exp(predicted_log_return))
    predicted_return = float(np.exp(predicted_log_return) - 1.0)
    return {
        "forecast_horizon": forecast_horizon,
        "predicted_close": predicted_close,
        "predicted_return": predicted_return,
        "last_close": last_close,
        "training_rows": int(len(training_frame)),
    }


@st.cache_data(max_entries=100)
def forecast_feature_model(price_history: pd.DataFrame, points_ahead: int = 30) -> pd.DataFrame:
    rows = []
    for forecast_horizon in range(1, points_ahead + 1):
        prediction = predict_horizon_close(price_history, forecast_horizon)
        if not prediction:
            break
        rows.append(
            {
                "projection_point": forecast_horizon,
                "pred_close": prediction["predicted_close"],
                "pred_return": prediction["predicted_return"],
                "training_rows": prediction["training_rows"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(max_entries=100)
def backtest_forecast_model(
    price_history: pd.DataFrame,
    forecast_horizon: int,
    training_points: int = BACKTEST_TRAINING_POINTS,
    max_test_points: int = MAX_BACKTEST_POINTS,
) -> pd.DataFrame:
    history = prepare_model_history(price_history)
    if history.empty or forecast_horizon < 1:
        return pd.DataFrame()

    available_test_points = len(history) - training_points - forecast_horizon + 1
    test_points = min(max_test_points, available_test_points)
    if test_points < MIN_BACKTEST_POINTS:
        return pd.DataFrame()

    test_start_training_end = len(history) - forecast_horizon - test_points
    rows = []
    for training_end_index in range(test_start_training_end, len(history) - forecast_horizon):
        training_start_index = training_end_index - training_points + 1
        training_slice = history.iloc[training_start_index : training_end_index + 1].copy()
        prediction = predict_horizon_close(training_slice, forecast_horizon)
        if not prediction:
            continue

        baseline_close = float(training_slice["Close"].iloc[-1])
        predicted_close = float(prediction["predicted_close"])
        actual_close = float(history["Close"].iloc[training_end_index + forecast_horizon])
        predicted_direction = np.sign(predicted_close - baseline_close)
        actual_direction = np.sign(actual_close - baseline_close)
        rows.append(
            {
                "date": history["Date"].iloc[training_end_index + forecast_horizon],
                "actual_close": actual_close,
                "predicted_close": predicted_close,
                "baseline_close": baseline_close,
                "absolute_error": abs(actual_close - predicted_close),
                "baseline_absolute_error": abs(actual_close - baseline_close),
                "direction_correct": bool(predicted_direction == actual_direction),
            }
        )

    return pd.DataFrame(rows)


def confidence_label(mae_improvement: float, directional_accuracy: float) -> str:
    if np.isnan(mae_improvement) or np.isnan(directional_accuracy):
        return "Unavailable"
    if mae_improvement >= 10 and directional_accuracy >= 58:
        return "High"
    if mae_improvement >= 0 and directional_accuracy >= 52:
        return "Moderate"
    return "Low"


def summarize_backtest(ticker: str, backtest: pd.DataFrame, forecast_horizon: int) -> dict:
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
        "Horizon": forecast_horizon,
        "Forecasts": len(backtest),
        "Model MAE": model_mae,
        "MAPE": mape,
        "Directional accuracy": directional_accuracy,
        "No-change MAE": baseline_mae,
        "MAE improvement vs. no-change": mae_improvement,
        "Confidence": confidence_label(mae_improvement, directional_accuracy),
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


def load_news_frames_parallel(tickers: List[str], company_by_ticker: dict[str, str]) -> list[pd.DataFrame]:
    if not tickers:
        return []

    news_frames: list[pd.DataFrame] = []
    max_workers = min(4, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_news, ticker, company_by_ticker.get(ticker, "")): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                news_df = future.result()
            except Exception as ex:
                bac_log(f"News fetch error for {ticker}: {ex}")
                continue
            bac_log(f"Fetched news for {ticker}: rows={len(news_df)}")
            if not news_df.empty:
                news_frames.append(news_df)

    return news_frames


def render_overview_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
    realtime_mode: bool,
    period: str,
    interval: str,
    price_prefix: str,
    price_format: str,
) -> None:
    st.subheader("Overview")
    st.caption(
        "This view focuses on the current market universe and keeps heavier chart and sentiment work off the page until you need it."
    )

    if ticker_source in MARKET_SOURCES:
        if ticker_source == IRELAND_SOURCE:
            st.caption(
                "The leaderboard ranks the latest available daily close from the tracked ISEQ 20 Euronext Dublin universe."
            )
        else:
            st.caption(
                "The leaderboard uses Yahoo Finance's predefined U.S. equity filter for liquid daily gainers."
            )

        leaderboard_columns = {
            "Daily change": st.column_config.NumberColumn("Daily change", format="%.2f%%"),
            "Last price": st.column_config.NumberColumn("Last price", format=price_format),
        }
        if "Last session" in detected_performers.columns:
            leaderboard_columns["Last session"] = st.column_config.DateColumn("Last session")

        st.dataframe(detected_performers, column_config=leaderboard_columns, hide_index=True)
        return

    overview_tickers = tickers[:3]
    if not overview_tickers:
        st.warning("Add at least one ticker symbol.")
        return

    with st.spinner("Loading overview snapshot..."):
        price_data = get_price_history_batch(overview_tickers, period=period, interval=interval)

    valid_tickers = [ticker for ticker in overview_tickers if not price_data[ticker].empty]
    if not valid_tickers:
        st.info("No snapshot data is available yet for the selected tickers.")
        return

    scores = {ticker: growth_score(price_data[ticker]) for ticker in valid_tickers}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_ticker = ranked[0][0]
    top_value = ranked[0][1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracked tickers", len(valid_tickers))
    col2.metric("Top mover", top_ticker)
    col3.metric("Best momentum", f"{top_value:.2f}%")

    if realtime_mode:
        quote_cols = st.columns(min(3, len(valid_tickers)))
        for i, ticker in enumerate(valid_tickers[:3]):
            price_series = price_data[ticker]["Close"].dropna()
            if len(price_series) >= 2:
                current_price = float(price_series.iloc[-1])
                previous_price = float(price_series.iloc[-2])
                delta_value = current_price - previous_price
                quote_cols[i].metric(
                    f"{ticker} latest {interval} close",
                    f"{price_prefix}{current_price:.2f}",
                    f"{price_prefix}{delta_value:+.2f} vs. prior bar",
                )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Recent momentum": f"{scores[ticker]:.2f}%",
                    "Last close": price_data[ticker]["Close"].dropna().iloc[-1],
                }
                for ticker in valid_tickers
            ]
        ),
        hide_index=True,
    )


def render_charts_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
    realtime_mode: bool,
    period: str,
    interval: str,
    forecast_points: int,
    price_prefix: str,
    price_axis_label: str,
    price_format: str,
) -> None:
    active_tickers = tickers[:MAX_CHARTED_PERFORMERS]
    if len(tickers) > MAX_CHARTED_PERFORMERS:
        st.info(f"Charting the first {MAX_CHARTED_PERFORMERS} symbols from the selected source.")

    with st.spinner("Loading price history for charting..."):
        price_data = get_price_history_batch(active_tickers, period=period, interval=interval)

    valid_tickers = [ticker for ticker in active_tickers if not price_data[ticker].empty]
    bac_log(f"Valid tickers with price data={valid_tickers}")

    if realtime_mode and not valid_tickers:
        bac_log("Realtime fetch returned empty. Falling back to daily history for display stability")
        st.warning("Real-time data is temporarily unavailable. Showing recent daily history instead.")
        price_data = get_price_history_batch(active_tickers, period="6mo", interval="1d")
        valid_tickers = [ticker for ticker in active_tickers if not price_data[ticker].empty]

    if not valid_tickers:
        st.error("No price data was returned. Check ticker symbols and try again.")
        st.stop()

    if ticker_source in MARKET_SOURCES:
        daily_change_by_ticker = detected_performers.set_index("Ticker")["Daily change"].to_dict()
        top_performers = [ticker for ticker in active_tickers if ticker in valid_tickers]
        if ticker_source == IRELAND_SOURCE:
            leader_label = "Top ISEQ 20 daily mover"
            performance_label = "Best ISEQ 20 daily change"
        else:
            leader_label = "Top detected daily gainer"
            performance_label = "Best detected daily change"
        leader_change = daily_change_by_ticker.get(top_performers[0], np.nan)
        performance_value = f"{leader_change:.2f}%" if pd.notna(leader_change) else "Unavailable"
        bac_log(
            f"Detected top performers={[(ticker, daily_change_by_ticker.get(ticker)) for ticker in top_performers]}"
        )
    else:
        scores = {ticker: growth_score(price_data[ticker]) for ticker in valid_tickers}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_performers = [ticker for ticker, _ in ranked[:MAX_CHARTED_PERFORMERS]]
        current_momentum_label = momentum_label(realtime_mode, interval)
        leader_label = f"Top {current_momentum_label} mover"
        performance_label = f"Best {current_momentum_label} growth"
        performance_value = f"{ranked[0][1]:.2f}%"
        bac_log(f"Top manual movers ({current_momentum_label})={ranked[:MAX_CHARTED_PERFORMERS]}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Charted performers", len(top_performers))
    col2.metric(leader_label, top_performers[0])
    col3.metric(performance_label, performance_value)

    if ticker_source in MARKET_SOURCES:
        if ticker_source == IRELAND_SOURCE:
            st.subheader("ISEQ 20 top daily performers")
            st.caption(
                "The leaderboard ranks the latest available daily close from the tracked ISEQ 20 Euronext Dublin universe. It is not a complete ranking of every Irish or European stock."
            )
        else:
            st.subheader("Detected top 10 daily gainers")
            st.caption(
                "The screener is Yahoo Finance's predefined U.S. equity filter for liquid, large-cap daily gainers. It is not a complete ranking of every listed stock."
            )
        leaderboard_columns = {
            "Daily change": st.column_config.NumberColumn("Daily change", format="%.2f%%"),
            "Last price": st.column_config.NumberColumn("Last price", format=price_format),
        }
        if "Last session" in detected_performers.columns:
            leaderboard_columns["Last session"] = st.column_config.DateColumn("Last session")
        st.dataframe(detected_performers, column_config=leaderboard_columns, hide_index=True)

    if realtime_mode:
        quote_cols = st.columns(min(3, len(top_performers)))
        for i, ticker in enumerate(top_performers[:3]):
            price_series = price_data[ticker]["Close"].dropna()
            if len(price_series) >= 2:
                current_price = float(price_series.iloc[-1])
                previous_price = float(price_series.iloc[-2])
                delta_value = current_price - previous_price
                quote_cols[i].metric(
                    f"{ticker} latest {interval} close",
                    f"{price_prefix}{current_price:.2f}",
                    f"{price_prefix}{delta_value:+.2f} vs. prior bar",
                )
        st.caption(
            "Intraday figures use the latest returned bar close. The delta is versus the prior bar, not a live tick or daily change."
        )

    if realtime_mode:
        selected_horizon_label = f"{forecast_points} {interval} bars"
    else:
        selected_horizon_label = f"{forecast_points} business days"

    if ticker_source == IRELAND_SOURCE:
        chart_heading = "ISEQ 20 daily leaders - history and feature-based forecast"
    elif ticker_source == US_SOURCE:
        chart_heading = "Detected top 10 daily gainers - history and feature-based forecast"
    else:
        chart_heading = "Top momentum stocks - history and feature-based forecast"
    st.subheader(chart_heading)
    st.caption(
        f"The dashed forecast estimates returns from recent momentum, volatility, RSI, price structure, and volume features. The backtest below scores the same {selected_horizon_label} horizon against a no-change baseline, so the chart and validation stay aligned."
    )
    backtest_rows = []
    for ticker in top_performers:
        df = price_data[ticker]
        fc = forecast_feature_model(df, points_ahead=forecast_points)
        backtest = backtest_forecast_model(df, forecast_horizon=forecast_points)
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
                    name=f"{ticker} Forecast model",
                    line={"dash": "dash", "width": 2},
                )
            )

        fig.update_layout(
            title=f"{ticker}: Price History & Feature Forecast",
            xaxis_title="Timestamp" if realtime_mode else "Date",
            yaxis_title=price_axis_label,
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig)

        if not fc.empty:
            current_projection = fc.iloc[-1]
            current_return_pct = float(current_projection["pred_return"] * 100)
            current_close = float(current_projection["pred_close"])
            confidence = "Unavailable"
            directional_accuracy = np.nan
            mae_improvement = np.nan
            if not backtest.empty:
                summary = summarize_backtest(ticker, backtest, forecast_points)
                confidence = summary["Confidence"]
                directional_accuracy = float(summary["Directional accuracy"])
                mae_improvement = float(summary["MAE improvement vs. no-change"])
                summary["Projected return"] = current_return_pct
                summary["Projected close"] = current_close
                backtest_rows.append(summary)

            accuracy_text = (
                f"{directional_accuracy:.1f}% directional accuracy"
                if pd.notna(directional_accuracy)
                else "directional accuracy unavailable"
            )
            improvement_text = (
                f"{mae_improvement:.1f}% vs. baseline"
                if pd.notna(mae_improvement)
                else "MAE comparison unavailable"
            )
            st.caption(
                f"{ticker}: current {selected_horizon_label} forecast {current_return_pct:+.2f}% to {price_prefix}{current_close:.2f}. Confidence: {confidence}. Recent backtest: {accuracy_text}, {improvement_text}."
            )
        elif not backtest.empty:
            backtest_rows.append(summarize_backtest(ticker, backtest, forecast_points))

    st.subheader("Forecast backtest")
    st.caption(
        f"Walk-forward test of up to {MAX_BACKTEST_POINTS} unseen {selected_horizon_label} forecasts. Each forecast is trained only on the preceding {BACKTEST_TRAINING_POINTS} observations and compared with a no-change baseline."
    )

    if backtest_rows:
        st.dataframe(
            pd.DataFrame(backtest_rows),
            column_config={
                "Projected return": st.column_config.NumberColumn(
                    "Projected return", format="%.2f%%"
                ),
                "Projected close": st.column_config.NumberColumn(
                    "Projected close", format=price_format
                ),
                "Model MAE": st.column_config.NumberColumn("Model MAE", format=price_format),
                "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                "Directional accuracy": st.column_config.NumberColumn(
                    "Directional accuracy", format="%.1f%%"
                ),
                "No-change MAE": st.column_config.NumberColumn("No-change MAE", format=price_format),
                "MAE improvement vs. no-change": st.column_config.NumberColumn(
                    "MAE improvement vs. no-change", format="%.1f%%"
                ),
            },
            hide_index=True,
        )
        st.caption(
            "Positive MAE improvement means the feature model beat the no-change baseline on the selected horizon; negative values mean it performed worse."
        )
    else:
        st.info(
            "Not enough price observations to backtest this forecast model. "
            f"At least {BACKTEST_TRAINING_POINTS + forecast_points + MIN_BACKTEST_POINTS - 1} observations are required."
        )


def render_news_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
) -> None:
    st.subheader("Investing news and sentiment")

    news_tickers = tickers[:MAX_CHARTED_PERFORMERS]
    company_by_ticker: dict[str, str] = (
        {str(ticker): str(company) for ticker, company in detected_performers.set_index("Ticker")["Company"].to_dict().items()}
        if not detected_performers.empty
        else {}
    )

    with st.spinner("Fetching news and sentiment..."):
        news_frames = load_news_frames_parallel(news_tickers, company_by_ticker)

    if news_frames:
        all_news = pd.concat(news_frames, ignore_index=True)
        bac_log(f"Total combined news rows={len(all_news)}")

        sentiment_by_ticker = (
            all_news.groupby("ticker", as_index=False)
            .agg(sentiment=("sentiment", "mean"))
            .sort_values(by="sentiment", ascending=False)
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
            title="Average news sentiment by ticker",
            xaxis_title="Ticker",
            yaxis_title="Compound sentiment score",
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


st.title("Stock Market Intelligence Dashboard")
st.caption("Ireland-focused market data, investing news, sentiment trends, and feature-based forecast charts")

if st.session_state.get("ticker_source") not in {*MARKET_SOURCES, MANUAL_SOURCE}:
    st.session_state["ticker_source"] = IRELAND_SOURCE
if st.session_state.get("active_view") not in VIEW_OPTIONS:
    st.session_state["active_view"] = "Overview"

with st.sidebar:
    st.header("Configuration")
    ticker_source = st.segmented_control(
        "Ticker source",
        [IRELAND_SOURCE, US_SOURCE, MANUAL_SOURCE],
        required=True,
        key="ticker_source",
        width="stretch",
    )
    raw_tickers = ""
    if ticker_source == MANUAL_SOURCE:
        raw_tickers = st.text_input(
            "Tickers (comma-separated)",
            help="Example: AAPL, MSFT, NVDA",
        )
    elif ticker_source == IRELAND_SOURCE:
        st.caption(
            "Ranks the tracked ISEQ 20 Euronext Dublin listings by their latest available daily close."
        )
    else:
        st.caption(
            "Uses Yahoo Finance's U.S. large-cap daily-gainers screen and charts the top 10 equities."
        )

    realtime_mode = st.toggle("Real-time Mode", value=False)

    if realtime_mode:
        period = st.selectbox("Intraday Window", ["1d", "5d"], index=0)
        interval = st.selectbox("Intraday Interval", ["1m", "2m", "5m"], index=0)
        forecast_points = st.slider("Forecast horizon bars", min_value=7, max_value=60, value=30)
    else:
        period = st.selectbox("History Window", ["6mo", "1y", "2y"], index=1)
        interval = "1d"
        forecast_points = st.slider(
            "Forecast horizon business days", min_value=7, max_value=60, value=30
        )

    if st.button("Refresh now", type="primary"):
        bac_log("Manual refresh requested by user")
        st.cache_data.clear()
        st.rerun()

    active_view = st.segmented_control(
        "View",
        VIEW_OPTIONS,
        required=True,
        key="active_view",
        width="stretch",
    )
    active_view = active_view or "Overview"

if ticker_source == IRELAND_SOURCE:
    detected_performers = get_iseq20_top_performers()
    tickers = detected_performers["Ticker"].tolist()
elif ticker_source == US_SOURCE:
    detected_performers = get_us_top_performers()
    tickers = detected_performers["Ticker"].tolist()
else:
    detected_performers = pd.DataFrame()
    tickers = parse_tickers(raw_tickers)

ticker_source = ticker_source or IRELAND_SOURCE

if ticker_source == IRELAND_SOURCE:
    price_prefix = "€"
    price_format = "€%.2f"
    price_axis_label = "Price (EUR)"
elif ticker_source == US_SOURCE:
    price_prefix = "$"
    price_format = "$%.2f"
    price_axis_label = "Price (USD)"
else:
    price_prefix = ""
    price_format = "%.2f"
    price_axis_label = "Price (listing currency)"

if realtime_mode:
    st.info("Real-time mode is using manual refresh. Click 'Refresh now' to update values.")

if ticker_source == IRELAND_SOURCE:
    st.caption(
        "This is an Ireland-focused market universe, not a broker eligibility check. "
        "Confirm that your broker gives your account access to Euronext Dublin."
    )


# Search terminal for [BAC_LOG] to track what the app is processing.
bac_log(
    f"Ticker source={ticker_source}, input tickers={tickers}, period={period}, interval={interval}, "
    f"realtime_mode={realtime_mode}, forecast_points={forecast_points}, view={active_view}"
)

if not tickers:
    if ticker_source == IRELAND_SOURCE:
        st.error("No ISEQ 20 price data was returned. Try refreshing in a moment.")
    elif ticker_source == US_SOURCE:
        st.error("No top performers were returned by the market screener. Try refreshing in a moment.")
    else:
        st.warning("Add at least one ticker symbol.")
    st.stop()

if active_view == "Overview":
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
    render_news_view(
        ticker_source=ticker_source,
        tickers=tickers,
        detected_performers=detected_performers,
    )

st.caption(
    "Data sources: Yahoo Finance (prices) and Google News RSS (headlines). "
    "Ireland mode ranks a tracked ISEQ 20 Euronext Dublin universe by the latest available daily close. "
    "Forecast quality is measured with a horizon-matched walk-forward backtest and no-change baseline. "
    "This dashboard provides directional insight only, not investment advice."
)
