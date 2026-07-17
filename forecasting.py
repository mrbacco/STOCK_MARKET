#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: forecasting.py
#############################

"""Forecast-model helpers and backtesting utilities.

This module owns the technical feature engineering, model fitting, prediction,
and walk-forward validation logic. The Streamlit UI calls into these helpers
instead of carrying the model internals inline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app_config import (
    BACKTEST_TRAINING_POINTS,
    INTRADAY_FREQUENCIES,
    MAX_BACKTEST_POINTS,
    MIN_BACKTEST_POINTS,
    MIN_MODEL_TRAINING_ROWS,
    MODEL_LOOKBACK_POINTS,
    PRICE_FEATURE_COLUMNS,
    RSI_PERIOD,
    SENTIMENT_FEATURE_COLUMNS,
)
from app_logging import bac_log_kv, bac_log_section
from sentiment_features import build_sentiment_feature_frame, has_sufficient_sentiment_history


def prepare_model_history(price_history: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize history before feature engineering begins."""
    bac_log_kv(
        "forecast.prepare_model_history",
        incoming_rows=len(price_history),
        incoming_columns=list(price_history.columns),
    )

    required_columns = ("Date", "Open", "High", "Low", "Close", "Volume")
    if not set(required_columns).issubset(price_history.columns):
        bac_log_kv(
            "forecast.prepare_model_history",
            missing_columns=sorted(set(required_columns).difference(price_history.columns)),
        )
        return pd.DataFrame()

    # The model expects deterministic column order and numeric price fields.
    history = price_history.loc[:, required_columns].copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        history[column] = pd.to_numeric(history[column], errors="coerce")

    # Remove unusable rows before rolling features are built.
    history = history.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    history = history.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(
        drop=True
    )
    history = history[(history["Open"] > 0) & (history["High"] > 0) & (history["Low"] > 0)]
    history = history[history["Close"] > 0].copy()
    history["Volume"] = history["Volume"].fillna(0.0).clip(lower=0.0)

    bac_log_kv("forecast.prepare_model_history", cleaned_rows=len(history))
    return history.reset_index(drop=True)


def compute_rsi(close_series: pd.Series, window: int = RSI_PERIOD) -> pd.Series:
    """Compute a smoothed RSI signal and keep the output bounded and stable."""
    bac_log_kv("forecast.compute_rsi", rows=len(close_series), window=window)

    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)

    # When there have been only gains or only losses, the classic RSI formula
    # needs explicit handling to avoid divide-by-zero artifacts.
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss.ne(0), 100.0)
    rsi = rsi.where(average_gain.ne(0), 0.0)
    rsi = rsi.mask(average_gain.eq(0) & average_loss.eq(0), 50.0)

    bac_log_kv("forecast.compute_rsi", output_rows=len(rsi))
    return rsi.fillna(50.0)


def build_feature_frame(
    history: pd.DataFrame,
    sentiment_history: pd.DataFrame | None = None,
    include_sentiment: bool = False,
    latest_sentiment_as_of: object | None = None,
) -> pd.DataFrame:
    """Create the ordered technical and optional point-in-time sentiment matrix."""
    feature_columns = (
        (*PRICE_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS)
        if include_sentiment
        else PRICE_FEATURE_COLUMNS
    )
    bac_log_kv(
        "forecast.build_feature_frame",
        history_rows=len(history),
        include_sentiment=include_sentiment,
    )
    if history.empty:
        bac_log_section("forecast.build_feature_frame", "History was empty.")
        return pd.DataFrame(columns=feature_columns)

    close = history["Close"]
    volume = history["Volume"].replace(0, np.nan)
    one_bar_log_return = np.log(close).diff()
    sma_5 = close.rolling(5).mean()
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    volume_mean_5 = volume.rolling(5).mean()

    # The feature set intentionally blends price trend, volatility, participation,
    # and intraday structure so the model is not relying on a single signal type.
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

    cleaned_features = features.replace([np.inf, -np.inf], np.nan)
    if include_sentiment:
        sentiment_features = build_sentiment_feature_frame(
            history,
            sentiment_history,
            latest_as_of=latest_sentiment_as_of,
        )
        cleaned_features = pd.concat([cleaned_features, sentiment_features], axis=1)
    bac_log_kv(
        "forecast.build_feature_frame",
        feature_rows=len(cleaned_features),
        feature_columns=list(cleaned_features.columns),
    )
    return cleaned_features


def build_forecast_training_frame(
    price_history: pd.DataFrame,
    forecast_horizon: int,
    sentiment_history: pd.DataFrame | None = None,
    include_sentiment: bool = False,
) -> pd.DataFrame:
    """Join features and target returns into one training table for the model."""
    bac_log_kv(
        "forecast.build_forecast_training_frame",
        history_rows=len(price_history),
        forecast_horizon=forecast_horizon,
        include_sentiment=include_sentiment,
    )

    history = prepare_model_history(price_history)
    if history.empty or forecast_horizon < 1:
        bac_log_section(
            "forecast.build_forecast_training_frame",
            "Training frame could not be created because inputs were invalid.",
        )
        return pd.DataFrame()

    feature_columns = (
        (*PRICE_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS)
        if include_sentiment
        else PRICE_FEATURE_COLUMNS
    )
    feature_frame = build_feature_frame(
        history,
        sentiment_history=sentiment_history,
        include_sentiment=include_sentiment,
    )
    target_log_return = np.log(history["Close"].shift(-forecast_horizon) / history["Close"])
    training_frame = pd.concat([history[["Date", "Close"]], feature_frame], axis=1)
    training_frame["target_log_return"] = target_log_return

    # Rows with incomplete rolling windows or missing future targets cannot be used.
    training_frame = training_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*feature_columns, "target_log_return"]
    )
    if len(training_frame) > MODEL_LOOKBACK_POINTS:
        training_frame = training_frame.iloc[-MODEL_LOOKBACK_POINTS:]

    if include_sentiment and not has_sufficient_sentiment_history(training_frame):
        bac_log_kv(
            "forecast.build_forecast_training_frame",
            message="Sentiment model is still collecting a historical baseline.",
            observed_sentiment_bars=int((training_frame["news_count_24h"] > 0).sum()),
        )
        return pd.DataFrame()

    bac_log_kv(
        "forecast.build_forecast_training_frame",
        training_rows=len(training_frame),
    )
    return training_frame.reset_index(drop=True)


def fit_forecast_model(
    training_frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = PRICE_FEATURE_COLUMNS,
) -> Pipeline | None:
    """Fit a small regularized regression model on the engineered features."""
    bac_log_kv("forecast.fit_forecast_model", training_rows=len(training_frame))
    if len(training_frame) < MIN_MODEL_TRAINING_ROWS:
        bac_log_kv(
            "forecast.fit_forecast_model",
            message="Not enough rows to fit the model.",
            minimum_rows=MIN_MODEL_TRAINING_ROWS,
        )
        return None

    # Scaling keeps the different feature magnitudes comparable, while Ridge
    # regularization reduces the chance that one noisy feature dominates.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(training_frame.loc[:, feature_columns], training_frame["target_log_return"])
    bac_log_section("forecast.fit_forecast_model", "Model fitting completed.")
    return model


def predict_horizon_close(
    price_history: pd.DataFrame,
    forecast_horizon: int,
    sentiment_history: pd.DataFrame | None = None,
    include_sentiment: bool = False,
    prediction_as_of: object | None = None,
) -> dict:
    """Predict the close price at the requested horizon using the latest feature row."""
    bac_log_kv(
        "forecast.predict_horizon_close",
        history_rows=len(price_history),
        forecast_horizon=forecast_horizon,
        include_sentiment=include_sentiment,
    )

    history = prepare_model_history(price_history)
    if history.empty:
        bac_log_section("forecast.predict_horizon_close", "Prediction aborted because history was empty.")
        return {}

    feature_columns = (
        (*PRICE_FEATURE_COLUMNS, *SENTIMENT_FEATURE_COLUMNS)
        if include_sentiment
        else PRICE_FEATURE_COLUMNS
    )
    feature_frame = build_feature_frame(
        history,
        sentiment_history=sentiment_history,
        include_sentiment=include_sentiment,
        latest_sentiment_as_of=prediction_as_of,
    )
    if feature_frame.empty:
        bac_log_section("forecast.predict_horizon_close", "Prediction aborted because features were empty.")
        return {}

    latest_features = feature_frame.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if latest_features.loc[:, feature_columns].isna().any(axis=None):
        bac_log_section(
            "forecast.predict_horizon_close",
            "Latest feature row still had missing values after cleaning.",
        )
        return {}

    training_frame = build_forecast_training_frame(
        history,
        forecast_horizon,
        sentiment_history=sentiment_history,
        include_sentiment=include_sentiment,
    )
    model = fit_forecast_model(training_frame, feature_columns=feature_columns)
    if model is None:
        bac_log_section("forecast.predict_horizon_close", "Prediction aborted because model fitting failed.")
        return {}

    predicted_log_return = float(model.predict(latest_features.loc[:, feature_columns])[0])
    last_close = float(history["Close"].iloc[-1])
    predicted_close = last_close * float(np.exp(predicted_log_return))
    predicted_return = float(np.exp(predicted_log_return) - 1.0)

    bac_log_kv(
        "forecast.predict_horizon_close",
        last_close=last_close,
        predicted_close=predicted_close,
        predicted_return=predicted_return,
        training_rows=len(training_frame),
    )
    return {
        "forecast_horizon": forecast_horizon,
        "predicted_close": predicted_close,
        "predicted_return": predicted_return,
        "last_close": last_close,
        "training_rows": int(len(training_frame)),
        "model_type": "Price + sentiment" if include_sentiment else "Price only",
    }


@st.cache_data(ttl="5m", max_entries=100)
def forecast_feature_model(
    price_history: pd.DataFrame,
    points_ahead: int = 30,
    sentiment_history: pd.DataFrame | None = None,
    include_sentiment: bool = False,
) -> pd.DataFrame:
    """Build a forward curve by predicting each future horizon independently."""
    bac_log_kv("forecast.forecast_feature_model", points_ahead=points_ahead, history_rows=len(price_history))
    rows = []
    for forecast_horizon in range(1, points_ahead + 1):
        prediction = predict_horizon_close(
            price_history,
            forecast_horizon,
            sentiment_history=sentiment_history,
            include_sentiment=include_sentiment,
            prediction_as_of=pd.Timestamp.now(tz="UTC"),
        )
        if not prediction:
            bac_log_kv(
                "forecast.forecast_feature_model",
                stopping_horizon=forecast_horizon,
                message="Stopped building the forecast curve because a prediction was unavailable.",
            )
            break

        rows.append(
            {
                "projection_point": forecast_horizon,
                "pred_close": prediction["predicted_close"],
                "pred_return": prediction["predicted_return"],
                "training_rows": prediction["training_rows"],
                "model_type": prediction["model_type"],
            }
        )

    result = pd.DataFrame(rows)
    bac_log_kv("forecast.forecast_feature_model", forecast_rows=len(result))
    return result


@st.cache_data(ttl="5m", max_entries=100)
def backtest_forecast_model(
    price_history: pd.DataFrame,
    forecast_horizon: int,
    training_points: int = BACKTEST_TRAINING_POINTS,
    max_test_points: int = MAX_BACKTEST_POINTS,
    sentiment_history: pd.DataFrame | None = None,
    include_sentiment: bool = False,
) -> pd.DataFrame:
    """Run a walk-forward backtest that matches the user-selected forecast horizon."""
    bac_log_kv(
        "forecast.backtest_forecast_model",
        history_rows=len(price_history),
        forecast_horizon=forecast_horizon,
        training_points=training_points,
        max_test_points=max_test_points,
        include_sentiment=include_sentiment,
    )

    history = prepare_model_history(price_history)
    if history.empty or forecast_horizon < 1:
        bac_log_section(
            "forecast.backtest_forecast_model",
            "Backtest aborted because history or horizon was invalid.",
        )
        return pd.DataFrame()

    available_test_points = len(history) - training_points - forecast_horizon + 1
    test_points = min(max_test_points, available_test_points)
    bac_log_kv(
        "forecast.backtest_forecast_model",
        available_test_points=available_test_points,
        chosen_test_points=test_points,
    )
    if test_points < MIN_BACKTEST_POINTS:
        bac_log_kv(
            "forecast.backtest_forecast_model",
            message="Not enough unseen samples for backtesting.",
            minimum_points=MIN_BACKTEST_POINTS,
        )
        return pd.DataFrame()

    test_start_training_end = len(history) - forecast_horizon - test_points
    rows = []
    for training_end_index in range(test_start_training_end, len(history) - forecast_horizon):
        training_start_index = training_end_index - training_points + 1
        training_slice = history.iloc[training_start_index : training_end_index + 1].copy()
        prediction = predict_horizon_close(
            training_slice,
            forecast_horizon,
            sentiment_history=sentiment_history,
            include_sentiment=include_sentiment,
        )
        if not prediction:
            bac_log_kv(
                "forecast.backtest_forecast_model",
                training_end_index=training_end_index,
                message="Skipped one backtest step because the prediction was unavailable.",
            )
            continue

        baseline_close = float(training_slice["Close"].iloc[-1])
        predicted_close = float(prediction["predicted_close"])
        actual_close = float(history["Close"].iloc[training_end_index + forecast_horizon])
        predicted_direction = np.sign(predicted_close - baseline_close)
        actual_direction = np.sign(actual_close - baseline_close)
        row = {
            "date": history["Date"].iloc[training_end_index + forecast_horizon],
            "actual_close": actual_close,
            "predicted_close": predicted_close,
            "baseline_close": baseline_close,
            "absolute_error": abs(actual_close - predicted_close),
            "baseline_absolute_error": abs(actual_close - baseline_close),
            "direction_correct": bool(predicted_direction == actual_direction),
        }
        rows.append(row)
        bac_log_kv(
            "forecast.backtest_forecast_model.step",
            training_end_index=training_end_index,
            actual_close=actual_close,
            predicted_close=predicted_close,
            baseline_close=baseline_close,
            absolute_error=row["absolute_error"],
            direction_correct=row["direction_correct"],
        )

    result = pd.DataFrame(rows)
    bac_log_kv("forecast.backtest_forecast_model", result_rows=len(result))
    return result


def confidence_label(mae_improvement: float, directional_accuracy: float) -> str:
    """Convert numeric backtest quality into a plain-English confidence label."""
    if np.isnan(mae_improvement) or np.isnan(directional_accuracy):
        label = "Unavailable"
    elif mae_improvement >= 10 and directional_accuracy >= 58:
        label = "High"
    elif mae_improvement >= 0 and directional_accuracy >= 52:
        label = "Moderate"
    else:
        label = "Low"

    bac_log_kv(
        "forecast.confidence_label",
        mae_improvement=mae_improvement,
        directional_accuracy=directional_accuracy,
        label=label,
    )
    return label


def summarize_backtest(ticker: str, backtest: pd.DataFrame, forecast_horizon: int) -> dict:
    """Summarize walk-forward results into one row for the Streamlit score table."""
    bac_log_kv(
        "forecast.summarize_backtest",
        ticker=ticker,
        backtest_rows=len(backtest),
        forecast_horizon=forecast_horizon,
    )

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
    bac_log_kv(
        "forecast.summarize_backtest",
        ticker=ticker,
        model_mae=model_mae,
        baseline_mae=baseline_mae,
        mape=mape,
        directional_accuracy=directional_accuracy,
        mae_improvement=mae_improvement,
    )

    summary = {
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
    bac_log_kv(
        "forecast.summarize_backtest",
        ticker=ticker,
        model_mae=model_mae,
        baseline_mae=baseline_mae,
        directional_accuracy=directional_accuracy,
    )
    return summary


def summarize_model_comparison(
    ticker: str,
    price_backtest: pd.DataFrame,
    sentiment_backtest: pd.DataFrame,
    forecast_horizon: int,
) -> dict:
    """Compare the price-only baseline with the sentiment-augmented candidate."""
    price_summary = summarize_backtest(ticker, price_backtest, forecast_horizon)
    comparison = {
        **price_summary,
        "Active model": "Price only",
        "Sentiment status": "Collecting history",
        "Price-only MAE": price_summary["Model MAE"],
        "Price-only directional accuracy": price_summary["Directional accuracy"],
        "Sentiment MAE": np.nan,
        "Sentiment directional accuracy": np.nan,
        "Sentiment MAE lift vs. price-only": np.nan,
    }
    if sentiment_backtest.empty:
        return comparison

    sentiment_summary = summarize_backtest(ticker, sentiment_backtest, forecast_horizon)
    price_mae = float(price_summary["Model MAE"])
    sentiment_mae = float(sentiment_summary["Model MAE"])
    lift = ((price_mae - sentiment_mae) / price_mae) * 100 if price_mae > 0 else np.nan
    promoted = bool(pd.notna(lift) and lift > 0)
    active_summary = sentiment_summary if promoted else price_summary
    comparison.update(active_summary)
    comparison.update(
        {
            "Active model": "Price + sentiment" if promoted else "Price only",
            "Sentiment status": "Promoted" if promoted else "Evaluated, not promoted",
            "Price-only MAE": price_mae,
            "Price-only directional accuracy": price_summary["Directional accuracy"],
            "Sentiment MAE": sentiment_mae,
            "Sentiment directional accuracy": sentiment_summary["Directional accuracy"],
            "Sentiment MAE lift vs. price-only": lift,
        }
    )
    return comparison


def future_projection_dates(
    last_date: pd.Timestamp,
    points_ahead: int,
    realtime_mode: bool,
    interval: str,
) -> pd.DatetimeIndex:
    """Generate future timestamps that line up with the selected operating mode."""
    bac_log_kv(
        "forecast.future_projection_dates",
        last_date=str(last_date),
        points_ahead=points_ahead,
        realtime_mode=realtime_mode,
        interval=interval,
    )

    last_timestamp = pd.Timestamp(last_date)
    if realtime_mode:
        frequency = INTRADAY_FREQUENCIES[interval]
        bac_log_kv(
            "forecast.future_projection_dates",
            projection_mode="intraday",
            frequency=frequency,
        )
        future_dates = pd.date_range(
            start=last_timestamp + pd.Timedelta(frequency),
            periods=points_ahead,
            freq=frequency,
        )
    else:
        bac_log_kv(
            "forecast.future_projection_dates",
            projection_mode="business_days",
            frequency="B",
        )
        future_dates = pd.bdate_range(
            start=last_timestamp + pd.offsets.BDay(1),
            periods=points_ahead,
        )

    bac_log_kv("forecast.future_projection_dates", generated_points=len(future_dates))
    return future_dates
