#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: market_model.py
#############################

"""Cross-sectional market model, ensemble weighting, and top-stock ranking.

The older forecasting path fits one Ridge model to one ticker at a time.  This
module complements it with a pooled panel: all candidates from the selected
market contribute training examples, while market-relative features help the
model separate broad moves from stock-specific opportunity.

All validation is chronological.  Model weights are learned on a tuning period,
then measured on a later evaluation period that was not used to choose those
weights.  Forecast-horizon gaps prevent labels from crossing a split boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app_config import (
    MIN_BACKTEST_POINTS,
    PANEL_EVALUATION_DATES,
    PANEL_FEATURE_COLUMNS,
    PANEL_MIN_BASE_TRAINING_DATES,
    PANEL_RANDOM_STATE,
    PANEL_TUNING_DATES,
    PRICE_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    resolve_market_calendar,
)
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from cache_control import (
    SharedCacheMiss,
    current_cache_scope,
    enqueue_analytics_job,
    get_cache_generation,
    shared_cache_get_or_compute,
)
from runtime_config import ANALYTICS_READ_ONLY
from forecasting import build_feature_frame, prepare_model_history


ModelFactory = Callable[[], Pipeline]


def _regression_model_factories() -> dict[str, ModelFactory]:
    """Return fresh, deterministic ensemble members for each chronological fit."""
    return {
        "Ridge": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0)),
            ]
        ),
        "Elastic Net": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.0005,
                        l1_ratio=0.15,
                        max_iter=5_000,
                        random_state=PANEL_RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Histogram gradient boosting": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=120,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=0.5,
                        random_state=PANEL_RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _direction_classifier() -> Pipeline:
    """Build the separate classifier used for probability of outperformance."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=PANEL_RANDOM_STATE,
                ),
            ),
        ]
    )


def _ticker_panel_frame(
    ticker: str,
    history: pd.DataFrame,
    sentiment_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Prepare one ticker once, including a current point-in-time sentiment row."""
    cleaned = prepare_model_history(history)
    if cleaned.empty:
        return pd.DataFrame()

    # Sentiment columns always exist in the pooled model.  A ticker with no news
    # receives neutral zeros, allowing the model to learn whether *news presence*
    # itself carries useful information as the persistent store grows.
    features = build_feature_frame(
        cleaned,
        sentiment_history=sentiment_history,
        include_sentiment=True,
        latest_sentiment_as_of=pd.Timestamp.now(tz="UTC"),
        market_calendar=resolve_market_calendar(None, ticker),
    )
    frame = pd.concat(
        [cleaned[["Date", "Close", "Volume"]], features],
        axis=1,
    )
    frame["Ticker"] = ticker
    frame["one_bar_log_return"] = np.log(frame["Close"]).diff()
    return frame.replace([np.inf, -np.inf], np.nan)


def build_market_panel(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Create the leakage-safe pooled feature and target table."""
    bac_log_kv(
        "market_model.build_market_panel",
        ticker_count=len(price_data),
        forecast_horizon=forecast_horizon,
        sentiment_ticker_count=len(sentiment_by_ticker or {}),
    )
    if forecast_horizon < 1:
        return pd.DataFrame()

    ticker_frames: list[pd.DataFrame] = []
    for ticker, history in price_data.items():
        frame = _ticker_panel_frame(
            ticker,
            history,
            (sentiment_by_ticker or {}).get(ticker),
        )
        if not frame.empty:
            ticker_frames.append(frame)

    if not ticker_frames:
        bac_log_section("market_model.build_market_panel", "No usable ticker histories were found.")
        return pd.DataFrame()

    panel = pd.concat(ticker_frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel = panel.dropna(subset=["Date", "Close"]).sort_values(["Date", "Ticker"])

    # Equal-weight context is calculated only from values observable on the same
    # date.  No future aggregate is used as a feature.
    market_by_date = (
        panel.groupby("Date", as_index=False)
        .agg(
            market_ret_1=("ret_1", "mean"),
            market_ret_5=("ret_5", "mean"),
            market_ret_20=("ret_20", "mean"),
            market_one_bar_log_return=("one_bar_log_return", "mean"),
        )
        .sort_values("Date")
    )
    breadth = (
        panel.assign(positive_bar=panel["ret_1"].gt(0).astype(float))
        .groupby("Date", as_index=False)["positive_bar"]
        .mean()
        .rename(columns={"positive_bar": "market_breadth_1"})
    )
    market_by_date = market_by_date.merge(breadth, on="Date", how="left")
    market_by_date["market_volatility_20"] = market_by_date[
        "market_one_bar_log_return"
    ].rolling(20, min_periods=10).std()
    panel = panel.merge(market_by_date, on="Date", how="left", validate="many_to_one")

    enriched_frames: list[pd.DataFrame] = []
    for _ticker, ticker_frame in panel.groupby("Ticker", sort=False):
        ticker_frame = ticker_frame.sort_values("Date").copy()
        market_variance = ticker_frame["market_one_bar_log_return"].rolling(
            20, min_periods=10
        ).var()
        rolling_covariance = ticker_frame["one_bar_log_return"].rolling(
            20, min_periods=10
        ).cov(ticker_frame["market_one_bar_log_return"])
        ticker_frame["rolling_beta_20"] = rolling_covariance / market_variance.replace(
            0.0, np.nan
        )
        ticker_frame["rolling_correlation_20"] = ticker_frame[
            "one_bar_log_return"
        ].rolling(20, min_periods=10).corr(ticker_frame["market_one_bar_log_return"])
        ticker_frame["relative_strength_5"] = (
            ticker_frame["ret_5"] - ticker_frame["market_ret_5"]
        )
        ticker_frame["relative_strength_20"] = (
            ticker_frame["ret_20"] - ticker_frame["market_ret_20"]
        )
        ticker_frame["log_dollar_volume"] = np.log1p(
            ticker_frame["Close"].clip(lower=0)
            * ticker_frame["Volume"].clip(lower=0)
        )

        # Targets are forward log returns.  Subtracting the same-date universe
        # average makes the objective "outperform this market" rather than simply
        # "rise when the whole market rises".
        ticker_frame["target_log_return"] = np.log(
            ticker_frame["Close"].shift(-forecast_horizon) / ticker_frame["Close"]
        )
        enriched_frames.append(ticker_frame)

    panel = pd.concat(enriched_frames, ignore_index=True)
    panel["target_market_log_return"] = panel.groupby("Date")[
        "target_log_return"
    ].transform("mean")
    panel["target_excess_log_return"] = (
        panel["target_log_return"] - panel["target_market_log_return"]
    )
    panel["target_outperformed"] = panel["target_excess_log_return"].gt(0).astype(int)
    panel = panel.replace([np.inf, -np.inf], np.nan)

    bac_log_kv(
        "market_model.build_market_panel",
        panel_rows=len(panel),
        panel_dates=panel["Date"].nunique(),
        usable_target_rows=int(panel["target_excess_log_return"].notna().sum()),
        sentiment_rows=int(panel["news_count_24h"].gt(0).sum()),
    )
    return panel.reset_index(drop=True)


def split_panel_dates(
    labeled_dates: pd.DatetimeIndex,
    forecast_horizon: int,
) -> dict[str, pd.DatetimeIndex]:
    """Create base, tuning, pre-evaluation, and evaluation periods with gaps."""
    dates = pd.DatetimeIndex(sorted(pd.unique(labeled_dates)))
    evaluation_count = min(PANEL_EVALUATION_DATES, max(MIN_BACKTEST_POINTS, len(dates) // 5))
    tuning_count = min(PANEL_TUNING_DATES, max(MIN_BACKTEST_POINTS, len(dates) // 5))

    evaluation_start = len(dates) - evaluation_count
    tuning_end = evaluation_start - forecast_horizon
    tuning_start = tuning_end - tuning_count
    base_end = tuning_start - forecast_horizon
    pre_evaluation_end = evaluation_start - forecast_horizon

    if base_end < PANEL_MIN_BASE_TRAINING_DATES or tuning_start < 0:
        bac_log_kv(
            "market_model.split_panel_dates",
            status="insufficient_dates",
            available_dates=len(dates),
            required_base_dates=PANEL_MIN_BASE_TRAINING_DATES,
            forecast_horizon=forecast_horizon,
        )
        return {}

    split = {
        "base": dates[:base_end],
        "tuning": dates[tuning_start:tuning_end],
        "pre_evaluation": dates[:pre_evaluation_end],
        "evaluation": dates[evaluation_start:],
    }
    bac_log_kv(
        "market_model.split_panel_dates",
        base_dates=len(split["base"]),
        tuning_dates=len(split["tuning"]),
        pre_evaluation_dates=len(split["pre_evaluation"]),
        evaluation_dates=len(split["evaluation"]),
        forecast_horizon_gap=forecast_horizon,
    )
    return split


def _fit_predict_regressors(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Pipeline]]:
    """Fit every healthy ensemble member and return its prediction vector."""
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}
    x_train = training_frame.loc[:, PANEL_FEATURE_COLUMNS]
    y_train = training_frame["target_excess_log_return"]
    x_predict = prediction_frame.loc[:, PANEL_FEATURE_COLUMNS]

    for model_name, factory in _regression_model_factories().items():
        try:
            model = factory()
            model.fit(x_train, y_train)
            predictions[model_name] = np.asarray(model.predict(x_predict), dtype=float)
            fitted_models[model_name] = model
            bac_log_kv(
                "market_model.fit_regressor",
                model=model_name,
                training_rows=len(training_frame),
                prediction_rows=len(prediction_frame),
            )
        except Exception as ex:
            bac_log_kv(
                "market_model.fit_regressor",
                model=model_name,
                fitting_error=str(ex),
            )
    return predictions, fitted_models


def _ensemble_weights(
    actual: pd.Series,
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert tuning MAE into normalized inverse-error model weights."""
    model_mae = {
        model_name: float(mean_absolute_error(actual, values))
        for model_name, values in predictions.items()
    }
    inverse_error = {
        name: 1.0 / max(error, 1e-8)
        for name, error in model_mae.items()
    }
    total = sum(inverse_error.values())
    weights = {
        name: value / total
        for name, value in inverse_error.items()
    } if total > 0 else {}
    bac_log_kv(
        "market_model.ensemble_weights",
        tuning_mae=model_mae,
        weights=weights,
    )
    return weights, model_mae


def _weighted_prediction(
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Blend model vectors using only weights learned on the tuning period."""
    if not predictions or not weights:
        return np.array([], dtype=float)
    first = next(iter(predictions.values()))
    blended = np.zeros(len(first), dtype=float)
    for model_name, values in predictions.items():
        blended += values * float(weights.get(model_name, 0.0))
    return blended


def _fit_probability_pipeline(
    base_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    full_labeled_frame: pd.DataFrame,
    latest_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit, calibrate, evaluate, and refresh probability-of-outperformance."""
    default_evaluation = np.full(len(evaluation_frame), 0.5, dtype=float)
    default_latest = np.full(len(latest_frame), 0.5, dtype=float)
    if base_frame["target_outperformed"].nunique() < 2:
        return default_evaluation, default_latest, np.nan

    try:
        base_classifier = _direction_classifier()
        base_classifier.fit(
            base_frame.loc[:, PANEL_FEATURE_COLUMNS],
            base_frame["target_outperformed"],
        )
        tuning_raw = base_classifier.predict_proba(
            tuning_frame.loc[:, PANEL_FEATURE_COLUMNS]
        )[:, 1]
        evaluation_raw = base_classifier.predict_proba(
            evaluation_frame.loc[:, PANEL_FEATURE_COLUMNS]
        )[:, 1]

        # Platt-style calibration is learned only on the tuning window.  If that
        # window contains a single class, raw probabilities remain the safest
        # deterministic fallback.
        calibrator: LogisticRegression | None = None
        if tuning_frame["target_outperformed"].nunique() >= 2:
            calibrator = LogisticRegression(random_state=PANEL_RANDOM_STATE)
            calibrator.fit(
                tuning_raw.reshape(-1, 1),
                tuning_frame["target_outperformed"],
            )
            evaluation_probability = calibrator.predict_proba(
                evaluation_raw.reshape(-1, 1)
            )[:, 1]
        else:
            evaluation_probability = evaluation_raw

        # Refresh the base classifier with every now-labeled row for the current
        # ranking.  The calibrator remains frozen from the historical tuning set.
        latest_classifier = _direction_classifier()
        latest_classifier.fit(
            full_labeled_frame.loc[:, PANEL_FEATURE_COLUMNS],
            full_labeled_frame["target_outperformed"],
        )
        latest_raw = latest_classifier.predict_proba(
            latest_frame.loc[:, PANEL_FEATURE_COLUMNS]
        )[:, 1]
        latest_probability = (
            calibrator.predict_proba(latest_raw.reshape(-1, 1))[:, 1]
            if calibrator is not None
            else latest_raw
        )
        brier = float(
            brier_score_loss(
                evaluation_frame["target_outperformed"],
                evaluation_probability,
            )
        )
        bac_log_kv(
            "market_model.probability_pipeline",
            evaluation_brier=brier,
            evaluation_rows=len(evaluation_frame),
            latest_rows=len(latest_frame),
        )
        return evaluation_probability, latest_probability, brier
    except Exception as ex:
        bac_log_kv("market_model.probability_pipeline", fitting_error=str(ex))
        return default_evaluation, default_latest, np.nan


def _compute_rank_market_candidates(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Fit the pooled ensemble and rank the strongest current candidates."""
    bac_log_section("market_model.rank_market_candidates", "Pooled ranking started.")
    panel = build_market_panel(price_data, forecast_horizon, sentiment_by_ticker)
    if panel.empty:
        return {"ranking": pd.DataFrame(), "evaluation": pd.DataFrame(), "diagnostics": {}}

    labeled = panel.dropna(
        subset=[*PANEL_FEATURE_COLUMNS, "target_excess_log_return"]
    ).copy()
    latest = (
        panel.dropna(subset=list(PANEL_FEATURE_COLUMNS))
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .copy()
    )
    split = split_panel_dates(pd.DatetimeIndex(labeled["Date"].unique()), forecast_horizon)
    if not split or latest.empty:
        bac_log_kv(
            "market_model.rank_market_candidates",
            status="insufficient_panel_history",
            labeled_rows=len(labeled),
            latest_rows=len(latest),
        )
        return {"ranking": pd.DataFrame(), "evaluation": pd.DataFrame(), "diagnostics": {}}

    by_dates = lambda dates: labeled[labeled["Date"].isin(dates)].copy()
    base_frame = by_dates(split["base"])
    tuning_frame = by_dates(split["tuning"])
    pre_evaluation_frame = by_dates(split["pre_evaluation"])
    evaluation_frame = by_dates(split["evaluation"])

    tuning_predictions, _ = _fit_predict_regressors(base_frame, tuning_frame)
    weights, tuning_mae = _ensemble_weights(
        tuning_frame["target_excess_log_return"],
        tuning_predictions,
    )
    if not weights:
        return {"ranking": pd.DataFrame(), "evaluation": pd.DataFrame(), "diagnostics": {}}

    # Final metrics come from a later untouched block, with a horizon-length gap
    # between its first origin and the preceding training origins.
    evaluation_predictions, _ = _fit_predict_regressors(
        pre_evaluation_frame,
        evaluation_frame,
    )
    evaluation_blend = _weighted_prediction(evaluation_predictions, weights)

    # Tuning residuals define finite-sample 50% and 80% prediction intervals.
    # Because those residuals precede evaluation, interval coverage is measured
    # on data that did not set the interval widths.
    tuning_blend = _weighted_prediction(tuning_predictions, weights)
    tuning_residuals = (
        tuning_frame["target_excess_log_return"].to_numpy(dtype=float) - tuning_blend
    )
    residual_q10, residual_q25, residual_q75, residual_q90 = np.quantile(
        tuning_residuals,
        [0.10, 0.25, 0.75, 0.90],
    )
    evaluation_probability, latest_probability, probability_brier = _fit_probability_pipeline(
        base_frame,
        tuning_frame,
        evaluation_frame,
        labeled,
        latest,
    )

    evaluation = evaluation_frame[
        ["Date", "Ticker", "target_excess_log_return", "target_outperformed"]
    ].copy()
    evaluation["predicted_excess_return"] = evaluation_blend
    evaluation["probability_outperform"] = evaluation_probability
    evaluation["lower_80"] = evaluation_blend + residual_q10
    evaluation["upper_80"] = evaluation_blend + residual_q90
    evaluation["interval_hit_80"] = evaluation["target_excess_log_return"].between(
        evaluation["lower_80"], evaluation["upper_80"]
    )

    latest_predictions, _ = _fit_predict_regressors(labeled, latest)
    latest_blend = _weighted_prediction(latest_predictions, weights)
    component_matrix = np.column_stack(list(latest_predictions.values()))
    latest_agreement_spread = np.std(component_matrix, axis=1)

    ranking = latest[["Ticker", "Date", "Close", *SENTIMENT_FEATURE_COLUMNS, "vol_20"]].copy()
    ranking["Expected excess return"] = latest_blend * 100.0
    ranking["Probability outperform"] = latest_probability * 100.0
    ranking["Lower 50"] = (latest_blend + residual_q25) * 100.0
    ranking["Upper 50"] = (latest_blend + residual_q75) * 100.0
    ranking["Lower 80"] = (latest_blend + residual_q10) * 100.0
    ranking["Upper 80"] = (latest_blend + residual_q90) * 100.0
    ranking["Predicted volatility"] = (
        latest["vol_20"].clip(lower=0).to_numpy(dtype=float)
        * np.sqrt(forecast_horizon)
        * 100.0
    )
    ranking["Model disagreement"] = latest_agreement_spread * 100.0
    ranking["Sentiment score"] = ranking["sentiment_24h"]

    # The score rewards expected market-relative return and calibrated odds, and
    # penalizes model disagreement.  It is a ranking device, not a promised gain.
    ranking["Model score"] = (
        ranking["Expected excess return"]
        + 0.04 * (ranking["Probability outperform"] - 50.0)
        - 0.25 * ranking["Model disagreement"]
    )
    ranking["Signal"] = np.select(
        [
            (ranking["Lower 80"] > 0) & (ranking["Probability outperform"] >= 55),
            (ranking["Upper 80"] < 0) & (ranking["Probability outperform"] < 45),
            (ranking["Expected excess return"] > 0)
            & (ranking["Probability outperform"] >= 50),
        ],
        ["Qualified", "Avoid", "Watch"],
        default="Abstain - insufficient edge",
    )
    ranking = ranking.sort_values(
        ["Model score", "Probability outperform"],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking = ranking.head(max(1, int(top_n))).copy()

    evaluation_mae = float(
        mean_absolute_error(
            evaluation["target_excess_log_return"],
            evaluation["predicted_excess_return"],
        )
    )
    baseline_mae = float(evaluation["target_excess_log_return"].abs().mean())
    direction_accuracy = float(
        (
            np.sign(evaluation["predicted_excess_return"])
            == np.sign(evaluation["target_excess_log_return"])
        ).mean()
        * 100.0
    )
    interval_coverage = float(evaluation["interval_hit_80"].mean() * 100.0)

    # This directly backtests the app's selection rule: on every evaluation date,
    # rank the candidate panel, take the best ten, and measure realized excess.
    selected_evaluation = (
        evaluation.sort_values(["Date", "predicted_excess_return"], ascending=[True, False])
        .groupby("Date", as_index=False)
        .head(max(1, int(top_n)))
    )
    selection_by_date = selected_evaluation.groupby("Date").agg(
        selected_mean_excess_return=("target_excess_log_return", "mean"),
        selected_hit_rate=("target_outperformed", "mean"),
        selected_count=("Ticker", "size"),
    )
    selection_mean_excess = float(
        selection_by_date["selected_mean_excess_return"].mean() * 100.0
    )
    selection_hit_rate = float(selection_by_date["selected_hit_rate"].mean() * 100.0)

    diagnostics = {
        "Forecast horizon": int(forecast_horizon),
        "Candidate tickers": int(panel["Ticker"].nunique()),
        "Training rows": int(len(labeled)),
        "Tuning dates": int(len(split["tuning"])),
        "Evaluation dates": int(len(split["evaluation"])),
        "Evaluation MAE": evaluation_mae * 100.0,
        "Zero-excess baseline MAE": baseline_mae * 100.0,
        "Directional accuracy": direction_accuracy,
        "Probability Brier score": probability_brier,
        "80% interval coverage": interval_coverage,
        "Top-10 realized mean excess": selection_mean_excess,
        "Top-10 realized hit rate": selection_hit_rate,
        "Sentiment-observed rows": int(panel["news_count_24h"].gt(0).sum()),
        "Universe note": "Backtest uses the supplied candidate universe; historical constituent snapshots are not available from Yahoo Finance.",
        "Model weights": weights,
        "Tuning MAE": tuning_mae,
    }
    bac_log_kv(
        "market_model.rank_market_candidates",
        ranking_rows=len(ranking),
        evaluation_mae_pct=diagnostics["Evaluation MAE"],
        directional_accuracy=direction_accuracy,
        interval_coverage=interval_coverage,
        selection_mean_excess=selection_mean_excess,
        selection_hit_rate=selection_hit_rate,
    )
    bac_log_list_preview(
        "market_model.rank_market_candidates",
        "ranked_tickers",
        ranking["Ticker"].tolist(),
    )
    return {
        "ranking": ranking,
        "evaluation": evaluation.reset_index(drop=True),
        "selection_backtest": selection_by_date.reset_index(),
        "diagnostics": diagnostics,
    }


@st.cache_data(ttl="15m", max_entries=24)
def _rank_market_candidates_cached(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None,
    top_n: int,
    cache_generation: int,
) -> dict[str, object]:
    """Use process and Redis cache layers around the pooled model fitting path."""
    return shared_cache_get_or_compute(
        "market-ranking",
        (
            price_data,
            forecast_horizon,
            sentiment_by_ticker,
            top_n,
            cache_generation,
        ),
        900,
        lambda: _compute_rank_market_candidates(
            price_data,
            forecast_horizon,
            sentiment_by_ticker,
            top_n,
        ),
        allow_compute=not ANALYTICS_READ_ONLY,
    )


def rank_market_candidates(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Read worker-precomputed ranking results or compute them in local mode."""
    generation = get_cache_generation(f"model:{current_cache_scope()}")
    try:
        return _rank_market_candidates_cached(
            price_data,
            forecast_horizon,
            sentiment_by_ticker,
            top_n,
            generation,
        )
    except SharedCacheMiss as ex:
        scope = current_cache_scope()
        enqueue_analytics_job(
            "market-ranking",
            scope,
            (price_data, forecast_horizon, sentiment_by_ticker, top_n),
        )
        bac_log_kv("market_model.rank_market_candidates", status="worker_queued", error=str(ex))
        return {"ranking": pd.DataFrame(), "evaluation": pd.DataFrame(), "diagnostics": {}}
