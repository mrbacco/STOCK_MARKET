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
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
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
)
from app_logging import (
    bac_debug_kv,
    bac_debug_section,
    bac_log_kv,
    bac_log_list_preview,
    bac_log_section,
)
from cache_control import cached_result
from runtime_config import ANALYTICS_READ_ONLY
from conformal import adaptive_conformal_bands
from forecasting import build_feature_frame, prepare_model_history
from market_sources import resolve_market_calendar
from volatility import horizon_volatility


ModelFactory = Callable[[], Pipeline]

# Documented equity factors (Jegadeesh-Titman momentum skipping the latest month,
# George-Hwang 52-week-high proximity, low volatility, Bali-Cakici-Whitelaw MAX).
FACTOR_FEATURE_COLUMNS = (
    "momentum_12_1",
    "momentum_6_1",
    "high_52w_gap",
    "volatility_60",
    "max_return_21",
)
# Market-wide stress: how unusual today's cross-section of returns is versus the
# trailing year (the turbulence index FinRL uses to cut risk).
TURBULENCE_FEATURE_COLUMNS = ("turbulence", "turbulence_20d")
# Columns identical for every stock on a date; ranking them within a date is
# meaningless, so the cross-sectional transform leaves them unchanged.
MARKET_LEVEL_COLUMNS = (
    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "market_volatility_20",
    "market_breadth_1",
    *TURBULENCE_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class PanelConfig:
    """Which feature groups the pooled model uses."""

    factor_features: bool = False
    # Replace each stock-level feature with its percentile within the date.
    cross_sectional_ranks: bool = False
    turbulence: bool = False

    @property
    def label(self) -> str:
        """Short name stored with test results, so stale results can be spotted."""
        groups = [
            name
            for name, enabled in (
                ("factors", self.factor_features),
                ("ranks", self.cross_sectional_ranks),
                ("turbulence", self.turbulence),
            )
            if enabled
        ]
        return "+".join(["base", *groups])

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return (
            *PANEL_FEATURE_COLUMNS,
            *(FACTOR_FEATURE_COLUMNS if self.factor_features else ()),
            *(TURBULENCE_FEATURE_COLUMNS if self.turbulence else ()),
        )


# Production keeps a configuration until the multi-year walk-forward test shows
# that a variant raises rank IC across universes. Factors plus per-date ranks
# beat the base features on the same test dates in all five tested universes
# at the 21-day horizon (mean rank IC 0.050 vs 0.019).
PRODUCTION_PANEL_CONFIG = PanelConfig(factor_features=True, cross_sectional_ranks=True)


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
    # Factor columns use only past closes; they matter only when the active
    # PanelConfig includes them.
    close = frame["Close"].astype(float)
    frame["momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0
    frame["momentum_6_1"] = close.shift(21) / close.shift(126) - 1.0
    frame["high_52w_gap"] = close / close.rolling(252, min_periods=126).max() - 1.0
    frame["volatility_60"] = frame["one_bar_log_return"].rolling(60, min_periods=40).std()
    frame["max_return_21"] = close.pct_change().rolling(21, min_periods=15).max()
    return frame.replace([np.inf, -np.inf], np.nan)


def _turbulence_by_date(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Daily Mahalanobis turbulence of the universe's returns, point in time.

    Each date's cross-section of returns is compared with the mean and
    covariance of the preceding `window` dates only.
    """
    returns = panel.pivot_table(index="Date", columns="Ticker", values="one_bar_log_return")
    values = returns.to_numpy(dtype=float)
    turbulence = np.full(len(returns), np.nan)
    for row in range(window, len(returns)):
        history = values[row - window:row]
        usable = np.isfinite(history).all(axis=0) & np.isfinite(values[row])
        if usable.sum() < 5:
            continue
        sample = history[:, usable]
        deviation = values[row, usable] - sample.mean(axis=0)
        covariance = np.cov(sample, rowvar=False)
        turbulence[row] = float(deviation @ np.linalg.pinv(covariance) @ deviation) / usable.sum()
    frame = pd.DataFrame({"Date": returns.index, "turbulence": np.log1p(turbulence)})
    frame["turbulence_20d"] = frame["turbulence"].rolling(20, min_periods=10).mean()
    return frame


def build_market_panel(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    config: PanelConfig = PRODUCTION_PANEL_CONFIG,
) -> pd.DataFrame:
    """Create the leakage-safe pooled feature and target table."""
    bac_debug_kv(
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
        bac_debug_section("market_model.build_market_panel", "No usable ticker histories were found.")
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
    # Raw values stay available after ranking: volatility for band fallbacks,
    # sentiment for display.
    panel["vol_20_raw"] = panel["vol_20"]
    panel["sentiment_24h_raw"] = panel["sentiment_24h"]
    panel["news_count_24h_raw"] = panel["news_count_24h"]
    if config.turbulence:
        panel = panel.merge(_turbulence_by_date(panel), on="Date", how="left")
    if config.cross_sectional_ranks:
        stock_columns = [
            column for column in config.feature_columns if column not in MARKET_LEVEL_COLUMNS
        ]
        panel[stock_columns] = (
            panel.groupby("Date")[stock_columns].rank(pct=True) - 0.5
        )

    bac_debug_kv(
        "market_model.build_market_panel",
        panel_rows=len(panel),
        panel_dates=panel["Date"].nunique(),
        usable_target_rows=int(panel["target_excess_log_return"].notna().sum()),
        sentiment_rows=int(panel["news_count_24h_raw"].gt(0).sum()),
    )
    return panel.reset_index(drop=True)


# A stock-level feature must vary across stocks on at least this share of
# training dates to be used.
MIN_INFORMATIVE_DATE_SHARE = 0.05


def informative_feature_columns(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Drop stock-level features that barely vary across stocks in training data.

    News sentiment exists only since collection started, so long-horizon
    training rows carry almost none of it while today's rows do. A model fitted
    on near-constant values extrapolates wildly when they suddenly vary, so
    such features are left out until enough history exists. Market-level
    features are the same for every stock on a date and are always kept.
    """
    stock_columns = [column for column in feature_columns if column not in MARKET_LEVEL_COLUMNS]
    if frame.empty or not stock_columns:
        return feature_columns
    varies = frame.groupby("Date")[stock_columns].std().fillna(0.0).gt(1e-12).mean()
    kept = tuple(
        column
        for column in feature_columns
        if column in MARKET_LEVEL_COLUMNS or float(varies[column]) >= MIN_INFORMATIVE_DATE_SHARE
    )
    dropped = [column for column in feature_columns if column not in kept]
    if dropped:
        bac_debug_kv("market_model.informative_features", dropped=dropped)
    return kept


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
        bac_debug_kv(
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
    bac_debug_kv(
        "market_model.split_panel_dates",
        base_dates=len(split["base"]),
        tuning_dates=len(split["tuning"]),
        pre_evaluation_dates=len(split["pre_evaluation"]),
        evaluation_dates=len(split["evaluation"]),
        forecast_horizon_gap=forecast_horizon,
    )
    return split


RANKER_MODEL = "LightGBM ranker"
# Within each date, targets are graded 0..4 by their cross-sectional quintile.
RANKER_RELEVANCE_GRADES = 5
# Share of the latest training dates held out to calibrate the ranker's scale.
RANKER_CALIBRATION_SHARE = 0.3


def mean_rank_ic(
    dates: ArrayLike,
    predicted: ArrayLike,
    realized: ArrayLike,
) -> tuple[float, float]:
    """Return the mean daily Spearman rank IC and its t-statistic.

    Rank IC measures, for each date, how well the predicted ordering of stocks
    matches the ordering of their realized returns (+1 perfect, 0 none).
    """
    data = pd.DataFrame(
        {
            "Date": np.asarray(dates),
            "predicted": np.asarray(predicted, dtype=float),
            "realized": np.asarray(realized, dtype=float),
        }
    ).dropna(subset=["predicted", "realized"])
    daily_ic_values: list[float] = []
    for _date, group in data.groupby("Date"):
        if len(group) < 3:
            continue
        ranks = group[["predicted", "realized"]].rank().to_numpy(dtype=float)
        if np.std(ranks[:, 0]) == 0 or np.std(ranks[:, 1]) == 0:
            continue
        daily_ic_values.append(float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1]))
    daily_ic = np.asarray(daily_ic_values, dtype=float)
    if daily_ic.size == 0:
        return float("nan"), float("nan")
    mean_ic = float(np.mean(daily_ic))
    spread = float(np.std(daily_ic, ddof=1)) if daily_ic.size > 1 else float("nan")
    t_stat = (
        mean_ic / spread * float(np.sqrt(len(daily_ic)))
        if spread and np.isfinite(spread) and spread > 0
        else float("nan")
    )
    return mean_ic, t_stat


def _cross_sectional_zscore(values: ArrayLike, dates: ArrayLike) -> np.ndarray:
    """Standardize scores within each date; single-row dates score zero."""
    series = pd.Series(np.asarray(values, dtype=float))
    by_date = series.groupby(np.asarray(dates))
    spread = by_date.transform("std").replace(0.0, np.nan)
    return ((series - by_date.transform("mean")) / spread).fillna(0.0).to_numpy()


def _relevance_grades(frame: pd.DataFrame) -> np.ndarray:
    """Grade each row 0..4 by its target's quintile within its date."""
    percentile = frame.groupby("Date")["target_excess_log_return"].rank(
        pct=True, method="first"
    )
    grades = np.floor(percentile.to_numpy() * RANKER_RELEVANCE_GRADES).astype(int)
    return np.minimum(grades, RANKER_RELEVANCE_GRADES - 1)


def _new_ranker():
    from lightgbm import LGBMRanker

    return LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=PANEL_RANDOM_STATE,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def _fit_ranker(frame: pd.DataFrame, feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS):
    """Fit a LambdaRank model with one query group per date."""
    ordered = frame.sort_values(["Date", "Ticker"])
    ranker = _new_ranker()
    ranker.fit(
        ordered.loc[:, list(feature_columns)],
        _relevance_grades(ordered),
        group=ordered.groupby("Date", sort=True).size().to_numpy(),
    )
    return ranker


def _fit_predict_ranker(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    forecast_horizon: int,
    feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS,
) -> np.ndarray | None:
    """Predict excess returns from a LambdaRank model trained on daily order.

    A ranker outputs scores, not returns. Its per-date z-scores are converted
    to the return scale with a slope fitted on the latest training dates, which
    a separate earlier-dates ranker never saw (with a horizon-length gap), so
    the scale is not inflated by in-sample fit. Returns None when LightGBM is
    unavailable or the held-out scores carry no positive information.
    """
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return None

    dates = np.sort(np.asarray(training_frame["Date"].unique()))
    holdout_count = int(len(dates) * RANKER_CALIBRATION_SHARE)
    fit_end = len(dates) - holdout_count - int(forecast_horizon)
    if holdout_count < 5 or fit_end < 20:
        return None
    early = pd.DataFrame(
        training_frame.loc[training_frame["Date"].isin(dates[:fit_end].tolist())]
    )
    holdout = pd.DataFrame(
        training_frame.loc[training_frame["Date"].isin(dates[-holdout_count:].tolist())]
    )

    holdout_z = _cross_sectional_zscore(
        np.asarray(
            _fit_ranker(early, feature_columns).predict(holdout.loc[:, list(feature_columns)])
        ),
        holdout["Date"],
    )
    realized = np.asarray(holdout["target_excess_log_return"], dtype=float)
    denominator = float(np.dot(holdout_z, holdout_z))
    slope = float(np.dot(holdout_z, realized) / denominator) if denominator > 0 else 0.0
    if slope <= 0:
        bac_debug_kv("market_model.ranker", status="no_holdout_signal", slope=slope)
        return None

    final_ranker = _fit_ranker(training_frame, feature_columns)
    prediction_z = _cross_sectional_zscore(
        np.asarray(final_ranker.predict(prediction_frame.loc[:, list(feature_columns)])),
        prediction_frame["Date"],
    )
    bac_debug_kv("market_model.ranker", status="fitted", slope=slope)
    return slope * prediction_z


def _attach_horizon_volatility(
    panel: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    horizon: int,
) -> tuple[pd.Series, dict[str, int]]:
    """Forecast each row's excess-return volatility over the horizon.

    The prediction target is a stock's return relative to the market, so the
    model is fitted on daily excess returns. Rows without a forecast fall back
    to the 20-bar volatility scaled by the square root of the horizon.
    """
    sigma = pd.Series(np.nan, index=panel.index, dtype=float)
    methods = {"garch": 0, "ewma": 0}
    excess_returns = panel["one_bar_log_return"] - panel["market_one_bar_log_return"]
    for _ticker, rows in panel.groupby("Ticker", sort=False):
        ordered = rows.sort_values("Date")
        forecast, method = horizon_volatility(
            pd.Series(excess_returns.loc[ordered.index]),
            pd.Series(ordered["Date"]),
            fit_end=fit_end,
            horizon=horizon,
        )
        sigma.loc[ordered.index] = forecast.to_numpy()
        methods[method] += 1
    fallback = panel["vol_20_raw"] * np.sqrt(float(horizon))
    sigma = sigma.where(sigma > 0, fallback)
    sigma = sigma.where(sigma > 0, float(np.nanmedian(sigma.to_numpy())))
    return sigma, methods


def volatility_scaling_exponent(residuals: np.ndarray, sigma: np.ndarray) -> float:
    """Estimate how strongly prediction errors grow with volatility.

    Fits ``log|error| = a + gamma * log(sigma)`` and returns gamma clipped to
    [0, 1]: 1 means errors grow in proportion to volatility, 0 means they do
    not depend on it. Bands then scale with ``sigma ** gamma``.
    """
    residuals = np.asarray(residuals, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    usable = np.isfinite(residuals) & np.isfinite(sigma) & (sigma > 0) & (residuals != 0)
    if usable.sum() < 50:
        return 1.0
    log_sigma = np.log(sigma[usable])
    variance = float(np.var(log_sigma))
    if variance <= 0:
        return 1.0
    covariance = float(np.cov(log_sigma, np.log(np.abs(residuals[usable])), ddof=0)[0, 1])
    return float(np.clip(covariance / variance, 0.0, 1.0))


def _fit_predict_regressors(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    include_ranker: bool = False,
    forecast_horizon: int = 1,
    feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS,
) -> tuple[dict[str, np.ndarray], dict[str, Pipeline]]:
    """Fit every healthy ensemble member and return its prediction vector."""
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}
    x_train = training_frame.loc[:, list(feature_columns)]
    y_train = training_frame["target_excess_log_return"]
    x_predict = prediction_frame.loc[:, list(feature_columns)]

    for model_name, factory in _regression_model_factories().items():
        try:
            model = factory()
            model.fit(x_train, y_train)
            predictions[model_name] = np.asarray(model.predict(x_predict), dtype=float)
            fitted_models[model_name] = model
            bac_debug_kv(
                "market_model.fit_regressor",
                model=model_name,
                training_rows=len(training_frame),
                prediction_rows=len(prediction_frame),
            )
        except Exception as ex:
            bac_debug_kv(
                "market_model.fit_regressor",
                model=model_name,
                fitting_error=str(ex),
            )
    if include_ranker:
        try:
            ranker_prediction = _fit_predict_ranker(
                training_frame,
                prediction_frame,
                forecast_horizon,
                feature_columns,
            )
        except Exception as ex:
            bac_debug_kv("market_model.ranker", fitting_error=str(ex))
            ranker_prediction = None
        if ranker_prediction is not None:
            predictions[RANKER_MODEL] = ranker_prediction
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
    """Blend model vectors using only weights learned on the tuning period.

    A member that fitted during tuning can still fail on a later window. Its
    weight is then dropped and the remaining weights are rescaled to sum to
    one, so a partial ensemble is not silently shrunk towards zero.
    """
    available_weights = {
        model_name: float(weights.get(model_name, 0.0))
        for model_name in predictions
        if float(weights.get(model_name, 0.0)) > 0.0
    }
    total_weight = sum(available_weights.values())
    if total_weight <= 0.0:
        return np.array([], dtype=float)
    first = next(iter(predictions.values()))
    blended = np.zeros(len(first), dtype=float)
    for model_name, weight in available_weights.items():
        blended += predictions[model_name] * (weight / total_weight)
    return blended


def _empty_ranking_result() -> dict[str, object]:
    """Return the shape callers expect when no ranking can be produced."""
    return {"ranking": pd.DataFrame(), "evaluation": pd.DataFrame(), "diagnostics": {}}


@dataclass(frozen=True)
class EnsembleSelection:
    """Members, weights, and ranker decision learned on a tuning period."""

    use_ranker: bool
    weights: dict[str, float]
    tuning_mae: dict[str, float]
    tuning_predictions: dict[str, np.ndarray]
    tuning_rank_ic_with: float
    tuning_rank_ic_without: float

    @property
    def ranker_status(self) -> str:
        if self.use_ranker:
            return "included"
        if np.isfinite(self.tuning_rank_ic_with):
            return "not helpful on tuning period"
        return "unavailable"


def select_ensemble(
    base_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    forecast_horizon: int,
    feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS,
) -> EnsembleSelection:
    """Fit members on the base period and choose weights on the tuning period.

    The ranker joins only if the tuning period's rank IC is positive with it
    and better than without it. Production ranking and the walk-forward test
    both call this, so they evaluate exactly the same selection rule.
    """
    tuning_target = pd.Series(tuning_frame["target_excess_log_return"])
    tuning_predictions, _ = _fit_predict_regressors(
        base_frame,
        tuning_frame,
        include_ranker=True,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
    )
    without_ranker = {
        name: values for name, values in tuning_predictions.items() if name != RANKER_MODEL
    }
    rank_ic_without, _ = mean_rank_ic(
        tuning_frame["Date"],
        _weighted_prediction(without_ranker, _ensemble_weights(tuning_target, without_ranker)[0]),
        tuning_target,
    )
    rank_ic_with = float("nan")
    if RANKER_MODEL in tuning_predictions:
        rank_ic_with, _ = mean_rank_ic(
            tuning_frame["Date"],
            _weighted_prediction(
                tuning_predictions,
                _ensemble_weights(tuning_target, tuning_predictions)[0],
            ),
            tuning_target,
        )
    use_ranker = bool(
        RANKER_MODEL in tuning_predictions
        and np.isfinite(rank_ic_with)
        and rank_ic_with > 0
        and (not np.isfinite(rank_ic_without) or rank_ic_with > rank_ic_without)
    )
    chosen = tuning_predictions if use_ranker else without_ranker
    weights, tuning_mae = _ensemble_weights(tuning_target, chosen)
    return EnsembleSelection(
        use_ranker=use_ranker,
        weights=weights,
        tuning_mae=tuning_mae,
        tuning_predictions=chosen,
        tuning_rank_ic_with=rank_ic_with,
        tuning_rank_ic_without=rank_ic_without,
    )


def blend_predictions(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    selection: EnsembleSelection,
    forecast_horizon: int,
    feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Refit the selected members and return the weighted blend and components."""
    predictions, _ = _fit_predict_regressors(
        training_frame,
        prediction_frame,
        include_ranker=selection.use_ranker,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
    )
    return _weighted_prediction(predictions, selection.weights), predictions


def _fit_probability_pipeline(
    base_frame: pd.DataFrame,
    tuning_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    full_labeled_frame: pd.DataFrame,
    latest_frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = PANEL_FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit, calibrate, evaluate, and refresh probability-of-outperformance."""
    default_evaluation = np.full(len(evaluation_frame), 0.5, dtype=float)
    default_latest = np.full(len(latest_frame), 0.5, dtype=float)
    if base_frame["target_outperformed"].nunique() < 2:
        return default_evaluation, default_latest, np.nan

    try:
        base_classifier = _direction_classifier()
        base_classifier.fit(
            base_frame.loc[:, list(feature_columns)],
            base_frame["target_outperformed"],
        )
        tuning_raw = base_classifier.predict_proba(
            tuning_frame.loc[:, list(feature_columns)]
        )[:, 1]
        evaluation_raw = base_classifier.predict_proba(
            evaluation_frame.loc[:, list(feature_columns)]
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
            full_labeled_frame.loc[:, list(feature_columns)],
            full_labeled_frame["target_outperformed"],
        )
        latest_raw = latest_classifier.predict_proba(
            latest_frame.loc[:, list(feature_columns)]
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


@cached_result(
    "market-ranking",
    ttl_seconds=900,
    max_entries=24,
    generation="model",
    allow_compute=not ANALYTICS_READ_ONLY,
    # The pooled ranking is an enhancement over the per-ticker forecast curves.
    # Keep those curves available if an estimator or shared-cache entry fails.
    on_failure=_empty_ranking_result,
)
def rank_market_candidates(
    price_data: Mapping[str, pd.DataFrame],
    forecast_horizon: int,
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Fit the pooled ensemble and rank the strongest current candidates.

    Read-only web replicas serve worker-warmed rankings and queue cold ones.
    """
    bac_log_section("market_model.rank_market_candidates", "Pooled ranking started.")
    feature_columns = PRODUCTION_PANEL_CONFIG.feature_columns
    panel = build_market_panel(
        price_data, forecast_horizon, sentiment_by_ticker, PRODUCTION_PANEL_CONFIG
    )
    if panel.empty:
        return _empty_ranking_result()

    labeled = panel.dropna(
        subset=[*feature_columns, "target_excess_log_return"]
    ).copy()
    latest = (
        panel.dropna(subset=list(feature_columns))
        .sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .copy()
    )
    feature_columns = informative_feature_columns(labeled, feature_columns)
    split = split_panel_dates(pd.DatetimeIndex(labeled["Date"].unique()), forecast_horizon)
    if not split or latest.empty:
        bac_log_kv(
            "market_model.rank_market_candidates",
            status="insufficient_panel_history",
            labeled_rows=len(labeled),
            latest_rows=len(latest),
        )
        return _empty_ranking_result()

    # Volatility parameters are estimated on base-period data only.
    sigma, volatility_methods = _attach_horizon_volatility(
        panel,
        # The base period is non-empty here, so its last date is a real Timestamp.
        fit_end=cast(pd.Timestamp, pd.Timestamp(str(split["base"][-1]))),
        horizon=forecast_horizon,
    )
    labeled["horizon_volatility"] = sigma.reindex(labeled.index)
    latest["horizon_volatility"] = sigma.reindex(latest.index)

    def by_dates(dates: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(labeled.loc[labeled["Date"].isin(dates.tolist())]).copy()

    base_frame = by_dates(split["base"])
    tuning_frame = by_dates(split["tuning"])
    pre_evaluation_frame = by_dates(split["pre_evaluation"])
    evaluation_frame = by_dates(split["evaluation"])

    selection = select_ensemble(base_frame, tuning_frame, forecast_horizon, feature_columns)
    if not selection.weights:
        return _empty_ranking_result()
    tuning_target = pd.Series(tuning_frame["target_excess_log_return"])
    tuning_predictions = selection.tuning_predictions
    weights, tuning_mae = selection.weights, selection.tuning_mae
    use_ranker = selection.use_ranker
    ranker_status = selection.ranker_status
    tuning_rank_ic_with = selection.tuning_rank_ic_with
    tuning_rank_ic_without = selection.tuning_rank_ic_without

    # Final metrics come from a later untouched block, with a horizon-length gap
    # between its first origin and the preceding training origins.
    evaluation_predictions, _ = _fit_predict_regressors(
        pre_evaluation_frame,
        evaluation_frame,
        include_ranker=use_ranker,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
    )
    evaluation_blend = _weighted_prediction(evaluation_predictions, weights)
    if evaluation_blend.size == 0:
        bac_log_kv(
            "market_model.rank_market_candidates",
            status="no_evaluation_models",
            evaluation_rows=len(evaluation_frame),
        )
        return _empty_ranking_result()

    # Bands are prediction +/- q * sigma, with sigma each row's own GARCH
    # excess-return volatility. Tuning errors calibrate q; adaptive conformal
    # inference then walks the evaluation dates, learning only from outcomes
    # already realized, so the reported coverage is out-of-sample.
    tuning_blend = _weighted_prediction(tuning_predictions, weights)
    tuning_residuals = np.asarray(tuning_target, dtype=float) - tuning_blend
    tuning_sigma = np.asarray(tuning_frame["horizon_volatility"], dtype=float)
    # Errors rarely grow one-for-one with volatility; the exponent is learned
    # on tuning errors and the band scale is sigma ** exponent.
    scaling_exponent = volatility_scaling_exponent(tuning_residuals, tuning_sigma)
    tuning_scores = np.abs(tuning_residuals) / tuning_sigma**scaling_exponent
    evaluation_sigma = (
        np.asarray(evaluation_frame["horizon_volatility"], dtype=float) ** scaling_exponent
    )
    evaluation_scores = np.abs(
        np.asarray(evaluation_frame["target_excess_log_return"], dtype=float)
        - evaluation_blend
    ) / evaluation_sigma
    bands = {
        coverage: adaptive_conformal_bands(
            calibration_scores=tuning_scores,
            evaluation_dates=pd.Series(evaluation_frame["Date"]),
            evaluation_scores=evaluation_scores,
            target_coverage=coverage,
            horizon=forecast_horizon,
        )
        for coverage in (0.8, 0.5)
    }
    evaluation_probability, latest_probability, probability_brier = _fit_probability_pipeline(
        base_frame,
        tuning_frame,
        evaluation_frame,
        labeled,
        latest,
        feature_columns,
    )

    evaluation = evaluation_frame[
        ["Date", "Ticker", "target_excess_log_return", "target_outperformed"]
    ].copy()
    evaluation["predicted_excess_return"] = evaluation_blend
    evaluation["probability_outperform"] = evaluation_probability
    evaluation_half_width = bands[0.8].evaluation_quantile * evaluation_sigma
    evaluation["lower_80"] = evaluation_blend - evaluation_half_width
    evaluation["upper_80"] = evaluation_blend + evaluation_half_width
    evaluation["interval_hit_80"] = evaluation["target_excess_log_return"].between(
        evaluation["lower_80"], evaluation["upper_80"]
    )

    latest_predictions, _ = _fit_predict_regressors(
        labeled,
        latest,
        include_ranker=use_ranker,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
    )
    latest_blend = _weighted_prediction(latest_predictions, weights)
    if latest_blend.size == 0:
        bac_log_kv(
            "market_model.rank_market_candidates",
            status="no_latest_models",
            latest_rows=len(latest),
        )
        return _empty_ranking_result()
    component_matrix = np.column_stack(list(latest_predictions.values()))
    latest_agreement_spread = np.std(component_matrix, axis=1)

    ranking = latest[
        ["Ticker", "Date", "Close", *SENTIMENT_FEATURE_COLUMNS, "sentiment_24h_raw", "vol_20"]
    ].copy()
    ranking["Expected excess return"] = latest_blend * 100.0
    ranking["Probability outperform"] = latest_probability * 100.0
    latest_sigma = np.asarray(latest["horizon_volatility"], dtype=float)
    for coverage, label in ((0.5, "50"), (0.8, "80")):
        half_width = bands[coverage].latest_quantile * latest_sigma**scaling_exponent
        ranking[f"Lower {label}"] = (latest_blend - half_width) * 100.0
        ranking[f"Upper {label}"] = (latest_blend + half_width) * 100.0
    # GARCH (or EWMA fallback) excess-return volatility over the horizon.
    ranking["Predicted volatility"] = latest_sigma * 100.0
    ranking["Model disagreement"] = latest_agreement_spread * 100.0
    ranking["Sentiment score"] = ranking["sentiment_24h_raw"]

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
    # Every stock is ranked; top_n only defines the top-N selection backtest.
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))

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
    rank_ic, rank_ic_t_stat = mean_rank_ic(
        evaluation["Date"],
        evaluation["predicted_excess_return"],
        evaluation["target_excess_log_return"],
    )

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
        "50% interval coverage": bands[0.5].evaluation_coverage * 100.0,
        "Rank IC": rank_ic,
        "Rank IC t-stat": rank_ic_t_stat,
        "LightGBM ranker": ranker_status,
        "Tuning rank IC without ranker": tuning_rank_ic_without,
        "Tuning rank IC with ranker": tuning_rank_ic_with,
        "Interval method": "GARCH-scaled adaptive conformal",
        "Volatility scaling exponent": scaling_exponent,
        "Volatility models": volatility_methods,
        "Adaptive 80% miss rate": bands[0.8].latest_alpha,
        "Top-10 realized mean excess": selection_mean_excess,
        "Top-10 realized hit rate": selection_hit_rate,
        "Sentiment-observed rows": int(np.count_nonzero(np.asarray(panel["news_count_24h_raw"]) > 0)),
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
