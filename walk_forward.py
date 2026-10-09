#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: walk_forward.py
#############################

"""Multi-year rolling walk-forward test of the pooled ranking model.

The production ranking is validated on a single 30-date window, which is too
short to separate skill from luck. This test replays the model over years:

1. Every `retrain_every` trading dates, the ensemble is rebuilt exactly as in
   production (`market_model.select_ensemble`): members are fitted on a base
   period, weights and the ranker decision are chosen on a later tuning
   period, and the selected members are refitted on all training dates.
2. Training uses only dates whose targets were realized before the first test
   date (a gap of `horizon` dates), at most the last `max_train_dates` dates.
3. The refitted model predicts every date of the next block; then the window
   rolls forward.

Every prediction is therefore out of sample. The design follows FinRL's
rolling-window retraining (MIT licensed); the code is written for this app's
point-in-time panel. Rank IC significance is measured on non-overlapping dates,
because overlapping multi-day targets would overstate it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app_logging import bac_log_kv
from market_model import (
    PRODUCTION_PANEL_CONFIG,
    PanelConfig,
    blend_predictions,
    build_market_panel,
    informative_feature_columns,
    select_ensemble,
)
from portfolio_backtest import top_n_backtest

DEFAULT_RETRAIN_EVERY = 63
DEFAULT_MIN_TRAIN_DATES = 252
DEFAULT_MAX_TRAIN_DATES = 504
DEFAULT_TUNING_DATES = 63

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class WalkForwardResult:
    # One row per test date and stock: predictions, baselines, and outcomes.
    predictions: pd.DataFrame
    # One row per retraining point.
    folds: pd.DataFrame
    summary: dict[str, object]


def _fold_starts(date_count: int, *, min_train: int, retrain_every: int) -> list[int]:
    return list(range(min_train, date_count, max(int(retrain_every), 1)))


def run_walk_forward(
    price_data: Mapping[str, pd.DataFrame],
    sentiment_by_ticker: Mapping[str, pd.DataFrame] | None,
    *,
    horizon: int,
    retrain_every: int = DEFAULT_RETRAIN_EVERY,
    min_train_dates: int = DEFAULT_MIN_TRAIN_DATES,
    max_train_dates: int = DEFAULT_MAX_TRAIN_DATES,
    tuning_dates: int = DEFAULT_TUNING_DATES,
    top_n: int = 10,
    cost_bps: float = 10.0,
    progress: ProgressCallback | None = None,
    config: PanelConfig = PRODUCTION_PANEL_CONFIG,
) -> WalkForwardResult:
    """Replay the production ensemble over the full history, fold by fold."""
    feature_columns = config.feature_columns
    panel = build_market_panel(price_data, horizon, sentiment_by_ticker, config)
    if panel.empty:
        return WalkForwardResult(pd.DataFrame(), pd.DataFrame(), {})
    labeled = pd.DataFrame(
        panel.dropna(subset=[*feature_columns, "target_excess_log_return"])
    )
    dates = np.sort(np.asarray(labeled["Date"].unique()))
    starts = _fold_starts(len(dates), min_train=min_train_dates, retrain_every=retrain_every)
    if not starts:
        return WalkForwardResult(pd.DataFrame(), pd.DataFrame(), {})

    blocks: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for fold, start in enumerate(starts, start=1):
        if progress is not None:
            progress((fold - 1) / len(starts), f"Retraining {fold} of {len(starts)}")
        # Training targets must be realized before the first test origin.
        train_end = start - int(horizon)
        train_dates = dates[max(0, train_end - max_train_dates):train_end]
        tuning_count = min(int(tuning_dates), len(train_dates) // 4)
        base_dates = train_dates[: len(train_dates) - tuning_count - int(horizon)]
        tune_dates = train_dates[len(train_dates) - tuning_count:]
        test_dates = dates[start:start + int(retrain_every)]
        if len(base_dates) < 60 or tuning_count < 20 or len(test_dates) == 0:
            continue

        def rows(selected: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(labeled.loc[labeled["Date"].isin(selected.tolist())])

        # Same rule as production: features too sparse in training are left out.
        fold_columns = informative_feature_columns(rows(train_dates), feature_columns)
        selection = select_ensemble(rows(base_dates), rows(tune_dates), horizon, fold_columns)
        if not selection.weights:
            continue
        test = rows(test_dates)
        blend, _components = blend_predictions(
            rows(train_dates), test, selection, horizon, fold_columns
        )
        if blend.size == 0:
            continue
        blocks.append(
            pd.DataFrame(
                {
                    "Date": test["Date"].to_numpy(),
                    "Ticker": test["Ticker"].to_numpy(),
                    "predicted_excess_return": blend,
                    # Simple rules the model must beat to justify its complexity.
                    "momentum_20d": np.asarray(test["ret_20"], dtype=float),
                    "reversal_5d": -np.asarray(test["ret_5"], dtype=float),
                    "target_excess_log_return": np.asarray(
                        test["target_excess_log_return"], dtype=float
                    ),
                }
            )
        )
        fold_rows.append(
            {
                "Retrained on": pd.Timestamp(test_dates[0]),
                "Trained through": pd.Timestamp(train_dates[-1]),
                "Training dates": len(train_dates),
                "Test dates": len(test_dates),
                "Ranker": selection.ranker_status,
                "Weights": selection.weights,
            }
        )
    if progress is not None:
        progress(1.0, "Summarizing")

    predictions = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    summary = summarize_walk_forward(
        predictions, horizon=horizon, top_n=top_n, cost_bps=cost_bps
    )
    summary["Retraining points"] = len(folds)
    summary["Model features"] = config.label
    bac_log_kv(
        "walk_forward.run",
        horizon=horizon,
        folds=len(folds),
        prediction_rows=len(predictions),
        rank_ic=summary.get("Rank IC"),
    )
    return WalkForwardResult(predictions, folds, summary)


def _daily_rank_ic(predictions: pd.DataFrame, score_column: str) -> pd.Series:
    """Spearman rank IC for each date between a score and the realized target."""
    dates: list[str] = []
    values: list[float] = []
    for date, group in predictions.groupby("Date"):
        ranks = group[[score_column, "target_excess_log_return"]].rank().to_numpy(dtype=float)
        ranks = ranks[np.isfinite(ranks).all(axis=1)]
        if len(ranks) < 3 or np.std(ranks[:, 0]) == 0 or np.std(ranks[:, 1]) == 0:
            continue
        dates.append(str(date))
        values.append(float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1]))
    return pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(dates)), dtype=float).sort_index()


def _mean(series: pd.Series) -> float:
    values = np.asarray(series, dtype=float)
    return float(values.mean()) if values.size else float("nan")


def _quarterly_means(daily_ic: pd.Series) -> dict[str, float]:
    """Average daily rank IC per calendar quarter, keyed like '2025Q3'."""
    by_quarter: dict[str, list[float]] = {}
    for date, value in zip(pd.DatetimeIndex(daily_ic.index), np.asarray(daily_ic, dtype=float)):
        by_quarter.setdefault(f"{date.year}Q{(date.month - 1) // 3 + 1}", []).append(value)
    return {quarter: float(np.mean(values)) for quarter, values in by_quarter.items()}


def _non_overlapping_t_stat(daily_ic: pd.Series, horizon: int) -> float:
    """t-statistic of mean IC on every `horizon`-th date (independent targets)."""
    sample = np.asarray(daily_ic.iloc[:: max(int(horizon), 1)], dtype=float)
    if sample.size < 3:
        return float("nan")
    spread = float(np.std(sample, ddof=1))
    return float(np.mean(sample) / spread * np.sqrt(sample.size)) if spread > 0 else float("nan")


def summarize_walk_forward(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    top_n: int,
    cost_bps: float,
) -> dict[str, object]:
    """Headline statistics of a walk-forward run."""
    if predictions.empty:
        return {}
    # Hold at most a quarter of the universe so top and bottom never overlap.
    stocks_per_date = int(np.median(np.asarray(predictions.groupby("Date")["Ticker"].nunique())))
    top_n = max(1, min(int(top_n), stocks_per_date // 4))
    model_ic = _daily_rank_ic(predictions, "predicted_excess_return")
    momentum_ic = _daily_rank_ic(predictions, "momentum_20d")
    reversal_ic = _daily_rank_ic(predictions, "reversal_5d")
    quarterly = _quarterly_means(model_ic)
    quarterly_values = np.asarray(list(quarterly.values()), dtype=float)
    backtest = top_n_backtest(predictions, horizon=horizon, top_n=top_n, cost_bps=cost_bps)
    dates = pd.to_datetime(predictions["Date"])
    span_years = (dates.max() - dates.min()).days / 365.25
    return {
        "Test start": str(dates.min().date()),
        "Test end": str(dates.max().date()),
        "Test years": round(float(span_years), 1),
        "Test dates": int(model_ic.size),
        "Rank IC": _mean(model_ic),
        "Rank IC t-stat": _non_overlapping_t_stat(model_ic, horizon),
        "Positive quarters": (
            float(np.mean(quarterly_values > 0) * 100.0) if quarterly_values.size else float("nan")
        ),
        "Quarterly rank IC": quarterly,
        "Momentum rank IC": _mean(momentum_ic),
        "Reversal rank IC": _mean(reversal_ic),
        "Top-N": int(top_n),
        "Cost bps": float(cost_bps),
        **{f"Portfolio {key}": value for key, value in backtest.summary.items()},
        "Top-N mean period excess": (
            float(np.asarray(backtest.periods["Gross excess"], dtype=float).mean() * 100.0)
            if not backtest.periods.empty
            else float("nan")
        ),
    }
