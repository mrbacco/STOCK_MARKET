#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: conformal.py
#############################

"""Volatility-scaled, adaptive conformal prediction intervals.

Each band is ``prediction +/- q * sigma``, where ``sigma`` is the row's own
volatility forecast, so calm stocks get narrow bands and volatile ones wide
ones. ``q`` is a conformal quantile of past normalized errors
``|actual - prediction| / sigma``:

1. Calibration starts from the tuning period's errors.
2. Walking through the evaluation dates in order, the band for each date uses
   only errors whose outcome had already been realized. A forecast made
   ``horizon`` bars ago is the newest one whose result is known.
3. Adaptive conformal inference (Gibbs and Candes, 2021) nudges the miss rate
   it aims for after every realized date: misses widen later bands, an
   unbroken run of hits narrows them. Coverage therefore tracks the target
   even when market behaviour drifts.

The reported evaluation coverage is the honest, sequential coverage of those
bands. The final state then produces the band for today's forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Learning rate of the adaptive miss-rate update.
ACI_STEP_SIZE = 0.05
# Keep the adaptive miss rate inside sensible bounds (never an infinite band).
MIN_ALPHA, MAX_ALPHA = 0.005, 0.95


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample conformal quantile for miss rate `alpha`."""
    finite = np.sort(scores[np.isfinite(scores)])
    if finite.size == 0:
        return float("nan")
    rank = math.ceil((finite.size + 1) * (1.0 - alpha))
    return float(finite[min(max(rank, 1), finite.size) - 1])


@dataclass(frozen=True)
class ConformalBands:
    """Sequential evaluation bands plus the state used for today's band."""

    # Half-width multipliers applied to each evaluation row's sigma.
    evaluation_quantile: np.ndarray
    # Share of evaluation rows whose outcome fell inside its band.
    evaluation_coverage: float
    # Quantile for new forecasts, after learning from every realized error.
    latest_quantile: float
    latest_alpha: float


def adaptive_conformal_bands(
    *,
    calibration_scores: np.ndarray,
    evaluation_dates: pd.Series,
    evaluation_scores: np.ndarray,
    target_coverage: float,
    horizon: int,
    step_size: float = ACI_STEP_SIZE,
) -> ConformalBands:
    """Run adaptive conformal inference over chronologically ordered dates.

    `evaluation_scores` are each row's normalized absolute error
    ``|actual - prediction| / sigma`` (NaN when sigma is unknown). They are
    used only after the row's outcome would have been realized.
    """
    target_alpha = 1.0 - float(target_coverage)
    alpha = target_alpha
    dates = pd.to_datetime(pd.Series(evaluation_dates).reset_index(drop=True))
    scores = np.asarray(evaluation_scores, dtype=float)
    unique_dates = list(pd.unique(dates.sort_values()))
    rows_by_date = [np.flatnonzero((dates == date).to_numpy()) for date in unique_dates]

    pool = list(np.asarray(calibration_scores, dtype=float))
    quantiles = np.full(len(dates), np.nan)
    miss_rate_by_date: list[float] = []

    def realize(date_index: int) -> None:
        """Add one date's outcomes to the pool and adapt the target miss rate."""
        nonlocal alpha
        rows = rows_by_date[date_index]
        pool.extend(scores[rows].tolist())
        alpha = float(
            np.clip(alpha + step_size * (target_alpha - miss_rate_by_date[date_index]),
                    MIN_ALPHA, MAX_ALPHA)
        )

    for position, rows in enumerate(rows_by_date):
        realized = position - int(horizon)
        if realized >= 0:
            realize(realized)
        quantile = conformal_quantile(np.asarray(pool), alpha)
        quantiles[rows] = quantile
        row_scores = scores[rows]
        known = np.isfinite(row_scores)
        miss_rate_by_date.append(
            float(np.mean(row_scores[known] > quantile)) if known.any() else target_alpha
        )

    # Every evaluation outcome is realized by now, so today's band learns from
    # the dates the sequential pass was still waiting on.
    for pending in range(max(len(rows_by_date) - int(horizon), 0), len(rows_by_date)):
        realize(pending)

    finite_scores = np.isfinite(scores) & np.isfinite(quantiles)
    coverage = (
        float(np.mean(scores[finite_scores] <= quantiles[finite_scores]))
        if finite_scores.any()
        else float("nan")
    )
    return ConformalBands(
        evaluation_quantile=quantiles,
        evaluation_coverage=coverage,
        latest_quantile=conformal_quantile(np.asarray(pool), alpha),
        latest_alpha=alpha,
    )
