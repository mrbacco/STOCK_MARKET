#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: volatility.py
#############################

"""Point-in-time forecasts of each stock's volatility over the forecast horizon.

A GARCH(1,1) model reacts to volatility clustering: forecasts rise right after
a shock and decay back as markets calm, unlike a flat 20-day standard
deviation. Its parameters are estimated once per ticker on an early "fit"
window only, then held fixed while the model filters every later return, so
each date's forecast uses information available on that date and no
validation period leaks into the parameters.

When GARCH is unavailable or degenerate (too little data, no convergence, no
reaction to shocks, or explosive persistence), the forecast falls back to an
exponentially weighted (RiskMetrics, lambda = 0.94) volatility, which also
adapts to recent shocks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from app_logging import bac_debug_kv, bac_log_kv

# Fewer fit-window returns than this make GARCH estimates unreliable.
MIN_GARCH_FIT_RETURNS = 100
# A shock coefficient below this means the fit ignores shocks entirely.
MIN_GARCH_ALPHA = 0.01
# Persistence at or above this is (near) explosive and not mean reverting.
MAX_GARCH_PERSISTENCE = 0.999
EWMA_LAMBDA = 0.94


def ewma_horizon_volatility(log_returns: pd.Series, horizon: int) -> pd.Series:
    """Return the RiskMetrics volatility of the next `horizon` log returns.

    The value at each date uses returns up to and including that date.
    """
    squared = log_returns.astype(float).pow(2)
    variance = squared.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False, min_periods=10).mean()
    return pd.Series(
        np.sqrt(np.asarray(variance, dtype=float) * float(horizon)),
        index=variance.index,
    )


def _garch_horizon_volatility(
    percent_returns: pd.Series,
    fit_mask: pd.Series,
    horizon: int,
) -> pd.Series | None:
    """Fit GARCH(1,1) on the fit window and forecast h-step volatility everywhere."""
    try:
        from arch import arch_model
    except ImportError:
        return None

    fit_sample = percent_returns[fit_mask]
    if len(fit_sample) < MIN_GARCH_FIT_RETURNS:
        return None
    with warnings.catch_warnings():
        # Convergence problems are detected from the result instead.
        warnings.simplefilter("ignore")
        fitted = arch_model(
            fit_sample, mean="Zero", vol="GARCH", p=1, q=1, rescale=False
        ).fit(disp="off")
    parameters = dict(
        zip(map(str, fitted.params.index), np.asarray(fitted.params, dtype=float))
    )
    alpha = parameters.get("alpha[1]", 0.0)
    beta = parameters.get("beta[1]", 0.0)
    if (
        fitted.convergence_flag != 0
        or alpha < MIN_GARCH_ALPHA
        or alpha + beta >= MAX_GARCH_PERSISTENCE
    ):
        bac_debug_kv(
            "volatility.garch",
            status="degenerate_fit",
            alpha=alpha,
            beta=beta,
            convergence_flag=fitted.convergence_flag,
        )
        return None

    # Fixed parameters filter the full series; each row's forecast only uses
    # returns observed up to that row.
    forecast = (
        arch_model(percent_returns, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
        .fix(fitted.params)
        .forecast(horizon=horizon, start=0, reindex=True)
    )
    horizon_variance = forecast.variance.iloc[:, :horizon].sum(axis=1)
    return pd.Series(
        np.sqrt(np.asarray(horizon_variance, dtype=float)) / 100.0,
        index=horizon_variance.index,
    )


def horizon_volatility(
    log_returns: pd.Series,
    dates: pd.Series,
    *,
    fit_end: pd.Timestamp,
    horizon: int,
) -> tuple[pd.Series, str]:
    """Forecast one ticker's log-return volatility over the next `horizon` bars.

    Returns the per-row forecast (aligned to `log_returns`) and the method used,
    "garch" or "ewma". Rows without enough history are NaN.
    """
    returns = log_returns.astype(float)
    observed = returns.notna()
    result = pd.Series(np.nan, index=returns.index, dtype=float)
    if not observed.any():
        return result, "ewma"

    percent_returns = returns[observed] * 100.0
    fit_mask = pd.to_datetime(dates[observed]) <= pd.Timestamp(fit_end)
    try:
        garch = _garch_horizon_volatility(percent_returns, fit_mask, horizon)
    except Exception as ex:
        bac_log_kv("volatility.garch", status="failed", error=str(ex))
        garch = None

    if garch is not None:
        result.loc[garch.index] = garch.to_numpy()
        return result, "garch"
    observed_returns = pd.Series(returns[observed])
    result.loc[observed_returns.index] = ewma_horizon_volatility(
        observed_returns, horizon
    ).to_numpy()
    return result, "ewma"
