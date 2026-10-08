#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: portfolio_backtest.py
#############################

"""Backtest of holding the model's top picks over the untouched evaluation period.

On every `horizon`-th evaluation date the strategy buys the `top_n` stocks with
the highest predicted excess return, equally weighted, and holds them for one
horizon. Holding periods therefore never overlap. Returns are excess returns
versus the universe's equal-weight average, so a strategy with no skill earns
about zero before costs.

Trading costs are charged on turnover: replacing a name costs one sale and one
purchase, each at `cost_bps` basis points of the traded weight. The bottom
`top_n` picks are tracked as a control; a working ranking should see the top
picks beat the bottom ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    periods: pd.DataFrame
    summary: dict[str, float]


def _mean_simple_return(log_returns: object) -> float:
    """Average simple return of equally weighted holdings given log returns."""
    values = np.asarray(log_returns, dtype=float)
    return float(np.mean(np.expm1(values))) if values.size else float("nan")


def top_n_backtest(
    evaluation: pd.DataFrame,
    *,
    horizon: int,
    top_n: int,
    cost_bps: float,
) -> BacktestResult:
    """Simulate equal-weight top-N holdings rebalanced every `horizon` dates."""
    required = {"Date", "Ticker", "predicted_excess_return", "target_excess_log_return"}
    if evaluation.empty or not required.issubset(evaluation.columns):
        return BacktestResult(pd.DataFrame(), {})

    dates = sorted(pd.to_datetime(evaluation["Date"]).unique())
    rebalance_dates = dates[:: max(int(horizon), 1)]
    rows: list[dict[str, object]] = []
    previous_holdings: set[str] = set()
    for date in rebalance_dates:
        day = pd.DataFrame(evaluation.loc[pd.to_datetime(evaluation["Date"]) == date])
        if len(day) < 2 * top_n:
            continue
        ranked = day.sort_values("predicted_excess_return", ascending=False)
        top = ranked.head(top_n)
        bottom = ranked.tail(top_n)
        holdings = set(top["Ticker"])
        # Weight replaced this period: everything at entry, then changed names.
        replaced = 1.0 if not previous_holdings else len(holdings - previous_holdings) / top_n
        # Changing a name means selling the old one and buying the new one.
        turnover_traded = replaced if not previous_holdings else 2.0 * replaced
        cost = turnover_traded * cost_bps / 10_000.0
        gross = _mean_simple_return(top["target_excess_log_return"])
        rows.append(
            {
                "Date": pd.Timestamp(date),
                "Holdings": ", ".join(top["Ticker"]),
                "Gross excess": gross,
                "Cost": cost,
                "Net excess": gross - cost,
                "Bottom-N excess": _mean_simple_return(bottom["target_excess_log_return"]),
                "Turnover": replaced,
            }
        )
        previous_holdings = holdings

    periods = pd.DataFrame(rows)
    if periods.empty:
        return BacktestResult(periods, {})
    periods["Cumulative net"] = (1.0 + periods["Net excess"]).cumprod() - 1.0
    periods["Cumulative gross"] = (1.0 + periods["Gross excess"]).cumprod() - 1.0
    periods["Cumulative bottom-N"] = (1.0 + periods["Bottom-N excess"]).cumprod() - 1.0

    periods_per_year = TRADING_DAYS_PER_YEAR / max(int(horizon), 1)
    net = periods["Net excess"].to_numpy(dtype=float)
    spread = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
    wealth = 1.0 + periods["Cumulative net"].to_numpy(dtype=float)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    summary = {
        "Periods": float(len(periods)),
        "Cumulative net excess": float(periods["Cumulative net"].iloc[-1] * 100.0),
        "Cumulative gross excess": float(periods["Cumulative gross"].iloc[-1] * 100.0),
        "Cumulative bottom-N excess": float(periods["Cumulative bottom-N"].iloc[-1] * 100.0),
        "Annualized net excess": float(np.mean(net) * periods_per_year * 100.0),
        "Information ratio": (
            float(np.mean(net) / spread * np.sqrt(periods_per_year))
            if spread and np.isfinite(spread) and spread > 0
            else float("nan")
        ),
        "Hit rate": float(np.mean(net > 0) * 100.0),
        "Max drawdown": float(drawdown.min() * 100.0),
        "Average turnover": float(periods["Turnover"].iloc[1:].mean() * 100.0)
        if len(periods) > 1
        else float("nan"),
        "Total cost": float(periods["Cost"].sum() * 100.0),
    }
    return BacktestResult(periods, summary)
