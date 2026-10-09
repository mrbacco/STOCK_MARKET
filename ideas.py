#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: ideas.py
#############################

"""Plain-language highlights for the Today page.

The pooled model already ranks every stock in a market. This module turns
that ranking into a few readable ideas: which stocks to look at, which to be
careful with, and short reasons describing what stands out about each one.
The reasons describe the stock's recent behaviour in simple terms; the
ranking itself comes from the model.

It also summarises the market's mood from the same price histories, so the
page can open with one sentence about the overall backdrop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
MAX_REASONS = 3


@dataclass(frozen=True)
class Idea:
    """One stock highlighted for the reader."""

    ticker: str
    company: str
    # Predicted return relative to the market over the holding period, in %.
    expected_excess: float
    # 80% band of that return, in %.
    low: float
    high: float
    # Chance of beating the market, in %.
    probability: float
    reasons: list[str] = field(default_factory=list)
    # "high", "low", or "" for typical price swings within this market.
    risk: str = ""


@dataclass(frozen=True)
class MarketMood:
    """A one-line read of the overall market."""

    # "positive", "mixed", "weak", "nervous", or "unknown".
    level: str
    headline: str
    detail: str


def _close(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame["Close"], dtype=float)


def _ratio(close: np.ndarray, back: int, end: int = 1) -> float:
    """Return close[-end] / close[-back] - 1, or NaN when history is too short."""
    if len(close) < back or close[-back] <= 0:
        return float("nan")
    return float(close[-end] / close[-back] - 1.0)


def stock_traits(price_data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple descriptive statistics per stock, plus their percentile in the market."""
    rows = []
    for ticker, frame in price_data.items():
        close = _close(frame)
        close = close[np.isfinite(close)]
        if close.size < TRADING_DAYS_PER_MONTH + 1:
            continue
        year = close[-TRADING_DAYS_PER_YEAR:]
        returns = np.diff(np.log(close[-61:]))
        rows.append(
            {
                "Ticker": str(ticker),
                # 12-month trend without the last month, like the model's factor.
                "trend_12_1": _ratio(close, TRADING_DAYS_PER_YEAR + 1, TRADING_DAYS_PER_MONTH + 1),
                "month": _ratio(close, TRADING_DAYS_PER_MONTH + 1),
                "from_high": float(close[-1] / year.max() - 1.0),
                "volatility": float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            }
        )
    traits = pd.DataFrame(rows, columns=["Ticker", "trend_12_1", "month", "from_high", "volatility"])
    for column in ("trend_12_1", "month", "volatility"):
        traits[f"{column}_pct"] = traits[column].rank(pct=True)
    return traits.set_index("Ticker")


def describe_stock(traits: pd.Series, sentiment: float | None) -> list[str]:
    """Up to three short phrases about what stands out for one stock."""
    reasons: list[tuple[float, str]] = []

    def add(strength: float, text: str) -> None:
        reasons.append((strength, text))

    def trait(name: str) -> float:
        value = traits.get(name)
        return float(value) if value is not None else float("nan")

    trend = trait("trend_12_1_pct")
    if trend >= 0.7:
        add(trend, "Strong uptrend over the past year")
    elif trend <= 0.3:
        add(1 - trend, "Weak trend over the past year")
    from_high = trait("from_high")
    if from_high >= -0.05:
        add(0.9, "Trading near its 52-week high")
    elif from_high <= -0.25:
        add(0.8, f"{abs(from_high) * 100:.0f}% below its 52-week high")
    month = trait("month_pct")
    if month >= 0.8:
        add(month, "One of the strongest stocks this past month")
    elif month <= 0.2:
        add(1 - month, "Pulled back over the past month")
    if sentiment is not None and np.isfinite(sentiment):
        if sentiment >= 0.2:
            add(0.75, "Recent news has been positive")
        elif sentiment <= -0.2:
            add(0.75, "Recent news has been negative")
    reasons.sort(key=lambda item: item[0], reverse=True)
    return [text for _strength, text in reasons[:MAX_REASONS]]


def risk_level(traits: pd.Series) -> str:
    """'high' for the jumpiest fifth of the market, 'low' for the steadiest third."""
    value = traits.get("volatility_pct")
    volatility = float(value) if value is not None else float("nan")
    if volatility >= 0.8:
        return "high"
    if volatility <= 0.3:
        return "low"
    return ""


def _ideas(
    rows: pd.DataFrame,
    traits: pd.DataFrame,
    companies: Mapping[str, str],
) -> list[Idea]:
    ideas = []
    for row in rows.to_dict("records"):
        ticker = str(row["Ticker"])
        sentiment = row.get("Sentiment score")
        stock = traits.loc[ticker] if ticker in traits.index else pd.Series(dtype=float)
        ideas.append(
            Idea(
                ticker=ticker,
                company=companies.get(ticker, ticker),
                expected_excess=float(row["Expected excess return"]),
                low=float(row["Lower 80"]),
                high=float(row["Upper 80"]),
                probability=float(row["Probability outperform"]),
                reasons=describe_stock(
                    stock, float(sentiment) if isinstance(sentiment, (int, float)) else None
                ),
                risk=risk_level(stock),
            )
        )
    return ideas


def pick_ideas(
    ranking: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    companies: Mapping[str, str],
    *,
    count: int = 3,
) -> tuple[list[Idea], list[Idea]]:
    """Return the stocks to look at and the stocks to be careful with.

    Ideas to look at must be expected to beat the market; stocks to be careful
    with must be expected to lag it. Either list may be shorter than `count`.
    """
    if ranking.empty:
        return [], []
    ordered = ranking.sort_values("Expected excess return", ascending=False)
    traits = stock_traits(price_data)
    best = ordered.loc[ordered["Expected excess return"] > 0].head(count)
    worst = ordered.loc[ordered["Expected excess return"] < 0].tail(count).iloc[::-1]
    return _ideas(best, traits, companies), _ideas(worst, traits, companies)


def market_mood(price_data: Mapping[str, pd.DataFrame]) -> MarketMood:
    """Summarise trend breadth and volatility across one market's stocks."""
    closes = {
        ticker: pd.Series(_close(frame), index=pd.DatetimeIndex(frame["Date"]))
        for ticker, frame in price_data.items()
        if len(frame) > 60
    }
    if len(closes) < 3:
        return MarketMood("unknown", "Not enough data", "The market's mood needs more price history.")
    prices = pd.DataFrame(closes).sort_index().ffill()
    above = (prices.iloc[-1] > prices.tail(50).mean()).mean()
    month = (prices.iloc[-1] / prices.iloc[-TRADING_DAYS_PER_MONTH - 1] - 1.0).median()
    index_returns = prices.apply(np.log).diff().mean(axis=1).dropna()
    rolling = index_returns.rolling(20).std().dropna()
    volatility_ratio = (
        float(rolling.iloc[-1] / rolling.tail(TRADING_DAYS_PER_YEAR).median())
        if len(rolling) > 20
        else 1.0
    )
    detail = (
        f"{above * 100:.0f}% of stocks are above their 50-day average, and the typical "
        f"stock moved {month * 100:+.1f}% over the past month."
    )
    if volatility_ratio >= 1.5:
        return MarketMood(
            "nervous",
            "Nervous",
            f"Prices are swinging about {volatility_ratio:.1f}x more than usual. {detail}",
        )
    if above >= 0.6 and month > 0:
        return MarketMood("positive", "Positive", f"Most stocks are rising. {detail}")
    if above <= 0.4 and month < 0:
        return MarketMood("weak", "Weak", f"Most stocks are falling. {detail}")
    return MarketMood("mixed", "Mixed", f"No clear direction. {detail}")
