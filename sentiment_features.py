#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_features.py
#############################

"""Point-in-time sentiment aggregates aligned to price-history rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app_config import (
    MIN_SENTIMENT_TRAINING_BARS,
    SENTIMENT_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_WINDOW_HOURS,
)


def _utc_naive(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _price_cutoffs(price_dates: pd.Series) -> list[pd.Timestamp]:
    dates = [_utc_naive(value) for value in price_dates]
    is_daily = bool(dates) and all(timestamp == timestamp.normalize() for timestamp in dates)
    if is_daily:
        return [timestamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) for timestamp in dates]
    return dates


def build_sentiment_feature_frame(
    price_history: pd.DataFrame,
    sentiment_history: pd.DataFrame | None,
    latest_as_of: object | None = None,
    window_hours: int = SENTIMENT_FEATURE_WINDOW_HOURS,
) -> pd.DataFrame:
    """Build leakage-safe 24-hour aggregates for every price row.

    An article is usable only after both its publication timestamp and its
    first-seen timestamp. The latter prevents an old article discovered today
    from appearing in a historical backtest.
    """
    empty_result = pd.DataFrame(
        0.0,
        index=price_history.index,
        columns=SENTIMENT_FEATURE_COLUMNS,
    )
    if price_history.empty or "Date" not in price_history.columns:
        return empty_result
    if sentiment_history is None or sentiment_history.empty:
        return empty_result

    required = {"published_at", "first_seen_at", "sentiment"}
    if not required.issubset(sentiment_history.columns):
        return empty_result

    news = sentiment_history.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce").dt.tz_localize(None)
    news["first_seen_at"] = pd.to_datetime(news["first_seen_at"], utc=True, errors="coerce").dt.tz_localize(None)
    news["sentiment"] = pd.to_numeric(news["sentiment"], errors="coerce")
    news = news.dropna(subset=["published_at", "first_seen_at", "sentiment"])
    if news.empty:
        return empty_result

    cutoffs = _price_cutoffs(price_history["Date"])
    if latest_as_of is not None and cutoffs:
        cutoffs[-1] = _utc_naive(latest_as_of)

    window = pd.Timedelta(hours=max(int(window_hours), 1))
    rows: list[dict[str, float]] = []
    for cutoff in cutoffs:
        observable = news[
            (news["published_at"] <= cutoff)
            & (news["first_seen_at"] <= cutoff)
        ]
        current = observable[
            (observable["published_at"] > cutoff - window)
            & (observable["published_at"] <= cutoff)
        ]["sentiment"]
        previous = observable[
            (observable["published_at"] > cutoff - (2 * window))
            & (observable["published_at"] <= cutoff - window)
        ]["sentiment"]

        current_mean = float(current.mean()) if not current.empty else 0.0
        previous_mean = float(previous.mean()) if not previous.empty else 0.0
        disagreement = float(current.std(ddof=0)) if len(current) > 1 else 0.0
        rows.append(
            {
                "sentiment_24h": current_mean,
                "sentiment_change_24h": current_mean - previous_mean,
                "news_count_24h": float(len(current)),
                "sentiment_disagreement_24h": disagreement,
            }
        )

    result = pd.DataFrame(rows, index=price_history.index)
    return result.loc[:, SENTIMENT_FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def has_sufficient_sentiment_history(feature_frame: pd.DataFrame) -> bool:
    """Require enough distinct bars with observable news before model fitting."""
    if "news_count_24h" not in feature_frame.columns:
        return False
    observed_bars = int((feature_frame["news_count_24h"] > 0).sum())
    return observed_bars >= MIN_SENTIMENT_TRAINING_BARS
