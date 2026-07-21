#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_features.py
#############################

"""Point-in-time, novelty-aware sentiment aligned to price-history rows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app_config import (
    MIN_SENTIMENT_TRAINING_BARS,
    SENTIMENT_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_WINDOW_HOURS,
)
from app_logging import bac_log_kv, bac_log_section

try:
    import pandas_market_calendars as market_calendars
except ImportError:  # pragma: no cover - degraded dependency installations only.
    market_calendars = None


# Common corporate words do not prove that a story is specifically about the
# selected company.  More distinctive name/ticker tokens receive the relevance
# credit below.
_RELEVANCE_STOP_WORDS = {
    "and",
    "bank",
    "company",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "plc",
    "sa",
    "spa",
    "the",
}

# Event classes intentionally use transparent keywords.  The model receives an
# event-intensity feature rather than a hard-coded bullish/bearish assumption.
_EVENT_KEYWORDS = {
    "earnings": {"earnings", "profit", "revenue", "results", "guidance", "forecast"},
    "capital": {"dividend", "buyback", "offering", "debt", "bond", "capital raise"},
    "merger": {"acquisition", "acquire", "merger", "takeover", "bid"},
    "legal": {"lawsuit", "probe", "investigation", "regulator", "fine", "court"},
    "management": {"ceo", "cfo", "chair", "resigns", "appointed", "management"},
    "operations": {"launch", "contract", "partnership", "plant", "production", "orders"},
}

# Source weights are intentionally bounded.  They reduce the influence of
# anonymous/unknown outlets without pretending that any publisher is infallible.
_HIGH_QUALITY_SOURCE_TERMS = {
    "reuters",
    "bloomberg",
    "financial times",
    "wall street journal",
    "associated press",
    "rte",
    "irish times",
    "ansa",
}
_ESTABLISHED_SOURCE_TERMS = {
    "bbc",
    "cnbc",
    "marketwatch",
    "morningstar",
    "barron's",
    "business insider",
    "seeking alpha",
    "yahoo finance",
}


def _utc_naive(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def _price_cutoffs(
    price_dates: pd.Series,
    market_calendar: str = "NYSE",
) -> list[pd.Timestamp]:
    """Map daily bars to actual exchange closes and preserve intraday times."""
    dates = [_utc_naive(value) for value in price_dates]
    is_daily = bool(dates) and all(timestamp == timestamp.normalize() for timestamp in dates)
    if not is_daily:
        return dates

    if market_calendars is not None and dates:
        try:
            calendar = market_calendars.get_calendar(market_calendar)
            schedule = calendar.schedule(
                start_date=min(dates).normalize(),
                end_date=max(dates).normalize(),
            )
            close_by_date = {
                pd.Timestamp(session_date).normalize(): pd.Timestamp(close_time)
                .tz_convert("UTC")
                .tz_localize(None)
                for session_date, close_time in schedule["market_close"].items()
            }
            cutoffs = [
                close_by_date.get(
                    timestamp.normalize(),
                    timestamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
                )
                for timestamp in dates
            ]
            bac_log_kv(
                "sentiment_features.price_cutoffs",
                market_calendar=market_calendar,
                daily_rows=len(dates),
                exchange_close_rows=sum(
                    timestamp.normalize() in close_by_date for timestamp in dates
                ),
            )
            return cutoffs
        except Exception as ex:
            bac_log_kv(
                "sentiment_features.price_cutoffs",
                market_calendar=market_calendar,
                calendar_error=str(ex),
            )

    bac_log_section(
        "sentiment_features.price_cutoffs",
        "Using end-of-day fallback because the exchange schedule was unavailable.",
    )
    return [timestamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1) for timestamp in dates]


def _normalized_words(value: object) -> list[str]:
    """Return lowercase alphanumeric words used by novelty and relevance checks."""
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _novelty_key(title: object) -> str:
    """Collapse punctuation and publisher suffixes in syndicated headlines."""
    title_without_source = re.split(r"\s[-|]\s", str(title), maxsplit=1)[0]
    return " ".join(_normalized_words(title_without_source))


def _source_quality(source: object) -> float:
    normalized = str(source).strip().lower()
    if any(term in normalized for term in _HIGH_QUALITY_SOURCE_TERMS):
        return 1.0
    if any(term in normalized for term in _ESTABLISHED_SOURCE_TERMS):
        return 0.8
    return 0.6 if normalized else 0.45


def _relevance_score(row: pd.Series) -> float:
    """Estimate whether the text directly identifies the requested company."""
    text = " ".join(_normalized_words(f"{row.get('title', '')} {row.get('summary', '')}"))
    ticker_token = str(row.get("ticker", "")).split(".", maxsplit=1)[0].lower()
    company_tokens = {
        token
        for token in _normalized_words(row.get("company_name", ""))
        if len(token) >= 3 and token not in _RELEVANCE_STOP_WORDS
    }
    matched_tokens = sum(token in text for token in company_tokens)
    if ticker_token and len(ticker_token) >= 3 and re.search(rf"\b{re.escape(ticker_token)}\b", text):
        return 1.0
    if company_tokens and matched_tokens == len(company_tokens):
        return 1.0
    if matched_tokens >= 1:
        return 0.75
    # The upstream query is company-specific, so do not discard unmatched items;
    # simply make them less influential than an explicitly identified article.
    return 0.45


def _event_intensity(text: object) -> float:
    normalized = str(text).lower()
    matched_classes = sum(
        any(keyword in normalized for keyword in keywords)
        for keywords in _EVENT_KEYWORDS.values()
    )
    return min(float(matched_classes) / 2.0, 1.0)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    if clean.empty or float(clean["weight"].sum()) <= 0:
        return 0.0
    return float(np.average(clean["value"], weights=clean["weight"]))


def build_sentiment_feature_frame(
    price_history: pd.DataFrame,
    sentiment_history: pd.DataFrame | None,
    latest_as_of: object | None = None,
    window_hours: int = SENTIMENT_FEATURE_WINDOW_HOURS,
    market_calendar: str = "NYSE",
) -> pd.DataFrame:
    """Build leakage-safe, relevance-weighted aggregates for every price row.

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
        bac_log_section("sentiment_features.build", "Price history was empty or missing Date.")
        return empty_result
    if sentiment_history is None or sentiment_history.empty:
        bac_log_kv("sentiment_features.build", sentiment_rows=0, price_rows=len(price_history))
        return empty_result

    required = {"published_at", "first_seen_at", "sentiment"}
    if not required.issubset(sentiment_history.columns):
        bac_log_kv(
            "sentiment_features.build",
            missing_columns=sorted(required.difference(sentiment_history.columns)),
        )
        return empty_result

    news = sentiment_history.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce").dt.tz_localize(None)
    news["first_seen_at"] = pd.to_datetime(news["first_seen_at"], utc=True, errors="coerce").dt.tz_localize(None)
    news["sentiment"] = pd.to_numeric(news["sentiment"], errors="coerce")
    news = news.dropna(subset=["published_at", "first_seen_at", "sentiment"])
    if news.empty:
        return empty_result

    # Optional persistence fields receive safe defaults so older databases and
    # small unit-test fixtures remain compatible with the richer features.
    for column, default in {
        "ticker": "",
        "company_name": "",
        "source": "",
        "title": "",
        "summary": "",
    }.items():
        if column not in news.columns:
            news[column] = default
    if "negative_probability" not in news.columns:
        news["negative_probability"] = news["sentiment"].lt(-0.05).astype(float)
    news["negative_probability"] = pd.to_numeric(
        news["negative_probability"], errors="coerce"
    ).fillna(0.0)

    news["novelty_key"] = news["title"].map(_novelty_key)
    # Older fixtures/databases can lack title text.  Their publication/first-seen
    # pair remains a stable fallback identity instead of collapsing every row
    # into one empty novelty bucket.
    missing_novelty_key = news["novelty_key"].eq("")
    news.loc[missing_novelty_key, "novelty_key"] = (
        news.loc[missing_novelty_key, "published_at"].astype(str)
        + "|"
        + news.loc[missing_novelty_key, "first_seen_at"].astype(str)
    )
    news = (
        news.sort_values(["first_seen_at", "published_at"])
        .drop_duplicates("novelty_key", keep="first")
        .reset_index(drop=True)
    )
    news["source_quality"] = news["source"].map(_source_quality)
    news["relevance"] = news.apply(_relevance_score, axis=1)
    news["event_intensity"] = (
        news["title"].astype(str) + ". " + news["summary"].astype(str)
    ).map(_event_intensity)

    cutoffs = _price_cutoffs(price_history["Date"], market_calendar=market_calendar)
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
        ].copy()
        previous = observable[
            (observable["published_at"] > cutoff - (2 * window))
            & (observable["published_at"] <= cutoff - window)
        ].copy()

        current_mean = float(current["sentiment"].mean()) if not current.empty else 0.0
        previous_mean = float(previous["sentiment"].mean()) if not previous.empty else 0.0
        disagreement = (
            float(current["sentiment"].std(ddof=0)) if len(current) > 1 else 0.0
        )
        if current.empty:
            recency_weighted_sentiment = 0.0
            negative_share = 0.0
            source_quality = 0.0
            source_diversity = 0.0
            relevance = 0.0
            event_intensity = 0.0
        else:
            age_hours = (
                cutoff - current["published_at"]
            ).dt.total_seconds().clip(lower=0) / 3_600.0
            recency = np.exp(-np.log(2.0) * age_hours / 6.0)
            influence_weight = (
                recency * current["source_quality"] * current["relevance"]
            )
            recency_weighted_sentiment = _weighted_mean(
                current["sentiment"], influence_weight
            )
            negative_share = _weighted_mean(
                current["negative_probability"], influence_weight
            )
            source_quality = _weighted_mean(
                current["source_quality"], recency
            )
            source_diversity = float(
                current["source"].replace("", np.nan).nunique() / max(len(current), 1)
            )
            relevance = _weighted_mean(current["relevance"], recency)
            event_intensity = _weighted_mean(
                current["event_intensity"], influence_weight
            )

        volume_shock = float(np.log1p(len(current)) - np.log1p(len(previous)))
        rows.append(
            {
                "sentiment_24h": current_mean,
                "sentiment_change_24h": current_mean - previous_mean,
                "news_count_24h": float(len(current)),
                "sentiment_disagreement_24h": disagreement,
                "recency_weighted_sentiment_24h": recency_weighted_sentiment,
                "negative_share_24h": negative_share,
                "news_volume_shock_24h": volume_shock,
                "source_quality_24h": source_quality,
                "source_diversity_24h": source_diversity,
                "sentiment_relevance_24h": relevance,
                "event_intensity_24h": event_intensity,
            }
        )

    result = pd.DataFrame(rows, index=price_history.index)
    result = result.loc[:, SENTIMENT_FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bac_log_kv(
        "sentiment_features.build",
        price_rows=len(price_history),
        incoming_news_rows=len(sentiment_history),
        novel_news_rows=len(news),
        observed_feature_rows=int(result["news_count_24h"].gt(0).sum()),
        market_calendar=market_calendar,
    )
    return result


def has_sufficient_sentiment_history(feature_frame: pd.DataFrame) -> bool:
    """Require enough distinct bars with observable news before model fitting."""
    if "news_count_24h" not in feature_frame.columns:
        return False
    observed_bars = int((feature_frame["news_count_24h"] > 0).sum())
    return observed_bars >= MIN_SENTIMENT_TRAINING_BARS
