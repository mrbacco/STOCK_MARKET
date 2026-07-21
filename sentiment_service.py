#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_service.py
#############################

"""Continuous RSS collection, FinBERT scoring, and background orchestration."""

from __future__ import annotations

import calendar
import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
import streamlit as st

from app_config import (
    SENTIMENT_COLLECTION_INTERVAL_SECONDS,
    SENTIMENT_MAX_NEWS_ITEMS,
)
from app_logging import bac_log_kv, bac_log_section
from provider_runtime import call_provider
from runtime_config import NEWS_MIN_INTERVAL_SECONDS
from sentiment_analysis import get_sentiment_analyzer
from sentiment_store import (
    existing_content_hashes,
    load_active_watchlist,
    save_news_sentiment,
    set_collector_state,
    update_watchlist,
)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _content_hash(title: str, _source: str) -> str:
    # Syndicated headlines commonly appear under several publisher names.  Hash
    # the normalized headline itself so those copies are scored and stored once.
    title_without_source = re.split(r"\s[-|]\s", str(title), maxsplit=1)[0]
    normalized = re.sub(r"[^a-z0-9]+", " ", _normalized_text(title_without_source)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _google_news_locale(ticker: str) -> tuple[str, str, str]:
    """Prefer local-market publishers while keeping FinBERT's English input."""
    ticker_upper = str(ticker).upper()
    if ticker_upper.endswith(".IR"):
        return "en-IE", "IE", "IE:en"
    if ticker_upper.endswith(".MI"):
        return "en", "IT", "IT:en"
    if ticker_upper.endswith(".L"):
        return "en-GB", "GB", "GB:en"
    return "en-US", "US", "US:en"


def _published_timestamp(entry: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    parsed_time = getattr(entry, "published_parsed", None)
    if parsed_time:
        seconds = calendar.timegm(parsed_time)
        return pd.Timestamp(datetime.fromtimestamp(seconds, tz=timezone.utc))

    published = getattr(entry, "published", "")
    parsed = pd.to_datetime(published, utc=True, errors="coerce")
    return fallback if pd.isna(parsed) else pd.Timestamp(parsed)


def _source_name(entry: Any) -> str:
    source = getattr(entry, "source", None)
    if isinstance(source, dict):
        return str(source.get("title", ""))
    return str(getattr(source, "title", "")) if source is not None else ""


def fetch_news_candidates(
    ticker: str,
    company_name: str = "",
    max_items: int = SENTIMENT_MAX_NEWS_ITEMS,
) -> list[dict[str, Any]]:
    """Fetch and normalize recent RSS entries without scoring them yet."""
    query = quote_plus(f'"{company_name or ticker}" stock investing when:2d')
    language, country, edition = _google_news_locale(ticker)
    url = (
        "https://news.google.com/rss/search"
        f"?q={query}&hl={language}&gl={country}&ceid={edition}"
    )
    first_seen = pd.Timestamp.now(tz="UTC")
    # Fetch through the shared provider guard rather than letting feedparser
    # perform an unbounded URL request.  This supplies explicit timeouts,
    # retry/backoff, rate limiting, and circuit-breaking BAC_LOG events.
    response = call_provider(
        "google-news-rss",
        "headline-search",
        lambda: requests.get(
            url,
            timeout=(3.05, 8),
            headers={"User-Agent": "stock-market-intelligence/1.0"},
        ),
        minimum_interval=NEWS_MIN_INTERVAL_SECONDS,
    )
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    candidates: list[dict[str, Any]] = []

    for entry in feed.entries[:max_items]:
        title = str(getattr(entry, "title", "")).strip()
        if not title:
            continue
        source = _source_name(entry)
        summary = str(getattr(entry, "summary", "")).strip()
        candidates.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "published_at": _published_timestamp(entry, first_seen),
                "first_seen_at": first_seen,
                "last_seen_at": first_seen,
                "source": source,
                "title": title,
                "summary": summary,
                "link": str(getattr(entry, "link", "")),
                "content_hash": _content_hash(title, source),
            }
        )
    bac_log_kv(
        "sentiment.fetch",
        ticker=ticker,
        candidates=len(candidates),
        language=language,
        country=country,
    )
    return candidates


def collect_tickers_once(
    company_by_ticker: Mapping[str, str],
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch, deduplicate, score, and persist one watchlist collection cycle."""
    clean_watchlist = {
        str(ticker).upper().strip(): str(company).strip()
        for ticker, company in company_by_ticker.items()
        if str(ticker).strip()
    }
    if not clean_watchlist:
        return {"tickers": 0, "fetched": 0, "new_articles": 0, "model": "unavailable"}

    update_watchlist(clean_watchlist, db_path=db_path)
    candidates_by_ticker: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(clean_watchlist))) as executor:
        futures = {
            executor.submit(fetch_news_candidates, ticker, company): ticker
            for ticker, company in clean_watchlist.items()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                candidates_by_ticker[ticker] = future.result()
            except Exception as ex:
                bac_log_kv("sentiment.collect.fetch", ticker=ticker, error=str(ex))
                candidates_by_ticker[ticker] = []

    fetched_count = sum(len(rows) for rows in candidates_by_ticker.values())
    new_candidates: list[dict[str, Any]] = []
    for ticker, candidates in candidates_by_ticker.items():
        known = existing_content_hashes(
            ticker,
            [candidate["content_hash"] for candidate in candidates],
            db_path=db_path,
        )
        seen_in_batch: set[str] = set()
        for candidate in candidates:
            content_hash = str(candidate["content_hash"])
            if content_hash in known or content_hash in seen_in_batch:
                continue
            seen_in_batch.add(content_hash)
            new_candidates.append(candidate)

    active_model = "no-new-articles"
    if new_candidates:
        analyzer = get_sentiment_analyzer()
        active_model = analyzer.active_model_name
        texts = [f"{row['title']}. {row['summary']}" for row in new_candidates]
        scores = analyzer.score_many(texts)
        scored_rows = []
        for candidate, score in zip(new_candidates, scores):
            scored_rows.append(
                {
                    **candidate,
                    "sentiment_label": score.label,
                    "sentiment": score.sentiment,
                    "positive_probability": score.positive_probability,
                    "neutral_probability": score.neutral_probability,
                    "negative_probability": score.negative_probability,
                    "model_name": score.model_name,
                }
            )
        save_news_sentiment(scored_rows, db_path=db_path)

    completed_at = pd.Timestamp.now(tz="UTC").isoformat()
    set_collector_state("last_completed_at", completed_at, db_path=db_path)
    set_collector_state("last_new_articles", len(new_candidates), db_path=db_path)
    set_collector_state("active_model", active_model, db_path=db_path)
    set_collector_state("last_error", "", db_path=db_path)
    result = {
        "tickers": len(clean_watchlist),
        "fetched": fetched_count,
        "new_articles": len(new_candidates),
        "model": active_model,
    }
    bac_log_kv("sentiment.collect", **result)
    return result


def collect_active_watchlist_once(db_path: str | Path | None = None) -> dict[str, Any]:
    """Collect all active persisted watchlist entries."""
    return collect_tickers_once(load_active_watchlist(db_path=db_path), db_path=db_path)


class BackgroundSentimentCollector:
    """One daemon thread that repeatedly collects the persisted watchlist."""

    def __init__(
        self,
        poll_seconds: int = SENTIMENT_COLLECTION_INTERVAL_SECONDS,
        db_path: str | Path | None = None,
    ) -> None:
        self.poll_seconds = max(int(poll_seconds), 60)
        self.db_path = db_path
        self._stop_event = threading.Event()
        # A separate wake event lets Streamlit notify the worker immediately
        # when the selected market changes, instead of waiting up to five minutes
        # for the next scheduled cycle.
        self._wake_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sentiment-collector",
            daemon=True,
        )
        self._thread.start()
        bac_log_kv("sentiment.background", status="started", poll_seconds=self.poll_seconds)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    def request_collection(self) -> None:
        """Wake the daemon so a newly updated watchlist is collected promptly."""
        self._wake_event.set()
        bac_log_section("sentiment.background", "Immediate collection requested.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                collect_active_watchlist_once(db_path=self.db_path)
            except Exception as ex:
                bac_log_kv("sentiment.background", status="cycle_failed", error=str(ex))
                try:
                    set_collector_state("last_error", str(ex), db_path=self.db_path)
                except Exception:
                    pass
            # Clear only a wake that this wait actually observed.  If a request
            # arrives just after a timeout, leaving the flag set guarantees the
            # next cycle sees it instead of losing the notification in a race.
            collection_requested = self._wake_event.wait(self.poll_seconds)
            if collection_requested:
                self._wake_event.clear()
        bac_log_section("sentiment.background", "Collector thread stopped.")


@st.cache_resource(show_spinner=False)
def ensure_background_sentiment_collector() -> BackgroundSentimentCollector:
    """Start exactly one collector per Streamlit process."""
    return BackgroundSentimentCollector()
