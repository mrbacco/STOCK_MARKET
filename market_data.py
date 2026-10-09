#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: market_data.py
#############################

"""Market-data, ticker parsing, and news-loading helpers.

This module isolates the public-data integration points. That makes it easier
to debug API behavior, cache usage, and malformed market payloads without
mixing those concerns into the Streamlit layout code.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
from typing import List, Mapping, cast

import numpy as np
import pandas as pd
import yfinance as yf

from app_config import MOMENTUM_PERIODS
from app_logging import (
    bac_debug_kv,
    bac_debug_list_preview,
    bac_debug_section,
    bac_log_kv,
    bac_log_list_preview,
    bac_log_section,
)
from cache_control import cached_result, current_cache_scope
from market_snapshot_store import (
    PriceSnapshot,
    load_price_history_snapshots,
    save_price_history_snapshots,
)
from marketstack_provider import fetch_marketstack_history
from provider_runtime import call_provider
from runtime_config import (
    MARKETSTACK_API_KEY,
    MARKETSTACK_BASE_URL,
    MARKETSTACK_MIN_INTERVAL_SECONDS,
    MARKET_DATA_LICENSE_CONFIRMED,
    MARKET_DATA_PROVIDER,
    RUN_IN_PROCESS_SENTIMENT,
    SNAPSHOT_PREVIEW_MAX_AGE_HOURS,
    YAHOO_MIN_INTERVAL_SECONDS,
)
from sentiment_service import collect_tickers_once
from sentiment_store import load_sentiment_history


def configure_yfinance_cache(
    cache_directory: str | Path | None = None,
) -> Path | None:
    """Point yfinance's SQLite caches at a verified writable directory.

    yfinance stores timezone, cookie, and ISIN metadata in small SQLite files.
    On Windows sandboxes, containers, and some hosted environments its default
    user-cache location can exist but still be unwritable. The resulting
    ``unable to open database file`` exception prevents price history from
    loading, which in turn leaves every downstream forecast empty.

    A caller-provided directory wins, followed by ``YFINANCE_CACHE_DIR``. The
    project data directory is the normal local/Docker location, while the OS
    temporary directory is a final portable fallback for read-only deployments.
    """
    configured_value = (
        str(cache_directory)
        if cache_directory is not None
        else os.getenv("YFINANCE_CACHE_DIR", "").strip()
    )
    candidates = [
        Path(configured_value) if configured_value else None,
        Path(__file__).resolve().parent / "data" / "yfinance-cache",
        Path(tempfile.gettempdir()) / "stock-market-yfinance-cache",
    ]

    # Preserve candidate order while avoiding duplicate work when the system
    # temporary directory happens to resolve inside the project directory.
    checked_paths: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate.resolve())
        if normalized in checked_paths:
            continue
        checked_paths.add(normalized)

        try:
            candidate.mkdir(parents=True, exist_ok=True)

            # Creating a directory alone is not a sufficient permission test on
            # all mounted filesystems. A tiny probe verifies that SQLite will be
            # able to create and update its database files here.
            write_probe = candidate / ".bac-yfinance-write-probe"
            write_probe.write_text("ok", encoding="utf-8")
            write_probe.unlink(missing_ok=True)

            # This must run before the first Ticker/download call. It configures
            # all yfinance SQLite caches despite the legacy function name.
            yf.set_tz_cache_location(str(candidate))
            bac_log_kv(
                "market_data.yfinance_cache",
                status="configured",
                directory=str(candidate),
            )
            return candidate
        except (OSError, RuntimeError) as ex:
            bac_log_kv(
                "market_data.yfinance_cache",
                status="candidate_unavailable",
                directory=str(candidate),
                error_type=type(ex).__name__,
                error=str(ex),
            )

    # Price calls retain their existing provider-level error handling, but this
    # explicit log explains why those calls may subsequently fail.
    bac_log_section(
        "market_data.yfinance_cache",
        "No writable yfinance cache directory was available.",
    )
    return None


# Configure the provider before any Streamlit cache or worker can fetch prices.
YFINANCE_CACHE_DIRECTORY = configure_yfinance_cache()

# A completely failed forty-symbol batch must not fan out into forty immediate
# provider calls on a laptop or production replica. Ten single-symbol retries
# fully cover manual portfolios and still recover enough automatic candidates
# to keep the forecast page useful.
SINGLE_TICKER_FALLBACK_LIMIT = 10


def format_price_history(history: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance history output to the columns the app expects."""
    bac_debug_kv(
        "market_data.format_price_history",
        incoming_rows=len(history),
        incoming_columns=list(history.columns),
    )

    if history.empty:
        bac_debug_section("market_data.format_price_history", "History frame was empty.")
        return pd.DataFrame()

    history = history.reset_index()
    date_column = "Datetime" if "Datetime" in history.columns else "Date"
    required_columns = {date_column, "Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(history.columns):
        bac_debug_kv(
            "market_data.format_price_history",
            missing_columns=sorted(required_columns.difference(history.columns)),
        )
        return pd.DataFrame()

    # The app uses one normalized "Date" column for both daily and intraday data.
    formatted = history[[date_column, "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={date_column: "Date"}
    )
    formatted["Date"] = pd.to_datetime(formatted["Date"]).dt.tz_localize(None)

    bac_debug_kv(
        "market_data.format_price_history",
        outgoing_rows=len(formatted),
        outgoing_columns=list(formatted.columns),
    )
    return formatted


def _mark_live_price_history(
    history: pd.DataFrame,
    *,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Attach provider provenance without changing the public dataframe shape."""
    result = history.copy()
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
    result.attrs["bac_data_status"] = "live"
    result.attrs["bac_data_provider"] = provider
    result.attrs["bac_fetched_at"] = fetched_at
    result.attrs["bac_latest_bar"] = (
        str(pd.Timestamp(result["Date"].iloc[-1])) if not result.empty else ""
    )
    return result


def assess_price_history_freshness(
    history: pd.DataFrame,
    *,
    realtime_mode: bool,
    now: object | None = None,
) -> dict[str, object]:
    """Classify severe provider staleness without misreading normal closures.

    Daily data gets a seven-day allowance for weekends and clustered exchange
    holidays. Intraday data gets four days so a Friday close remains valid over
    a long weekend, while a frozen endpoint is still detected promptly.
    """
    if history.empty or "Date" not in history.columns:
        return {
            "status": "unavailable",
            "latest_bar": None,
            "age_hours": np.nan,
        }

    latest_bar = pd.to_datetime(history["Date"], errors="coerce").max()
    if pd.isna(latest_bar):
        return {
            "status": "unavailable",
            "latest_bar": None,
            "age_hours": np.nan,
        }

    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("UTC").tz_localize(None)
    if latest_bar.tzinfo is not None:
        latest_bar = latest_bar.tz_convert("UTC").tz_localize(None)
    age_hours = max(float((current - latest_bar).total_seconds() / 3600.0), 0.0)
    maximum_age_hours = 96.0 if realtime_mode else 24.0 * 7.0
    status = "fresh" if age_hours <= maximum_age_hours else "stale"
    diagnosis = {
        "status": status,
        "latest_bar": latest_bar,
        "age_hours": age_hours,
        "maximum_age_hours": maximum_age_hours,
    }
    bac_debug_kv(
        "market_data.freshness",
        status=status,
        latest_bar=str(latest_bar),
        age_hours=age_hours,
        maximum_age_hours=maximum_age_hours,
        realtime_mode=realtime_mode,
    )
    return diagnosis


@dataclass(frozen=True)
class PriceDataHealth:
    """Which histories are fresh enough to train on, and why the rest are not."""

    live_tickers: list[str]
    stale_tickers: list[str]
    freshness_by_ticker: dict[str, dict[str, object]]


def classify_price_histories(
    price_data: Mapping[str, pd.DataFrame],
    *,
    realtime_mode: bool,
) -> PriceDataHealth:
    """Split non-empty histories into fresh and stale/recovered sets.

    Only provider-tagged histories receive wall-clock freshness gating.
    Deterministic tests and caller-supplied dataframes intentionally have no
    provider metadata and remain usable as explicit offline inputs. A live
    history that is too old is re-tagged ``provider_stale`` in place so later
    stages label its forecast as recovery output.

    The web tier and the analytics worker both rank only ``live_tickers``, so
    their shared-cache keys address the same candidate pool.
    """
    valid_tickers = [ticker for ticker, frame in price_data.items() if not frame.empty]
    freshness_by_ticker = {
        ticker: assess_price_history_freshness(
            price_data[ticker],
            realtime_mode=realtime_mode,
        )
        for ticker in valid_tickers
    }
    stale_tickers: list[str] = []
    for ticker in valid_tickers:
        provider_status = price_data[ticker].attrs.get("bac_data_status")
        freshness_status = freshness_by_ticker[ticker]["status"]
        if provider_status == "last_known_good" or (
            provider_status == "live" and freshness_status != "fresh"
        ):
            stale_tickers.append(ticker)
            if provider_status != "last_known_good":
                price_data[ticker].attrs["bac_data_status"] = "provider_stale"
    live_tickers = [ticker for ticker in valid_tickers if ticker not in stale_tickers]
    bac_log_list_preview("market_data.data_health", "live_tickers", live_tickers)
    bac_log_list_preview("market_data.data_health", "stale_tickers", stale_tickers)
    return PriceDataHealth(live_tickers, stale_tickers, freshness_by_ticker)


def _save_price_snapshots_safely(snapshots: list[PriceSnapshot]) -> None:
    """Persist recovery data without letting a disk issue block live prices."""
    if not snapshots:
        return
    try:
        save_price_history_snapshots(snapshots)
    except Exception as ex:
        bac_log_kv(
            "market_data.snapshot",
            snapshots=len(snapshots),
            status="save_failed",
            error_type=type(ex).__name__,
            error=str(ex),
        )


def _load_price_snapshots_safely(
    tickers: List[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Return last complete histories, keyed by the requested ticker spelling."""
    try:
        snapshots = load_price_history_snapshots(tickers, period, interval)
    except Exception as ex:
        bac_log_kv(
            "market_data.snapshot",
            tickers=len(tickers),
            period=period,
            interval=interval,
            status="load_failed",
            error_type=type(ex).__name__,
            error=str(ex),
        )
        return {}
    return {
        ticker: snapshots[ticker.upper()]
        for ticker in tickers
        if ticker.upper() in snapshots
    }


def _persist_and_backfill_snapshots(
    result: dict[str, pd.DataFrame],
    period: str,
    interval: str,
) -> None:
    """Save every fresh history, then fill empty ones from their last snapshot.

    Fresh frames are saved before stale fallbacks load, and a snapshot is only
    ever replaced by a complete, non-empty history. Both steps are single
    database round trips regardless of how many tickers are involved.
    """
    _save_price_snapshots_safely(
        [
            PriceSnapshot(
                ticker,
                period,
                interval,
                history,
                fetched_at=history.attrs.get("bac_fetched_at"),
            )
            for ticker, history in result.items()
            if not history.empty and history.attrs.get("bac_data_status") == "live"
        ]
    )
    missing_tickers = [ticker for ticker, history in result.items() if history.empty]
    if missing_tickers:
        result.update(_load_price_snapshots_safely(missing_tickers, period, interval))


def _fetch_single_price_history_from_provider(
    ticker: str,
    period: str,
    interval: str,
    operation: str,
) -> pd.DataFrame:
    """Execute and normalize one direct ticker request with common protection."""
    bac_debug_kv(
        "market_data.provider_selection",
        provider=MARKET_DATA_PROVIDER,
        ticker=ticker,
        period=period,
        interval=interval,
        license_confirmed=MARKET_DATA_LICENSE_CONFIRMED,
    )
    if MARKET_DATA_PROVIDER == "marketstack":
        # The licensing flag is intentionally visible in logs but does not
        # block a developer from evaluating the free API locally. Deployment
        # documentation requires it to be true before serving paying users.
        history = call_provider(
            "marketstack",
            operation,
            lambda: fetch_marketstack_history(
                ticker,
                period,
                interval,
                api_key=MARKETSTACK_API_KEY,
                base_url=MARKETSTACK_BASE_URL,
            ),
            minimum_interval=MARKETSTACK_MIN_INTERVAL_SECONDS,
        )
        return (
            _mark_live_price_history(history, provider="marketstack")
            if not history.empty
            else history
        )

    history = call_provider(
        "yahoo-finance",
        operation,
        lambda: yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        ),
        minimum_interval=YAHOO_MIN_INTERVAL_SECONDS,
    )
    formatted = format_price_history(history)
    return (
        _mark_live_price_history(formatted, provider="yfinance")
        if not formatted.empty
        else formatted
    )


def _compute_price_history_batch(
    tickers: List[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Fetch many ticker histories in one request to reduce repeated network work."""
    bac_debug_list_preview("market_data.get_price_history_batch", "requested_tickers", tickers)
    bac_debug_kv(
        "market_data.get_price_history_batch",
        period=period,
        interval=interval,
    )

    result = {ticker: pd.DataFrame() for ticker in tickers}
    if not tickers:
        bac_debug_section("market_data.get_price_history_batch", "No tickers were provided.")
        return result

    # Marketstack's EOD API is normalized one ticker at a time. Its commercial
    # request accounting is per symbol, so pretending this is one bulk request
    # would obscure quota consumption. Snapshot persistence below is shared.
    if MARKET_DATA_PROVIDER == "marketstack":
        for ticker in tickers:
            try:
                result[ticker] = _fetch_single_price_history_from_provider(
                    ticker,
                    period,
                    interval,
                    "price-history",
                )
            except Exception as ex:
                bac_debug_kv(
                    "market_data.get_price_history_batch",
                    provider=MARKET_DATA_PROVIDER,
                    ticker=ticker,
                    status="provider_failed",
                    error_type=type(ex).__name__,
                    error=str(ex),
                )
        _persist_and_backfill_snapshots(result, period, interval)
        return result

    data = pd.DataFrame()
    try:
        data = call_provider(
            "yahoo-finance",
            "price-history-batch",
            lambda: yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                # Parallel downloads are about three times faster for a whole
                # market; a ticker that times out is retried singly below and
                # otherwise backfilled from its last saved snapshot.
                threads=True,
                progress=False,
                timeout=10,
            ),
            minimum_interval=YAHOO_MIN_INTERVAL_SECONDS,
        )
    except Exception as ex:
        bac_debug_kv("market_data.get_price_history_batch", download_error=str(ex))

    if data is None or data.empty:
        bac_debug_section("market_data.get_price_history_batch", "Download returned no rows.")
    else:
        bac_debug_kv(
            "market_data.get_price_history_batch",
            raw_shape=data.shape,
            raw_is_multiindex=isinstance(data.columns, pd.MultiIndex),
        )

        # yfinance can return either ticker-first or field-first MultiIndex
        # layouts. Every successfully normalized frame is marked live so a
        # downstream forecast can distinguish it from outage recovery data.
        if isinstance(data.columns, pd.MultiIndex):
            first_level = set(data.columns.get_level_values(0))
            second_level = set(data.columns.get_level_values(1))
            for ticker in tickers:
                if ticker in first_level:
                    ticker_data = data[ticker].copy()
                elif ticker in second_level:
                    ticker_data = data.xs(ticker, axis=1, level=1).copy()
                else:
                    bac_debug_kv(
                        "market_data.get_price_history_batch",
                        ticker=ticker,
                        message="Ticker was missing from the MultiIndex payload.",
                    )
                    continue

                ticker_frame = (
                    ticker_data.to_frame().T
                    if isinstance(ticker_data, pd.Series)
                    else ticker_data
                )
                formatted = format_price_history(ticker_frame.dropna(how="all"))
                result[ticker] = (
                    _mark_live_price_history(formatted)
                    if not formatted.empty
                    else formatted
                )
                bac_debug_kv(
                    "market_data.get_price_history_batch",
                    ticker=ticker,
                    formatted_rows=len(formatted),
                )
        else:
            formatted = format_price_history(data.copy().dropna(how="all"))
            result[tickers[0]] = (
                _mark_live_price_history(formatted)
                if not formatted.empty
                else formatted
            )
            bac_debug_kv(
                "market_data.get_price_history_batch",
                ticker=tickers[0],
                formatted_rows=len(formatted),
            )

    missing_tickers = [ticker for ticker in tickers if result[ticker].empty]
    retry_tickers = missing_tickers[:SINGLE_TICKER_FALLBACK_LIMIT]
    bac_debug_list_preview(
        "market_data.get_price_history_batch",
        "single_ticker_retries",
        retry_tickers,
    )
    for ticker in retry_tickers:
        try:
            result[ticker] = _fetch_single_price_history_from_provider(
                ticker,
                period,
                interval,
                "single-price-history-fallback",
            )
        except Exception as ex:
            bac_debug_kv(
                "market_data.get_price_history_batch",
                ticker=ticker,
                status="single_ticker_fallback_failed",
                error_type=type(ex).__name__,
                error=str(ex),
            )

    _persist_and_backfill_snapshots(result, period, interval)

    non_empty_tickers = [ticker for ticker, frame in result.items() if not frame.empty]
    stale_tickers = [
        ticker
        for ticker, frame in result.items()
        if frame.attrs.get("bac_data_status") == "last_known_good"
    ]
    bac_debug_list_preview(
        "market_data.get_price_history_batch",
        "non_empty_tickers",
        non_empty_tickers,
    )
    bac_debug_list_preview(
        "market_data.get_price_history_batch",
        "last_known_good_tickers",
        stale_tickers,
    )
    return result


@cached_result(
    "price-history-batch",
    ttl_seconds=30,
    max_entries=100,
    generation="market",
)
def get_price_history_batch(
    tickers: List[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Fetch histories with targeted generation invalidation for this market."""
    return _compute_price_history_batch(list(tickers), period, interval)


# Fixed-universe leaderboards are the first page after startup. Their first
# load can show recent saved snapshots instantly while live prices download in
# a background thread, instead of blocking on Yahoo for several seconds.
SNAPSHOT_PREVIEW_STATUS = "snapshot_preview"
# Below this share of tickers with a usable snapshot, the preview is skipped.
SNAPSHOT_PREVIEW_MIN_COVERAGE = 0.8
_PREVIEW_REFRESH_STATE: dict[tuple, str] = {}
_PREVIEW_REFRESH_LOCK = threading.Lock()


def _snapshot_preview(
    tickers: List[str],
    period: str,
    interval: str,
) -> tuple[dict[str, pd.DataFrame], str] | None:
    """Return recent saved histories and their oldest fetch time, if usable."""
    if SNAPSHOT_PREVIEW_MAX_AGE_HOURS <= 0:
        return None
    snapshots = _load_price_snapshots_safely(tickers, period, interval)
    if len(snapshots) < SNAPSHOT_PREVIEW_MIN_COVERAGE * len(tickers):
        return None
    fetched_times = pd.to_datetime(
        [frame.attrs.get("bac_fetched_at") for frame in snapshots.values()],
        utc=True,
        errors="coerce",
    )
    if fetched_times.isna().any():
        return None
    oldest_fetch = cast(pd.Timestamp, fetched_times.min())
    max_age = pd.Timedelta(hours=SNAPSHOT_PREVIEW_MAX_AGE_HOURS)
    if pd.Timestamp.now(tz="UTC") - oldest_fetch > max_age:
        return None
    return snapshots, oldest_fetch.isoformat()


def _refresh_in_background(
    key: tuple,
    tickers: List[str],
    period: str,
    interval: str,
) -> None:
    """Download live prices off the request thread and warm the shared caches."""
    context = contextvars.copy_context()

    def refresh() -> None:
        try:
            # Runs in the caller's cache scope, so the warmed entry is exactly
            # the one the next rerun reads.
            context.run(get_price_history_batch, tickers, period, interval)
            bac_log_kv("market_data.snapshot_preview", status="live_prices_ready")
        except Exception as ex:
            bac_log_kv(
                "market_data.snapshot_preview",
                status="refresh_failed",
                error_type=type(ex).__name__,
                error=str(ex),
            )
        finally:
            with _PREVIEW_REFRESH_LOCK:
                _PREVIEW_REFRESH_STATE[key] = "done"

    threading.Thread(target=refresh, name="leaderboard-refresh", daemon=True).start()


def snapshot_refresh_pending() -> bool:
    """Return True while a leaderboard's live prices are still downloading."""
    with _PREVIEW_REFRESH_LOCK:
        return "refreshing" in _PREVIEW_REFRESH_STATE.values()


def _leaderboard_histories(
    tickers: List[str],
    period: str,
    interval: str,
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Return leaderboard histories and, for a preview, the snapshot fetch time.

    The first request per cache scope uses recent snapshots when available
    and starts a background download; later requests (and every request once
    that download finishes) use the normal cached provider path.
    """
    key = (current_cache_scope(), tuple(sorted(tickers)), period, interval)
    with _PREVIEW_REFRESH_LOCK:
        state = _PREVIEW_REFRESH_STATE.get(key)
    if state != "done":
        preview = _snapshot_preview(tickers, period, interval)
        if preview is not None:
            with _PREVIEW_REFRESH_LOCK:
                start_refresh = key not in _PREVIEW_REFRESH_STATE
                if start_refresh:
                    _PREVIEW_REFRESH_STATE[key] = "refreshing"
            if start_refresh:
                _refresh_in_background(key, tickers, period, interval)
            snapshots, fetched_at = preview
            bac_log_kv(
                "market_data.snapshot_preview",
                status="served",
                tickers=len(snapshots),
                fetched_at=fetched_at,
            )
            return snapshots, fetched_at
        with _PREVIEW_REFRESH_LOCK:
            _PREVIEW_REFRESH_STATE.setdefault(key, "done")
    return get_price_history_batch(tickers, period=period, interval=interval), None


def _is_live_leaderboard(result: pd.DataFrame) -> bool:
    """Cache only leaderboards built from live prices, never previews."""
    return result.attrs.get("bac_data_status") != SNAPSHOT_PREVIEW_STATUS


def _rank_latest_daily_performers(
    listings: Mapping[str, str],
    limit: int,
    log_context: str,
) -> pd.DataFrame:
    """Rank a fixed market universe by its latest two available daily closes."""
    bac_log_kv(log_context, limit=limit, universe_size=len(listings))
    columns = ["Ticker", "Company", "Daily change", "Last price", "Last session"]
    price_data, preview_fetched_at = _leaderboard_histories(
        list(listings),
        period="5d",
        interval="1d",
    )

    rows = []
    for ticker, company in listings.items():
        history = price_data.get(ticker, pd.DataFrame())
        if history.empty:
            bac_log_kv(
                log_context,
                ticker=ticker,
                message="Skipped because no history was returned.",
            )
            continue

        closes = history[["Date", "Close"]].dropna().sort_values("Date")
        if len(closes) < 2:
            bac_log_kv(
                log_context,
                ticker=ticker,
                message="Skipped because fewer than two closes were available.",
            )
            continue

        previous_close = float(closes["Close"].iloc[-2])
        last_price = float(closes["Close"].iloc[-1])
        if previous_close == 0:
            bac_log_kv(
                log_context,
                ticker=ticker,
                message="Skipped because the previous close was zero.",
            )
            continue

        daily_change = ((last_price - previous_close) / previous_close) * 100
        if not np.isfinite(daily_change):
            bac_log_kv(
                log_context,
                ticker=ticker,
                message="Skipped because the daily change was not finite.",
            )
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": daily_change,
                "Last price": last_price,
                "Last session": pd.Timestamp(closes["Date"].iloc[-1]).date(),
            }
        )

    result = (
        pd.DataFrame(rows).sort_values("Daily change", ascending=False).head(limit).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=columns)
    )
    if preview_fetched_at is not None:
        result.attrs["bac_data_status"] = SNAPSHOT_PREVIEW_STATUS
        result.attrs["bac_fetched_at"] = preview_fetched_at
    bac_log_kv(log_context, result_rows=len(result), preview=preview_fetched_at is not None)
    return result


# Leaderboards are cheap to rebuild from the shared price-batch cache, so they
# stay process-local rather than adding another Redis entry.
@cached_result(
    "universe-leaderboard",
    ttl_seconds=300,
    max_entries=32,
    generation="market",
    shared=False,
    cache_if=_is_live_leaderboard,
)
def get_universe_leaderboard(universe_key: str) -> pd.DataFrame:
    """Rank one registered universe by each stock's latest daily change."""
    from market_sources import MARKET_SOURCE_REGISTRY

    source = MARKET_SOURCE_REGISTRY[universe_key]
    return _rank_latest_daily_performers(
        source.listings,
        len(source.listings),
        f"market_data.leaderboard.{universe_key}",
    )


@cached_result("news", ttl_seconds=60, max_entries=200, shared=False)
def get_news(ticker: str, company_name: str = "", max_items: int = 20) -> pd.DataFrame:
    """Collect current headlines and return persistent financial sentiment."""
    bac_debug_kv(
        "market_data.get_news",
        ticker=ticker,
        company_name=company_name,
        max_items=max_items,
    )
    # Production web replicas only read the shared store.  The dedicated worker
    # owns network collection and FinBERT inference, preventing one RSS request
    # per user/session.  Local mode preserves the convenient on-demand behavior.
    if RUN_IN_PROCESS_SENTIMENT:
        collect_tickers_once({ticker: company_name or ticker})
    result = load_sentiment_history(ticker, limit=max_items)
    if result.empty:
        bac_debug_kv("market_data.get_news", ticker=ticker, result_rows=0)
        return pd.DataFrame()
    result = result.rename(columns={"published_at": "published"})
    bac_debug_kv("market_data.get_news", ticker=ticker, result_rows=len(result))
    return result


def growth_score(df: pd.DataFrame, periods: int = MOMENTUM_PERIODS) -> float:
    """Compute a simple percentage-change score over the requested lookback window."""
    bac_debug_kv("market_data.growth_score", rows=len(df), periods=periods)
    if df.empty or len(df) < periods + 1:
        bac_debug_section("market_data.growth_score", "Insufficient history for momentum score.")
        return float("-inf")

    start = df["Close"].iloc[-(periods + 1)]
    end = df["Close"].iloc[-1]
    if start == 0:
        bac_debug_section("market_data.growth_score", "Start price was zero; returning -inf.")
        return float("-inf")

    score = ((end - start) / start) * 100.0
    bac_debug_kv("market_data.growth_score", start=float(start), end=float(end), score=score)
    return score
