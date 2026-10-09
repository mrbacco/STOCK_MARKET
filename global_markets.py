#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: global_markets.py
#############################

"""Cross-asset and cross-universe snapshots for the Global markets page.

All functions are Streamlit-free and cached through `cached_result`, so the
page renders from shared results and the analytics worker could warm them.
Performance figures are computed from daily closes:

- 1D, 1W, 1M, 3M, YTD, and 1Y returns (bars of 1, 5, 21, 63, and 252 sessions);
- 20-session realized volatility, annualized;
- the last 63 closes as a sparkline.

Yields move in percentage points, not percent, so rate changes are reported as
level differences.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from app_logging import bac_log_kv
from cache_control import cached_result
from market_data import get_price_history_batch

SPARKLINE_BARS = 63
RETURN_WINDOWS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: str


GLOBAL_INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("^GSPC", "S&P 500", "Equity indices"),
    Instrument("^IXIC", "Nasdaq Composite", "Equity indices"),
    Instrument("^DJI", "Dow Jones Industrial Average", "Equity indices"),
    Instrument("^RUT", "Russell 2000", "Equity indices"),
    Instrument("^GSPTSE", "S&P/TSX Composite", "Equity indices"),
    Instrument("^BVSP", "Ibovespa", "Equity indices"),
    Instrument("^STOXX50E", "Euro Stoxx 50", "Equity indices"),
    Instrument("^FTSE", "FTSE 100", "Equity indices"),
    Instrument("^GDAXI", "DAX", "Equity indices"),
    Instrument("^FCHI", "CAC 40", "Equity indices"),
    Instrument("FTSEMIB.MI", "FTSE MIB", "Equity indices"),
    Instrument("^IBEX", "IBEX 35", "Equity indices"),
    Instrument("^ISEQ", "ISEQ Overall", "Equity indices"),
    Instrument("^N225", "Nikkei 225", "Equity indices"),
    Instrument("^HSI", "Hang Seng", "Equity indices"),
    Instrument("000001.SS", "Shanghai Composite", "Equity indices"),
    Instrument("^KS11", "KOSPI", "Equity indices"),
    Instrument("^BSESN", "BSE Sensex", "Equity indices"),
    Instrument("^AXJO", "S&P/ASX 200", "Equity indices"),
    Instrument("^VIX", "CBOE VIX (S&P 500)", "Volatility"),
    Instrument("^VXN", "CBOE VXN (Nasdaq-100)", "Volatility"),
    Instrument("^IRX", "US 13-week T-bill", "Rates"),
    Instrument("^FVX", "US 5-year Treasury", "Rates"),
    Instrument("^TNX", "US 10-year Treasury", "Rates"),
    Instrument("^TYX", "US 30-year Treasury", "Rates"),
    Instrument("DX-Y.NYB", "US dollar index", "Currencies"),
    Instrument("EURUSD=X", "EUR/USD", "Currencies"),
    Instrument("GBPUSD=X", "GBP/USD", "Currencies"),
    Instrument("USDJPY=X", "USD/JPY", "Currencies"),
    Instrument("EURGBP=X", "EUR/GBP", "Currencies"),
    Instrument("USDCNY=X", "USD/CNY", "Currencies"),
    Instrument("GC=F", "Gold", "Commodities"),
    Instrument("SI=F", "Silver", "Commodities"),
    Instrument("HG=F", "Copper", "Commodities"),
    Instrument("CL=F", "WTI crude oil", "Commodities"),
    Instrument("BZ=F", "Brent crude oil", "Commodities"),
    Instrument("NG=F", "Natural gas", "Commodities"),
    Instrument("BTC-USD", "Bitcoin", "Crypto"),
    Instrument("ETH-USD", "Ether", "Crypto"),
)
ASSET_CLASSES = tuple(dict.fromkeys(instrument.asset_class for instrument in GLOBAL_INSTRUMENTS))
LEVEL_CHANGE_CLASSES = {"Rates"}


def performance_metrics(history: pd.DataFrame, *, level_changes: bool = False) -> dict[str, Any]:
    """Summarize one daily close history; changes are percent or level points."""
    if not {"Date", "Close"}.issubset(history.columns):
        history = pd.DataFrame(columns=["Date", "Close"])
    frame = (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(history["Date"], errors="coerce"),
                "Close": pd.to_numeric(history["Close"], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("Date")
    )
    metrics: dict[str, Any] = {"Last": np.nan, "Last session": None, "Trend": []}
    for label in (*RETURN_WINDOWS, "1Y", "YTD"):
        metrics[label] = np.nan
    metrics["Volatility 20D"] = np.nan
    if len(frame) < 2:
        return metrics

    close = frame["Close"].to_numpy(dtype=float)
    last = float(close[-1])

    def change(reference: float) -> float:
        if level_changes:
            return last - reference
        return (last / reference - 1.0) * 100.0 if reference > 0 else np.nan

    metrics["Last"] = last
    metrics["Last session"] = pd.Timestamp(frame["Date"].iloc[-1]).date()
    for label, bars in RETURN_WINDOWS.items():
        if len(close) > bars:
            metrics[label] = change(float(close[-1 - bars]))
    # One year is measured by calendar date: exchanges trade 245-255 sessions a
    # year, so a fixed bar count fails on a one-year download. A download that
    # starts a few days after the anniversary still counts.
    year_ago = frame["Date"].iloc[-1] - pd.DateOffset(years=1)
    before = frame.loc[frame["Date"] <= year_ago, "Close"]
    near = frame.loc[frame["Date"] <= year_ago + pd.Timedelta(days=4), "Close"]
    if not before.empty:
        metrics["1Y"] = change(float(before.iloc[-1]))
    elif not near.empty:
        metrics["1Y"] = change(float(near.iloc[0]))
    current_year = pd.Timestamp(frame["Date"].iloc[-1]).year
    prior_year = frame.loc[frame["Date"].dt.year < current_year, "Close"]
    if not prior_year.empty:
        metrics["YTD"] = change(float(prior_year.iloc[-1]))
    if not level_changes and len(close) > 20:
        log_returns = np.diff(np.log(close[-21:]))
        metrics["Volatility 20D"] = float(np.std(log_returns, ddof=1) * np.sqrt(252) * 100.0)
    metrics["Trend"] = [float(value) for value in close[-SPARKLINE_BARS:]]
    return metrics


@cached_result("global-market-snapshot", ttl_seconds=300, max_entries=4, generation="market")
def global_market_snapshot() -> pd.DataFrame:
    """Return one row of performance metrics per global instrument."""
    symbols = [instrument.symbol for instrument in GLOBAL_INSTRUMENTS]
    # Two years, so the one-year change always has a starting close.
    snapshot = _instrument_frame(get_price_history_batch(symbols, period="2y", interval="1d"))
    bac_log_kv(
        "global_markets.snapshot",
        instruments=len(snapshot),
        priced=int(np.isfinite(np.asarray(snapshot["Last"], dtype=float)).sum()),
    )
    return snapshot


@cached_result("universe-snapshot", ttl_seconds=900, max_entries=16, generation="market")
def universe_snapshot(universe_key: str) -> pd.DataFrame:
    """Return per-stock performance metrics for one registered universe."""
    histories = get_price_history_batch(
        _job_symbols(universe_key), period="1y", interval="1d"
    )
    return _universe_frame(universe_key, histories)


def universe_summary(snapshot: pd.DataFrame) -> dict[str, object]:
    """Aggregate one universe snapshot into breadth and median performance."""
    priced = snapshot.dropna(subset=["1D"])
    if priced.empty:
        return {"Stocks": len(snapshot), "Priced": 0}
    best = priced.loc[priced["1D"].idxmax()]
    worst = priced.loc[priced["1D"].idxmin()]
    def median(column: str) -> float:
        values = np.asarray(priced[column], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")

    daily = np.asarray(priced["1D"], dtype=float)
    return {
        "Stocks": len(snapshot),
        "Priced": len(priced),
        "Advancing": float(np.mean(daily > 0) * 100.0),
        "Median 1D": median("1D"),
        "Median 1M": median("1M"),
        "Median YTD": median("YTD"),
        "Best today": f"{best['Company']} ({best['1D']:+.1f}%)",
        "Worst today": f"{worst['Company']} ({worst['1D']:+.1f}%)",
    }


def global_movers(snapshots: list[pd.DataFrame], count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the biggest daily gainers and losers across all universes."""
    combined = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    if combined.empty or "1D" not in combined.columns:
        return pd.DataFrame(), pd.DataFrame()
    # A stock listed in several universes (e.g. Euro Stoxx 50 and DAX) counts once.
    combined = combined.dropna(subset=["1D"]).drop_duplicates("Ticker")
    gainers = combined.nlargest(count, "1D").reset_index(drop=True)
    losers = combined.nsmallest(count, "1D").reset_index(drop=True)
    return gainers, losers


# --- Background overview -------------------------------------------------------
# The Global markets page needs ~45 Yahoo requests on a cold start (about two
# minutes). A background thread fills the overview instead: recent saved
# snapshots appear immediately, then each section switches to live data as its
# download completes. The page polls this process-wide state and never blocks.

INSTRUMENTS_JOB = "instruments"
OVERVIEW_REFRESH_SECONDS = 900


@dataclass(frozen=True)
class OverviewEntry:
    frame: pd.DataFrame
    live: bool
    updated_at: float
    # For saved-snapshot entries, when the underlying prices were fetched.
    saved_at: str | None = None


_OVERVIEW: dict[str, OverviewEntry] = {}
_OVERVIEW_LOCK = threading.Lock()
_WARMUP_THREAD: threading.Thread | None = None


def overview_jobs() -> list[str]:
    """Instruments first, then every universe in registry order."""
    from market_sources import MARKET_SOURCE_REGISTRY

    return [INSTRUMENTS_JOB, *MARKET_SOURCE_REGISTRY]


def _job_symbols(job: str) -> list[str]:
    if job == INSTRUMENTS_JOB:
        return [instrument.symbol for instrument in GLOBAL_INSTRUMENTS]
    from market_sources import MARKET_SOURCE_REGISTRY

    return MARKET_SOURCE_REGISTRY[job].tickers


def _instrument_frame(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": instrument.symbol,
                "Name": instrument.name,
                "Asset class": instrument.asset_class,
                **performance_metrics(
                    histories.get(instrument.symbol, pd.DataFrame()),
                    level_changes=instrument.asset_class in LEVEL_CHANGE_CLASSES,
                ),
            }
            for instrument in GLOBAL_INSTRUMENTS
        ]
    )


def _universe_frame(job: str, histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from market_sources import MARKET_SOURCE_REGISTRY

    source = MARKET_SOURCE_REGISTRY[job]
    return pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Company": company,
                "Universe": source.label,
                "Universe key": source.key,
                **performance_metrics(histories.get(ticker, pd.DataFrame())),
            }
            for ticker, company in source.listings.items()
        ]
    )


def _saved_entry(job: str) -> OverviewEntry | None:
    """Build an entry from recent saved snapshots, if they cover the job."""
    from market_data import SNAPSHOT_PREVIEW_MIN_COVERAGE, _load_price_snapshots_safely
    from runtime_config import SNAPSHOT_PREVIEW_MAX_AGE_HOURS

    if SNAPSHOT_PREVIEW_MAX_AGE_HOURS <= 0:
        return None
    symbols = _job_symbols(job)
    histories = _load_price_snapshots_safely(symbols, "1y", "1d")
    if len(histories) < SNAPSHOT_PREVIEW_MIN_COVERAGE * len(symbols):
        return None
    fetched = pd.to_datetime(
        [frame.attrs.get("bac_fetched_at") for frame in histories.values()],
        utc=True,
        errors="coerce",
    )
    if fetched.isna().any():
        return None
    # No NaT remains, so the minimum is a real timestamp.
    oldest = cast(pd.Timestamp, fetched.min())
    if pd.Timestamp.now(tz="UTC") - oldest > pd.Timedelta(hours=SNAPSHOT_PREVIEW_MAX_AGE_HOURS):
        return None
    frame = _instrument_frame(histories) if job == INSTRUMENTS_JOB else _universe_frame(job, histories)
    return OverviewEntry(frame, live=False, updated_at=time.monotonic(), saved_at=str(oldest))


def _live_entry(job: str) -> OverviewEntry:
    frame = global_market_snapshot() if job == INSTRUMENTS_JOB else universe_snapshot(job)
    return OverviewEntry(frame, live=True, updated_at=time.monotonic())


def _needs_live_refresh(entry: OverviewEntry | None) -> bool:
    return (
        entry is None
        or not entry.live
        or time.monotonic() - entry.updated_at > OVERVIEW_REFRESH_SECONDS
    )


def _warm_overview() -> None:
    jobs = overview_jobs()
    for job in jobs:
        with _OVERVIEW_LOCK:
            known = job in _OVERVIEW
        if not known:
            saved = _saved_entry(job)
            if saved is not None:
                with _OVERVIEW_LOCK:
                    _OVERVIEW.setdefault(job, saved)
    for job in jobs:
        with _OVERVIEW_LOCK:
            entry = _OVERVIEW.get(job)
        if not _needs_live_refresh(entry):
            continue
        try:
            live = _live_entry(job)
        except Exception as ex:
            bac_log_kv("global_markets.overview", job=job, status="failed", error=str(ex))
            continue
        with _OVERVIEW_LOCK:
            _OVERVIEW[job] = live
    bac_log_kv("global_markets.overview", status="warm", jobs=len(jobs))


def ensure_overview_warmup() -> None:
    """Start the background loader unless it is running or everything is fresh."""
    global _WARMUP_THREAD
    with _OVERVIEW_LOCK:
        if _WARMUP_THREAD is not None and _WARMUP_THREAD.is_alive():
            return
        if not any(_needs_live_refresh(_OVERVIEW.get(job)) for job in overview_jobs()):
            return
        context = contextvars.copy_context()
        _WARMUP_THREAD = threading.Thread(
            target=context.run, args=(_warm_overview,), name="global-overview", daemon=True
        )
        _WARMUP_THREAD.start()


def overview_entries() -> dict[str, OverviewEntry]:
    """Return a copy of whatever overview sections are ready."""
    with _OVERVIEW_LOCK:
        return dict(_OVERVIEW)


def overview_loading() -> bool:
    """True while any section is missing or still showing saved data."""
    entries = overview_entries()
    return any(job not in entries or not entries[job].live for job in overview_jobs())


def reset_overview() -> None:
    """Forget cached sections so the next visit reloads live data."""
    with _OVERVIEW_LOCK:
        _OVERVIEW.clear()
