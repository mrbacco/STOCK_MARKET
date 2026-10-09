#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: ui_state.py
#############################

"""Shared Streamlit state: the sidebar selection and the market analysis.

`app.py` renders the sidebar once per run and stores the resulting `Selection`
in session state; every page reads it with `current_selection()`. The pooled
ranking is expensive, so `market_analysis()` computes it once per selection
and refresh generation and shares it across the ranking, stock, portfolio,
and model-health pages.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from app_logging import bac_log_kv
from cache_control import get_cache_generation, invalidate_market_scope, set_cache_scope
from global_markets import reset_overview
from walk_forward_store import load_walk_forward
from market_data import classify_price_histories
from ranking_store import clear_rankings
from market_sources import (
    DEFAULT_UNIVERSE,
    MARKET_SOURCES,
    WATCHLIST_KEY,
    MarketSource,
    all_listed_tickers,
    company_name,
    get_market_source,
    universe_label,
)

if TYPE_CHECKING:
    # The model stack loads only when a page needs a ranking or projection,
    # so the Global markets page starts without scikit-learn.
    from chart_pipeline import ChartPrices, MarketRankingResult
    from market_data import PriceDataHealth

# The 12-month momentum factors need a year of prices before the first
# training row; five years matches the training size the walk-forward test validated.
HISTORY_PERIODS = ("2y", "5y")
DEFAULT_HISTORY_PERIOD = "5y"
# Holding periods in trading days; the walk-forward test favours one month.
HOLDING_PERIODS = {5: "1 week", 21: "1 month"}
DEFAULT_HORIZON = 21
MAX_WATCHLIST = 25
WATCHLIST_FILE = Path(__file__).resolve().parent / "data" / "watchlist.json"
STOCK_PAGE = "app_pages/stock.py"
# A finished analysis is shared by every browser session for this long, so a
# new tab or a page reload does not retrain the model. It matches the ranking
# cache TTL; Refresh data starts a new generation and recomputes at once.
SHARED_ANALYSIS_TTL_SECONDS = 900
MAX_SHARED_ANALYSES = 24
_shared_analyses: dict[tuple, tuple[float, MarketAnalysis]] = {}
_shared_analyses_lock = threading.Lock()


@dataclass(frozen=True)
class Selection:
    """What the user is looking at, as chosen in the sidebar."""

    universe_key: str
    source: MarketSource | None
    tickers: list[str]
    companies: dict[str, str]
    period: str
    horizon: int
    cache_scope: str

    @property
    def label(self) -> str:
        return universe_label(self.universe_key)

    @property
    def is_watchlist(self) -> bool:
        return self.source is None

    def price_display(self) -> tuple[str, str, str]:
        if self.source is not None:
            return self.source.price_display()
        return "", "%.2f", "Price (listing currency)"


@dataclass(frozen=True)
class MarketAnalysis:
    """Prices, data health, and the pooled ranking for one selection."""

    prices: ChartPrices
    health: PriceDataHealth
    ranking: MarketRankingResult
    resolved_forecasts: int = 0

    @property
    def has_ranking(self) -> bool:
        return not self.ranking.ranking.empty


def load_saved_watchlist() -> list[str]:
    """Return the persisted watchlist, or an empty list."""
    try:
        tickers = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(ticker) for ticker in tickers][:MAX_WATCHLIST]


def save_watchlist(tickers: list[str]) -> None:
    """Persist the watchlist so it survives app restarts."""
    try:
        WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        WATCHLIST_FILE.write_text(json.dumps(tickers, indent=2), encoding="utf-8")
    except OSError as ex:
        bac_log_kv("ui_state.watchlist", status="save_failed", error=str(ex))


def initialize_session_state() -> None:
    """Seed every shared session key in one place."""
    defaults = {
        "universe": DEFAULT_UNIVERSE,
        "history_period": DEFAULT_HISTORY_PERIOD,
        "forecast_horizon": DEFAULT_HORIZON,
        "watchlist_tickers": load_saved_watchlist(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    # Values from older app versions fall back to the current defaults.
    if st.session_state["history_period"] not in HISTORY_PERIODS:
        st.session_state["history_period"] = DEFAULT_HISTORY_PERIOD
    if st.session_state["forecast_horizon"] not in HOLDING_PERIODS:
        st.session_state["forecast_horizon"] = DEFAULT_HORIZON
    # A universe removed from the registry must not leave the app stuck.
    if st.session_state["universe"] not in (*MARKET_SOURCES, WATCHLIST_KEY):
        st.session_state["universe"] = DEFAULT_UNIVERSE


def _normalize_tickers(tickers: list[str]) -> list[str]:
    cleaned = []
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned[:MAX_WATCHLIST]


def _on_watchlist_change() -> None:
    tickers = _normalize_tickers(st.session_state.get("watchlist_tickers", []))
    st.session_state["watchlist_tickers"] = tickers
    save_watchlist(tickers)


def render_sidebar() -> Selection:
    """Render the shared controls and return the resulting selection."""
    with st.sidebar:
        universe_key = st.selectbox(
            "Market",
            (*MARKET_SOURCES, WATCHLIST_KEY),
            format_func=universe_label,
            key="universe",
        )
        source = get_market_source(universe_key)
        if source is not None:
            st.caption(source.description)
            tickers = source.tickers
            companies = dict(source.listings)
        else:
            listed = all_listed_tickers()
            options = sorted({*listed, *st.session_state.get("watchlist_tickers", [])})
            st.multiselect(
                "Watchlist tickers",
                options,
                format_func=lambda ticker: f"{ticker} - {listed.get(ticker, ticker)}",
                max_selections=MAX_WATCHLIST,
                accept_new_options=True,
                placeholder="Add stocks or type a Yahoo Finance symbol",
                key="watchlist_tickers",
                on_change=_on_watchlist_change,
            )
            tickers = _normalize_tickers(st.session_state.get("watchlist_tickers", []))
            companies = {ticker: company_name(ticker) for ticker in tickers}

        horizon = st.segmented_control(
            "Holding period",
            tuple(HOLDING_PERIODS),
            format_func=HOLDING_PERIODS.__getitem__,
            required=True,
            key="forecast_horizon",
            help="How long you plan to hold. The model is strongest over one month.",
        )
        with st.expander("Advanced settings"):
            period = st.segmented_control(
                "History window",
                HISTORY_PERIODS,
                required=True,
                key="history_period",
                help="Longer history gives the models more data to learn from and validate on.",
            )
        cache_scope = f"{universe_key}:{period}:1d"
        if universe_key == WATCHLIST_KEY:
            # Each watchlist gets its own refresh scope.
            cache_scope = f"{cache_scope}:{'|'.join(sorted(tickers)) or 'empty'}"
        set_cache_scope(cache_scope)
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            invalidate_market_scope(cache_scope)
            clear_rankings(str(universe_key))
            reset_overview()
            st.session_state.pop("market_analysis", None)
            st.rerun()

    selection = Selection(
        universe_key=str(universe_key),
        source=source,
        tickers=list(tickers),
        companies=companies,
        period=str(period or DEFAULT_HISTORY_PERIOD),
        horizon=int(horizon or DEFAULT_HORIZON),
        cache_scope=cache_scope,
    )
    st.session_state["selection"] = selection
    return selection


def current_selection() -> Selection:
    """Return the selection rendered by `app.py` for this run."""
    selection = st.session_state.get("selection")
    if not isinstance(selection, Selection):
        raise RuntimeError("The sidebar selection is rendered by app.py before any page.")
    set_cache_scope(selection.cache_scope)
    return selection


def market_analysis(selection: Selection, *, compute: bool = True) -> MarketAnalysis | None:
    """Return this selection's prices and ranking, computing them at most once.

    The result is reused until the selection, horizon, or a refresh changes.
    With ``compute=False`` it returns None instead of starting the work.
    """
    key = (
        selection.cache_scope,
        selection.horizon,
        get_cache_generation(f"market:{selection.cache_scope}"),
        get_cache_generation(f"model:{selection.cache_scope}"),
    )
    cached = st.session_state.get("market_analysis")
    if cached is not None and cached[0] == key:
        return cached[1]
    shared = _shared_analysis(key)
    if shared is not None:
        st.session_state["market_analysis"] = (key, shared)
        return shared
    if not compute or not selection.tickers:
        return None

    from chart_pipeline import (
        MarketRankingResult,
        load_chart_prices,
        rank_live_candidates,
        resolve_matured_forecasts,
    )

    with st.spinner("Loading price history..."):
        prices = load_chart_prices(
            selection.tickers,
            period=selection.period,
            interval="1d",
            realtime_mode=False,
            forecast_points=selection.horizon,
        )
    health = classify_price_histories(
        {ticker: prices.price_data[ticker] for ticker in prices.valid_tickers},
        realtime_mode=False,
    )
    monitoring_market = selection.universe_key
    resolved = resolve_matured_forecasts(prices.price_data, health, monitoring_market)
    ranking = MarketRankingResult()
    if not selection.is_watchlist and len(health.live_tickers) >= 2:
        with st.spinner(
            "Training the model on this market. This happens once a day per market and takes "
            "about half a minute; after that it loads in seconds.",
            show_time=True,
        ):
            ranking = rank_live_candidates(
                prices.price_data,
                health,
                forecast_horizon=selection.horizon,
                monitoring_market=monitoring_market,
            )
    analysis = MarketAnalysis(prices, health, ranking, resolved)
    st.session_state["market_analysis"] = (key, analysis)
    _store_shared_analysis(key, analysis)
    return analysis


def _shared_analysis(key: tuple) -> MarketAnalysis | None:
    with _shared_analyses_lock:
        entry = _shared_analyses.get(key)
        if entry is None or time.monotonic() - entry[0] > SHARED_ANALYSIS_TTL_SECONDS:
            return None
        return entry[1]


def _store_shared_analysis(key: tuple, analysis: MarketAnalysis) -> None:
    with _shared_analyses_lock:
        now = time.monotonic()
        expired = [
            entry_key
            for entry_key, (stored_at, _analysis) in _shared_analyses.items()
            if now - stored_at > SHARED_ANALYSIS_TTL_SECONDS
        ]
        for entry_key in expired:
            del _shared_analyses[entry_key]
        _shared_analyses[key] = (now, analysis)
        while len(_shared_analyses) > MAX_SHARED_ANALYSES:
            del _shared_analyses[min(_shared_analyses, key=lambda k: _shared_analyses[k][0])]


def walk_forward_summary(selection: Selection) -> dict | None:
    """Return the stored multi-year walk-forward summary for this selection."""
    if selection.is_watchlist:
        return None
    stored = load_walk_forward(selection.universe_key, selection.horizon, with_predictions=False)
    # A run of an older feature set says nothing about today's model.
    return stored.summary if stored is not None and stored.is_current else None


def markets_with_evidence(horizon: int) -> list[tuple[str, float]]:
    """Universes whose current model shows at least tentative tested evidence.

    Returns (universe key, rank IC) pairs, strongest first.
    """
    from model_evidence import assess_walk_forward_evidence

    found = []
    for key in MARKET_SOURCES:
        stored = load_walk_forward(key, horizon, with_predictions=False)
        if stored is None or not stored.is_current:
            continue
        if assess_walk_forward_evidence(stored.summary).level in ("supported", "tentative"):
            found.append((key, float(stored.summary.get("Rank IC", 0.0))))
    return sorted(found, key=lambda item: item[1], reverse=True)


def open_stock(ticker: str) -> None:
    """Show one stock on the Stock page."""
    st.session_state["stock_ticker"] = ticker
    st.switch_page(STOCK_PAGE)


def leaderboard_frame(selection: Selection) -> pd.DataFrame:
    """Return the selection's daily-move leaderboard (or the watchlist list)."""
    if selection.source is not None:
        return selection.source.load_performers()
    return pd.DataFrame(
        {"Ticker": selection.tickers, "Company": [selection.companies[t] for t in selection.tickers]}
    )
