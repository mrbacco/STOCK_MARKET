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

HISTORY_PERIODS = ("1y", "2y", "5y")
DEFAULT_HISTORY_PERIOD = "1y"
DEFAULT_HORIZON = 3
MAX_WATCHLIST = 25
WATCHLIST_FILE = Path(__file__).resolve().parent / "data" / "watchlist.json"
STOCK_PAGE = "app_pages/stock.py"


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
        st.header("Market", divider="gray")
        universe_key = st.selectbox(
            "Stock universe",
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

        st.header("Model", divider="gray")
        period = st.segmented_control(
            "History window",
            HISTORY_PERIODS,
            required=True,
            key="history_period",
            help="Longer history gives the models more data to learn from and validate on.",
        )
        horizon = st.slider(
            "Forecast horizon (trading days)",
            min_value=1,
            max_value=5,
            key="forecast_horizon",
        )
        cache_scope = f"{universe_key}:{period}:1d"
        if universe_key == WATCHLIST_KEY:
            # Each watchlist gets its own refresh scope.
            cache_scope = f"{cache_scope}:{'|'.join(sorted(tickers)) or 'empty'}"
        set_cache_scope(cache_scope)
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            invalidate_market_scope(cache_scope)
            reset_overview()
            st.session_state.pop("market_analysis", None)
            st.rerun()

    selection = Selection(
        universe_key=str(universe_key),
        source=source,
        tickers=list(tickers),
        companies=companies,
        period=str(period or DEFAULT_HISTORY_PERIOD),
        horizon=int(horizon),
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
        with st.spinner("Training the market-wide ensemble and ranking the universe..."):
            ranking = rank_live_candidates(
                prices.price_data,
                health,
                forecast_horizon=selection.horizon,
                monitoring_market=monitoring_market,
            )
    analysis = MarketAnalysis(prices, health, ranking, resolved)
    st.session_state["market_analysis"] = (key, analysis)
    return analysis


def walk_forward_summary(selection: Selection) -> dict | None:
    """Return the stored multi-year walk-forward summary for this selection."""
    if selection.is_watchlist:
        return None
    stored = load_walk_forward(selection.universe_key, selection.horizon, with_predictions=False)
    return stored.summary if stored is not None else None


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
