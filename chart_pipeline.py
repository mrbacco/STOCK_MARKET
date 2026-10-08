#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: chart_pipeline.py
#############################

"""Data and model pipeline behind the Charts view, free of Streamlit.

`views.render_charts_view` used to load prices, rank the market, fit and
compare models, choose the active forecast, and write monitoring records
inline with its Plotly and table rendering. Those steps now live here and
return small dataclasses, so they can be tested without a Streamlit runtime
and the view module only decides how results are displayed.

Pipeline order, as used by the view:

1. `load_chart_prices` - fetch histories, falling back to daily bars when an
   intraday request returns nothing.
2. `market_data.classify_price_histories` - split fresh and stale histories.
3. `resolve_matured_forecasts` - close earlier forecasts whose target arrived.
4. `rank_live_candidates` - pooled market ranking (automatic daily sources).
5. `select_charted_tickers` - which tickers get charts, plus headline metrics.
6. `build_ticker_forecast` / `record_displayed_forecast` - one ticker's
   forecast curve, its uncertainty bands, and its monitoring record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app_config import MAX_CHARTED_PERFORMERS
from app_logging import bac_debug_kv, bac_log_kv, bac_log_list_preview, bac_log_section
from forecasting import (
    add_forecast_intervals,
    backtest_forecast_model,
    diagnose_forecast_readiness,
    forecast_feature_model,
    future_projection_dates,
    summarize_model_comparison,
)
from market_data import (
    PriceDataHealth,
    get_price_history_batch,
    growth_score,
    momentum_label,
)
from market_model import rank_market_candidates
from market_sources import MarketSource, resolve_market_calendar
from model_monitoring import (
    record_forecast,
    record_market_model_run,
    resolve_pending_forecasts,
)
from sentiment_store import load_sentiment_history

# Live prices can change each minute, but refitting and walk-forward testing
# every model on each poll would create unnecessary CPU load. The model uses
# only bars before the currently active five-minute bucket; the solid observed
# line still displays every price bar returned by the provider.
REALTIME_MODEL_REFRESH_FREQUENCY = "5min"

PRICE_ONLY_MODEL = "Price only"
SENTIMENT_MODEL = "Price + sentiment"


def prepare_realtime_forecast_history(
    price_history: pd.DataFrame,
    *,
    realtime_mode: bool,
) -> pd.DataFrame:
    """Freeze intraday model input within each five-minute refresh bucket.

    A 60-second fragment rerun should redraw new observed prices, but it should
    not continuously refit the model against a bar that may still be forming.
    Excluding the latest five-minute bucket gives the forecast a stable origin
    until the next bucket begins. Daily mode returns the original history
    unchanged.
    """
    if (
        not realtime_mode
        or price_history.empty
        or "Date" not in price_history.columns
    ):
        return price_history

    dates = pd.to_datetime(price_history["Date"], errors="coerce")
    latest_bar = dates.max()
    if pd.isna(latest_bar):
        return price_history

    active_bucket_start = latest_bar.floor(REALTIME_MODEL_REFRESH_FREQUENCY)
    completed_history = price_history.loc[dates < active_bucket_start].copy()

    # Very short market sessions or newly listed symbols may not yet have
    # enough completed bars. In that case retain the full frame so the existing
    # readiness diagnostics, rather than this optimization, decide whether a
    # forecast is possible.
    if len(completed_history) < 30:
        completed_history = price_history.copy()
        status = "full_history_short_fallback"
    else:
        status = "completed_bucket"

    completed_history.attrs.update(price_history.attrs)
    bac_debug_kv(
        "chart_pipeline.realtime_model_input",
        status=status,
        live_rows=len(price_history),
        model_rows=len(completed_history),
        latest_live_bar=str(latest_bar),
        latest_model_bar=(
            str(completed_history["Date"].iloc[-1])
            if not completed_history.empty
            else None
        ),
        refresh_frequency=REALTIME_MODEL_REFRESH_FREQUENCY,
    )
    return completed_history


def volatility_regime(price_history: pd.DataFrame) -> str:
    """Classify the latest realized volatility relative to the ticker's history."""
    close = pd.to_numeric(price_history.get("Close"), errors="coerce")
    rolling_volatility = np.log(close).diff().rolling(20, min_periods=10).std().dropna()
    if rolling_volatility.empty:
        return "Unknown"
    latest = float(rolling_volatility.iloc[-1])
    lower_quartile = float(rolling_volatility.quantile(0.25))
    upper_quartile = float(rolling_volatility.quantile(0.75))
    regime = (
        "High volatility"
        if latest >= upper_quartile
        else "Low volatility"
        if latest <= lower_quartile
        else "Normal volatility"
    )
    bac_debug_kv(
        "chart_pipeline.volatility_regime",
        latest=latest,
        lower_quartile=lower_quartile,
        upper_quartile=upper_quartile,
        regime=regime,
    )
    return regime


@dataclass(frozen=True)
class ChartPrices:
    """Loaded histories plus the mode actually used to load them."""

    price_data: dict[str, pd.DataFrame]
    valid_tickers: list[str]
    realtime_mode: bool
    interval: str
    forecast_points: int
    # True when an empty intraday request was replaced by recent daily bars.
    daily_fallback: bool = False


def load_chart_prices(
    tickers: list[str],
    *,
    period: str,
    interval: str,
    realtime_mode: bool,
    forecast_points: int,
) -> ChartPrices:
    """Load chart histories, falling back to daily bars if intraday is empty."""
    price_data = get_price_history_batch(tickers, period=period, interval=interval)
    valid_tickers = [ticker for ticker in tickers if not price_data[ticker].empty]
    bac_log_list_preview("chart_pipeline.prices", "valid_tickers", valid_tickers)
    if not realtime_mode or valid_tickers:
        return ChartPrices(price_data, valid_tickers, realtime_mode, interval, forecast_points)

    # Intraday endpoints are the most fragile, so the fallback keeps the page
    # usable. Labels, forecast dates, and horizons must then match daily bars:
    # keeping minute settings would place daily forecasts minutes apart and
    # misstate the model's horizon.
    bac_log_section(
        "chart_pipeline.prices",
        "Intraday fetch was empty; falling back to daily history.",
    )
    price_data = get_price_history_batch(tickers, period="6mo", interval="1d")
    valid_tickers = [ticker for ticker in tickers if not price_data[ticker].empty]
    bac_log_list_preview("chart_pipeline.prices", "fallback_valid_tickers", valid_tickers)
    return ChartPrices(
        price_data,
        valid_tickers,
        realtime_mode=False,
        interval="1d",
        forecast_points=min(forecast_points, 5),
        daily_fallback=True,
    )


def resolve_matured_forecasts(
    price_data: Mapping[str, pd.DataFrame],
    health: PriceDataHealth,
    monitoring_market: str,
) -> int:
    """Close earlier forecasts whose target bar is now in fresh history.

    Stale snapshots are excluded so an outage cannot create a misleading
    resolution event. This runs before the current forecasts are recorded.
    """
    return resolve_pending_forecasts(
        {ticker: price_data[ticker] for ticker in health.live_tickers},
        monitoring_market,
    )


@dataclass(frozen=True)
class MarketRankingResult:
    """Pooled ranking output; empty when the ranking is unavailable."""

    ranking: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, object] = field(default_factory=dict)
    sentiment_by_ticker: dict[str, pd.DataFrame] = field(default_factory=dict)


def rank_live_candidates(
    price_data: Mapping[str, pd.DataFrame],
    health: PriceDataHealth,
    *,
    forecast_horizon: int,
    monitoring_market: str,
) -> MarketRankingResult:
    """Rank the fresh candidate pool and record the pooled model's diagnostics."""
    # Local SQLite reads are fast; passing every available history makes
    # sentiment part of the ranking as soon as the collector stores articles.
    sentiment_by_ticker = {
        ticker: load_sentiment_history(ticker)
        for ticker in health.live_tickers
    }
    result = rank_market_candidates(
        {ticker: price_data[ticker] for ticker in health.live_tickers},
        forecast_horizon=forecast_horizon,
        sentiment_by_ticker=sentiment_by_ticker,
        top_n=MAX_CHARTED_PERFORMERS,
    )
    ranking = result.get("ranking")
    if not isinstance(ranking, pd.DataFrame):
        ranking = pd.DataFrame()
    raw_diagnostics = result.get("diagnostics")
    diagnostics = dict(raw_diagnostics) if isinstance(raw_diagnostics, Mapping) else {}
    if diagnostics:
        ranking_as_of = max(
            pd.Timestamp(price_data[ticker]["Date"].max())
            for ticker in health.freshness_by_ticker
        )
        record_market_model_run(
            monitoring_market,
            forecast_horizon,
            ranking_as_of,
            diagnostics,
        )
    return MarketRankingResult(ranking, diagnostics, sentiment_by_ticker)


@dataclass(frozen=True)
class ChartSelection:
    """Tickers to chart, in display order, and the headline metric strip."""

    tickers: list[str]
    leader_label: str
    performance_label: str
    performance_value: str


def select_charted_tickers(
    *,
    market_source: MarketSource | None,
    ranking: pd.DataFrame,
    candidate_tickers: list[str],
    valid_tickers: list[str],
    detected_performers: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    realtime_mode: bool,
    interval: str,
) -> ChartSelection:
    """Choose charted tickers from the model ranking, daily moves, or momentum."""
    if market_source is not None and not ranking.empty:
        selection = ChartSelection(
            tickers=ranking["Ticker"].tolist(),
            leader_label="Top predicted ticker",
            performance_label="Expected excess return",
            performance_value=f"{float(ranking['Expected excess return'].iloc[0]):+.2f}%",
        )
    elif market_source is not None:
        # A transparent fallback preserves chart access when the candidate
        # history is too short for the embargoed pooled validation.
        tickers = [
            ticker for ticker in candidate_tickers if ticker in valid_tickers
        ][:MAX_CHARTED_PERFORMERS]
        daily_change_by_ticker = (
            detected_performers.set_index("Ticker")["Daily change"].to_dict()
        )
        leader_change = daily_change_by_ticker.get(tickers[0], np.nan)
        selection = ChartSelection(
            tickers=tickers,
            leader_label=market_source.fallback_leader_label,
            performance_label=market_source.fallback_performance_label,
            performance_value=(
                f"{leader_change:.2f}%" if pd.notna(leader_change) else "Unavailable"
            ),
        )
    else:
        scores = {ticker: growth_score(price_data[ticker]) for ticker in valid_tickers}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        current_momentum_label = momentum_label(realtime_mode, interval)
        selection = ChartSelection(
            tickers=[ticker for ticker, _ in ranked[:MAX_CHARTED_PERFORMERS]],
            leader_label=f"Top {current_momentum_label} mover",
            performance_label=f"Best {current_momentum_label} growth",
            performance_value=f"{ranked[0][1]:.2f}%",
        )

    bac_log_kv(
        "chart_pipeline.selection",
        leader_label=selection.leader_label,
        performance_value=selection.performance_value,
        ranking_available=not ranking.empty,
    )
    bac_log_list_preview("chart_pipeline.selection", "charted_tickers", selection.tickers)
    return selection


@dataclass(frozen=True)
class TickerForecast:
    """Everything the view needs to chart and caption one ticker's forecast."""

    ticker: str
    # Every observed bar, drawn as the solid line.
    history: pd.DataFrame
    # Bars the model was fitted on; may end before `history` in live mode.
    model_history: pd.DataFrame
    # Forecast curve, with 50%/80% band columns when backtests calibrate them.
    forecast: pd.DataFrame
    future_dates: pd.DatetimeIndex
    active_model: str
    comparison: dict | None
    sentiment_history: pd.DataFrame
    using_stale_data: bool
    requested_points: int
    # Set when the curve is missing or shorter than requested.
    diagnosis: dict[str, object] | None = None

    @property
    def status(self) -> str:
        if self.forecast.empty:
            return "unavailable"
        return "partial" if self.diagnosis else "ready"

    def backtest_summary(self) -> dict | None:
        """Return this ticker's backtest-table row, or None without a backtest."""
        if self.comparison is None:
            return None
        summary = dict(self.comparison)
        if not self.forecast.empty:
            final_projection = self.forecast.iloc[-1]
            summary["Projected return"] = float(final_projection["pred_return"] * 100)
            summary["Projected close"] = float(final_projection["pred_close"])
        return summary


def _resolve_sentiment_history(
    ticker: str,
    realtime_mode: bool,
    preloaded: pd.DataFrame | None,
) -> pd.DataFrame:
    # Intraday forecasts are price-only; daily mode reuses ranking inputs.
    if realtime_mode:
        return pd.DataFrame()
    if preloaded is not None:
        return preloaded
    return load_sentiment_history(ticker)


def build_ticker_forecast(
    ticker: str,
    history: pd.DataFrame,
    *,
    ticker_source: str | None,
    realtime_mode: bool,
    interval: str,
    forecast_points: int,
    preloaded_sentiment: pd.DataFrame | None = None,
) -> TickerForecast:
    """Fit, compare, and calibrate one ticker's forecast curve.

    The sentiment-augmented model replaces the price-only curve only when the
    paired walk-forward comparison promotes it.
    """
    model_history = prepare_realtime_forecast_history(history, realtime_mode=realtime_mode)
    using_stale_data = history.attrs.get("bac_data_status") in {
        "last_known_good",
        "provider_stale",
    }
    bac_debug_kv(
        "chart_pipeline.ticker",
        ticker=ticker,
        price_rows=len(history),
        forecast_points=forecast_points,
        data_status=history.attrs.get("bac_data_status", "live"),
        data_fetched_at=history.attrs.get("bac_fetched_at"),
    )

    calendar_name = resolve_market_calendar(ticker_source, ticker)
    price_forecast = forecast_feature_model(
        model_history,
        points_ahead=forecast_points,
        market_calendar=calendar_name,
    )
    price_backtest = backtest_forecast_model(
        model_history,
        forecast_horizon=forecast_points,
        market_calendar=calendar_name,
    )
    sentiment_history = _resolve_sentiment_history(ticker, realtime_mode, preloaded_sentiment)
    sentiment_forecast = pd.DataFrame()
    sentiment_backtest = pd.DataFrame()
    if not sentiment_history.empty:
        sentiment_forecast = forecast_feature_model(
            model_history,
            points_ahead=forecast_points,
            sentiment_history=sentiment_history,
            include_sentiment=True,
            market_calendar=calendar_name,
        )
        sentiment_backtest = backtest_forecast_model(
            model_history,
            forecast_horizon=forecast_points,
            sentiment_history=sentiment_history,
            include_sentiment=True,
            market_calendar=calendar_name,
        )

    comparison = (
        summarize_model_comparison(ticker, price_backtest, sentiment_backtest, forecast_points)
        if not price_backtest.empty
        else None
    )
    sentiment_promoted = bool(
        comparison is not None
        and comparison["Active model"] == SENTIMENT_MODEL
        and not sentiment_forecast.empty
    )
    forecast = add_forecast_intervals(
        sentiment_forecast if sentiment_promoted else price_forecast,
        sentiment_backtest if sentiment_promoted else price_backtest,
        last_close=float(model_history["Close"].iloc[-1]),
    )

    # An empty chart used to look like "forecasting did nothing". Diagnose only
    # after the fast path failed or stopped early, so the view can show why.
    diagnosis = None
    if forecast.empty or len(forecast) < forecast_points:
        diagnosis = diagnose_forecast_readiness(
            model_history,
            forecast_horizon=len(forecast) + 1,
            market_calendar=calendar_name,
        )
    bac_log_kv(
        "chart_pipeline.forecast_status",
        ticker=ticker,
        requested_points=forecast_points,
        returned_points=len(forecast),
        **(diagnosis or {"status": "ready"}),
    )

    future_dates = (
        future_projection_dates(
            model_history["Date"].iloc[-1],
            len(forecast),
            realtime_mode,
            interval,
            market_calendar=calendar_name,
        )
        if not forecast.empty
        else pd.DatetimeIndex([])
    )
    bac_debug_kv(
        "chart_pipeline.ticker",
        ticker=ticker,
        forecast_rows=len(forecast),
        sentiment_articles=len(sentiment_history),
        sentiment_promoted=sentiment_promoted,
    )
    return TickerForecast(
        ticker=ticker,
        history=history,
        model_history=model_history,
        forecast=forecast,
        future_dates=future_dates,
        active_model=SENTIMENT_MODEL if sentiment_promoted else PRICE_ONLY_MODEL,
        comparison=comparison,
        sentiment_history=sentiment_history,
        using_stale_data=using_stale_data,
        requested_points=forecast_points,
        diagnosis=diagnosis,
    )


def record_displayed_forecast(
    forecast: TickerForecast,
    *,
    monitoring_market: str,
    ranking: pd.DataFrame,
) -> None:
    """Persist a displayed forecast so production accuracy can be measured.

    Recovery forecasts built from stale data stay visible but are not recorded
    as new production predictions.
    """
    if forecast.forecast.empty or not len(forecast.future_dates):
        return
    forecast_origin = forecast.model_history["Date"].iloc[-1]
    if forecast.using_stale_data:
        bac_debug_kv(
            "chart_pipeline.ticker",
            ticker=forecast.ticker,
            status="stale_forecast_not_recorded",
            forecast_origin=str(forecast_origin),
        )
        return

    ranking_row = (
        ranking[ranking["Ticker"] == forecast.ticker]
        if "Ticker" in ranking.columns
        else pd.DataFrame()
    )
    sentiment = forecast.sentiment_history
    if not ranking_row.empty:
        monitored_sentiment = float(ranking_row["Sentiment score"].iloc[0])
    elif not sentiment.empty and "sentiment" in sentiment.columns:
        monitored_sentiment = float(
            pd.to_numeric(sentiment["sentiment"], errors="coerce").tail(20).mean()
        )
    else:
        monitored_sentiment = np.nan

    final_projection = forecast.forecast.iloc[-1]
    record_forecast(
        market_source=monitoring_market,
        ticker=forecast.ticker,
        forecast_origin=forecast_origin,
        target_at=forecast.future_dates[-1],
        horizon=int(final_projection["projection_point"]),
        model_name=forecast.active_model,
        regime=volatility_regime(forecast.model_history),
        origin_close=float(forecast.model_history["Close"].iloc[-1]),
        predicted_close=float(final_projection["pred_close"]),
        predicted_return=float(final_projection["pred_return"]),
        lower_80=final_projection.get("lower_80", np.nan),
        upper_80=final_projection.get("upper_80", np.nan),
        sentiment_score=monitored_sentiment,
    )
