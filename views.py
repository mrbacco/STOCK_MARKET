#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: views.py
#############################

"""Streamlit view-rendering helpers.

Each public function in this module renders one high-level page view. The
main app script chooses which view to render based on sidebar state and passes
the already-selected data context into these helpers. Data loading, model
fitting, and monitoring writes for the Charts view live in `chart_pipeline`;
this module only decides how their results are displayed.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_config import (
    BACKTEST_TRAINING_POINTS,
    MAX_BACKTEST_POINTS,
    MAX_CHARTED_PERFORMERS,
    MIN_BACKTEST_POINTS,
    selected_horizon_label,
)
from app_logging import bac_debug_kv, bac_log_kv, bac_log_list_preview, bac_log_section
from cache_control import set_cache_scope
from chart_pipeline import (
    REALTIME_MODEL_REFRESH_FREQUENCY,
    MarketRankingResult,
    TickerForecast,
    build_ticker_forecast,
    load_chart_prices,
    rank_live_candidates,
    record_displayed_forecast,
    resolve_matured_forecasts,
    select_charted_tickers,
)
from market_data import (
    PriceDataHealth,
    classify_price_histories,
    company_names_by_ticker,
    get_price_history_batch,
    growth_score,
    load_news_frames_parallel,
)
from market_sources import MANUAL_CHART_HEADING, MarketSource, get_market_source
from model_monitoring import (
    latest_drift_summary,
    load_forecast_quality,
    load_market_model_history,
)
from runtime_config import ANALYTICS_READ_ONLY, LIVE_CHART_REFRESH_SECONDS


def _leaderboard_column_config(
    performers: pd.DataFrame,
    price_format: str,
) -> dict[str, Any]:
    """Column formats shared by every daily-move leaderboard table."""
    columns: dict[str, Any] = {
        "Daily change": st.column_config.NumberColumn("Daily change", format="%.2f%%"),
        "Last price": st.column_config.NumberColumn("Last price", format=price_format),
    }
    if "Last session" in performers.columns:
        columns["Last session"] = st.column_config.DateColumn("Last session")
    return columns


def _render_latest_bar_quotes(
    price_data: dict[str, pd.DataFrame],
    tickers: List[str],
    interval: str,
    price_prefix: str,
) -> None:
    """Show up to three latest-bar closes with their change versus the prior bar."""
    shown_tickers = tickers[:3]
    if not shown_tickers:
        return
    quote_cols = st.columns(len(shown_tickers))
    for index, ticker in enumerate(shown_tickers):
        price_series = price_data[ticker]["Close"].dropna()
        if len(price_series) < 2:
            bac_debug_kv(
                "views.quote",
                ticker=ticker,
                message="Skipped metric because fewer than two close values were available.",
                close_points=len(price_series),
            )
            continue
        current_price = float(price_series.iloc[-1])
        previous_price = float(price_series.iloc[-2])
        delta_value = current_price - previous_price
        bac_debug_kv(
            "views.quote",
            ticker=ticker,
            current_price=current_price,
            previous_price=previous_price,
            delta_value=delta_value,
        )
        quote_cols[index].metric(
            f"{ticker} latest {interval} close",
            f"{price_prefix}{current_price:.2f}",
            f"{price_prefix}{delta_value:+.2f} vs. prior bar",
        )


def render_overview_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
    realtime_mode: bool,
    period: str,
    interval: str,
    price_prefix: str,
    price_format: str,
) -> None:
    """Render the lightest-weight page, focused on quick market context."""
    bac_log_kv(
        "views.render_overview_view",
        ticker_source=ticker_source,
        realtime_mode=realtime_mode,
        period=period,
        interval=interval,
        ticker_count=len(tickers),
    )
    bac_log_list_preview("views.render_overview_view", "incoming_tickers", tickers)

    st.subheader("Overview")
    st.caption(
        "This view focuses on the current market universe and keeps heavier chart and sentiment work off the page until you need it."
    )

    # Market-sourced views simply show the ranked leaderboard because the source
    # selection itself already decided which universe is relevant.
    market_source = get_market_source(ticker_source)
    if market_source is not None:
        bac_log_section("views.render_overview_view", "Rendering leaderboard-only overview.")
        st.caption(market_source.overview_caption)

        bac_log_kv(
            "views.render_overview_view",
            detected_rows=len(detected_performers),
            detected_columns=list(detected_performers.columns),
        )
        st.dataframe(
            detected_performers,
            column_config=_leaderboard_column_config(detected_performers, price_format),
            hide_index=True,
        )
        return

    # Manual mode uses only the first few tickers so the overview stays compact.
    overview_tickers = tickers[:3]
    bac_log_list_preview("views.render_overview_view", "overview_tickers", overview_tickers)
    if not overview_tickers:
        bac_log_section("views.render_overview_view", "No overview tickers were available.")
        st.warning("Add at least one ticker symbol.")
        return

    with st.spinner("Loading overview snapshot..."):
        price_data = get_price_history_batch(overview_tickers, period=period, interval=interval)

    valid_tickers = [ticker for ticker in overview_tickers if not price_data[ticker].empty]
    bac_log_list_preview("views.render_overview_view", "valid_overview_tickers", valid_tickers)
    if not valid_tickers:
        bac_log_section("views.render_overview_view", "No valid overview price data was returned.")
        st.info("No snapshot data is available yet for the selected tickers.")
        return

    scores = {ticker: growth_score(price_data[ticker]) for ticker in valid_tickers}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_ticker = ranked[0][0]
    top_value = ranked[0][1]
    bac_log_kv(
        "views.render_overview_view",
        top_ticker=top_ticker,
        top_value=top_value,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracked tickers", len(valid_tickers))
    col2.metric("Top mover", top_ticker)
    col3.metric("Best momentum", f"{top_value:.2f}%")

    if realtime_mode:
        bac_log_section("views.render_overview_view", "Rendering intraday quote metrics.")
        _render_latest_bar_quotes(price_data, valid_tickers, interval, price_prefix)

    summary_frame = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Recent momentum": f"{scores[ticker]:.2f}%",
                "Last close": price_data[ticker]["Close"].dropna().iloc[-1],
            }
            for ticker in valid_tickers
        ]
    )
    bac_log_kv("views.render_overview_view", summary_rows=len(summary_frame))
    st.dataframe(summary_frame, hide_index=True)
    bac_log_section("views.render_overview_view", "Overview rendering completed.")


def render_charts_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
    realtime_mode: bool,
    period: str,
    interval: str,
    forecast_points: int,
    price_prefix: str,
    price_axis_label: str,
    price_format: str,
) -> None:
    """Render the heavier charting and forecasting page."""
    bac_log_kv(
        "views.render_charts_view",
        ticker_source=ticker_source,
        realtime_mode=realtime_mode,
        period=period,
        interval=interval,
        forecast_points=forecast_points,
        ticker_count=len(tickers),
    )

    # Automatic daily markets are intentionally loaded as a wider candidate
    # pool.  The pooled model, rather than today's price move, decides which
    # ten tickers deserve charts.  Manual and intraday modes remain bounded
    # because users have already selected/order-ranked those symbols.
    market_source = get_market_source(ticker_source)
    automatic_daily_ranking = market_source is not None and not realtime_mode
    candidate_tickers = tickers if automatic_daily_ranking else tickers[:MAX_CHARTED_PERFORMERS]
    if automatic_daily_ranking:
        st.info(
            f"Evaluating {len(candidate_tickers)} market candidates, then automatically charting the best {MAX_CHARTED_PERFORMERS} forward predictions."
        )
    elif len(tickers) > MAX_CHARTED_PERFORMERS:
        st.info(f"Charting the first {MAX_CHARTED_PERFORMERS} selected symbols.")

    with st.spinner("Loading price history for charting..."):
        prices = load_chart_prices(
            candidate_tickers,
            period=period,
            interval=interval,
            realtime_mode=realtime_mode,
            forecast_points=forecast_points,
        )
    if prices.daily_fallback:
        st.warning("Real-time data is temporarily unavailable. Showing recent daily history instead.")
    if not prices.valid_tickers:
        bac_log_section("views.render_charts_view", "No valid chart data was available.")
        st.error(
            "No live or last-known-good price history is available. Check ticker "
            "symbols, provider connectivity, and [BAC_LOG] market-data entries."
        )
        st.stop()
    # A daily fallback changes the mode, interval, and horizon for every later step.
    realtime_mode = prices.realtime_mode
    interval = prices.interval
    forecast_points = prices.forecast_points
    price_data = prices.price_data

    health = classify_price_histories(
        {ticker: price_data[ticker] for ticker in prices.valid_tickers},
        realtime_mode=realtime_mode,
    )
    _render_data_health(health, price_data)

    monitoring_market = str(ticker_source or "Manual tickers")
    resolved_forecasts = resolve_matured_forecasts(price_data, health, monitoring_market)
    if resolved_forecasts:
        st.toast(f"Resolved {resolved_forecasts} earlier forecast observations.")

    ranking = MarketRankingResult()
    if automatic_daily_ranking and len(health.live_tickers) >= 2:
        with st.spinner(
            "Training the market-wide ensemble and ranking forward opportunities..."
        ):
            ranking = rank_live_candidates(
                price_data,
                health,
                forecast_horizon=forecast_points,
                monitoring_market=monitoring_market,
            )

    selection = select_charted_tickers(
        market_source=market_source,
        ranking=ranking.ranking,
        candidate_tickers=candidate_tickers,
        valid_tickers=prices.valid_tickers,
        detected_performers=detected_performers,
        price_data=price_data,
        realtime_mode=realtime_mode,
        interval=interval,
    )
    if market_source is not None and ranking.ranking.empty and automatic_daily_ranking:
        _render_ranking_pending_notice(ticker_source, period, forecast_points)

    col1, col2, col3 = st.columns(3)
    col1.metric("Charted performers", len(selection.tickers))
    col2.metric(selection.leader_label, selection.tickers[0])
    col3.metric(selection.performance_label, selection.performance_value)

    if market_source is not None:
        _render_market_ranking(
            market_source,
            ranking,
            detected_performers,
            forecast_points,
            price_format,
        )

    if realtime_mode:
        bac_log_section("views.render_charts_view", "Rendering realtime quote metrics.")
        _render_latest_bar_quotes(price_data, selection.tickers, interval, price_prefix)
        st.caption(
            "Intraday figures use the latest returned bar close. The delta is versus the prior bar, not a live tick or daily change."
        )

    horizon_label = selected_horizon_label(realtime_mode, interval, forecast_points)
    if market_source is not None and not ranking.ranking.empty:
        chart_heading = "Predicted top 10 - history, forecast, and uncertainty"
    elif market_source is not None:
        chart_heading = market_source.chart_heading
    else:
        chart_heading = MANUAL_CHART_HEADING
    st.subheader(chart_heading)
    st.caption(
        f"The dashed line is the ticker-level {horizon_label} forecast. Shaded 50% and 80% bands are calibrated from earlier walk-forward return residuals. On daily horizons, point-in-time sentiment is continuously evaluated and only replaces the price-only curve after at least {MIN_BACKTEST_POINTS} paired forecasts improve MAE."
    )

    forecasts: list[TickerForecast] = []
    for ticker in selection.tickers:
        forecast = build_ticker_forecast(
            ticker,
            price_data[ticker],
            ticker_source=ticker_source,
            realtime_mode=realtime_mode,
            interval=interval,
            forecast_points=forecast_points,
            preloaded_sentiment=ranking.sentiment_by_ticker.get(ticker),
        )
        record_displayed_forecast(
            forecast,
            monitoring_market=monitoring_market,
            ranking=ranking.ranking,
        )
        _render_forecast_status_warning(forecast)
        st.plotly_chart(
            _build_forecast_figure(
                forecast,
                realtime_mode=realtime_mode,
                interval=interval,
                price_axis_label=price_axis_label,
            ),
            key=f"price_forecast_chart_{ticker}_{interval}",
        )
        _render_forecast_caption(
            forecast,
            horizon_label=horizon_label,
            price_prefix=price_prefix,
            realtime_mode=realtime_mode,
        )
        forecasts.append(forecast)

    _render_forecast_pipeline_summary(forecasts, health)
    _render_backtest_table(forecasts, horizon_label, forecast_points, price_format)
    _render_model_monitoring(monitoring_market, forecast_points, price_format)
    bac_log_section("views.render_charts_view", "Charts rendering completed.")


def _render_data_health(health: PriceDataHealth, price_data: dict[str, pd.DataFrame]) -> None:
    """Keep operational data quality beside the forecasts it affects.

    A paying user should never have to infer data quality from whether a
    dashed line appeared.
    """
    if health.stale_tickers:
        stale_details = ", ".join(
            (
                f"{ticker} (latest bar "
                f"{health.freshness_by_ticker[ticker].get('latest_bar', 'unknown')}; "
                f"last fetched "
                f"{price_data[ticker].attrs.get('bac_fetched_at', 'unknown')})"
            )
            for ticker in health.stale_tickers
        )
        st.warning(
            "Market-data recovery/staleness mode is active. Forecasts remain "
            f"visible but are not treated as fresh for: {stale_details}."
        )

    latest_market_bar = max(
        (
            diagnosis["latest_bar"]
            for diagnosis in health.freshness_by_ticker.values()
            if diagnosis.get("latest_bar") is not None
        ),
        default=None,
    )
    data_health_columns = st.columns(4)
    data_health_columns[0].metric(
        "Data pipeline",
        "Healthy" if not health.stale_tickers else "Recovery mode",
    )
    data_health_columns[1].metric("Fresh histories", len(health.live_tickers))
    data_health_columns[2].metric("Stale/recovered", len(health.stale_tickers))
    data_health_columns[3].metric(
        "Latest market bar",
        (
            pd.Timestamp(latest_market_bar).strftime("%Y-%m-%d %H:%M")
            if latest_market_bar is not None
            else "Unavailable"
        ),
    )


def _render_ranking_pending_notice(
    ticker_source: str | None,
    period: str,
    forecast_points: int,
) -> None:
    """Explain why an automatic market shows its daily ordering instead of ranks."""
    if ANALYTICS_READ_ONLY:
        st.info(
            "The analytics worker is preparing this market, period, and horizon. "
            "Showing the current daily ordering until its shared result is ready."
        )
        bac_log_kv(
            "views.render_charts_view",
            status="analytics_worker_pending",
            ticker_source=ticker_source,
            period=period,
            forecast_points=forecast_points,
        )
    else:
        st.warning(
            "The market-wide model does not yet have enough embargoed history; using the current daily ordering temporarily."
        )


def _render_market_ranking(
    market_source: MarketSource,
    ranking: MarketRankingResult,
    detected_performers: pd.DataFrame,
    forecast_points: int,
    price_format: str,
) -> None:
    """Show the pooled-model ranking (or its fallback) and the candidate pool."""
    bac_log_section("views.render_charts_view", "Rendering automatic market ranking.")
    if ranking.ranking.empty:
        st.subheader(market_source.fallback_heading)
        st.caption(
            "The pooled ranking is temporarily unavailable, so this table shows the current daily-move candidates."
        )
    else:
        st.subheader("Model-ranked top 10 forward opportunities")
        st.caption(
            f"These are the ten strongest {forecast_points}-session market-relative forecasts from the full loaded candidate pool. The score blends expected excess return, calibrated probability, model agreement, market context, liquidity, and continuously collected sentiment. 'Abstain' means the point estimate is not strong enough relative to uncertainty."
        )
        company_lookup = (
            detected_performers[["Ticker", "Company"]].drop_duplicates("Ticker")
            if "Company" in detected_performers.columns
            else pd.DataFrame(columns=["Ticker", "Company"])
        )
        ranking_display = ranking.ranking.merge(
            company_lookup,
            on="Ticker",
            how="left",
            validate="one_to_one",
        )
        ranking_display = ranking_display[
            [
                "Rank",
                "Ticker",
                "Company",
                "Signal",
                "Expected excess return",
                "Probability outperform",
                "Lower 80",
                "Upper 80",
                "Predicted volatility",
                "Model disagreement",
                "Sentiment score",
            ]
        ]
        st.dataframe(
            ranking_display,
            column_config={
                "Expected excess return": st.column_config.NumberColumn(
                    "Expected excess return", format="%+.2f%%"
                ),
                "Probability outperform": st.column_config.ProgressColumn(
                    "Probability outperform", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Lower 80": st.column_config.NumberColumn("80% lower", format="%+.2f%%"),
                "Upper 80": st.column_config.NumberColumn("80% upper", format="%+.2f%%"),
                "Predicted volatility": st.column_config.NumberColumn(
                    "Predicted volatility", format="%.2f%%"
                ),
                "Model disagreement": st.column_config.NumberColumn(
                    "Model disagreement", format="%.2f%%"
                ),
                "Sentiment score": st.column_config.NumberColumn(
                    "24h sentiment", format="%+.3f"
                ),
            },
            hide_index=True,
            width="stretch",
        )

        # A compact metric strip surfaces the untouched evaluation period,
        # including a direct backtest of selecting the ten best each date.
        diagnostics = ranking.diagnostics
        diagnostic_columns = st.columns(5)
        diagnostic_columns[0].metric(
            "Evaluation direction",
            f"{float(diagnostics.get('Directional accuracy', np.nan)):.1f}%",
        )
        diagnostic_columns[1].metric(
            "80% band coverage",
            f"{float(diagnostics.get('80% interval coverage', np.nan)):.1f}%",
        )
        diagnostic_columns[2].metric(
            "Excess-return MAE",
            f"{float(diagnostics.get('Evaluation MAE', np.nan)):.2f}%",
        )
        diagnostic_columns[3].metric(
            "Top-10 realized excess",
            f"{float(diagnostics.get('Top-10 realized mean excess', np.nan)):+.2f}%",
        )
        diagnostic_columns[4].metric(
            "Top-10 realized hit rate",
            f"{float(diagnostics.get('Top-10 realized hit rate', np.nan)):.1f}%",
        )
        with st.expander("Model validation and weighting details"):
            st.json(diagnostics)

    # Keep source selection visible without confusing it with the predictive
    # ranking.  Users can inspect the underlying movers in a collapsed area.
    with st.expander("Current daily-move candidate pool"):
        st.dataframe(
            detected_performers,
            column_config=_leaderboard_column_config(detected_performers, price_format),
            hide_index=True,
            width="stretch",
        )


def _render_forecast_status_warning(forecast: TickerForecast) -> None:
    """Tell the user exactly why a curve is missing or shorter than requested."""
    if forecast.diagnosis is None:
        return
    message = forecast.diagnosis.get("message", "")
    if forecast.status == "unavailable":
        st.warning(f"{forecast.ticker}: forecast unavailable. {message}")
    elif forecast.status == "partial":
        st.warning(
            f"{forecast.ticker}: showing {len(forecast.forecast)} of "
            f"{forecast.requested_points} requested forecast points. {message}"
        )


def _add_band(
    fig: go.Figure,
    forecast: TickerForecast,
    level: int,
    fillcolor: str,
) -> None:
    """Shade one uncertainty band between its upper and lower curves."""
    lower, upper = f"lower_{level}", f"upper_{level}"
    if not {lower, upper}.issubset(forecast.forecast.columns):
        return
    legend_group = f"{forecast.ticker}-{level}-band"
    fig.add_trace(
        go.Scatter(
            x=forecast.future_dates,
            y=forecast.forecast[upper],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=legend_group,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.future_dates,
            y=forecast.forecast[lower],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fillcolor,
            name=f"{forecast.ticker} {level}% interval",
            legendgroup=legend_group,
        )
    )


def _build_forecast_figure(
    forecast: TickerForecast,
    *,
    realtime_mode: bool,
    interval: str,
    price_axis_label: str,
) -> go.Figure:
    """Draw observed history, the latest bar, uncertainty bands, and the curve."""
    ticker = forecast.ticker
    history = forecast.history
    fig = go.Figure()
    # The solid line anchors the user in observed history before the forecast starts.
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Close"],
            mode="lines",
            name=f"{ticker} Close",
            line={"width": 2},
        )
    )
    # A separate marker makes the most recent observed point visually obvious.
    fig.add_trace(
        go.Scatter(
            x=[history["Date"].iloc[-1]],
            y=[history["Close"].iloc[-1]],
            mode="markers",
            name=f"{ticker} Latest",
            marker={"size": 10},
        )
    )
    if not forecast.forecast.empty:
        # The forecast origin is the last stable model bar. The observed line
        # may already extend a few minutes beyond it because live prices redraw
        # more frequently than the forecast is recomputed. Bands are drawn from
        # widest to narrowest so both remain visible under the central line.
        _add_band(fig, forecast, 80, "rgba(99, 110, 250, 0.12)")
        _add_band(fig, forecast, 50, "rgba(99, 110, 250, 0.24)")
        fig.add_trace(
            go.Scatter(
                x=forecast.future_dates,
                y=forecast.forecast["pred_close"],
                mode="lines",
                name=f"{ticker} {forecast.active_model} forecast",
                line={"dash": "dash", "width": 2},
            )
        )

    fig.update_layout(
        title=f"{ticker}: Price History & Feature Forecast",
        xaxis_title="Timestamp" if realtime_mode else "Date",
        yaxis_title=price_axis_label,
        template="plotly_white",
        height=420,
        # A stable uirevision tells Plotly to retain zoom, pan, and legend
        # choices when Streamlit replaces the figure with fresh live data.
        uirevision=f"{ticker}:{interval}:{'live' if realtime_mode else 'daily'}",
    )
    return fig


def _render_forecast_caption(
    forecast: TickerForecast,
    *,
    horizon_label: str,
    price_prefix: str,
    realtime_mode: bool,
) -> None:
    """Connect the chart to its backtest scorecard in one sentence."""
    if forecast.forecast.empty:
        return
    final_projection = forecast.forecast.iloc[-1]
    projected_return_pct = float(final_projection["pred_return"] * 100)
    projected_close = float(final_projection["pred_close"])
    confidence = "Unavailable"
    directional_accuracy = np.nan
    mae_improvement = np.nan
    sentiment_status = "Intraday model is price-only" if realtime_mode else "Collecting history"
    if forecast.comparison is not None:
        confidence = forecast.comparison["Confidence"]
        directional_accuracy = float(forecast.comparison["Directional accuracy"])
        mae_improvement = float(forecast.comparison["MAE improvement vs. no-change"])
        sentiment_status = str(forecast.comparison["Sentiment status"])

    accuracy_text = (
        f"{directional_accuracy:.1f}% directional accuracy"
        if pd.notna(directional_accuracy)
        else "directional accuracy unavailable"
    )
    improvement_text = (
        f"{mae_improvement:.1f}% vs. baseline"
        if pd.notna(mae_improvement)
        else "MAE comparison unavailable"
    )
    data_label = (
        "RECOVERY FORECAST using last-known-good market data"
        if forecast.using_stale_data
        else "current forecast"
    )
    bac_debug_kv(
        "views.forecast_caption",
        ticker=forecast.ticker,
        projected_return_pct=projected_return_pct,
        projected_close=projected_close,
        confidence=confidence,
    )
    st.caption(
        f"{forecast.ticker}: {data_label}; {horizon_label} {forecast.active_model.lower()} "
        f"projection {projected_return_pct:+.2f}% to "
        f"{price_prefix}{projected_close:.2f}. Backtest rating: "
        f"{confidence}. Recent backtest: {accuracy_text}, "
        f"{improvement_text}. Sentiment: {sentiment_status}."
    )


def _render_forecast_pipeline_summary(
    forecasts: list[TickerForecast],
    health: PriceDataHealth,
) -> None:
    """Summarize how many charted tickers produced a forecast curve."""
    successes = sum(1 for forecast in forecasts if not forecast.forecast.empty)
    bac_log_kv(
        "views.render_charts_view.forecast_summary",
        requested_tickers=len(forecasts),
        successful_tickers=successes,
        failed_or_partial_tickers=sum(1 for forecast in forecasts if forecast.diagnosis),
        live_data_tickers=len(health.live_tickers),
        stale_data_tickers=len(health.stale_tickers),
    )
    if successes == len(forecasts):
        st.caption(
            f"Forecast pipeline ready: {successes}/{len(forecasts)} "
            "ticker curves generated."
        )
    elif successes:
        st.warning(
            f"Forecast pipeline partially available: {successes}/"
            f"{len(forecasts)} ticker curves generated. See ticker warnings above."
        )
    else:
        st.error(
            "Forecast pipeline unavailable for this selection. The ticker warnings "
            "above and [BAC_LOG] entries contain the exact failure reasons."
        )


def _render_backtest_table(
    forecasts: list[TickerForecast],
    horizon_label: str,
    forecast_points: int,
    price_format: str,
) -> None:
    """Show walk-forward backtest quality for every charted ticker."""
    st.subheader("Forecast backtest")
    st.caption(
        f"Walk-forward test of up to {MAX_BACKTEST_POINTS} unseen {horizon_label} forecasts. Each forecast is trained only on the preceding {BACKTEST_TRAINING_POINTS} observations and compared with a no-change baseline."
    )

    backtest_rows = [
        summary
        for summary in (forecast.backtest_summary() for forecast in forecasts)
        if summary is not None
    ]
    if not backtest_rows:
        bac_log_section("views.render_charts_view", "No backtest summary rows were available.")
        if ANALYTICS_READ_ONLY:
            st.info(
                "Backtests for this selection are not in the shared analytics cache yet. "
                "The worker will prepare them without blocking this browser session."
            )
        else:
            st.info(
                "Not enough price observations to backtest this forecast model. "
                f"At least {BACKTEST_TRAINING_POINTS + forecast_points + MIN_BACKTEST_POINTS - 1} observations are required."
            )
        return

    backtest_frame = pd.DataFrame(backtest_rows)
    bac_log_kv("views.render_charts_view", backtest_summary_rows=len(backtest_frame))
    st.dataframe(
        backtest_frame,
        column_config={
            "Projected return": st.column_config.NumberColumn(
                "Projected return", format="%.2f%%"
            ),
            "Projected close": st.column_config.NumberColumn(
                "Projected close", format=price_format
            ),
            "Model MAE": st.column_config.NumberColumn("Model MAE", format=price_format),
            "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
            "Directional accuracy": st.column_config.NumberColumn(
                "Directional accuracy", format="%.1f%%"
            ),
            "No-change MAE": st.column_config.NumberColumn("No-change MAE", format=price_format),
            "MAE improvement vs. no-change": st.column_config.NumberColumn(
                "MAE improvement vs. no-change", format="%.1f%%"
            ),
            "Price-only MAE": st.column_config.NumberColumn(
                "Price-only MAE", format=price_format
            ),
            "Sentiment MAE": st.column_config.NumberColumn(
                "Sentiment MAE", format=price_format
            ),
            "Price-only directional accuracy": st.column_config.NumberColumn(
                "Price-only directional accuracy", format="%.1f%%"
            ),
            "Sentiment directional accuracy": st.column_config.NumberColumn(
                "Sentiment directional accuracy", format="%.1f%%"
            ),
            "Sentiment MAE lift vs. price-only": st.column_config.NumberColumn(
                "Sentiment MAE lift vs. price-only", format="%.1f%%"
            ),
        },
        hide_index=True,
    )
    st.caption(
        "Positive sentiment MAE lift means the augmented model beat the otherwise identical price-only model. Sentiment remains under evaluation until enough point-in-time history exists and is promoted only when that lift is positive."
    )


def _render_model_monitoring(
    monitoring_market: str,
    forecast_points: int,
    price_format: str,
) -> None:
    """Show resolved production-forecast quality and pooled-model drift."""
    st.subheader("Production model monitoring")
    st.caption(
        "Displayed forecasts are stored locally and resolved automatically once their target session arrives. The table is grouped by model, selected horizon, and the volatility regime present at forecast time."
    )
    forecast_quality = load_forecast_quality(
        monitoring_market,
        horizon=forecast_points,
    )
    if forecast_quality.empty:
        st.info(
            "No production forecasts have reached this target horizon yet. Monitoring has started and will populate on future reruns."
        )
    else:
        st.dataframe(
            forecast_quality,
            column_config={
                "MAE": st.column_config.NumberColumn("Close MAE", format=price_format),
                "Return MAE": st.column_config.NumberColumn("Return MAE", format="%.2f%%"),
                "Directional accuracy": st.column_config.NumberColumn(
                    "Directional accuracy", format="%.1f%%"
                ),
                "80% interval coverage": st.column_config.NumberColumn(
                    "80% interval coverage", format="%.1f%%"
                ),
                "Last resolved": st.column_config.DatetimeColumn("Last resolved"),
            },
            hide_index=True,
            width="stretch",
        )

    model_history = load_market_model_history(
        monitoring_market,
        horizon=forecast_points,
    )
    if model_history.empty:
        return
    drift = latest_drift_summary(model_history)
    if drift:
        drift_columns = st.columns(4)
        drift_columns[0].metric(
            "MAE drift",
            f"{drift['MAE drift']:+.2f} pp",
            delta_color="inverse",
        )
        drift_columns[1].metric(
            "Direction drift",
            f"{drift['Direction drift']:+.1f} pp",
        )
        drift_columns[2].metric(
            "Brier drift",
            f"{drift['Brier drift']:+.3f}",
            delta_color="inverse",
        )
        drift_columns[3].metric(
            "Coverage drift",
            f"{drift['Coverage drift']:+.1f} pp",
        )
    with st.expander("Stored market-model run history"):
        model_history_display = model_history.rename(
            columns={
                "as_of": "As of",
                "candidate_tickers": "Candidates",
                "evaluation_dates": "Evaluation dates",
                "evaluation_mae": "Evaluation MAE",
                "baseline_mae": "Baseline MAE",
                "directional_accuracy": "Directional accuracy",
                "probability_brier": "Probability Brier",
                "interval_coverage_80": "80% interval coverage",
                "selection_mean_excess": "Top-10 mean excess",
                "selection_hit_rate": "Top-10 hit rate",
                "sentiment_observed_rows": "Sentiment rows",
            }
        )
        st.dataframe(
            model_history_display.drop(columns=["model_weights_json"]),
            hide_index=True,
            width="stretch",
        )


@st.fragment(run_every=f"{LIVE_CHART_REFRESH_SECONDS}s")
def render_live_charts_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
    period: str,
    interval: str,
    forecast_points: int,
    price_prefix: str,
    price_axis_label: str,
    price_format: str,
    cache_scope: str,
) -> None:
    """Auto-refresh only the real-time chart section on a bounded cadence.

    Streamlit fragments leave the sidebar, news collector status, and the rest
    of the page untouched. Keeping this wrapper separate also lets the user
    disable live updates and fall back to a normal one-shot chart render.
    """
    # Fragment reruns may execute in a fresh script context, so restore the
    # market-specific cache namespace before any data or model lookup.
    set_cache_scope(cache_scope)
    refresh_started_at = pd.Timestamp.now(tz="UTC")
    bac_log_kv(
        "views.live_charts.refresh",
        status="started",
        ticker_count=len(tickers),
        period=period,
        interval=interval,
        refresh_seconds=LIVE_CHART_REFRESH_SECONDS,
        cache_scope=cache_scope,
        refreshed_at=refresh_started_at.isoformat(),
    )
    st.caption(
        ":green[● Live updates on] · Free Yahoo polling every "
        f"{LIVE_CHART_REFRESH_SECONDS} seconds · Forecast model refreshes from "
        f"completed {REALTIME_MODEL_REFRESH_FREQUENCY} buckets · "
        f"Last refresh started {refresh_started_at.strftime('%H:%M:%S UTC')}"
    )
    render_charts_view(
        ticker_source=ticker_source,
        tickers=tickers,
        detected_performers=detected_performers,
        realtime_mode=True,
        period=period,
        interval=interval,
        forecast_points=forecast_points,
        price_prefix=price_prefix,
        price_axis_label=price_axis_label,
        price_format=price_format,
    )
    bac_log_kv(
        "views.live_charts.refresh",
        status="completed",
        ticker_count=len(tickers),
        refreshed_at=pd.Timestamp.now(tz="UTC").isoformat(),
    )


def render_news_view(
    ticker_source: str | None,
    tickers: List[str],
    detected_performers: pd.DataFrame,
) -> None:
    """Render the headline and sentiment page."""
    bac_log_kv(
        "views.render_news_view",
        ticker_source=ticker_source,
        ticker_count=len(tickers),
        detected_rows=len(detected_performers),
    )
    bac_log_list_preview("views.render_news_view", "incoming_tickers", tickers)

    st.subheader("Investing news and sentiment")

    news_tickers = tickers[:MAX_CHARTED_PERFORMERS]
    company_by_ticker = company_names_by_ticker(detected_performers, news_tickers)
    bac_log_list_preview("views.render_news_view", "news_tickers", news_tickers)

    with st.spinner("Fetching news and sentiment..."):
        news_frames = load_news_frames_parallel(news_tickers, company_by_ticker)

    if news_frames:
        all_news = pd.concat(news_frames, ignore_index=True)
        bac_log_kv("views.render_news_view", combined_news_rows=len(all_news))

        sentiment_by_ticker = (
            all_news.groupby("ticker", as_index=False)
            .agg(sentiment=("sentiment", "mean"))
            .sort_values(by="sentiment", ascending=False)
        )
        bac_log_kv("views.render_news_view", sentiment_rows=len(sentiment_by_ticker))

        bar = go.Figure(
            data=[
                go.Bar(
                    x=sentiment_by_ticker["ticker"],
                    y=sentiment_by_ticker["sentiment"],
                    marker_color=[
                        "#2ca02c" if score > 0.05 else "#d62728" if score < -0.05 else "#7f7f7f"
                        for score in sentiment_by_ticker["sentiment"]
                    ],
                    name="Average Sentiment",
                )
            ]
        )
        bar.update_layout(
            title="Average news sentiment by ticker",
            xaxis_title="Ticker",
            yaxis_title="Financial sentiment score",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(bar)

        news_table = (
            all_news[
                [
                    "ticker",
                    "published",
                    "first_seen_at",
                    "source",
                    "title",
                    "sentiment_label",
                    "sentiment",
                    "positive_probability",
                    "neutral_probability",
                    "negative_probability",
                    "model_name",
                    "link",
                ]
            ]
            .sort_values(by="published", ascending=False)
            .reset_index(drop=True)
        )
        bac_log_kv("views.render_news_view", news_table_rows=len(news_table))
        st.dataframe(news_table)
        bac_log_section("views.render_news_view", "News rendering completed with rows.")
    else:
        bac_log_section("views.render_news_view", "No news rows were available.")
        st.info("No news items were fetched right now. Try again in a moment.")
