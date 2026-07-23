#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: views.py
#############################

"""Streamlit view-rendering helpers.

Each function in this module is responsible for one high-level page view. The
main app script chooses which view to render based on sidebar state and passes
the already-selected data context into these helpers.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_config import (
    BACKTEST_TRAINING_POINTS,
    FTSE_MIB_SOURCE,
    IRELAND_SOURCE,
    MARKET_SOURCES,
    MAX_BACKTEST_POINTS,
    MAX_CHARTED_PERFORMERS,
    MIN_BACKTEST_POINTS,
    US_SOURCE,
    resolve_market_calendar,
    selected_horizon_label,
)
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from forecasting import (
    add_forecast_intervals,
    backtest_forecast_model,
    diagnose_forecast_readiness,
    forecast_feature_model,
    future_projection_dates,
    summarize_model_comparison,
)
from market_model import rank_market_candidates
from market_data import (
    get_price_history_batch,
    growth_score,
    load_news_frames_parallel,
    momentum_label,
)
from model_monitoring import (
    latest_drift_summary,
    load_forecast_quality,
    load_market_model_history,
    record_forecast,
    record_market_model_run,
    resolve_pending_forecasts,
)
from sentiment_store import load_sentiment_history
from runtime_config import ANALYTICS_READ_ONLY


def _volatility_regime(price_history: pd.DataFrame) -> str:
    """Classify the latest realized volatility relative to the ticker's history."""
    close = pd.to_numeric(price_history.get("Close"), errors="coerce")
    rolling_volatility = np.log(close).diff().rolling(20, min_periods=10).std().dropna()
    if rolling_volatility.empty:
        return "Unknown"
    latest = float(rolling_volatility.iloc[-1])
    lower_quartile = float(rolling_volatility.quantile(0.25))
    upper_quartile = float(rolling_volatility.quantile(0.75))
    regime = "High volatility" if latest >= upper_quartile else "Low volatility" if latest <= lower_quartile else "Normal volatility"
    bac_log_kv(
        "views.volatility_regime",
        latest=latest,
        lower_quartile=lower_quartile,
        upper_quartile=upper_quartile,
        regime=regime,
    )
    return regime


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
    if ticker_source in MARKET_SOURCES:
        bac_log_section("views.render_overview_view", "Rendering leaderboard-only overview.")
        if ticker_source == IRELAND_SOURCE:
            st.caption(
                "The leaderboard ranks the latest available daily close from the tracked ISEQ 20 Euronext Dublin universe."
            )
        elif ticker_source == FTSE_MIB_SOURCE:
            st.caption(
                "The leaderboard ranks the latest available daily close across Yahoo-supported FTSE MIB constituents."
            )
        else:
            st.caption(
                "The leaderboard uses Yahoo Finance's predefined U.S. equity filter for liquid daily gainers."
            )

        leaderboard_columns = {
            "Daily change": st.column_config.NumberColumn("Daily change", format="%.2f%%"),
            "Last price": st.column_config.NumberColumn("Last price", format=price_format),
        }
        if "Last session" in detected_performers.columns:
            leaderboard_columns["Last session"] = st.column_config.DateColumn("Last session")

        bac_log_kv(
            "views.render_overview_view",
            detected_rows=len(detected_performers),
            detected_columns=list(detected_performers.columns),
        )
        st.dataframe(detected_performers, column_config=leaderboard_columns, hide_index=True)
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
        quote_cols = st.columns(min(3, len(valid_tickers)))
        for index, ticker in enumerate(valid_tickers[:3]):
            price_series = price_data[ticker]["Close"].dropna()
            if len(price_series) >= 2:
                current_price = float(price_series.iloc[-1])
                previous_price = float(price_series.iloc[-2])
                delta_value = current_price - previous_price
                bac_log_kv(
                    "views.render_overview_view.quote",
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
            else:
                bac_log_kv(
                    "views.render_overview_view.quote",
                    ticker=ticker,
                    message="Skipped metric because fewer than two close values were available.",
                    close_points=len(price_series),
                )

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
    bac_log_list_preview("views.render_charts_view", "incoming_tickers", tickers)

    # Automatic daily markets are intentionally loaded as a wider candidate
    # pool.  The pooled model below, rather than today's price move, decides
    # which ten tickers deserve charts.  Manual and intraday modes remain bounded
    # because users have already selected/order-ranked those symbols.
    automatic_daily_ranking = ticker_source in MARKET_SOURCES and not realtime_mode
    active_tickers = tickers if automatic_daily_ranking else tickers[:MAX_CHARTED_PERFORMERS]
    if automatic_daily_ranking:
        st.info(
            f"Evaluating {len(active_tickers)} market candidates, then automatically charting the best {MAX_CHARTED_PERFORMERS} forward predictions."
        )
    elif len(tickers) > MAX_CHARTED_PERFORMERS:
        st.info(f"Charting the first {MAX_CHARTED_PERFORMERS} selected symbols.")
    bac_log_kv(
        "views.render_charts_view",
        automatic_daily_ranking=automatic_daily_ranking,
        candidate_count=len(active_tickers),
        chart_limit=MAX_CHARTED_PERFORMERS,
    )

    with st.spinner("Loading price history for charting..."):
        price_data = get_price_history_batch(active_tickers, period=period, interval=interval)

    valid_tickers = [ticker for ticker in active_tickers if not price_data[ticker].empty]
    bac_log_list_preview("views.render_charts_view", "valid_tickers", valid_tickers)

    # Intraday endpoints are the most fragile, so the fallback keeps the page usable.
    if realtime_mode and not valid_tickers:
        bac_log_section(
            "views.render_charts_view",
            "Intraday fetch was empty; falling back to daily history.",
        )
        st.warning("Real-time data is temporarily unavailable. Showing recent daily history instead.")
        price_data = get_price_history_batch(active_tickers, period="6mo", interval="1d")
        valid_tickers = [ticker for ticker in active_tickers if not price_data[ticker].empty]
        # Downstream labels, forecast dates, and horizons must match the daily
        # fallback.  Keeping the original minute settings would place daily
        # forecasts only minutes apart and misstate the model's horizon.
        realtime_mode = False
        interval = "1d"
        forecast_points = min(forecast_points, 5)
        bac_log_list_preview("views.render_charts_view", "fallback_valid_tickers", valid_tickers)

    if not valid_tickers:
        bac_log_section("views.render_charts_view", "No valid chart data was available.")
        st.error("No price data was returned. Check ticker symbols and try again.")
        st.stop()

    monitoring_market = str(ticker_source or "Manual tickers")
    # Resolve old forecasts before writing the current run.  Only target bars
    # already present in the freshly loaded history can close an observation.
    resolved_forecasts = resolve_pending_forecasts(
        {ticker: price_data[ticker] for ticker in valid_tickers},
        monitoring_market,
    )
    if resolved_forecasts:
        st.toast(f"Resolved {resolved_forecasts} earlier forecast observations.")

    sentiment_by_ticker: dict[str, pd.DataFrame] = {}
    market_ranking: pd.DataFrame = pd.DataFrame()
    market_diagnostics: dict[str, object] = {}
    if automatic_daily_ranking and len(valid_tickers) >= 2:
        # SQLite reads are local and fast; passing all available histories makes
        # sentiment part of the ranking continuously as the collector adds data.
        sentiment_by_ticker = {
            ticker: load_sentiment_history(ticker)
            for ticker in valid_tickers
        }
        usable_price_data = {ticker: price_data[ticker] for ticker in valid_tickers}
        with st.spinner(
            "Training the market-wide ensemble and ranking forward opportunities..."
        ):
            ranking_result = rank_market_candidates(
                usable_price_data,
                forecast_horizon=forecast_points,
                sentiment_by_ticker=sentiment_by_ticker,
                top_n=MAX_CHARTED_PERFORMERS,
            )
        market_ranking = ranking_result.get("ranking", pd.DataFrame())
        market_diagnostics = ranking_result.get("diagnostics", {})
        if market_diagnostics:
            ranking_as_of = max(
                pd.Timestamp(price_data[ticker]["Date"].max())
                for ticker in valid_tickers
            )
            record_market_model_run(
                monitoring_market,
                forecast_points,
                ranking_as_of,
                market_diagnostics,
            )

    if ticker_source in MARKET_SOURCES:
        if not market_ranking.empty:
            top_performers = market_ranking["Ticker"].tolist()
            leader_label = "Top predicted ticker"
            performance_label = "Expected excess return"
            performance_value = f"{float(market_ranking['Expected excess return'].iloc[0]):+.2f}%"
        else:
            # A transparent fallback preserves chart access when the candidate
            # history is too short for the embargoed pooled validation.
            top_performers = [
                ticker for ticker in active_tickers if ticker in valid_tickers
            ][:MAX_CHARTED_PERFORMERS]
            daily_change_by_ticker = (
                detected_performers.set_index("Ticker")["Daily change"].to_dict()
            )
            if ticker_source == IRELAND_SOURCE:
                leader_label = "Top ISEQ 20 daily mover"
                performance_label = "Best ISEQ 20 daily change"
            elif ticker_source == FTSE_MIB_SOURCE:
                leader_label = "Top FTSE MIB daily mover"
                performance_label = "Best FTSE MIB daily change"
            else:
                leader_label = "Top detected daily gainer"
                performance_label = "Best detected daily change"
            leader_change = daily_change_by_ticker.get(top_performers[0], np.nan)
            performance_value = (
                f"{leader_change:.2f}%" if pd.notna(leader_change) else "Unavailable"
            )
            if automatic_daily_ranking:
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
        bac_log_kv(
            "views.render_charts_view",
            leader_label=leader_label,
            performance_value=performance_value,
            ranking_available=not market_ranking.empty,
        )
    else:
        scores = {ticker: growth_score(price_data[ticker]) for ticker in valid_tickers}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_performers = [ticker for ticker, _ in ranked[:MAX_CHARTED_PERFORMERS]]
        current_momentum_label = momentum_label(realtime_mode, interval)
        leader_label = f"Top {current_momentum_label} mover"
        performance_label = f"Best {current_momentum_label} growth"
        performance_value = f"{ranked[0][1]:.2f}%"
        bac_log_kv(
            "views.render_charts_view",
            leader_label=leader_label,
            performance_value=performance_value,
            top_ranked=ranked[:3],
        )

    bac_log_list_preview("views.render_charts_view", "top_performers", top_performers)
    col1, col2, col3 = st.columns(3)
    col1.metric("Charted performers", len(top_performers))
    col2.metric(leader_label, top_performers[0])
    col3.metric(performance_label, performance_value)

    if ticker_source in MARKET_SOURCES:
        bac_log_section("views.render_charts_view", "Rendering automatic market ranking.")
        if not market_ranking.empty:
            st.subheader("Model-ranked top 10 forward opportunities")
            st.caption(
                f"These are the ten strongest {forecast_points}-session market-relative forecasts from the full loaded candidate pool. The score blends expected excess return, calibrated probability, model agreement, market context, liquidity, and continuously collected sentiment. 'Abstain' means the point estimate is not strong enough relative to uncertainty."
            )
            company_lookup = (
                detected_performers[["Ticker", "Company"]].drop_duplicates("Ticker")
                if "Company" in detected_performers.columns
                else pd.DataFrame(columns=["Ticker", "Company"])
            )
            ranking_display = market_ranking.merge(
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
            diagnostic_columns = st.columns(5)
            diagnostic_columns[0].metric(
                "Evaluation direction",
                f"{float(market_diagnostics.get('Directional accuracy', np.nan)):.1f}%",
            )
            diagnostic_columns[1].metric(
                "80% band coverage",
                f"{float(market_diagnostics.get('80% interval coverage', np.nan)):.1f}%",
            )
            diagnostic_columns[2].metric(
                "Excess-return MAE",
                f"{float(market_diagnostics.get('Evaluation MAE', np.nan)):.2f}%",
            )
            diagnostic_columns[3].metric(
                "Top-10 realized excess",
                f"{float(market_diagnostics.get('Top-10 realized mean excess', np.nan)):+.2f}%",
            )
            diagnostic_columns[4].metric(
                "Top-10 realized hit rate",
                f"{float(market_diagnostics.get('Top-10 realized hit rate', np.nan)):.1f}%",
            )
            with st.expander("Model validation and weighting details"):
                st.json(market_diagnostics)
        else:
            fallback_heading = (
                "ISEQ 20 top daily performers"
                if ticker_source == IRELAND_SOURCE
                else "FTSE MIB top 10 daily performers"
                if ticker_source == FTSE_MIB_SOURCE
                else "Detected top 10 daily gainers"
            )
            st.subheader(fallback_heading)
            st.caption(
                "The pooled ranking is temporarily unavailable, so this table shows the current daily-move candidates."
            )

        # Keep source selection visible without confusing it with the predictive
        # ranking.  Users can inspect the underlying movers in a collapsed area.
        with st.expander("Current daily-move candidate pool"):
            leaderboard_columns = {
                "Daily change": st.column_config.NumberColumn(
                    "Daily change", format="%.2f%%"
                ),
                "Last price": st.column_config.NumberColumn(
                    "Last price", format=price_format
                ),
            }
            if "Last session" in detected_performers.columns:
                leaderboard_columns["Last session"] = st.column_config.DateColumn(
                    "Last session"
                )
            st.dataframe(
                detected_performers,
                column_config=leaderboard_columns,
                hide_index=True,
                width="stretch",
            )

    if realtime_mode:
        bac_log_section("views.render_charts_view", "Rendering realtime quote metrics.")
        quote_cols = st.columns(min(3, len(top_performers)))
        for index, ticker in enumerate(top_performers[:3]):
            price_series = price_data[ticker]["Close"].dropna()
            if len(price_series) >= 2:
                current_price = float(price_series.iloc[-1])
                previous_price = float(price_series.iloc[-2])
                delta_value = current_price - previous_price
                bac_log_kv(
                    "views.render_charts_view.quote",
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
            else:
                bac_log_kv(
                    "views.render_charts_view.quote",
                    ticker=ticker,
                    message="Skipped realtime metric because fewer than two close values were available.",
                    close_points=len(price_series),
                )
        st.caption(
            "Intraday figures use the latest returned bar close. The delta is versus the prior bar, not a live tick or daily change."
        )

    horizon_label = selected_horizon_label(realtime_mode, interval, forecast_points)
    if ticker_source in MARKET_SOURCES and not market_ranking.empty:
        chart_heading = "Predicted top 10 - history, forecast, and uncertainty"
    elif ticker_source == IRELAND_SOURCE:
        chart_heading = "ISEQ 20 candidates - history and feature-based forecast"
    elif ticker_source == FTSE_MIB_SOURCE:
        chart_heading = "FTSE MIB candidates - history and feature-based forecast"
    elif ticker_source == US_SOURCE:
        chart_heading = "Detected U.S. candidates - history and feature-based forecast"
    else:
        chart_heading = "Top momentum stocks - history and feature-based forecast"

    st.subheader(chart_heading)
    st.caption(
        f"The dashed line is the ticker-level {horizon_label} forecast. Shaded 50% and 80% bands are calibrated from earlier walk-forward return residuals. On daily horizons, point-in-time sentiment is continuously evaluated and only replaces the price-only curve after at least {MIN_BACKTEST_POINTS} paired forecasts improve MAE."
    )

    backtest_rows = []
    forecast_successes = 0
    forecast_failures: list[dict[str, object]] = []
    for ticker in top_performers:
        df = price_data[ticker]
        bac_log_kv(
            "views.render_charts_view.ticker",
            ticker=ticker,
            price_rows=len(df),
            forecast_points=forecast_points,
        )

        calendar_name = resolve_market_calendar(ticker_source, ticker)
        price_fc = forecast_feature_model(
            df,
            points_ahead=forecast_points,
            market_calendar=calendar_name,
        )
        price_backtest = backtest_forecast_model(
            df,
            forecast_horizon=forecast_points,
            market_calendar=calendar_name,
        )
        if realtime_mode:
            sentiment_history = pd.DataFrame()
        elif ticker in sentiment_by_ticker:
            sentiment_history = sentiment_by_ticker[ticker]
        else:
            sentiment_history = load_sentiment_history(ticker)
        sentiment_fc = pd.DataFrame()
        sentiment_backtest = pd.DataFrame()
        if not realtime_mode and not sentiment_history.empty:
            sentiment_fc = forecast_feature_model(
                df,
                points_ahead=forecast_points,
                sentiment_history=sentiment_history,
                include_sentiment=True,
                market_calendar=calendar_name,
            )
            sentiment_backtest = backtest_forecast_model(
                df,
                forecast_horizon=forecast_points,
                sentiment_history=sentiment_history,
                include_sentiment=True,
                market_calendar=calendar_name,
            )

        comparison = None
        if not price_backtest.empty:
            comparison = summarize_model_comparison(
                ticker,
                price_backtest,
                sentiment_backtest,
                forecast_points,
            )
        sentiment_promoted = bool(
            comparison is not None
            and comparison["Active model"] == "Price + sentiment"
            and not sentiment_fc.empty
        )
        fc = sentiment_fc if sentiment_promoted else price_fc
        backtest = sentiment_backtest if sentiment_promoted else price_backtest
        active_model = "Price + sentiment" if sentiment_promoted else "Price only"
        fc = add_forecast_intervals(
            fc,
            backtest,
            last_close=float(df["Close"].iloc[-1]),
        )
        if fc.empty:
            # An empty Plotly chart previously looked like "forecasting did
            # nothing." Diagnose the failed ticker only after the fast path has
            # failed, then expose the exact cause to both the user and BAC logs.
            diagnosis = diagnose_forecast_readiness(
                df,
                forecast_horizon=1,
                market_calendar=calendar_name,
            )
            forecast_failures.append({"ticker": ticker, **diagnosis})
            bac_log_kv(
                "views.render_charts_view.forecast_status",
                ticker=ticker,
                requested_points=forecast_points,
                returned_points=0,
                **diagnosis,
            )
            st.warning(f"{ticker}: forecast unavailable. {diagnosis['message']}")
        else:
            forecast_successes += 1
            if len(fc) < forecast_points:
                # A curve can stop at a later horizon when only the longest
                # target lacks enough realized training labels. Keep the useful
                # prefix and identify precisely where it became unavailable.
                diagnosis = diagnose_forecast_readiness(
                    df,
                    forecast_horizon=len(fc) + 1,
                    market_calendar=calendar_name,
                )
                forecast_failures.append({"ticker": ticker, **diagnosis})
                bac_log_kv(
                    "views.render_charts_view.forecast_status",
                    ticker=ticker,
                    requested_points=forecast_points,
                    returned_points=len(fc),
                    **diagnosis,
                )
                st.warning(
                    f"{ticker}: showing {len(fc)} of {forecast_points} requested "
                    f"forecast points. {diagnosis['message']}"
                )
            else:
                bac_log_kv(
                    "views.render_charts_view.forecast_status",
                    ticker=ticker,
                    requested_points=forecast_points,
                    returned_points=len(fc),
                    status="ready",
                )
        bac_log_kv(
            "views.render_charts_view.ticker",
            ticker=ticker,
            forecast_rows=len(fc),
            backtest_rows=len(backtest),
            sentiment_articles=len(sentiment_history),
            sentiment_promoted=sentiment_promoted,
        )

        fig = go.Figure()

        # The solid line anchors the user in observed history before the forecast starts.
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name=f"{ticker} Close",
                line={"width": 2},
            )
        )

        # A separate marker makes the most recent observed point visually obvious.
        fig.add_trace(
            go.Scatter(
                x=[df["Date"].iloc[-1]],
                y=[df["Close"].iloc[-1]],
                mode="markers",
                name=f"{ticker} Latest",
                marker={"size": 10},
            )
        )

        if not fc.empty:
            last_date = df["Date"].iloc[-1]
            future_dates = future_projection_dates(
                last_date,
                len(fc),
                realtime_mode,
                interval,
                market_calendar=resolve_market_calendar(ticker_source, ticker),
            )
            # Draw uncertainty from widest to narrowest so both bands remain
            # visible underneath the central forecast line.
            if {"lower_80", "upper_80"}.issubset(fc.columns):
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=fc["upper_80"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f"{ticker}-80-band",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=fc["lower_80"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(99, 110, 250, 0.12)",
                        name=f"{ticker} 80% interval",
                        legendgroup=f"{ticker}-80-band",
                    )
                )
            if {"lower_50", "upper_50"}.issubset(fc.columns):
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=fc["upper_50"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=f"{ticker}-50-band",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=fc["lower_50"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(99, 110, 250, 0.24)",
                        name=f"{ticker} 50% interval",
                        legendgroup=f"{ticker}-50-band",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=fc["pred_close"],
                    mode="lines",
                    name=f"{ticker} {active_model} forecast",
                    line={"dash": "dash", "width": 2},
                )
            )
            if len(future_dates):
                final_projection = fc.iloc[-1]
                ranking_row = (
                    market_ranking[market_ranking["Ticker"] == ticker]
                    if "Ticker" in market_ranking.columns
                    else pd.DataFrame()
                )
                if not ranking_row.empty:
                    monitored_sentiment = float(ranking_row["Sentiment score"].iloc[0])
                elif not sentiment_history.empty and "sentiment" in sentiment_history.columns:
                    monitored_sentiment = float(
                        pd.to_numeric(
                            sentiment_history["sentiment"], errors="coerce"
                        ).tail(20).mean()
                    )
                else:
                    monitored_sentiment = np.nan
                record_forecast(
                    market_source=monitoring_market,
                    ticker=ticker,
                    forecast_origin=last_date,
                    target_at=future_dates[-1],
                    horizon=int(final_projection["projection_point"]),
                    model_name=active_model,
                    regime=_volatility_regime(df),
                    origin_close=float(df["Close"].iloc[-1]),
                    predicted_close=float(final_projection["pred_close"]),
                    predicted_return=float(final_projection["pred_return"]),
                    lower_80=final_projection.get("lower_80", np.nan),
                    upper_80=final_projection.get("upper_80", np.nan),
                    sentiment_score=monitored_sentiment,
                )

        fig.update_layout(
            title=f"{ticker}: Price History & Feature Forecast",
            xaxis_title="Timestamp" if realtime_mode else "Date",
            yaxis_title=price_axis_label,
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig)

        # The caption under each chart helps connect the picture to the scorecard.
        if not fc.empty:
            current_projection = fc.iloc[-1]
            current_return_pct = float(current_projection["pred_return"] * 100)
            current_close = float(current_projection["pred_close"])
            confidence = "Unavailable"
            directional_accuracy = np.nan
            mae_improvement = np.nan
            sentiment_status = "Intraday model is price-only" if realtime_mode else "Collecting history"

            if comparison is not None:
                summary = dict(comparison)
                confidence = summary["Confidence"]
                directional_accuracy = float(summary["Directional accuracy"])
                mae_improvement = float(summary["MAE improvement vs. no-change"])
                sentiment_status = str(summary["Sentiment status"])
                summary["Projected return"] = current_return_pct
                summary["Projected close"] = current_close
                backtest_rows.append(summary)

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
            bac_log_kv(
                "views.render_charts_view.caption",
                ticker=ticker,
                current_return_pct=current_return_pct,
                current_close=current_close,
                confidence=confidence,
            )
            st.caption(
                f"{ticker}: current {horizon_label} {active_model.lower()} forecast {current_return_pct:+.2f}% to {price_prefix}{current_close:.2f}. Backtest rating: {confidence}. Recent backtest: {accuracy_text}, {improvement_text}. Sentiment: {sentiment_status}."
            )
        elif comparison is not None:
            backtest_rows.append(comparison)

    bac_log_kv(
        "views.render_charts_view.forecast_summary",
        requested_tickers=len(top_performers),
        successful_tickers=forecast_successes,
        failed_or_partial_tickers=len(forecast_failures),
    )
    if forecast_successes == len(top_performers):
        st.caption(
            f"Forecast pipeline ready: {forecast_successes}/{len(top_performers)} "
            "ticker curves generated."
        )
    elif forecast_successes:
        st.warning(
            f"Forecast pipeline partially available: {forecast_successes}/"
            f"{len(top_performers)} ticker curves generated. See ticker warnings above."
        )
    else:
        st.error(
            "Forecast pipeline unavailable for this selection. The ticker warnings "
            "above and [BAC_LOG] entries contain the exact failure reasons."
        )

    st.subheader("Forecast backtest")
    st.caption(
        f"Walk-forward test of up to {MAX_BACKTEST_POINTS} unseen {horizon_label} forecasts. Each forecast is trained only on the preceding {BACKTEST_TRAINING_POINTS} observations and compared with a no-change baseline."
    )

    if backtest_rows:
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
    else:
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
    if not model_history.empty:
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
    bac_log_section("views.render_charts_view", "Charts rendering completed.")


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
    company_by_ticker = (
        {
            str(ticker): str(company)
            for ticker, company in detected_performers.set_index("Ticker")["Company"].to_dict().items()
        }
        if not detected_performers.empty
        else {}
    )
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
