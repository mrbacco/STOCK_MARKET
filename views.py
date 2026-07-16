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
    selected_horizon_label,
)
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from forecasting import (
    backtest_forecast_model,
    forecast_feature_model,
    future_projection_dates,
    summarize_backtest,
)
from market_data import (
    get_price_history_batch,
    growth_score,
    load_news_frames_parallel,
    momentum_label,
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
    if ticker_source in MARKET_SOURCES:
        bac_log_section("views.render_overview_view", "Rendering leaderboard-only overview.")
        if ticker_source == IRELAND_SOURCE:
            st.caption(
                "The leaderboard ranks the latest available daily close from the tracked ISEQ 20 Euronext Dublin universe."
            )
        elif ticker_source == FTSE_MIB_SOURCE:
            st.caption(
                "This view tracks the latest available daily move for the FTSE MIB benchmark index."
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

    active_tickers = tickers[:MAX_CHARTED_PERFORMERS]
    if len(tickers) > MAX_CHARTED_PERFORMERS:
        bac_log_kv(
            "views.render_charts_view",
            message="Trimming ticker list for charting.",
            chart_limit=MAX_CHARTED_PERFORMERS,
        )
        st.info(f"Charting the first {MAX_CHARTED_PERFORMERS} symbols from the selected source.")

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
        bac_log_list_preview("views.render_charts_view", "fallback_valid_tickers", valid_tickers)

    if not valid_tickers:
        bac_log_section("views.render_charts_view", "No valid chart data was available.")
        st.error("No price data was returned. Check ticker symbols and try again.")
        st.stop()

    if ticker_source in MARKET_SOURCES:
        daily_change_by_ticker = detected_performers.set_index("Ticker")["Daily change"].to_dict()
        top_performers = [ticker for ticker in active_tickers if ticker in valid_tickers]
        if ticker_source == IRELAND_SOURCE:
            leader_label = "Top ISEQ 20 daily mover"
            performance_label = "Best ISEQ 20 daily change"
        elif ticker_source == FTSE_MIB_SOURCE:
            leader_label = "Tracked FTSE MIB index"
            performance_label = "Latest FTSE MIB daily change"
        else:
            leader_label = "Top detected daily gainer"
            performance_label = "Best detected daily change"
        leader_change = daily_change_by_ticker.get(top_performers[0], np.nan)
        performance_value = f"{leader_change:.2f}%" if pd.notna(leader_change) else "Unavailable"
        bac_log_kv(
            "views.render_charts_view",
            leader_label=leader_label,
            performance_value=performance_value,
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
        bac_log_section("views.render_charts_view", "Rendering detected-performer leaderboard.")
        if ticker_source == IRELAND_SOURCE:
            st.subheader("ISEQ 20 top daily performers")
            st.caption(
                "The leaderboard ranks the latest available daily close from the tracked ISEQ 20 Euronext Dublin universe. It is not a complete ranking of every Irish or European stock."
            )
        elif ticker_source == FTSE_MIB_SOURCE:
            st.subheader("FTSE MIB benchmark index")
            st.caption(
                "This source charts the FTSE MIB benchmark index directly. It is an index view, not a ranked list of Italian constituents."
            )
        else:
            st.subheader("Detected top 10 daily gainers")
            st.caption(
                "The screener is Yahoo Finance's predefined U.S. equity filter for liquid, large-cap daily gainers. It is not a complete ranking of every listed stock."
            )
        leaderboard_columns = {
            "Daily change": st.column_config.NumberColumn("Daily change", format="%.2f%%"),
            "Last price": st.column_config.NumberColumn("Last price", format=price_format),
        }
        if "Last session" in detected_performers.columns:
            leaderboard_columns["Last session"] = st.column_config.DateColumn("Last session")
        st.dataframe(detected_performers, column_config=leaderboard_columns, hide_index=True)

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
        st.caption(
            "Intraday figures use the latest returned bar close. The delta is versus the prior bar, not a live tick or daily change."
        )

    horizon_label = selected_horizon_label(realtime_mode, interval, forecast_points)
    if ticker_source == IRELAND_SOURCE:
        chart_heading = "ISEQ 20 daily leaders - history and feature-based forecast"
    elif ticker_source == FTSE_MIB_SOURCE:
        chart_heading = "FTSE MIB index - history and feature-based forecast"
    elif ticker_source == US_SOURCE:
        chart_heading = "Detected top 10 daily gainers - history and feature-based forecast"
    else:
        chart_heading = "Top momentum stocks - history and feature-based forecast"

    st.subheader(chart_heading)
    st.caption(
        f"The dashed forecast estimates returns from recent momentum, volatility, RSI, price structure, and volume features. The backtest below scores the same {horizon_label} horizon against a no-change baseline, so the chart and validation stay aligned."
    )

    backtest_rows = []
    for ticker in top_performers:
        df = price_data[ticker]
        bac_log_kv(
            "views.render_charts_view.ticker",
            ticker=ticker,
            price_rows=len(df),
            forecast_points=forecast_points,
        )

        fc = forecast_feature_model(df, points_ahead=forecast_points)
        backtest = backtest_forecast_model(df, forecast_horizon=forecast_points)
        bac_log_kv(
            "views.render_charts_view.ticker",
            ticker=ticker,
            forecast_rows=len(fc),
            backtest_rows=len(backtest),
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
            future_dates = future_projection_dates(last_date, len(fc), realtime_mode, interval)
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=fc["pred_close"],
                    mode="lines",
                    name=f"{ticker} Forecast model",
                    line={"dash": "dash", "width": 2},
                )
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

            if not backtest.empty:
                summary = summarize_backtest(ticker, backtest, forecast_points)
                confidence = summary["Confidence"]
                directional_accuracy = float(summary["Directional accuracy"])
                mae_improvement = float(summary["MAE improvement vs. no-change"])
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
                f"{ticker}: current {horizon_label} forecast {current_return_pct:+.2f}% to {price_prefix}{current_close:.2f}. Confidence: {confidence}. Recent backtest: {accuracy_text}, {improvement_text}."
            )
        elif not backtest.empty:
            backtest_rows.append(summarize_backtest(ticker, backtest, forecast_points))

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
            },
            hide_index=True,
        )
        st.caption(
            "Positive MAE improvement means the feature model beat the no-change baseline on the selected horizon; negative values mean it performed worse."
        )
    else:
        bac_log_section("views.render_charts_view", "No backtest summary rows were available.")
        st.info(
            "Not enough price observations to backtest this forecast model. "
            f"At least {BACKTEST_TRAINING_POINTS + forecast_points + MIN_BACKTEST_POINTS - 1} observations are required."
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
            yaxis_title="Compound sentiment score",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(bar)

        news_table = (
            all_news[["ticker", "published", "title", "sentiment_label", "sentiment", "link"]]
            .sort_values(by="published", ascending=False)
            .reset_index(drop=True)
        )
        bac_log_kv("views.render_news_view", news_table_rows=len(news_table))
        st.dataframe(news_table)
    else:
        bac_log_section("views.render_news_view", "No news rows were available.")
        st.info("No news items were fetched right now. Try again in a moment.")
