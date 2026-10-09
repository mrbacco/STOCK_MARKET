#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: ui_components.py
#############################

"""Reusable Streamlit renderers shared by the pages in `app_pages/`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_config import BACKTEST_TRAINING_POINTS, MAX_BACKTEST_POINTS, MIN_BACKTEST_POINTS
from market_data import PriceDataHealth
from model_evidence import EvidenceAssessment
from model_monitoring import latest_drift_summary, load_forecast_quality, load_market_model_history
from ui_state import open_stock

if TYPE_CHECKING:
    from chart_pipeline import TickerForecast
    from walk_forward_store import StoredWalkForward

PERCENT = "%+.2f%%"


def diagnostic_value(diagnostics: dict[str, object], key: str) -> float:
    """Return a numeric diagnostic, or NaN when it is missing or not numeric."""
    value = diagnostics.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def leaderboard_column_config(performers: pd.DataFrame, price_format: str) -> dict[str, Any]:
    """Column formats shared by every daily-move leaderboard table."""
    columns: dict[str, Any] = {
        "Daily change": st.column_config.NumberColumn("Daily change", format="%+.2f%%"),
        "Last price": st.column_config.NumberColumn("Last price", format=price_format),
    }
    if "Last session" in performers.columns:
        columns["Last session"] = st.column_config.DateColumn("Last session")
    return columns


def selectable_table(
    frame: pd.DataFrame,
    *,
    key: str,
    column_config: dict[str, Any] | None = None,
    column_order: list[str] | None = None,
    height: int | None = None,
) -> None:
    """Show a table whose rows open the Stock page when clicked."""
    if height is None:
        event = st.dataframe(
            frame, column_config=column_config, column_order=column_order, hide_index=True,
            on_select="rerun", selection_mode="single-row", key=key,
        )
    else:
        event = st.dataframe(
            frame, column_config=column_config, column_order=column_order, hide_index=True,
            on_select="rerun", selection_mode="single-row", key=key, height=height,
        )
    # Both state dictionaries are declared total=False, so every key is optional.
    rows = (event or {}).get("selection", {}).get("rows", [])
    if rows and "Ticker" in frame.columns:
        open_stock(str(frame.iloc[rows[0]]["Ticker"]))


def render_data_health(health: PriceDataHealth, price_data: dict[str, pd.DataFrame]) -> None:
    """Keep data quality next to the results it affects."""
    if health.stale_tickers:
        stale_details = ", ".join(
            f"{ticker} (latest bar {health.freshness_by_ticker[ticker].get('latest_bar', 'unknown')})"
            for ticker in health.stale_tickers
        )
        st.warning(
            "Market-data recovery mode: these histories are stale or recovered from the last "
            f"saved snapshot and are excluded from the ranking: {stale_details}.",
            icon=":material/warning:",
        )
    bars = [
        pd.Timestamp(str(diagnosis["latest_bar"]))
        for diagnosis in health.freshness_by_ticker.values()
        if diagnosis.get("latest_bar") is not None
    ]
    latest_market_bar = max(bars) if bars else None
    with st.container(horizontal=True):
        st.metric(
            "Data pipeline",
            "Healthy" if not health.stale_tickers else "Recovery mode",
            border=True,
        )
        st.metric("Fresh histories", len(health.live_tickers), border=True)
        st.metric("Stale or recovered", len(health.stale_tickers), border=True)
        st.metric(
            "Latest market bar",
            (
                pd.Timestamp(str(latest_market_bar)).strftime("%Y-%m-%d")
                if latest_market_bar is not None
                else "Unavailable"
            ),
            border=True,
        )


def render_evidence(evidence: EvidenceAssessment, *, compact: bool = False) -> None:
    """Show how far the ranking can be trusted, before the ranking itself.

    The compact form is the two-line caveat used above rankings; the full form,
    with every supporting number, is used on the Model health page.
    """
    if compact and evidence.caveat:
        text = evidence.caveat
    else:
        text = f"**{evidence.headline}.** {evidence.detail}"
    if evidence.level == "supported":
        st.success(text, icon=":material/verified:")
    elif evidence.level == "tentative":
        st.info(text, icon=":material/info:")
    elif evidence.level == "none":
        st.warning(text, icon=":material/report:")
    else:
        st.info(text, icon=":material/hourglass:")


def ranking_column_config() -> dict[str, Any]:
    return {
        "Expected excess return": st.column_config.NumberColumn(
            "Expected excess", format=PERCENT,
            help="Predicted return relative to the universe average over the horizon.",
        ),
        "Probability outperform": st.column_config.ProgressColumn(
            "Probability outperform", min_value=0, max_value=100, format="%.0f%%",
        ),
        "Lower 80": st.column_config.NumberColumn("80% low", format=PERCENT),
        "Upper 80": st.column_config.NumberColumn("80% high", format=PERCENT),
        "Predicted volatility": st.column_config.NumberColumn(
            "Excess volatility", format="%.2f%%",
            help="GARCH forecast of how much this stock's return relative to the "
            "market may vary over the horizon. It also scales the bands.",
        ),
        "Model disagreement": st.column_config.NumberColumn("Model disagreement", format="%.2f%%"),
        "Sentiment score": st.column_config.NumberColumn("24h sentiment", format="%+.2f"),
    }


def render_validation_strip(diagnostics: dict[str, object]) -> None:
    """Summarize the untouched evaluation period in one metric row."""
    with st.container(horizontal=True):
        st.metric(
            "Rank IC",
            f"{diagnostic_value(diagnostics, 'Rank IC'):+.3f}",
            border=True,
            help="Average daily correlation between predicted and realized order of "
            "stocks. Above 0 beats chance; 0.03 or more is useful for stocks.",
        )
        st.metric(
            "Top-10 excess",
            f"{diagnostic_value(diagnostics, 'Top-10 realized mean excess'):+.2f}%",
            border=True,
            help="Average realized excess return of the model's ten best-ranked stocks.",
        )
        st.metric(
            "Top-10 hit rate",
            f"{diagnostic_value(diagnostics, 'Top-10 realized hit rate'):.0f}%",
            border=True,
        )
        st.metric(
            "80% band coverage",
            f"{diagnostic_value(diagnostics, '80% interval coverage'):.0f}%",
            border=True,
            help="Share of outcomes inside their adaptive, volatility-scaled 80% band.",
        )
        st.metric(
            "Direction accuracy",
            f"{diagnostic_value(diagnostics, 'Directional accuracy'):.0f}%",
            border=True,
        )


def render_forecast_status(forecast: TickerForecast) -> None:
    """Explain a missing or truncated forecast curve."""
    if forecast.diagnosis is None:
        return
    message = forecast.diagnosis.get("message", "")
    if forecast.status == "unavailable":
        st.warning(f"Projection unavailable. {message}", icon=":material/warning:")
    elif forecast.status == "partial":
        st.warning(
            f"Showing {len(forecast.forecast)} of {forecast.requested_points} projected "
            f"sessions. {message}",
            icon=":material/warning:",
        )


def _add_band(fig: go.Figure, forecast: TickerForecast, level: int, fillcolor: str) -> None:
    lower, upper = f"lower_{level}", f"upper_{level}"
    if not {lower, upper}.issubset(forecast.forecast.columns):
        return
    group = f"{forecast.ticker}-{level}-band"
    fig.add_trace(
        go.Scatter(
            x=forecast.future_dates, y=forecast.forecast[upper], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False, legendgroup=group,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.future_dates, y=forecast.forecast[lower], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor=fillcolor,
            name=f"{level}% band", legendgroup=group,
        )
    )


def forecast_figure(forecast: TickerForecast, *, price_axis_label: str, lookback: int = 180) -> go.Figure:
    """Observed history plus the projection and its 50%/80% bands."""
    history = forecast.history.tail(lookback)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=history["Date"], y=history["Close"], mode="lines", name="Close", line={"width": 2})
    )
    if not forecast.forecast.empty:
        _add_band(fig, forecast, 80, "rgba(99, 110, 250, 0.12)")
        _add_band(fig, forecast, 50, "rgba(99, 110, 250, 0.24)")
        fig.add_trace(
            go.Scatter(
                x=forecast.future_dates, y=forecast.forecast["pred_close"], mode="lines",
                name=f"{forecast.active_model} projection", line={"dash": "dash", "width": 2},
            )
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=price_axis_label,
        template="plotly_white",
        height=430,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.08},
        uirevision=forecast.ticker,
    )
    return fig


def render_forecast_caption(forecast: TickerForecast, *, horizon: int, price_prefix: str) -> None:
    """Connect the projection to its backtest scorecard in one sentence."""
    if forecast.forecast.empty:
        return
    final = forecast.forecast.iloc[-1]
    comparison = forecast.comparison or {}
    accuracy = comparison.get("Directional accuracy", np.nan)
    improvement = comparison.get("MAE improvement vs. no-change", np.nan)
    accuracy_text = (
        f"{float(accuracy):.0f}% directional accuracy" if pd.notna(accuracy) else "no backtest yet"
    )
    improvement_text = (
        f"{float(improvement):+.1f}% error vs. a no-change forecast" if pd.notna(improvement) else ""
    )
    stale = " Built from last-known-good data." if forecast.using_stale_data else ""
    st.caption(
        f"{horizon}-session {forecast.active_model.lower()} projection: "
        f"{float(final['pred_return']) * 100:+.2f}% to {price_prefix}{float(final['pred_close']):,.2f}. "
        f"Backtest: {accuracy_text}"
        f"{', ' + improvement_text if improvement_text else ''}. "
        f"Sentiment: {comparison.get('Sentiment status', 'not evaluated')}.{stale}"
    )


def backtest_column_config(price_format: str) -> dict[str, Any]:
    return {
        "Projected return": st.column_config.NumberColumn("Projected return", format="%.2f%%"),
        "Projected close": st.column_config.NumberColumn("Projected close", format=price_format),
        "Model MAE": st.column_config.NumberColumn("Model MAE", format=price_format),
        "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
        "Directional accuracy": st.column_config.NumberColumn("Direction", format="%.1f%%"),
        "No-change MAE": st.column_config.NumberColumn("No-change MAE", format=price_format),
        "MAE improvement vs. no-change": st.column_config.NumberColumn(
            "Improvement vs. no-change", format="%.1f%%"
        ),
        "Price-only MAE": st.column_config.NumberColumn("Price-only MAE", format=price_format),
        "Sentiment MAE": st.column_config.NumberColumn("Sentiment MAE", format=price_format),
        "Sentiment MAE lift vs. price-only": st.column_config.NumberColumn(
            "Sentiment lift", format="%.1f%%"
        ),
    }


def backtest_caption(horizon: int) -> str:
    return (
        f"Walk-forward test of up to {MAX_BACKTEST_POINTS} unseen {horizon}-session forecasts, "
        f"each trained only on the preceding {BACKTEST_TRAINING_POINTS} sessions and compared "
        f"with a no-change baseline. Sentiment replaces the price-only model only after "
        f"{MIN_BACKTEST_POINTS} paired forecasts improve its error."
    )


def render_model_monitoring(monitoring_market: str, horizon: int, price_format: str) -> None:
    """Resolved production-forecast quality, drift, and stored run history."""
    forecast_quality = load_forecast_quality(monitoring_market, horizon=horizon)
    if forecast_quality.empty:
        st.info(
            "No recorded projections have reached their target date yet. Projections shown on "
            "the Stock page are stored and scored automatically.",
            icon=":material/hourglass:",
        )
    else:
        st.dataframe(
            forecast_quality,
            column_config={
                "MAE": st.column_config.NumberColumn("Close MAE", format=price_format),
                "Return MAE": st.column_config.NumberColumn("Return MAE", format="%.2f%%"),
                "Directional accuracy": st.column_config.NumberColumn("Direction", format="%.1f%%"),
                "80% interval coverage": st.column_config.NumberColumn(
                    "80% band coverage", format="%.1f%%"
                ),
                "Last resolved": st.column_config.DatetimeColumn("Last resolved"),
            },
            hide_index=True,
        )

    model_history = load_market_model_history(monitoring_market, horizon=horizon)
    if model_history.empty:
        return
    drift = latest_drift_summary(model_history)
    if drift:
        st.markdown("**Latest run versus earlier runs**")
        with st.container(horizontal=True):
            st.metric("MAE drift", f"{drift['MAE drift']:+.2f} pp", delta_color="inverse", border=True)
            st.metric("Direction drift", f"{drift['Direction drift']:+.1f} pp", border=True)
            st.metric("Brier drift", f"{drift['Brier drift']:+.3f}", border=True)
            st.metric("Coverage drift", f"{drift['Coverage drift']:+.1f} pp", border=True)
            rank_ic_drift = float(drift.get("Rank IC drift", np.nan))
            st.metric(
                "Rank IC drift",
                f"{rank_ic_drift:+.3f}" if np.isfinite(rank_ic_drift) else "n/a",
                border=True,
            )
    st.markdown("**Stored model runs**")
    st.dataframe(
        model_history.drop(columns=["model_weights_json"]).rename(
            columns={
                "as_of": "As of",
                "candidate_tickers": "Candidates",
                "evaluation_dates": "Evaluation dates",
                "evaluation_mae": "Evaluation MAE",
                "baseline_mae": "Baseline MAE",
                "directional_accuracy": "Direction",
                "probability_brier": "Brier",
                "interval_coverage_80": "80% coverage",
                "selection_mean_excess": "Top-10 excess",
                "selection_hit_rate": "Top-10 hit rate",
                "sentiment_observed_rows": "Sentiment rows",
                "rank_ic": "Rank IC",
            }
        ),
        hide_index=True,
    )


def render_walk_forward(stored: StoredWalkForward, horizon: int) -> None:
    """Show a stored multi-year walk-forward result."""
    from portfolio_backtest import top_n_backtest

    summary = stored.summary
    tested = pd.Timestamp(stored.created_at).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(
        f"Tested {tested} on {summary.get('Test start')} to {summary.get('Test end')} "
        f"({summary.get('Test years')} years, {summary.get('Test dates')} test dates, "
        f"{summary.get('Retraining points')} retraining points)."
    )
    with st.container(horizontal=True):
        st.metric("Rank IC", f"{diagnostic_value(summary, 'Rank IC'):+.3f}", border=True)
        st.metric(
            "t-statistic",
            f"{diagnostic_value(summary, 'Rank IC t-stat'):.1f}",
            border=True,
            help="On non-overlapping dates. Above 2 is the usual significance bar.",
        )
        st.metric(
            "Positive quarters",
            f"{diagnostic_value(summary, 'Positive quarters'):.0f}%",
            border=True,
        )
        st.metric(
            f"Top-{int(diagnostic_value(summary, 'Top-N'))} net excess",
            f"{diagnostic_value(summary, 'Portfolio Cumulative net excess'):+.1f}%",
            border=True,
            help="Compounded return versus the universe after trading costs.",
        )
        st.metric(
            "Information ratio",
            f"{diagnostic_value(summary, 'Portfolio Information ratio'):.2f}",
            border=True,
        )

    st.markdown("**Model versus simple rules** (average daily rank IC)")
    st.dataframe(
        pd.DataFrame(
            {
                "Ranking": ["Pooled model", "20-day momentum", "5-day reversal"],
                "Rank IC": [
                    diagnostic_value(summary, "Rank IC"),
                    diagnostic_value(summary, "Momentum rank IC"),
                    diagnostic_value(summary, "Reversal rank IC"),
                ],
            }
        ),
        column_config={"Rank IC": st.column_config.NumberColumn(format="%+.3f")},
        hide_index=True,
    )

    quarterly = summary.get("Quarterly rank IC", {})
    chart_column, curve_column = st.columns(2)
    if isinstance(quarterly, dict) and quarterly:
        with chart_column:
            st.markdown("**Rank IC by quarter**")
            values = [float(value) for value in quarterly.values()]
            figure = go.Figure(
                go.Bar(
                    x=list(quarterly),
                    y=values,
                    marker_color=["#2e7d32" if value > 0 else "#c62828" for value in values],
                )
            )
            figure.update_layout(
                template="plotly_white", height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                yaxis_title="Rank IC",
            )
            st.plotly_chart(figure, key="walk_forward_quarters")
    if not stored.predictions.empty:
        backtest = top_n_backtest(
            stored.predictions,
            horizon=horizon,
            top_n=int(diagnostic_value(summary, "Top-N")),
            cost_bps=diagnostic_value(summary, "Cost bps"),
        )
        if not backtest.periods.empty:
            with curve_column:
                st.markdown("**Cumulative excess return**")
                figure = go.Figure()
                for column, name, dash in (
                    ("Cumulative net", "Top-N after costs", "solid"),
                    ("Cumulative gross", "Top-N before costs", "dot"),
                    ("Cumulative bottom-N", "Bottom-N (control)", "dash"),
                ):
                    figure.add_trace(
                        go.Scatter(
                            x=backtest.periods["Date"],
                            y=backtest.periods[column] * 100,
                            name=name,
                            mode="lines",
                            line={"dash": dash},
                        )
                    )
                figure.add_hline(y=0, line_color="gray", line_width=1)
                figure.update_layout(
                    template="plotly_white", height=300,
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    yaxis_title="%", legend={"orientation": "h", "y": 1.15},
                )
                st.plotly_chart(figure, key="walk_forward_curve")

    if not stored.folds.empty:
        with st.expander("Retraining log"):
            st.dataframe(
                stored.folds.assign(Weights=stored.folds["Weights"].astype(str)),
                hide_index=True,
            )
