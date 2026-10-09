#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/model_health.py
#############################

"""Model health: multi-year walk-forward test, validation, live scoring, drift."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_evidence import assess_ranking_evidence
from ui_components import (
    render_evidence,
    render_model_monitoring,
    render_validation_strip,
    render_walk_forward,
)
from ui_state import current_selection, market_analysis
from walk_forward_store import (
    WALK_FORWARD_HISTORY,
    load_walk_forward,
    start_walk_forward,
    walk_forward_status,
)

selection = current_selection()
_prefix, price_format, _axis = selection.price_display()
st.subheader(f"{selection.label} - {selection.horizon}-session horizon")

# --- Multi-year walk-forward ---------------------------------------------------
with st.container(border=True):
    st.markdown("**Multi-year walk-forward test**")
    st.caption(
        f"Replays the production ensemble over {WALK_FORWARD_HISTORY} of history: every quarter "
        "it is retrained only on data available at that time, then predicts the next quarter. "
        "Every prediction is out of sample, so this is the most reliable test of the ranking."
    )
    if selection.is_watchlist:
        st.info("Choose a stock universe in the sidebar to test the pooled ranking.",
                icon=":material/info:")
    else:
        universe, horizon = selection.universe_key, selection.horizon
        stored = load_walk_forward(universe, horizon)
        status = walk_forward_status(universe, horizon)
        if status.running:
            @st.fragment(run_every="2s")
            def walk_forward_progress() -> None:
                current = walk_forward_status(universe, horizon)
                if not current.running:
                    st.rerun()
                st.progress(current.progress, text=current.message or "Running")

            walk_forward_progress()
        else:
            if status.error:
                st.error(f"The last run failed: {status.error}", icon=":material/error:")
            label = "Re-run walk-forward test" if stored else "Run walk-forward test"
            if st.button(label, icon=":material/play_arrow:", key="run_walk_forward"):
                start_walk_forward(universe, horizon)
                st.rerun()
        if stored is not None and not stored.is_current:
            st.warning(
                "This result was produced by an earlier version of the model, so it no longer "
                "counts as evidence. Re-run the test.",
                icon=":material/history:",
            )
        if stored is not None:
            render_evidence(assess_ranking_evidence({}, stored.summary))
            render_walk_forward(stored, horizon)
        elif not status.running:
            st.caption("Not tested yet for this universe and horizon. A run takes about a minute.")

# --- Latest model --------------------------------------------------------------
st.subheader("Latest model")
analysis = market_analysis(selection, compute=False)
if analysis is None:
    st.info(
        "Open Market ranking first: it trains the model whose details are shown here.",
        icon=":material/leaderboard:",
    )
else:
    diagnostics = analysis.ranking.diagnostics
    if diagnostics:
        st.caption("Validation on the 30 most recent untouched dates (short and noisy).")
        render_validation_strip(diagnostics)
        weights = diagnostics.get("Model weights", {})
        with st.container(border=True):
            st.markdown("**Ensemble members and weights**")
            if isinstance(weights, dict) and weights:
                st.dataframe(
                    pd.DataFrame(
                        {"Model": list(weights), "Weight": [float(w) * 100 for w in weights.values()]}
                    ),
                    column_config={
                        "Weight": st.column_config.ProgressColumn(
                            "Weight", min_value=0, max_value=100, format="%.0f%%"
                        )
                    },
                    hide_index=True,
                )
            exponent = diagnostics.get("Volatility scaling exponent")
            st.caption(
                f"LightGBM ranker: {diagnostics.get('LightGBM ranker', 'n/a')}. "
                f"Volatility models: {diagnostics.get('Volatility models', 'n/a')}."
                + (f" Band scaling exponent: {exponent:.2f}." if isinstance(exponent, (int, float)) else "")
            )
        with st.expander("All validation diagnostics"):
            st.json(diagnostics)

st.subheader("Live scoring of recorded projections")
st.caption(
    "Every projection shown on the Stock page is stored and scored once its target date "
    "passes. The tables compare those real outcomes with the model's claims."
)
render_model_monitoring(selection.universe_key, selection.horizon, price_format)
