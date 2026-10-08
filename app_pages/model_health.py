#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/model_health.py
#############################

"""Model health: validation, ensemble composition, live scoring, and drift."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_evidence import assess_ranking_evidence
from ui_components import render_evidence, render_model_monitoring, render_validation_strip
from ui_state import current_selection, market_analysis

selection = current_selection()
_prefix, price_format, _axis = selection.price_display()
st.subheader(f"{selection.label} - {selection.horizon}-session horizon")

analysis = market_analysis(selection, compute=False)
if analysis is None:
    st.info(
        "Open Market ranking first: it trains the model whose validation is shown here.",
        icon=":material/leaderboard:",
    )
else:
    diagnostics = analysis.ranking.diagnostics
    render_evidence(assess_ranking_evidence(diagnostics))
    if diagnostics:
        render_validation_strip(diagnostics)
        weights = diagnostics.get("Model weights", {})
        with st.container(border=True):
            st.markdown("**Ensemble members and weights**")
            if isinstance(weights, dict) and weights:
                st.dataframe(
                    pd.DataFrame({"Model": list(weights), "Weight": [float(w) * 100 for w in weights.values()]}),
                    column_config={"Weight": st.column_config.ProgressColumn(
                        "Weight", min_value=0, max_value=100, format="%.0f%%")},
                    hide_index=True,
                )
            st.caption(
                f"LightGBM ranker: {diagnostics.get('LightGBM ranker', 'n/a')}. "
                f"Volatility models: {diagnostics.get('Volatility models', 'n/a')}. "
                f"Band scaling exponent: {diagnostics.get('Volatility scaling exponent', float('nan')):.2f}."
                if isinstance(diagnostics.get("Volatility scaling exponent"), (int, float))
                else f"LightGBM ranker: {diagnostics.get('LightGBM ranker', 'n/a')}."
            )
        with st.expander("All validation diagnostics"):
            st.json(diagnostics)

st.subheader("Live scoring of recorded projections")
st.caption(
    "Every projection shown on the Stock page is stored and scored once its target date "
    "passes. The tables compare those real outcomes with the model's claims."
)
render_model_monitoring(selection.universe_key, selection.horizon, price_format)
