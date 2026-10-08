#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/world_markets.py
#############################

"""Global markets: cross-asset dashboard and a summary of every stock universe."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from global_markets import (
    ASSET_CLASSES,
    INSTRUMENTS_JOB,
    LEVEL_CHANGE_CLASSES,
    ensure_overview_warmup,
    global_movers,
    overview_entries,
    overview_jobs,
    overview_loading,
    universe_summary,
)
from market_sources import MARKET_SOURCE_REGISTRY
from ui_components import selectable_table
from ui_state import current_selection

current_selection()
st.caption(
    "World indices, volatility, rates, currencies, commodities, and crypto from daily "
    "Yahoo Finance closes, then every tracked stock universe. Select a stock to open it."
)
ensure_overview_warmup()

# symbol: (label, value format, change is in percentage points)
HEADLINES = {
    "^GSPC": ("S&P 500", "{:,.0f}", False),
    "^STOXX50E": ("Euro Stoxx 50", "{:,.0f}", False),
    "^N225": ("Nikkei 225", "{:,.0f}", False),
    "^VIX": ("VIX", "{:.1f}", False),
    "^TNX": ("US 10-year yield", "{:.2f}%", True),
    "EURUSD=X": ("EUR/USD", "{:.4f}", False),
    "GC=F": ("Gold", "${:,.0f}", False),
    "BTC-USD": ("Bitcoin", "${:,.0f}", False),
}


def render_headlines(snapshot: pd.DataFrame) -> None:
    by_symbol = snapshot.set_index("Symbol")
    with st.container(horizontal=True):
        for symbol, (label, value_format, level_change) in HEADLINES.items():
            if symbol not in by_symbol.index or pd.isna(by_symbol.at[symbol, "Last"]):
                st.metric(label, "Unavailable", border=True)
                continue
            row = by_symbol.loc[symbol]
            change = row["1D"]
            delta = (
                None
                if pd.isna(change)
                else f"{change:+.2f} pp" if level_change else f"{change:+.2f}%"
            )
            st.metric(
                label,
                value_format.format(float(row["Last"])),
                delta,
                # A rising VIX or yield is not good news for equities.
                delta_color="inverse" if symbol in {"^VIX", "^TNX"} else "normal",
                border=True,
                chart_data=row["Trend"] or None,
                chart_type="line",
            )


def render_asset_tabs(snapshot: pd.DataFrame) -> None:
    for tab, asset_class in zip(st.tabs(list(ASSET_CLASSES)), ASSET_CLASSES):
        rows = snapshot[snapshot["Asset class"] == asset_class]
        level = asset_class in LEVEL_CHANGE_CLASSES
        change_format = "%+.2f pp" if level else "%+.2f%%"
        with tab:
            st.dataframe(
                rows,
                column_order=[
                    "Name", "Symbol", "Last", "1D", "1W", "1M", "YTD", "1Y",
                    *([] if level else ["Volatility 20D"]), "Trend", "Last session",
                ],
                column_config={
                    "Last": st.column_config.NumberColumn(
                        "Yield" if level else "Last", format="%.2f%%" if level else "%,.2f"
                    ),
                    **{
                        label: st.column_config.NumberColumn(label, format=change_format)
                        for label in ("1D", "1W", "1M", "YTD", "1Y")
                    },
                    "Volatility 20D": st.column_config.NumberColumn(
                        "Volatility (20D, annualized)", format="%.1f%%"
                    ),
                    "Trend": st.column_config.LineChartColumn("3 months"),
                    "Last session": st.column_config.DateColumn("Last session"),
                },
                hide_index=True,
            )


def render_overview() -> None:
    entries = overview_entries()
    jobs = overview_jobs()
    live = sum(1 for job in jobs if job in entries and entries[job].live)
    saved_times = [entry.saved_at for entry in entries.values() if not entry.live and entry.saved_at]
    if live < len(jobs):
        note = f"Loading live prices: {live} of {len(jobs)} sections ready."
        if saved_times:
            oldest = pd.Timestamp(min(saved_times)).strftime("%H:%M UTC")
            note += f" Sections marked saved show prices from {oldest} until then."
        st.caption(f":orange[{note}]")

    instruments = entries.get(INSTRUMENTS_JOB)
    if instruments is None:
        st.info("Loading world markets...", icon=":material/hourglass:")
    else:
        render_headlines(instruments.frame)
        render_asset_tabs(instruments.frame)

    st.subheader("Stock universes")
    st.caption(
        "Breadth and median performance of every tracked universe. Advancing is the share "
        "of stocks up on their latest session."
    )
    summaries = []
    snapshots = []
    for key, source in MARKET_SOURCE_REGISTRY.items():
        entry = entries.get(key)
        if entry is None:
            continue
        snapshots.append(entry.frame)
        summaries.append(
            {
                "Universe": source.label,
                "Region": source.region,
                "Data": "Live" if entry.live else "Saved",
                **universe_summary(entry.frame),
            }
        )
    if not summaries:
        st.info("Loading stock universes...", icon=":material/hourglass:")
        return
    st.dataframe(
        pd.DataFrame(summaries),
        column_config={
            "Advancing": st.column_config.ProgressColumn(
                "Advancing", min_value=0, max_value=100, format="%.0f%%"
            ),
            "Median 1D": st.column_config.NumberColumn("Median 1D", format="%+.2f%%"),
            "Median 1M": st.column_config.NumberColumn("Median 1M", format="%+.2f%%"),
            "Median YTD": st.column_config.NumberColumn("Median YTD", format="%+.2f%%"),
        },
        hide_index=True,
    )

    gainers, losers = global_movers(snapshots)
    mover_columns = ["Ticker", "Company", "Universe", "1D", "1M", "Trend"]
    mover_config = {
        "1D": st.column_config.NumberColumn("1D", format="%+.2f%%"),
        "1M": st.column_config.NumberColumn("1M", format="%+.2f%%"),
        "Trend": st.column_config.LineChartColumn("3 months"),
    }
    gainers_column, losers_column = st.columns(2)
    with gainers_column:
        st.markdown("**Top gainers on the latest session**")
        selectable_table(gainers, key="global_gainers", column_order=mover_columns,
                         column_config=mover_config)
    with losers_column:
        st.markdown("**Top decliners on the latest session**")
        selectable_table(losers, key="global_losers", column_order=mover_columns,
                         column_config=mover_config)


if overview_loading():
    @st.fragment(run_every="3s")
    def live_overview() -> None:
        """Poll the background loader; rerun the page once everything is live."""
        if not overview_loading():
            st.rerun()
        render_overview()

    live_overview()
else:
    render_overview()
