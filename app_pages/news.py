#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/news.py
#############################

"""News & sentiment: FinBERT-scored headlines across the selected universe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from runtime_config import RUN_IN_PROCESS_SENTIMENT
from sentiment_store import get_collector_status, load_sentiment_history
from ui_components import selectable_table
from ui_state import current_selection

selection = current_selection()
status = get_collector_status()
collector = "in this app, every 5 minutes" if RUN_IN_PROCESS_SENTIMENT else "by the sentiment worker"
st.caption(
    f"Headlines from Google News, scored by FinBERT and collected {collector}. "
    f"{status.get('article_count', 0)} articles stored for {status.get('watchlist_count', 0)} "
    "tracked stocks."
)
if not selection.tickers:
    st.info("Add stocks to your watchlist in the sidebar.", icon=":material/playlist_add:")
    st.stop()

now = pd.Timestamp.now(tz="UTC").tz_localize(None)


def mean_or_nan(values: object) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.mean()) if array.size else float("nan")

rows = []
headlines = []
for ticker in selection.tickers:
    articles = load_sentiment_history(ticker)
    if articles.empty:
        rows.append(
            {
                "Ticker": ticker,
                "Company": selection.companies.get(ticker, ticker),
                "Sentiment 24H": float("nan"),
                "Sentiment 7D": float("nan"),
                "Articles 24H": 0,
                "Articles 7D": 0,
                "Latest headline": "",
            }
        )
        continue
    published = pd.to_datetime(articles["published_at"])
    last_day = articles[published >= now - pd.Timedelta(days=1)]
    last_week = articles[published >= now - pd.Timedelta(days=7)]
    latest = articles.sort_values("published_at").iloc[-1]
    rows.append(
        {
            "Ticker": ticker,
            "Company": selection.companies.get(ticker, ticker),
            "Sentiment 24H": mean_or_nan(last_day["sentiment"]),
            "Sentiment 7D": mean_or_nan(last_week["sentiment"]),
            "Articles 24H": len(last_day),
            "Articles 7D": len(last_week),
            "Latest headline": str(latest["title"]),
        }
    )
    headlines.append(articles.assign(Company=selection.companies.get(ticker, ticker)))

summary = pd.DataFrame(rows).sort_values("Sentiment 7D", ascending=False, na_position="last")
covered = pd.DataFrame(summary.loc[summary["Articles 7D"] > 0])
st.markdown(f"**Sentiment by stock** ({len(covered)} of {len(summary)} with news in the last 7 days)")
selectable_table(
    summary,
    key="sentiment_summary",
    column_config={
        "Sentiment 24H": st.column_config.NumberColumn("Sentiment 24H", format="%+.2f"),
        "Sentiment 7D": st.column_config.NumberColumn("Sentiment 7D", format="%+.2f"),
    },
)

if not covered.empty:
    chart_rows = covered.dropna(subset=["Sentiment 7D"]).head(25)
    figure = go.Figure(
        go.Bar(
            x=chart_rows["Ticker"],
            y=chart_rows["Sentiment 7D"],
            marker_color=[
                "#2e7d32" if value > 0.05 else "#c62828" if value < -0.05 else "#8a8a8a"
                for value in chart_rows["Sentiment 7D"]
            ],
        )
    )
    figure.update_layout(
        template="plotly_white", height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10},
        yaxis_title="Average 7-day sentiment",
    )
    st.plotly_chart(figure, key="sentiment_bars")

if headlines:
    st.markdown("**Latest headlines**")
    combined = pd.concat(headlines, ignore_index=True).sort_values("published_at", ascending=False)
    st.dataframe(
        combined.head(200),
        column_order=["published_at", "ticker", "Company", "source", "title", "sentiment_label", "sentiment", "link"],
        column_config={
            "published_at": st.column_config.DatetimeColumn("Published", format="MMM DD, HH:mm"),
            "ticker": "Ticker",
            "source": "Source",
            "title": "Headline",
            "sentiment_label": "Label",
            "sentiment": st.column_config.NumberColumn("Score", format="%+.2f"),
            "link": st.column_config.LinkColumn("Link", display_text="Open"),
        },
        hide_index=True,
    )
else:
    st.info(
        "No headlines have been collected for this universe yet. The collector picks up "
        "newly viewed universes within a few minutes.",
        icon=":material/hourglass:",
    )
