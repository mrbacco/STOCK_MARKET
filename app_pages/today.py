#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: app_pages/today.py
#############################

"""Today: the market's mood and a few highlighted ideas, in plain language."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from call_tracker import BEATS, LAGS, Call, check_calls, live_record, record_calls
from ideas import Idea, market_mood, pick_ideas
from market_sources import universe_label
from model_evidence import assess_walk_forward_evidence
from runtime_config import ANALYTICS_READ_ONLY
from ui_state import (
    HOLDING_PERIODS,
    MIN_BASKET_HIT_RATE,
    basket_hit_rate,
    current_selection,
    market_analysis,
    markets_with_evidence,
    open_stock,
    track_record,
    walk_forward_summary,
)

RISK_BADGES = {
    "high": ":red-badge[:material/bolt: Big price swings]",
    "low": ":green-badge[:material/water: Steady price]",
    "": "",
}
MOOD_STYLE = {
    "positive": (st.success, ":material/trending_up:"),
    "mixed": (st.info, ":material/trending_flat:"),
    "weak": (st.warning, ":material/trending_down:"),
    "nervous": (st.error, ":material/warning:"),
    "unknown": (st.info, ":material/hourglass:"),
}

selection = current_selection()
period = HOLDING_PERIODS.get(selection.horizon, f"{selection.horizon} sessions")
st.subheader(f"Today in {selection.label}")
if not selection.tickers:
    st.info("Add stocks to your watchlist in the sidebar.", icon=":material/playlist_add:")
    st.stop()

analysis = market_analysis(selection)
if analysis is None or not analysis.prices.valid_tickers:
    st.error(
        "No price history is available for this market. Check your connection, then refresh.",
        icon=":material/error:",
    )
    st.stop()

prices = {ticker: analysis.prices.price_data[ticker] for ticker in analysis.prices.valid_tickers}
mood = market_mood(prices)
show_mood, mood_icon = MOOD_STYLE[mood.level]
show_mood(f"**Market mood: {mood.headline}.** {mood.detail}", icon=mood_icon)


def _switch_market(universe_key: str) -> None:
    st.session_state["universe"] = universe_key


def render_alternatives() -> None:
    """Point to markets where the current model has tested evidence."""
    alternatives = [
        key for key, _ic in markets_with_evidence(selection.horizon)
        if key != selection.universe_key
    ]
    if not alternatives:
        st.caption(
            "No market has tested evidence yet for this holding period. Try the other "
            "holding period in the sidebar."
        )
        return
    st.markdown("Markets where the model has tested evidence right now:")
    with st.container(horizontal=True):
        for key in alternatives[:5]:
            st.button(
                universe_label(key),
                key=f"switch_{key}",
                on_click=_switch_market,
                args=(key,),
                icon=":material/swap_horiz:",
            )


if selection.is_watchlist:
    st.info(
        "Ideas come from ranking a whole market, so they are not available for a watchlist. "
        "Choose a market in the sidebar, or open your stocks on the Stock page.",
        icon=":material/info:",
    )
    st.stop()

ranking = analysis.ranking
if ranking.ranking.empty:
    st.info(
        "The analytics worker is still preparing this market. Check back in a few minutes."
        if ANALYTICS_READ_ONLY
        else "There is not enough price history to rank this market.",
        icon=":material/hourglass:",
    )
    st.stop()

# Predictions need the multi-year test, and a top-3 basket that beat the
# market often enough there; the 30-date window alone is too short to trust.
tested = walk_forward_summary(selection)
evidence = assess_walk_forward_evidence(tested) if tested else None
record = track_record(selection)
hit_rate = basket_hit_rate(record)
if evidence is None or evidence.level not in ("supported", "tentative"):
    reason = (
        "The current model has not been tested on this market yet."
        if evidence is None
        else "In years of testing, the model's picks here did no better than chance."
    )
elif hit_rate < MIN_BASKET_HIT_RATE:
    reason = (
        f"In testing, its top 3 beat the market in only {int(record.get('Basket right', 0))} "
        f"of {int(record.get('Periods', 0))} months, close to a coin flip."
    )
else:
    reason = ""
if reason:
    st.warning(
        f"**No reliable ideas in {selection.label} for a {period} hold.** {reason} "
        "Rather than show a guess, this page stays quiet.",
        icon=":material/do_not_disturb_on:",
    )
    render_alternatives()
    st.stop()

best, worst = pick_ideas(ranking.ranking, prices, selection.companies)
market_name = selection.label.split(" - ")[-1]
badge = (
    ":green-badge[:material/verified: Tested edge]"
    if evidence is not None and evidence.level == "supported"
    else ":orange-badge[:material/science: Early evidence, not proven]"
)

# Every call shown here is saved and checked against the market once its
# holding period is over, so the page can show a live record next to the test.
as_of = pd.Timestamp(max(pd.to_datetime(frame["Date"]).max() for frame in prices.values()))
check_calls(selection.universe_key, selection.horizon, prices)
record_calls(
    selection.universe_key,
    selection.horizon,
    as_of,
    [Call(idea.ticker, BEATS, idea.expected_excess) for idea in best]
    + [Call(idea.ticker, LAGS, idea.expected_excess) for idea in worst],
)
live = live_record(selection.universe_key, selection.horizon)

st.markdown(f"#### Prediction for the next {period} &nbsp; {badge}")
with st.container(border=True):
    if best:
        names = ", ".join(f"**{idea.company}**" for idea in best[:-1])
        names = f"{names} and **{best[-1].company}**" if names else f"**{best[-1].company}**"
        st.markdown(f"##### {names} should beat the {market_name}.")
    else:
        st.markdown(f"##### No stock in the {market_name} is expected to beat it right now.")
    if record:
        periods = int(record["Periods"])
        st.markdown(
            f":material/history: **Track record in testing:** the model's top "
            f"{int(record['Basket size'])} beat the market in **{int(record['Basket right'])} of "
            f"{periods} months** ({100 * record['Basket right'] / periods:.0f}%), by "
            f"{record['Basket mean excess']:+.1f}% on average. Its #1 pick did in "
            f"{int(record['Top pick right'])} of {periods}."
        )
    if live.checked:
        st.markdown(
            f":material/fact_check: **Live record:** {live.right} of {live.checked} calls "
            f"checked so far came true."
        )
    elif live.next_check_after is not None:
        first_result = live.next_check_after + pd.offsets.BDay(selection.horizon)
        st.markdown(
            f":material/fact_check: **Live record:** {live.made} calls saved; the first are "
            f"checked around {first_result:%d %b %Y}."
        )


def render_idea(idea: Idea) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{idea.company}** &nbsp; :gray[{idea.ticker}] &nbsp; {RISK_BADGES[idea.risk]}"
        )
        st.metric(
            f"Prediction vs the market, {period}",
            f"{idea.expected_excess:+.1f}%",
            help="Predicted return minus the average stock in this market.",
        )
        st.markdown(f"Likely range: **{idea.low:+.1f}%** to **{idea.high:+.1f}%**")
        if idea.reasons:
            st.markdown("\n".join(f"- {reason}" for reason in idea.reasons))
        if st.button("See details", key=f"open_{idea.ticker}", icon=":material/arrow_forward:"):
            open_stock(idea.ticker)


columns = st.columns(max(len(best), 1))
for column, idea in zip(columns, best):
    with column:
        render_idea(idea)

if worst:
    names = ", ".join(
        f"**{idea.company}** ({idea.expected_excess:+.1f}%)" for idea in worst
    )
    st.markdown(f":material/front_hand: **Should lag the market:** {names}.")

st.caption(
    "A prediction here means beating or lagging the average stock in this market over the "
    "holding period, not a guaranteed price. The likely range covers 8 outcomes in 10. "
    "Spread your money across several ideas and keep each one small."
)
