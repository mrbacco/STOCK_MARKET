#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: model_evidence.py
#############################

"""Judge whether the pooled ranking has demonstrated an edge.

When a multi-year walk-forward test exists for the universe and horizon, the
verdict uses it: thousands of out-of-sample dates say far more than the single
30-date evaluation window. Otherwise it falls back to that window. Either way
it looks at:

- rank IC: the average daily rank correlation between predicted and realized
  order, with its t-statistic (on non-overlapping dates for walk-forward);
- the realized excess return of the model's top picks (after costs for
  walk-forward).

Buy/avoid signals are shown only when the evidence is supported. Otherwise
the app still shows the model's scores, clearly labelled as unproven.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# A rank IC of 0.03 or more is commonly treated as economically useful for
# daily stock ranking; a t-statistic of 2 is the usual significance bar.
SUPPORTED_MIN_RANK_IC = 0.03
SUPPORTED_MIN_T_STAT = 2.0
TENTATIVE_MIN_T_STAT = 1.0
_UNAVAILABLE_CAVEAT = (
    "**Caution: this ranking cannot be validated yet.** There is not enough price history "
    "to test it.  \nUse it for research only, not as buy or sell advice."
)


@dataclass(frozen=True)
class EvidenceAssessment:
    """The ranking's evidence level and how to describe it."""

    # "supported", "tentative", "none", or "unavailable".
    level: str
    headline: str
    detail: str
    # Two-line caution shown above rankings; the full detail lives on Model health.
    caveat: str = ""

    @property
    def show_signals(self) -> bool:
        """Only a supported ranking may present buy/avoid signals."""
        return self.level == "supported"


def _number(diagnostics: Mapping[str, object], key: str) -> float:
    value = diagnostics.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _caveat(level: str, rank_ic: float, tested_years: float | None) -> str:
    """Two markdown lines: what the evidence shows, and how to use the ranking."""
    period = (
        f"Over {tested_years:.1f} years of out-of-sample testing"
        if tested_years is not None
        else "On the latest test window"
    )
    if level == "supported":
        return (
            f"**Validated edge.** {period} the ranking beat chance (rank IC {rank_ic:+.3f}).  \n"
            "Past performance can fade; keep position sizes modest."
        )
    if level == "tentative":
        return (
            f"**Caution: weak evidence only.** {period} the ranking beat chance only slightly "
            f"(rank IC {rank_ic:+.3f}).  \nTreat it as a research hint, not as buy or sell advice."
        )
    if tested_years is None:
        return (
            "**Caution: this ranking is unproven.** It has not beaten chance on recent data, and "
            "no multi-year test has been run yet.  \nUse it for research only; run the multi-year "
            "test on the Model health page."
        )
    return (
        f"**Caution: this ranking is unproven.** {period} it did not beat chance "
        f"(rank IC {rank_ic:+.3f}).  \nUse it for research only, not as buy or sell advice; "
        "details are on the Model health page."
    )


def _verdict(
    rank_ic: float,
    t_stat: float,
    top_excess: float,
    summary: str,
    tested_years: float | None = None,
) -> EvidenceAssessment:
    if (
        rank_ic >= SUPPORTED_MIN_RANK_IC
        and not math.isnan(t_stat)
        and t_stat >= SUPPORTED_MIN_T_STAT
        and top_excess > 0
    ):
        return EvidenceAssessment(
            "supported",
            "The ranking has a validated edge in this universe",
            summary,
            _caveat("supported", rank_ic, tested_years),
        )
    if rank_ic > 0 and not math.isnan(t_stat) and t_stat >= TENTATIVE_MIN_T_STAT:
        return EvidenceAssessment(
            "tentative",
            "Some evidence of an edge, not yet conclusive",
            summary + " Treat the ordering as a weak hint, not a signal.",
            _caveat("tentative", rank_ic, tested_years),
        )
    return EvidenceAssessment(
        "none",
        "No demonstrated edge in this universe",
        summary
        + " The model's ordering has not beaten chance here, so no buy or avoid"
        " signals are shown.",
        _caveat("none", rank_ic, tested_years),
    )


def assess_walk_forward_evidence(summary: Mapping[str, object]) -> EvidenceAssessment:
    """Classify a multi-year walk-forward result."""
    rank_ic = _number(summary, "Rank IC")
    if math.isnan(rank_ic):
        return EvidenceAssessment(
            "unavailable",
            "The walk-forward test produced no predictions",
            "Try a universe with longer price history.",
            _UNAVAILABLE_CAVEAT,
        )
    t_stat = _number(summary, "Rank IC t-stat")
    net_excess = _number(summary, "Portfolio Cumulative net excess")
    reversal = _number(summary, "Reversal rank IC")
    text = (
        f"Over {_number(summary, 'Test years'):.1f} years of walk-forward testing: rank IC "
        f"{rank_ic:+.3f} (t = {t_stat:.1f}), positive in "
        f"{_number(summary, 'Positive quarters'):.0f}% of quarters; the top "
        f"{int(_number(summary, 'Top-N'))} picks returned {net_excess:+.1f}% versus the "
        f"universe after costs. A simple 5-day reversal rule scored rank IC {reversal:+.3f}."
    )
    return _verdict(rank_ic, t_stat, net_excess, text, _number(summary, "Test years"))


def assess_ranking_evidence(
    diagnostics: Mapping[str, object],
    walk_forward: Mapping[str, object] | None = None,
) -> EvidenceAssessment:
    """Classify the ranking's evidence, preferring a multi-year walk-forward."""
    if walk_forward:
        return assess_walk_forward_evidence(walk_forward)
    rank_ic = _number(diagnostics, "Rank IC")
    t_stat = _number(diagnostics, "Rank IC t-stat")
    top_excess = _number(diagnostics, "Top-10 realized mean excess")
    raw_dates = _number(diagnostics, "Evaluation dates")
    evaluation_dates = 0 if math.isnan(raw_dates) else int(raw_dates)

    if not diagnostics or math.isnan(rank_ic):
        return EvidenceAssessment(
            "unavailable",
            "Not enough history to validate the ranking",
            "The model needs more price history before its ranking can be tested.",
            _UNAVAILABLE_CAVEAT,
        )

    summary = (
        f"Rank IC {rank_ic:+.3f} (t = {t_stat:.1f}) and top-10 excess return "
        f"{top_excess:+.2f}% over only {evaluation_dates} evaluation dates. Run the "
        "multi-year walk-forward test on the Model health page for a reliable verdict."
    )
    return _verdict(rank_ic, t_stat, top_excess, summary)
