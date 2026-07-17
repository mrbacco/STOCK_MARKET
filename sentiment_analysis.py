#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: sentiment_analysis.py
#############################

"""Financial-domain headline scoring with a resilient VADER fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Callable, Iterable

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app_config import SENTIMENT_MODEL_NAME
from app_logging import bac_log_kv, bac_log_section


@dataclass(frozen=True)
class SentimentScore:
    """Normalized probabilities and scalar sentiment for one headline."""

    label: str
    sentiment: float
    positive_probability: float
    neutral_probability: float
    negative_probability: float
    model_name: str

    def as_dict(self) -> dict[str, Any]:
        """Return a database-ready dictionary."""
        return asdict(self)


def normalize_finbert_scores(
    label_scores: Iterable[dict[str, Any]],
    model_name: str = SENTIMENT_MODEL_NAME,
) -> SentimentScore:
    """Convert FinBERT's three softmax labels to the app's common schema."""
    probabilities = {
        str(item.get("label", "")).lower(): float(item.get("score", 0.0))
        for item in label_scores
    }
    positive = probabilities.get("positive", 0.0)
    neutral = probabilities.get("neutral", 0.0)
    negative = probabilities.get("negative", 0.0)
    label = max(
        {"Positive": positive, "Neutral": neutral, "Negative": negative},
        key={"Positive": positive, "Neutral": neutral, "Negative": negative}.get,
    )
    score = SentimentScore(
        label=label,
        sentiment=positive - negative,
        positive_probability=positive,
        neutral_probability=neutral,
        negative_probability=negative,
        model_name=model_name,
    )
    bac_log_kv(
        "sentiment.normalize_finbert_scores",
        model_name=model_name,
        label=score.label,
        sentiment=score.sentiment,
    )
    return score


class FinancialSentimentAnalyzer:
    """Batch FinBERT scorer that falls back without stopping collection."""

    def __init__(
        self,
        model_name: str = SENTIMENT_MODEL_NAME,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._lock = Lock()
        self._vader = SentimentIntensityAnalyzer()
        self._pipeline: Any | None = None
        self.load_error = ""

        try:
            if pipeline_factory is None:
                from transformers import pipeline

                pipeline_factory = pipeline
            self._pipeline = pipeline_factory(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=-1,
            )
            bac_log_kv("sentiment.analyzer", model=model_name, status="finbert_ready")
        except Exception as ex:
            self.load_error = str(ex)
            bac_log_kv(
                "sentiment.analyzer",
                model=model_name,
                status="vader_fallback",
                error=self.load_error,
            )

    @property
    def active_model_name(self) -> str:
        """Report which model will score the next batch."""
        return self.model_name if self._pipeline is not None else "vader-fallback"

    def _score_with_vader(self, text: str) -> SentimentScore:
        probabilities = self._vader.polarity_scores(text)
        positive = float(probabilities["pos"])
        neutral = float(probabilities["neu"])
        negative = float(probabilities["neg"])
        compound = float(probabilities["compound"])
        label = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
        score = SentimentScore(
            label=label,
            sentiment=compound,
            positive_probability=positive,
            neutral_probability=neutral,
            negative_probability=negative,
            model_name="vader-fallback",
        )
        bac_log_kv(
            "sentiment.analyzer.vader",
            text_length=len(text),
            label=score.label,
            sentiment=score.sentiment,
        )
        return score

    def score_many(self, texts: Iterable[str]) -> list[SentimentScore]:
        """Score a batch while serializing access to the shared model object."""
        clean_texts = [str(text).strip() for text in texts]
        bac_log_kv(
            "sentiment.analyzer.score_many",
            incoming_count=len(clean_texts),
            active_model=self.active_model_name,
        )
        if not clean_texts:
            bac_log_section("sentiment.analyzer.score_many", "Received an empty batch.")
            return []

        if self._pipeline is None:
            scores = [self._score_with_vader(text) for text in clean_texts]
            bac_log_kv(
                "sentiment.analyzer.score_many",
                output_count=len(scores),
                mode="vader_only",
            )
            return scores

        try:
            with self._lock:
                outputs = self._pipeline(
                    clean_texts,
                    top_k=None,
                    truncation=True,
                    batch_size=min(8, len(clean_texts)),
                )
            if outputs and isinstance(outputs[0], dict):
                outputs = [outputs]
            scores = [normalize_finbert_scores(output, self.model_name) for output in outputs]
            bac_log_kv(
                "sentiment.analyzer.score_many",
                output_count=len(scores),
                mode="finbert",
            )
            return scores
        except Exception as ex:
            bac_log_kv(
                "sentiment.analyzer",
                model=self.model_name,
                status="batch_fallback",
                error=str(ex),
            )
            scores = [self._score_with_vader(text) for text in clean_texts]
            bac_log_kv(
                "sentiment.analyzer.score_many",
                output_count=len(scores),
                mode="fallback_after_exception",
            )
            return scores


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> FinancialSentimentAnalyzer:
    """Load FinBERT once per process and share it across Streamlit reruns."""
    bac_log_section("sentiment.analyzer", "Loading the shared financial sentiment model.")
    return FinancialSentimentAnalyzer()
