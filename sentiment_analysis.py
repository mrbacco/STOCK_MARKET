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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app_config import SENTIMENT_MODEL_NAME
from app_logging import bac_debug_kv, bac_debug_section, bac_log_kv, bac_log_section


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
    probability_by_label = {"Positive": positive, "Neutral": neutral, "Negative": negative}
    label = max(probability_by_label, key=lambda name: probability_by_label[name])
    score = SentimentScore(
        label=label,
        sentiment=positive - negative,
        positive_probability=positive,
        neutral_probability=neutral,
        negative_probability=negative,
        model_name=model_name,
    )
    bac_debug_kv(
        "sentiment.normalize_finbert_scores",
        model_name=model_name,
        label=score.label,
        sentiment=score.sentiment,
    )
    return score


def _model_is_cached(model_name: str) -> bool:
    """Return True when the model's config is already in the local Hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(model_name, "config.json"), str)
    except Exception:
        return False


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
            self._pipeline = self._load_pipeline(pipeline_factory, model_name)
            bac_log_kv("sentiment.analyzer", model=model_name, status="finbert_ready")
        except Exception as ex:
            self.load_error = str(ex)
            bac_log_kv(
                "sentiment.analyzer",
                model=model_name,
                status="vader_fallback",
                error=self.load_error,
            )

    @staticmethod
    def _load_pipeline(pipeline_factory: Callable[..., Any], model_name: str) -> Any:
        """Build the classifier, skipping the Hugging Face Hub when cached.

        A cached model loads with ``local_files_only`` so startup does not wait
        on Hub metadata requests (about two seconds per load). An incomplete
        cache retries online, which also downloads the missing files.
        """
        arguments = {
            "model": model_name,
            "tokenizer": model_name,
            "device": -1,
        }
        if _model_is_cached(model_name):
            try:
                return pipeline_factory(
                    "text-classification",
                    local_files_only=True,
                    **arguments,
                )
            except Exception as ex:
                bac_log_kv(
                    "sentiment.analyzer",
                    model=model_name,
                    status="local_load_failed_retrying_online",
                    error=str(ex),
                )
        return pipeline_factory("text-classification", **arguments)

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
        bac_debug_kv(
            "sentiment.analyzer.vader",
            text_length=len(text),
            label=score.label,
            sentiment=score.sentiment,
        )
        return score

    def score_many(self, texts: Iterable[str]) -> list[SentimentScore]:
        """Score a batch while serializing access to the shared model object."""
        clean_texts = [str(text).strip() for text in texts]
        bac_debug_kv(
            "sentiment.analyzer.score_many",
            incoming_count=len(clean_texts),
            active_model=self.active_model_name,
        )
        if not clean_texts:
            bac_debug_section("sentiment.analyzer.score_many", "Received an empty batch.")
            return []

        if self._pipeline is None:
            scores = [self._score_with_vader(text) for text in clean_texts]
            bac_debug_kv(
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
            bac_debug_kv(
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
            bac_debug_kv(
                "sentiment.analyzer.score_many",
                output_count=len(scores),
                mode="fallback_after_exception",
            )
            return scores


_SHARED_ANALYZER: FinancialSentimentAnalyzer | None = None
_SHARED_ANALYZER_LOCK = Lock()


def get_sentiment_analyzer() -> FinancialSentimentAnalyzer:
    """Load FinBERT once per process and share it across reruns and threads."""
    global _SHARED_ANALYZER
    with _SHARED_ANALYZER_LOCK:
        if _SHARED_ANALYZER is None:
            bac_log_section("sentiment.analyzer", "Loading the shared financial sentiment model.")
            _SHARED_ANALYZER = FinancialSentimentAnalyzer()
        return _SHARED_ANALYZER
