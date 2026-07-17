#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_sentiment_pipeline.py
#############################

"""Offline tests for persistence, point-in-time features, and model comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sentiment_service
from app_config import SENTIMENT_FEATURE_COLUMNS
from forecasting import (
    build_forecast_training_frame,
    summarize_model_comparison,
)
from sentiment_analysis import SentimentScore, normalize_finbert_scores
from sentiment_features import build_sentiment_feature_frame
from sentiment_store import load_sentiment_history, save_news_sentiment


def _news_row(first_seen_at: str = "2026-01-03T14:00:00Z") -> dict:
    return {
        "ticker": "TEST",
        "content_hash": "abc123",
        "company_name": "Test plc",
        "published_at": "2026-01-03T12:00:00Z",
        "first_seen_at": first_seen_at,
        "last_seen_at": first_seen_at,
        "source": "Test News",
        "title": "Test company raises guidance",
        "summary": "Profit expectations increased.",
        "link": "https://example.test/story",
        "sentiment": 0.7,
        "sentiment_label": "Positive",
        "positive_probability": 0.8,
        "neutral_probability": 0.15,
        "negative_probability": 0.05,
        "model_name": "ProsusAI/finbert",
    }


class SentimentPipelineTest(unittest.TestCase):
    def test_finbert_probabilities_are_normalized(self) -> None:
        score = normalize_finbert_scores(
            [
                {"label": "positive", "score": 0.75},
                {"label": "negative", "score": 0.10},
                {"label": "neutral", "score": 0.15},
            ]
        )
        self.assertEqual("Positive", score.label)
        self.assertAlmostEqual(0.65, score.sentiment)

    def test_store_preserves_first_seen_time_on_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "sentiment.db"
            save_news_sentiment([_news_row()], db_path=database)
            updated = _news_row(first_seen_at="2026-01-04T14:00:00Z")
            updated["sentiment"] = 0.5
            save_news_sentiment([updated], db_path=database)

            stored = load_sentiment_history("TEST", db_path=database)
            self.assertEqual(1, len(stored))
            self.assertEqual(pd.Timestamp("2026-01-03 14:00:00"), stored["first_seen_at"].iloc[0])
            self.assertAlmostEqual(0.5, stored["sentiment"].iloc[0])

    def test_collection_deduplicates_before_scoring(self) -> None:
        candidate = {
            key: value
            for key, value in _news_row().items()
            if key
            in {
                "ticker",
                "content_hash",
                "company_name",
                "published_at",
                "first_seen_at",
                "last_seen_at",
                "source",
                "title",
                "summary",
                "link",
            }
        }

        class FakeAnalyzer:
            active_model_name = "fake-finbert"

            def __init__(self) -> None:
                self.scored_texts = 0

            def score_many(self, texts):
                text_list = list(texts)
                self.scored_texts += len(text_list)
                return [
                    SentimentScore("Positive", 0.7, 0.8, 0.15, 0.05, "fake-finbert")
                    for _ in text_list
                ]

        analyzer = FakeAnalyzer()
        original_fetch = sentiment_service.fetch_news_candidates
        original_analyzer = sentiment_service.get_sentiment_analyzer
        sentiment_service.fetch_news_candidates = lambda ticker, company: [candidate, dict(candidate)]
        sentiment_service.get_sentiment_analyzer = lambda: analyzer
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                database = Path(temporary_directory) / "sentiment.db"
                result = sentiment_service.collect_tickers_once(
                    {"TEST": "Test plc"},
                    db_path=database,
                )
                self.assertEqual(1, result["new_articles"])
                self.assertEqual(1, analyzer.scored_texts)
                self.assertEqual(1, len(load_sentiment_history("TEST", db_path=database)))
        finally:
            sentiment_service.fetch_news_candidates = original_fetch
            sentiment_service.get_sentiment_analyzer = original_analyzer

    def test_future_first_seen_time_is_excluded(self) -> None:
        price_history = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-01-03 13:00", "2026-01-03 15:00"]),
                "Close": [100.0, 101.0],
            }
        )
        sentiment_history = pd.DataFrame([_news_row()])
        features = build_sentiment_feature_frame(price_history, sentiment_history)

        self.assertEqual(0.0, features["news_count_24h"].iloc[0])
        self.assertEqual(1.0, features["news_count_24h"].iloc[1])
        self.assertAlmostEqual(0.7, features["sentiment_24h"].iloc[1])

    def test_sentiment_columns_join_training_frame(self) -> None:
        dates = pd.date_range("2025-01-01", periods=180, freq="B")
        close = 100 + np.linspace(0, 20, len(dates)) + np.sin(np.arange(len(dates)) / 5)
        price_history = pd.DataFrame(
            {
                "Date": dates,
                "Open": close - 0.2,
                "High": close + 0.8,
                "Low": close - 0.8,
                "Close": close,
                "Volume": 1_000_000 + (np.arange(len(dates)) * 1_000),
            }
        )
        sentiment_history = pd.DataFrame(
            {
                "published_at": dates + pd.Timedelta(hours=12),
                "first_seen_at": dates + pd.Timedelta(hours=12),
                "sentiment": np.sin(np.arange(len(dates)) / 7),
            }
        )
        training = build_forecast_training_frame(
            price_history,
            forecast_horizon=3,
            sentiment_history=sentiment_history,
            include_sentiment=True,
        )
        self.assertFalse(training.empty)
        self.assertTrue(set(SENTIMENT_FEATURE_COLUMNS).issubset(training.columns))

    def test_sentiment_is_promoted_only_when_mae_improves(self) -> None:
        common = {
            "date": pd.date_range("2026-01-01", periods=5),
            "actual_close": [100.0] * 5,
            "baseline_close": [98.0] * 5,
            "baseline_absolute_error": [2.0] * 5,
            "direction_correct": [True] * 5,
        }
        price = pd.DataFrame(
            {
                **common,
                "predicted_close": [99.0] * 5,
                "absolute_error": [1.0] * 5,
            }
        )
        sentiment = pd.DataFrame(
            {
                **common,
                "predicted_close": [99.5] * 5,
                "absolute_error": [0.5] * 5,
            }
        )
        comparison = summarize_model_comparison("TEST", price, sentiment, 3)
        self.assertEqual("Price + sentiment", comparison["Active model"])
        self.assertAlmostEqual(50.0, comparison["Sentiment MAE lift vs. price-only"])


if __name__ == "__main__":
    unittest.main()
