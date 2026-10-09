#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_ranking_store.py
#############################

"""Tests of the day-long ranking store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ranking_store import clear_rankings, load_ranking, ranking_cache_key, save_ranking


def _prices(days: int) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-01", periods=days)
    return {ticker: pd.DataFrame({"Date": dates, "Close": 1.0}) for ticker in ("A", "B")}


class RankingStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "rankings.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_a_new_bar_or_another_model_changes_the_key(self) -> None:
        key = ranking_cache_key("dax40", 21, "base", _prices(100))
        self.assertEqual(key, ranking_cache_key("dax40", 21, "base", _prices(100)))
        self.assertNotEqual(key, ranking_cache_key("dax40", 21, "base", _prices(101)))
        self.assertNotEqual(key, ranking_cache_key("dax40", 5, "base", _prices(100)))
        self.assertNotEqual(key, ranking_cache_key("dax40", 21, "base+ranks", _prices(100)))

    def test_round_trip_and_clear(self) -> None:
        ranking = pd.DataFrame(
            {"Ticker": ["A", "B"], "Rank": [1, 2], "Date": pd.to_datetime(["2026-01-02"] * 2)}
        )
        evaluation = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01"]), "value": [0.5]})
        diagnostics = {"Rank IC": 0.04, "Model weights": {"Ridge": 1.0}}
        save_ranking("k", "dax40", ranking=ranking, evaluation=evaluation,
                     diagnostics=diagnostics, db_path=self.db_path)
        loaded = load_ranking("k", db_path=self.db_path)
        assert loaded is not None
        pd.testing.assert_frame_equal(ranking, loaded[0], check_dtype=False)
        pd.testing.assert_frame_equal(evaluation, loaded[1], check_dtype=False)
        self.assertEqual(diagnostics, loaded[2])
        clear_rankings("dax40", db_path=self.db_path)
        self.assertIsNone(load_ranking("k", db_path=self.db_path))


if __name__ == "__main__":
    unittest.main()
