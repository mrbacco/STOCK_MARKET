#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tools/aligned_comparison.py
#############################

"""Compare two feature variants on exactly the same walk-forward test dates.

Variants with long-lookback factors start testing later than the baseline, so
their headline numbers cover different periods. This script runs both, keeps
only the dates both predicted, and compares rank IC and top-N returns there.

    python tools/aligned_comparison.py --baseline V0 --candidate V2 --horizons 5 21
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

from walk_forward_experiment import VARIANTS  # noqa: E402  (same directory)


def compare_universe(universe: str, horizons: list[int], baseline: str, candidate: str) -> list[dict]:
    import numpy as np
    import pandas as pd

    from market_data import classify_price_histories, get_price_history_batch
    from market_model import PanelConfig
    from market_sources import MARKET_SOURCE_REGISTRY
    from sentiment_store import load_sentiment_history
    from walk_forward import run_walk_forward, summarize_walk_forward

    prices = get_price_history_batch(MARKET_SOURCE_REGISTRY[universe].tickers, period="5y", interval="1d")
    health = classify_price_histories(prices, realtime_mode=False)
    live = {ticker: prices[ticker] for ticker in health.live_tickers}
    sentiment = {ticker: load_sentiment_history(ticker) for ticker in live}
    rows = []
    for horizon in horizons:
        predictions = {
            name: run_walk_forward(
                live, sentiment, horizon=horizon, config=PanelConfig(**VARIANTS[name])
            ).predictions
            for name in (baseline, candidate)
        }
        shared = set(predictions[baseline]["Date"]) & set(predictions[candidate]["Date"])
        for name, frame in predictions.items():
            aligned = pd.DataFrame(frame.loc[frame["Date"].isin(sorted(shared))])
            summary = summarize_walk_forward(aligned, horizon=horizon, top_n=10, cost_bps=10.0)
            rows.append(
                {
                    "Universe": universe,
                    "Horizon": horizon,
                    "Variant": name,
                    "Shared dates": len(shared),
                    "Rank IC": summary.get("Rank IC", np.nan),
                    "t-stat": summary.get("Rank IC t-stat", np.nan),
                    "Positive quarters": summary.get("Positive quarters", np.nan),
                    "Top-N net excess": summary.get("Portfolio Cumulative net excess", np.nan),
                    "Bottom-N excess": summary.get("Portfolio Cumulative bottom-N excess", np.nan),
                }
            )
    return rows


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Compare two feature variants on the same walk-forward test dates."
    )
    parser.add_argument("--universes", nargs="+", default=["eurostoxx50", "dow30", "nasdaq_leaders", "ftsemib", "iseq20"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 21])
    parser.add_argument("--baseline", default="V0")
    parser.add_argument("--candidate", default="V2")
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = [
            executor.submit(compare_universe, universe, arguments.horizons, arguments.baseline, arguments.candidate)
            for universe in arguments.universes
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    results = pd.DataFrame(rows).sort_values(["Horizon", "Universe", "Variant"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.round(3).to_string(index=False))
        print("\nMean rank IC on shared dates:")
        print(results.pivot_table(index="Variant", columns="Horizon", values="Rank IC").round(4))


if __name__ == "__main__":
    main()
