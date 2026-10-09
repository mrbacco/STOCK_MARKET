#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tools/walk_forward_experiment.py
#############################

"""Compare pooled-model feature variants with the multi-year walk-forward test.

Every variant is evaluated on the same universes, horizons, and five-year
history, so the comparison is apples to apples. Results are printed and saved
as CSV under data/experiments/. Run from the project root:

    python tools/walk_forward_experiment.py --universes eurostoxx50 dow30 \
        --horizons 5 21 --variants V0 V1 V2 V3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Several processes share the CPU; keep each one's numeric threads modest.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

VARIANTS = {
    "V0": {},
    "V1": {"factor_features": True},
    "V2": {"factor_features": True, "cross_sectional_ranks": True},
    "V3": {"factor_features": True, "cross_sectional_ranks": True, "turbulence": True},
}
REPORTED = (
    "Rank IC",
    "Rank IC t-stat",
    "Positive quarters",
    "Portfolio Cumulative net excess",
    "Portfolio Cumulative bottom-N excess",
    "Portfolio Information ratio",
    "Momentum rank IC",
    "Reversal rank IC",
    "Test years",
)


def run_universe(universe: str, horizons: list[int], variants: list[str]) -> list[dict]:
    """Load one universe once, then run every variant and horizon."""
    from market_data import classify_price_histories, get_price_history_batch
    from market_model import PanelConfig
    from market_sources import MARKET_SOURCE_REGISTRY
    from sentiment_store import load_sentiment_history
    from walk_forward import run_walk_forward

    prices = get_price_history_batch(MARKET_SOURCE_REGISTRY[universe].tickers, period="5y", interval="1d")
    health = classify_price_histories(prices, realtime_mode=False)
    live = {ticker: prices[ticker] for ticker in health.live_tickers}
    sentiment = {ticker: load_sentiment_history(ticker) for ticker in live}
    rows = []
    for horizon in horizons:
        for variant in variants:
            started = time.perf_counter()
            result = run_walk_forward(
                live, sentiment, horizon=horizon, config=PanelConfig(**VARIANTS[variant])
            )
            rows.append(
                {
                    "Universe": universe,
                    "Horizon": horizon,
                    "Variant": variant,
                    **{key: result.summary.get(key) for key in REPORTED},
                    "Seconds": round(time.perf_counter() - started),
                }
            )
    return rows


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Compare pooled-model feature variants with the walk-forward test."
    )
    parser.add_argument("--universes", nargs="+", default=["eurostoxx50", "dow30", "nasdaq_leaders", "ftsemib"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 21])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(run_universe, universe, arguments.horizons, arguments.variants): universe
            for universe in arguments.universes
        }
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"finished {futures[future]}", flush=True)

    results = pd.DataFrame(rows).sort_values(["Horizon", "Universe", "Variant"])
    output = PROJECT_ROOT / "data" / "experiments"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"walk_forward_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.round(3).to_string(index=False))
        print("\nMean rank IC by horizon and variant:")
        print(results.pivot_table(index="Variant", columns="Horizon", values="Rank IC", aggfunc="mean").round(4))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
