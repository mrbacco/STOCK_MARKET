#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tools/refresh_walk_forward.py
#############################

"""Re-run and store the production walk-forward test for many universes.

Run this after the production model changes, so every evidence badge in the
app describes the model that is actually ranking stocks. Run from the project
root:

    python tools/refresh_walk_forward.py --horizons 5 21
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


def refresh_universe(universe: str, horizons: list[int]) -> list[dict]:
    from walk_forward_store import run_and_store_walk_forward

    rows = []
    for horizon in horizons:
        started = time.perf_counter()
        try:
            summary = run_and_store_walk_forward(universe, horizon)
        except Exception as ex:
            rows.append({"Universe": universe, "Horizon": horizon, "Error": str(ex)})
            continue
        rows.append(
            {
                "Universe": universe,
                "Horizon": horizon,
                "Rank IC": summary.get("Rank IC"),
                "t-stat": summary.get("Rank IC t-stat"),
                "Top-N net excess": summary.get("Portfolio Cumulative net excess"),
                "Seconds": round(time.perf_counter() - started),
            }
        )
    return rows


def main() -> None:
    import pandas as pd

    from market_sources import MARKET_SOURCE_REGISTRY

    parser = argparse.ArgumentParser(description="Re-run and store production walk-forward tests.")
    parser.add_argument("--universes", nargs="+", default=list(MARKET_SOURCE_REGISTRY))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 21])
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(refresh_universe, universe, arguments.horizons): universe
            for universe in arguments.universes
        }
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"finished {futures[future]}", flush=True)
    results = pd.DataFrame(rows).sort_values(["Horizon", "Universe"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
