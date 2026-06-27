#!/usr/bin/env python3
"""Build positions, compute skill metrics, and rank traders by shrunk edge.

  python scripts/run_rank.py --synthetic
  python scripts/run_rank.py                 # uses ingested Parquet lake
"""

from __future__ import annotations

import argparse

import pandas as pd

from polycope.config import settings
from polycope.data.store import write_parquet
from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.model.ranking import rank_traders
from polycope.pipeline import load_dataset


def main(synthetic: bool, min_bets: int, top: int) -> int:
    trades, markets = load_dataset(synthetic=synthetic)
    positions = build_positions(trades, markets)
    metrics = trader_metrics(positions)
    ranked = rank_traders(metrics, min_bets=min_bets)

    settings.ensure_dirs()
    write_parquet(ranked, settings.data_dir / "ranking.parquet")

    pd.set_option("display.width", 140)
    cols = ["rank", "wallet", "n_bets", "hit_rate", "pooled_roi", "eb_edge", "eb_hit", "brier_skill"]
    show = ranked[ranked["eligible"]].head(top)
    print(f"\nTop {len(show)} traders by empirical-Bayes edge (min_bets={min_bets}):\n")
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFull ranking -> {settings.data_dir / 'ranking.parquet'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data instead of the lake")
    ap.add_argument("--min-bets", type=int, default=10)
    ap.add_argument("--top", type=int, default=15)
    raise SystemExit(main(**vars(ap.parse_args())))
