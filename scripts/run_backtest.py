#!/usr/bin/env python3
"""Validate edge persistence out-of-sample, then backtest copying the top cohort.

  python scripts/run_backtest.py --synthetic

Two outputs:
  1. OOS validation (the go/no-go gate): rank traders on older data, measure the
     top cohort vs. the population AND vs. a naive raw-PnL leaderboard on newer data.
  2. A friction-aware backtest of copying the top-ranked traders.
"""

from __future__ import annotations

import argparse

from polycope.backtest.engine import BacktestConfig, run_backtest
from polycope.backtest.report import format_report
from polycope.features.trades import build_positions
from polycope.model.validate import evaluate_oos
from polycope.pipeline import load_dataset


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%" if x == x else "n/a"


def main(synthetic: bool, top_k: int, min_bets: int) -> int:
    trades, markets = load_dataset(synthetic=synthetic)
    positions = build_positions(trades, markets)

    # ---- 1. Out-of-sample validation ----
    oos = evaluate_oos(positions, top_k=top_k, min_bets=min_bets)
    print("=" * 56)
    print("OUT-OF-SAMPLE VALIDATION (go/no-go gate)")
    print("=" * 56)
    print(f"train positions  : {oos['n_train_positions']}")
    print(f"test positions   : {oos['n_test_positions']}")
    print(f"population ROI    : {_pct(oos['population_test_roi'])}  (everyone, test window)")
    print(f"EB cohort ROI     : {_pct(oos['eb_cohort']['pooled_roi'])}  "
          f"(top {oos['eb_cohort']['n_wallets']} by shrunk edge)")
    print(f"raw-PnL cohort ROI: {_pct(oos['pnl_baseline_cohort']['pooled_roi'])}  "
          f"(top {oos['pnl_baseline_cohort']['n_wallets']} by leaderboard PnL)")
    edge = oos["eb_cohort"]["pooled_roi"]
    pop = oos["population_test_roi"]
    verdict = "PASS" if (edge == edge and pop == pop and edge > pop) else "FAIL"
    print(f"\nverdict: {verdict} — EB cohort {'beats' if verdict=='PASS' else 'does NOT beat'} "
          "the population out-of-sample.")

    # ---- 2. Friction-aware backtest (test window, train-derived wallets) ----
    # Wallet selection uses only training data (same cohort as OOS validation).
    # Simulated trades are restricted to the test window so no training-period
    # signal is replayed.
    wallets = oos["eb_wallets"]
    cutoff = oos["cutoff_ts"]
    test_trades = trades[trades["timestamp"].gt(cutoff)]
    print(f"\n(Backtest: {len(wallets)} wallets, {len(test_trades)} test-window signals)")
    result = run_backtest(test_trades, positions, markets, wallets, BacktestConfig())
    print(format_report(result))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-bets", type=int, default=10)
    raise SystemExit(main(**vars(ap.parse_args())))
