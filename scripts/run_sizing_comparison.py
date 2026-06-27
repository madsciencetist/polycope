#!/usr/bin/env python3
"""Compare position-sizing strategies for copy-trading.

Uses the same OOS train/test split as run_backtest.py so results are
directly comparable: wallets are selected on training data, signals are
from the test window only.

  python scripts/run_sizing_comparison.py [--synthetic] [--top-k N] [--min-bets N]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from polycope.backtest.engine import BacktestConfig, run_backtest
from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.model.ranking import rank_traders
from polycope.model.validate import evaluate_oos, time_split
from polycope.pipeline import load_dataset


def _calmar(total_return: float, max_dd: float) -> str:
    if max_dd <= 0:
        return "inf"
    return f"{total_return / max_dd:.2f}"


def main(top_k: int, min_bets: int, synthetic: bool) -> int:
    trades, markets = load_dataset(synthetic=synthetic)
    positions = build_positions(trades, markets)

    oos = evaluate_oos(positions, top_k=top_k, min_bets=min_bets)
    wallets = oos["eb_wallets"]
    cutoff = oos["cutoff_ts"]
    test_trades = trades[trades["timestamp"].gt(cutoff)]

    # Per-wallet metrics from the training split (needed for variable sizing).
    train, _ = time_split(positions)
    ranked = rank_traders(trader_metrics(train), min_bets=min_bets)
    wm = (
        ranked[ranked["wallet"].isin(wallets)]
        .set_index("wallet")[["eb_edge", "eb_hit"]]
        .to_dict("index")
    )

    strategies: list[tuple[str, BacktestConfig]] = [
        ("Fixed 2%",          BacktestConfig(sizing="fixed")),
        ("EB-weighted",       BacktestConfig(sizing="eb_weighted")),
        ("Kelly (cap 10%)",   BacktestConfig(sizing="kelly",        max_stake_fraction=0.10)),
        ("Kelly (cap 5%)",    BacktestConfig(sizing="kelly",        max_stake_fraction=0.05)),
        ("Proportional",      BacktestConfig(sizing="proportional")),
    ]

    rows = []
    for name, cfg in strategies:
        r = run_backtest(test_trades, positions, markets, wallets, cfg, wallet_metrics=wm)
        wr = r["win_rate"]
        rows.append(
            {
                "Strategy":       name,
                "Return":         f"{r['total_return'] * 100:+.1f}%",
                "Win rate":       f"{wr * 100:.1f}%" if wr == wr else "n/a",
                "Max DD":         f"{r['max_drawdown'] * 100:.1f}%",
                "Calmar":         _calmar(r["total_return"], r["max_drawdown"]),
                "N copied":       r["n_copied"],
                "Final equity":   f"${r['final_equity']:,.0f}",
            }
        )

    df = pd.DataFrame(rows).set_index("Strategy")

    print(f"\nOOS split: {oos['n_train_positions']} train positions / "
          f"{oos['n_test_positions']} test positions  "
          f"(cutoff ts={cutoff})")
    print(f"Copy list: {len(wallets)} wallets (top-{top_k} by EB edge, min {min_bets} bets)\n")
    print("=" * 72)
    print("SIZING STRATEGY COMPARISON  (test window, train-derived wallet list)")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)
    print("\nCalmar = total return / max drawdown  (higher = better risk-adjusted)")

    # Per-wallet breakdown for EB-weighted to show where capital concentrated.
    print("\n--- EB-weighted wallet weights ---")
    from polycope.backtest.engine import _build_sizing_tables, _signals
    sig = _signals(test_trades, wallets)
    eb_w, eb_h, _, eb_raw = _build_sizing_tables(sig, wallets, wm, BacktestConfig(sizing="eb_weighted"))
    for wallet, w in sorted(eb_w.items(), key=lambda x: -x[1]):
        edge = wm.get(wallet, {}).get("eb_edge", float("nan"))
        hit = wm.get(wallet, {}).get("eb_hit", float("nan"))
        print(f"  {wallet[:12]}...  weight={w:.3f}  eb_edge={edge:.4f}  eb_hit={hit:.3f}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-bets", type=int, default=10)
    raise SystemExit(main(**vars(ap.parse_args())))
