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


def _notional_stats(trades: pd.DataFrame, wallets: list[str]) -> dict[str, dict]:
    """Per-wallet mean and std of bet notional from the supplied trade slice."""
    buy = trades[trades["side"].eq("BUY") & trades["wallet"].isin(wallets)].copy()
    if buy.empty or "size" not in buy.columns:
        return {}
    buy["_notional"] = buy["price"] * buy["size"]
    grp = buy.groupby("wallet")["_notional"]
    means = grp.mean()
    stds  = grp.std(ddof=1).fillna(0)
    return {
        w: {"mean_notional": float(means.get(w, 1.0)), "std_notional": float(stds.get(w, 0.0))}
        for w in wallets
    }


def main(top_k: int, min_bets: int, synthetic: bool, metric: str) -> int:
    trades, markets = load_dataset(synthetic=synthetic)
    positions = build_positions(trades, markets)

    oos = evaluate_oos(positions, top_k=top_k, min_bets=min_bets, metric=metric)
    wallets  = oos["eb_wallets"]
    cutoff   = oos["cutoff_ts"]
    test_trades  = trades[trades["timestamp"].gt(cutoff)]
    train_trades = trades[trades["timestamp"].le(cutoff)]

    # Per-wallet EB metrics from the training split.
    train, _ = time_split(positions)
    ranked = rank_traders(trader_metrics(train), min_bets=min_bets, metric=metric)
    wm = (
        ranked[ranked["wallet"].isin(wallets)]
        .set_index("wallet")[["eb_edge", "eb_hit"]]
        .to_dict("index")
    )

    # Inject training-period notional stats so wallet_normalized uses no test data.
    for w, stats in _notional_stats(train_trades, wallets).items():
        wm.setdefault(w, {}).update(stats)

    strategies: list[tuple[str, BacktestConfig]] = [
        ("Fixed 2%",           BacktestConfig(sizing="fixed")),
        ("EB-weighted",        BacktestConfig(sizing="eb_weighted")),
        ("Kelly (cap 10%)",    BacktestConfig(sizing="kelly",             max_stake_fraction=0.10)),
        ("Kelly (cap 5%)",     BacktestConfig(sizing="kelly",             max_stake_fraction=0.05)),
        ("Proportional",       BacktestConfig(sizing="proportional")),
        ("Wallet-normalized",  BacktestConfig(sizing="wallet_normalized")),
    ]

    rows = []
    for name, cfg in strategies:
        r  = run_backtest(test_trades, positions, markets, wallets, cfg, wallet_metrics=wm)
        wr = r["win_rate"]
        rows.append(
            {
                "Strategy":     name,
                "Return":       f"{r['total_return'] * 100:+.1f}%",
                "Win rate":     f"{wr * 100:.1f}%" if wr == wr else "n/a",
                "Max DD":       f"{r['max_drawdown'] * 100:.1f}%",
                "Calmar":       _calmar(r["total_return"], r["max_drawdown"]),
                "N copied":     r["n_copied"],
                "Final equity": f"${r['final_equity']:,.0f}",
            }
        )

    df = pd.DataFrame(rows).set_index("Strategy")

    print(f"\nOOS split: {oos['n_train_positions']} train positions / "
          f"{oos['n_test_positions']} test positions  "
          f"(cutoff ts={cutoff})")
    print(f"Copy list: {len(wallets)} wallets (top-{top_k} by EB {metric} edge, min {min_bets} bets)\n")
    print("=" * 72)
    print("SIZING STRATEGY COMPARISON  (test window, train-derived wallet list)")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)
    print("\nCalmar = total return / max drawdown  (higher = better risk-adjusted)")

    # Show how wallet-normalized conviction varies per-wallet.
    print("\n--- Training-period notional stats (wallet_normalized inputs) ---")
    print(f"  {'Wallet':<16}  {'Mean notional':>14}  {'Std notional':>13}  {'CV':>6}")
    for w in wallets:
        mu  = wm.get(w, {}).get("mean_notional", float("nan"))
        sig = wm.get(w, {}).get("std_notional",  float("nan"))
        cv  = sig / mu if mu else float("nan")
        print(f"  {w[:14]:<16}  {mu:>14.2f}  {sig:>13.2f}  {cv:>6.2f}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k",    type=int, default=15)
    ap.add_argument("--min-bets", type=int, default=10)
    ap.add_argument("--metric",   choices=["roi", "irr"], default="roi",
                    help="Edge metric for wallet ranking: roi (default) or irr (capital velocity)")
    raise SystemExit(main(**vars(ap.parse_args())))
