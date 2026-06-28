#!/usr/bin/env python3
"""Compare position-sizing strategies for copy-trading — walk-forward, pooled ROI.

Why this shape:
  Ranking strategies on *compounded* equity is misleading.  Because each stake is
  a fraction of current cash, a strategy that wins big early compounds the winnings
  and runs away with the total return — an artifact of bet *ordering*, not of better
  sizing.  Walk-forward tests across several cutoffs showed the compounded "winner"
  changing window to window, and even *inverting* the order-independent pooled ROI
  (e.g. a strategy with 3x the per-dollar edge posting a lower compounded return).

  So this script headlines POOLED ROI = Sum(pnl) / Sum(cost) — the capital-weighted,
  order-independent edge per dollar — alongside MAX DRAWDOWN.  Compounded return is
  kept only as a footnote column.  It also runs a multi-cutoff WALK-FORWARD by
  default, because a single split is one coin-flip; a strategy worth choosing should
  hold up across windows.

  python scripts/run_sizing_comparison.py
  python scripts/run_sizing_comparison.py --cutoffs 2026-02-01,2026-04-01 --top-k 30
  python scripts/run_sizing_comparison.py --synthetic
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from polycope.backtest.engine import BacktestConfig, run_backtest
from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.model.ranking import rank_traders
from polycope.model.validate import evaluate_oos, time_split
from polycope.pipeline import load_dataset

DEFAULT_CUTOFFS = "2026-01-01,2026-02-01,2026-03-01,2026-04-01"


def _ts(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


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


def _strategies() -> list[tuple[str, BacktestConfig]]:
    return [
        ("Fixed 2%",          BacktestConfig(sizing="fixed")),
        ("EB-weighted",       BacktestConfig(sizing="eb_weighted")),
        ("Kelly (cap 10%)",   BacktestConfig(sizing="kelly", max_stake_fraction=0.10)),
        ("Proportional",      BacktestConfig(sizing="proportional")),
        ("Wallet-normalized", BacktestConfig(sizing="wallet_normalized")),
    ]


def _pooled_roi(ledger: pd.DataFrame) -> float:
    """Order-independent capital-weighted ROI: Sum(pnl) / Sum(cost)."""
    if ledger.empty or ledger["cost"].sum() <= 0:
        return float("nan")
    return float(ledger["pnl"].sum() / ledger["cost"].sum())


def _run_cutoff(positions, trades, markets, cutoff_ts, top_k, min_bets, metric):
    """Run every strategy at one cutoff. Returns (rows, oos) or (None, oos)."""
    oos = evaluate_oos(positions, top_k=top_k, min_bets=min_bets, metric=metric,
                       cutoff_ts=cutoff_ts)
    wallets = oos["eb_wallets"]
    if not wallets:
        return None, oos

    test_trades  = trades[trades["timestamp"].gt(cutoff_ts)]
    train_trades = trades[trades["timestamp"].le(cutoff_ts)]
    train, _ = time_split(positions, cutoff_ts=cutoff_ts)
    ranked = rank_traders(trader_metrics(train), min_bets=min_bets, metric=metric)
    wm = (ranked[ranked["wallet"].isin(wallets)]
          .set_index("wallet")[["eb_edge", "eb_hit"]].to_dict("index"))
    for w, stats in _notional_stats(train_trades, wallets).items():
        wm.setdefault(w, {}).update(stats)

    rows = []
    for name, cfg in _strategies():
        r = run_backtest(test_trades, positions, markets, wallets, cfg, wallet_metrics=wm)
        rows.append({
            "Strategy":   name,
            "pooled_roi": _pooled_roi(r["ledger"]),
            "max_dd":     r["max_drawdown"],
            "win_rate":   r["win_rate"],
            "n":          r["n_copied"],
            "comp_ret":   r["total_return"],
        })
    return rows, oos


def _fmt_cutoff_table(rows: list[dict], oos: dict, label: str) -> str:
    out = [
        f"\ncutoff {label}   train={oos['n_train_positions']}  test={oos['n_test_positions']}  "
        f"population pooled ROI={oos['population_test_roi']*100:+.1f}%",
        f"  {'Strategy':<18} {'POOLED ROI':>11} {'MAX DD':>8} {'win':>6} {'n':>7} {'(comp ret)':>11}",
    ]
    best = max((r["pooled_roi"] for r in rows if r["pooled_roi"] == r["pooled_roi"]), default=float("nan"))
    for r in rows:
        star = " *" if r["pooled_roi"] == best else "  "
        out.append(
            f"  {r['Strategy']:<18} {r['pooled_roi']*100:>+10.1f}%{star}"
            f" {r['max_dd']*100:>6.1f}% {r['win_rate']*100:>5.1f}%"
            f" {r['n']:>7} {r['comp_ret']*100:>+10.1f}%"
        )
    return "\n".join(out)


def main(top_k: int, min_bets: int, synthetic: bool, metric: str, cutoffs: str) -> int:
    trades, markets = load_dataset(synthetic=synthetic)
    positions = build_positions(trades, markets)

    cutoff_dates = [c.strip() for c in cutoffs.split(",") if c.strip()]

    print("=" * 78)
    print("SIZING STRATEGY WALK-FORWARD  (headline: pooled ROI = Sum(pnl)/Sum(cost))")
    print("=" * 78)
    print(f"top_k={top_k}  min_bets={min_bets}  metric={metric}  cutoffs={cutoff_dates}")
    print("Pooled ROI is order-independent; compounded return (rightmost) is a noisy")
    print("artifact of bet ordering and shown only for reference.  * = best pooled ROI.")

    # Accumulate per-strategy pooled ROI / DD across cutoffs for the summary.
    agg: dict[str, dict[str, list]] = {}
    n_windows = 0
    for cd in cutoff_dates:
        rows, oos = _run_cutoff(positions, trades, markets, _ts(cd), top_k, min_bets, metric)
        if rows is None:
            print(f"\ncutoff {cd}: no eligible wallets (need >= {min_bets} bets pre-cutoff)")
            continue
        n_windows += 1
        print(_fmt_cutoff_table(rows, oos, cd))
        for r in rows:
            a = agg.setdefault(r["Strategy"], {"pooled": [], "dd": [], "comp": []})
            a["pooled"].append(r["pooled_roi"])
            a["dd"].append(r["max_dd"])
            a["comp"].append(r["comp_ret"])

    if n_windows == 0:
        print("\nNo windows produced an eligible cohort; nothing to summarize.")
        return 1

    # Summary: rank by MEAN pooled ROI; show worst-window pooled ROI and mean DD.
    print("\n" + "=" * 78)
    print(f"SUMMARY ACROSS {n_windows} CUTOFFS  (ranked by mean pooled ROI)")
    print("=" * 78)
    print(f"  {'Strategy':<18} {'mean pooled':>12} {'worst window':>13} {'mean DD':>9} {'mean comp':>11}")
    summary = []
    for name, a in agg.items():
        summary.append((
            name,
            float(np.nanmean(a["pooled"])),
            float(np.nanmin(a["pooled"])),
            float(np.nanmean(a["dd"])),
            float(np.nanmean(a["comp"])),
        ))
    summary.sort(key=lambda t: t[1], reverse=True)
    for name, mean_p, worst_p, mean_dd, mean_c in summary:
        print(f"  {name:<18} {mean_p*100:>+11.1f}% {worst_p*100:>+12.1f}% "
              f"{mean_dd*100:>8.1f}% {mean_c*100:>+10.1f}%")
    print("=" * 78)
    print("\nPick by: highest mean pooled ROI that also holds up in its worst window,")
    print("then lowest mean drawdown.  Compounded return is NOT a selection criterion.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k",    type=int, default=15)
    ap.add_argument("--min-bets", type=int, default=50)
    ap.add_argument("--metric",   choices=["roi", "irr"], default="roi",
                    help="Edge metric for wallet ranking: roi (default) or irr (capital velocity)")
    ap.add_argument("--cutoffs",  default=DEFAULT_CUTOFFS,
                    help=f"Comma-separated calendar cutoffs YYYY-MM-DD (default: {DEFAULT_CUTOFFS})")
    raise SystemExit(main(**vars(ap.parse_args())))
