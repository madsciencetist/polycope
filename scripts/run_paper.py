#!/usr/bin/env python3
"""Forward paper-trade the copy cohort — no real money, legal anywhere.

Maintains a persistent simulated portfolio in data/paper_state.json. Run it
repeatedly over time; each run copies new BUYs from the cohort and settles
resolved positions, so live (paper) results accumulate across weeks.

Typical weekly loop:
  python scripts/run_ingest.py                 # refresh the lake
  python scripts/run_rank.py --min-bets 50 --top 15   # refresh copy-list
  python scripts/run_paper.py                  # advance the paper portfolio

First run starts the clock "now" (copies only future trades). To seed from recent
history for an immediate signal:
  python scripts/run_paper.py --replay-days 30 --reset

Monitor any time without advancing:
  python scripts/run_paper.py --status               # summary + per-wallet attribution
  python scripts/run_paper.py --positions --limit 50 # dump open positions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polycope.config import settings
from polycope.data.store import read_parquet
from polycope.features.trades import add_duration_bucket
from polycope.model.ranking import rank_traders, top_wallets
from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.paper import (
    PaperConfig,
    advance_portfolio,
    new_state,
    report,
    wallet_attribution,
)
from polycope.pipeline import load_dataset

STATE_PATH = settings.data_dir / "paper_state.json"


def _pct(x: float) -> str:
    return f"{x*100:+.2f}%" if x == x else "n/a"


def print_report(state: dict, rep: dict, top_n: int = 20) -> None:
    print("\n" + "=" * 64)
    print("PAPER PORTFOLIO")
    print("=" * 64)
    print(f"equity           : ${rep['equity']:,.2f}  ({_pct(rep['total_return'])})")
    print(f"cash             : ${rep['cash']:,.2f}")
    print(f"open positions   : {rep['open_positions']}  (${rep['open_cost']:,.2f} at risk, marked at cost)")
    print(f"settled          : {rep['n_settled']}")
    print(f"realized PnL     : ${rep['realized_pnl']:,.2f}  (pooled ROI {_pct(rep['pooled_roi'])})")
    print(f"win rate         : {_pct(rep['win_rate'])}")

    attr = wallet_attribution(state)
    if attr:
        print("\n--- per-wallet attribution (by realized PnL) ---")
        print(f"  {'wallet':<16} {'realized':>10} {'roi':>8} {'settled':>8} {'win%':>6} {'open':>5} {'open$':>10}")
        for r in attr[:top_n]:
            w = (r["wallet"] or "(untracked)")[:14]
            print(f"  {w:<16} {r['realized_pnl']:>+10.2f} {_pct(r['roi']):>8} "
                  f"{r['settled']:>8} {_pct(r['win_rate']):>6} {r['open']:>5} {r['open_cost']:>10.2f}")


def print_positions(state: dict, titles: dict[str, str], limit: int) -> None:
    """Dump open positions (largest first): wallet, market, entry, cost, age, status."""
    import time

    now = int(time.time())
    rows = sorted(state["open"], key=lambda p: p["cost"], reverse=True)
    print("\n" + "=" * 96)
    print(f"OPEN POSITIONS ({len(rows)} total, showing up to {limit}, largest first)")
    print("=" * 96)
    print(f"  {'wallet':<14} {'out':>3} {'entry':>6} {'cost':>9} {'age(d)':>7} {'status':>9}  market")
    for p in rows[:limit]:
        w = (p.get("wallet", "") or "?")[:12]
        ep = p.get("entry_price", float("nan"))
        age = (now - p["entry_ts"]) / 86400.0 if p.get("entry_ts") else float("nan")
        end_ts = p.get("end_ts", 0)
        status = "awaiting" if end_ts and end_ts <= now else "live"  # awaiting = past close, unresolved
        title = (titles.get(p["market_id"], "") or p["market_id"])[:46]
        ep_s = f"{ep:.2f}" if ep == ep else "  n/a"
        age_s = f"{age:.1f}" if age == age else "  n/a"
        print(f"  {w:<14} {p['outcome_index']:>3} {ep_s:>6} {p['cost']:>9.2f} {age_s:>7} {status:>9}  {title}")
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more (raise --limit to see them)")


def _load_copylist(top_k: int, min_bets: int) -> list[str]:
    """Top-k cohort: prefer a precomputed ranking.parquet, else rank on the fly."""
    rk = settings.data_dir / "ranking.parquet"
    if rk.exists():
        ranked = read_parquet(rk)
        return top_wallets(ranked, top_k, require_eligible=True)
    trades, markets = load_dataset(synthetic=False)
    ranked = rank_traders(trader_metrics(build_positions(trades, markets)), min_bets=min_bets)
    return top_wallets(ranked, top_k, require_eligible=True)


def main(top_k: int, min_bets: int, bankroll: float, fee_bps: float,
         replay_days: int, reset: bool, status: bool, positions: bool, limit: int) -> int:
    settings.ensure_dirs()
    cfg = PaperConfig(initial_bankroll=bankroll, fee_bps=fee_bps)

    # --status / --positions: read and display the saved portfolio; no fetch or advance.
    if status or positions:
        if not STATE_PATH.exists():
            print(f"No paper state at {STATE_PATH} — run a cycle first.")
            return 1
        state = json.loads(STATE_PATH.read_text())
        if status or not positions:
            print_report(state, report(state))
        if positions:
            titles: dict[str, str] = {}
            mk_path = settings.raw_dir / "markets.parquet"
            if mk_path.exists():
                mk = read_parquet(mk_path)
                titles = mk.set_index("market_id")["title"].to_dict()
            print_positions(state, titles, limit)
        print(f"\nstate: {STATE_PATH}  (last_ts={state['last_ts']})")
        return 0

    trades = read_parquet(settings.raw_dir / "trades.parquet")
    markets = read_parquet(settings.raw_dir / "markets.parquet")
    if "duration_bucket" not in markets or markets["duration_bucket"].eq("").all():
        markets = add_duration_bucket(markets)
    if trades.empty:
        print("No trades in the lake — run scripts/run_ingest.py first.")
        return 1

    wallets = _load_copylist(top_k, min_bets)
    print(f"copy-list: {len(wallets)} wallets (top-{top_k}, min_bets={min_bets})")

    now_max = int(trades["timestamp"].max())
    if reset or not STATE_PATH.exists():
        start = now_max - replay_days * 86400 if replay_days else now_max
        state = new_state(cfg, last_ts=start)
        mode = f"replay last {replay_days}d" if replay_days else "start now (future trades only)"
        print(f"initializing paper portfolio: {mode}  bankroll=${bankroll:,.0f}  fee={fee_bps:.0f}bps")
    else:
        state = json.loads(STATE_PATH.read_text())
        print(f"resuming paper portfolio from last_ts={state['last_ts']}")

    state, rep = advance_portfolio(state, trades, markets, wallets, cfg)
    STATE_PATH.write_text(json.dumps(state))

    print_report(state, rep)
    print(f"\nstate -> {STATE_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--min-bets", type=int, default=50)
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=100.0, help="entry taker fee (default 100 = Polycopy 1%)")
    ap.add_argument("--replay-days", type=int, default=0,
                    help="seed from this many days of recent history on init (0 = start now)")
    ap.add_argument("--reset", action="store_true", help="discard existing paper state and reinitialize")
    ap.add_argument("--status", action="store_true",
                    help="show the saved portfolio (incl. per-wallet attribution) without advancing")
    ap.add_argument("--positions", action="store_true",
                    help="dump open positions (market, entry, cost, age) without advancing")
    ap.add_argument("--limit", type=int, default=40, help="max open positions to print (--positions)")
    raise SystemExit(main(**vars(ap.parse_args())))
