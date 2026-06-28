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
from polycope.paper import PaperConfig, advance_portfolio, new_state, report
from polycope.pipeline import load_dataset

STATE_PATH = settings.data_dir / "paper_state.json"


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
         replay_days: int, reset: bool) -> int:
    settings.ensure_dirs()
    cfg = PaperConfig(initial_bankroll=bankroll, fee_bps=fee_bps)

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

    def pct(x):
        return f"{x*100:+.2f}%" if x == x else "n/a"

    print("\n" + "=" * 56)
    print("PAPER PORTFOLIO")
    print("=" * 56)
    print(f"equity           : ${rep['equity']:,.2f}  ({pct(rep['total_return'])})")
    print(f"cash             : ${rep['cash']:,.2f}")
    print(f"open positions   : {rep['open_positions']}  (${rep['open_cost']:,.2f} at risk)")
    print(f"settled          : {rep['n_settled']}")
    print(f"realized PnL     : ${rep['realized_pnl']:,.2f}  (pooled ROI {pct(rep['pooled_roi'])})")
    print(f"win rate         : {pct(rep['win_rate'])}")
    print(f"state -> {STATE_PATH}")
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
    raise SystemExit(main(**vars(ap.parse_args())))
