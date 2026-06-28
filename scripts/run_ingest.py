#!/usr/bin/env python3
"""Ingest live Polymarket data into the local Parquet lake.

  python scripts/run_ingest.py                     # 1000 wallets, min PnL $1000
  python scripts/run_ingest.py --wallets 2000      # wider pool
  python scripts/run_ingest.py --min-pnl 5000      # higher quality bar

Pulls the leaderboard (paginated across all time windows) -> candidate wallets
-> full trade history, then fetches resolution data from the CLOB API for every
market that appears in the trades.

The leaderboard returns 50 rows per page regardless of the limit parameter.
--wallets controls how many unique addresses to collect before stopping pagination.
--min-pnl filters out dust/bot accounts before spending API quota on their history.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from polycope.config import settings
from polycope.data.ingest import ingest_leaderboard, ingest_wallets
from polycope.data.resolutions import ingest_markets


async def main(wallets: int, min_pnl: float) -> int:
    settings.ensure_dirs()
    try:
        addrs = await ingest_leaderboard(n=wallets, min_pnl=min_pnl)
        if not addrs:
            print("No wallets found — check network access or lower --min-pnl", file=sys.stderr)
            return 1
        print(f"fetching trade history for {len(addrs)} wallets (batches of 100)...")
        trades = await ingest_wallets(addrs, out_path=settings.raw_dir / "trades.parquet")
        print(f"trades -> {len(trades):,} rows  ({trades['wallet'].nunique()} wallets with data)")
        condition_ids = trades["market_id"].unique().tolist()
        print(f"fetching resolution data for {len(condition_ids):,} markets from CLOB API...")
        mk = await ingest_markets(condition_ids, out_path=settings.raw_dir / "markets.parquet")
        print(f"markets -> {len(mk):,} rows -> {settings.raw_dir / 'markets.parquet'}")
    except (httpx.HTTPError, httpx.HTTPStatusError) as e:
        print(
            f"\nNetwork error reaching Polymarket: {e!r}\n"
            "If this is a 403/CONNECT denial, the environment's network policy is "
            "blocking *.polymarket.com.  Use --synthetic in run_rank.py / run_backtest.py.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallets",  type=int,   default=2000,
                    help="Max unique wallet addresses to collect from leaderboard (default: 1000)")
    ap.add_argument("--min-pnl",  type=float, default=100.0,
                    help="Skip wallets with lifetime PnL below this (default: $1000)")
    raise SystemExit(asyncio.run(main(**vars(ap.parse_args()))))
