#!/usr/bin/env python3
"""Ingest live Polymarket data into the local Parquet lake.

  python scripts/run_ingest.py --wallets 200

Pulls the leaderboard -> candidate wallets -> full trade history, then fetches
resolution data from the CLOB API for every market that appears in the trades.
Requires outbound access to *.polymarket.com; if the environment's network policy
blocks it, this exits with a clear message (use the synthetic path in
run_rank/run_backtest meanwhile).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from polycope.config import settings
from polycope.data.ingest import ingest_leaderboard, ingest_wallets
from polycope.data.resolutions import ingest_markets


async def main(wallets: int) -> int:
    settings.ensure_dirs()
    try:
        addrs = await ingest_leaderboard(limit=wallets)
        print(f"leaderboard -> {len(addrs)} wallets")
        trades = await ingest_wallets(addrs, out_path=settings.raw_dir / "trades.parquet")
        print(f"trades -> {len(trades)} rows -> {settings.raw_dir / 'trades.parquet'}")
        condition_ids = trades["market_id"].unique().tolist()
        print(f"fetching resolution data for {len(condition_ids)} markets from CLOB API...")
        mk = await ingest_markets(condition_ids, out_path=settings.raw_dir / "markets.parquet")
        print(f"markets -> {len(mk)} rows -> {settings.raw_dir / 'markets.parquet'}")
    except (httpx.HTTPError, httpx.HTTPStatusError) as e:
        print(
            f"\nNetwork error reaching Polymarket: {e!r}\n"
            "If this is a 403/CONNECT denial, the environment's network policy is "
            "blocking *.polymarket.com. Open it, or use the --synthetic path in "
            "run_rank.py / run_backtest.py.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallets", type=int, default=200)
    raise SystemExit(asyncio.run(main(**vars(ap.parse_args()))))
