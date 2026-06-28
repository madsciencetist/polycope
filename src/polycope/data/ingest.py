"""Normalize raw Polymarket API responses into the canonical lake schema.

The public API's exact field names drift over time and differ slightly between
endpoints, so the normalizers are deliberately tolerant: they accept a list of
candidate keys and coerce types. This keeps the rest of the pipeline stable and
makes the normalizers unit-testable without network access.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pandas as pd

from ..schema import TRADE_COLUMNS
from .client import PolymarketClient
from .store import write_parquet


def _first(d: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_trade(raw: dict) -> dict | None:
    """Map one raw /trades or /activity TRADE record to the canonical trade row."""
    side = str(_first(raw, ("side", "type", "action"), "")).upper()
    if side in {"", "TRADE"}:
        side = "BUY"  # /activity TRADE rows use a separate field; default conservatively
    price = _first(raw, ("price", "avgPrice", "fillPrice"))
    size = _first(raw, ("size", "shares", "amount", "quantity"))
    if price is None or size is None:
        return None
    return {
        "wallet": str(_first(raw, ("proxyWallet", "user", "wallet", "maker", "address"), "")).lower(),
        "market_id": str(_first(raw, ("conditionId", "market", "marketId", "condition_id"), "")),
        "outcome_index": int(_first(raw, ("outcomeIndex", "outcome_index", "outcome"), 0) or 0),
        "side": "SELL" if side.startswith("SELL") else "BUY",
        "price": float(price),
        "size": float(size),
        "timestamp": int(_first(raw, ("timestamp", "ts", "time", "matchTime"), 0) or 0),
        "tx_hash": str(_first(raw, ("transactionHash", "txHash", "hash"), "")),
    }


def trades_to_frame(raw_trades: list[dict]) -> pd.DataFrame:
    rows = [r for r in (normalize_trade(t) for t in raw_trades) if r is not None]
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if not df.empty:
        df = df[(df["price"] >= 0) & (df["price"] <= 1) & (df["size"] > 0)]
    return df.reset_index(drop=True)


def extract_wallets(leaderboard_rows: list[dict]) -> list[str]:
    out = []
    for row in leaderboard_rows:
        addr = _first(row, ("proxyWallet", "wallet", "address", "user", "proxy_address"))
        if addr:
            out.append(str(addr).lower())
    # de-dupe, preserve order
    return list(dict.fromkeys(out))


async def ingest_wallets(
    wallets: list[str],
    cfg=None,
    out_path=None,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Fetch full trade history for each wallet and persist a single trades table.

    Processes wallets in batches so we can print progress and write partial
    results incrementally rather than holding everything in memory at once.
    """
    async with PolymarketClient(cfg) as client:
        frames: list[pd.DataFrame] = []
        for i in range(0, len(wallets), batch_size):
            batch = wallets[i : i + batch_size]
            results = await asyncio.gather(*(client.all_trades(w) for w in batch))
            batch_frames = [trades_to_frame(r) for r in results]
            frames.extend(batch_frames)
            n_trades = sum(len(f) for f in batch_frames)
            print(
                f"  wallets {i+1}-{min(i+len(batch), len(wallets))}/{len(wallets)}"
                f"  +{n_trades} trades",
                flush=True,
            )

    df = (
        pd.concat(frames, ignore_index=True)
        if any(not f.empty for f in frames)
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    if out_path is not None:
        write_parquet(df, out_path)
    return df


async def ingest_leaderboard(
    n: int = 1000,
    min_pnl: float = 100.0,
    cfg=None,
) -> list[str]:
    """Fetch wallet addresses from the leaderboard with pagination across all time windows.

    min_pnl filters out dust/bot accounts before we spend API quota on their
    trade history.  The leaderboard is sorted descending so we stop each window
    once a page falls below the floor.
    """
    async with PolymarketClient(cfg) as client:
        rows = await client.leaderboard_wallets(n=n, min_pnl=min_pnl)
    wallets = extract_wallets(rows)
    print(
        f"leaderboard  n={n}  min_pnl=${min_pnl:,.0f}"
        f"  -> {len(rows)} qualifying rows  -> {len(wallets)} unique wallets"
    )
    return wallets
