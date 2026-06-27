"""Normalize raw Polymarket API responses into the canonical lake schema.

The public API's exact field names drift over time and differ slightly between
endpoints, so the normalizers are deliberately tolerant: they accept a list of
candidate keys and coerce types. This keeps the rest of the pipeline stable and
makes the normalizers unit-testable without network access.
"""

from __future__ import annotations

import asyncio
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
    wallets: list[str], cfg=None, out_path=None
) -> pd.DataFrame:
    """Fetch full trade history for each wallet and persist a single trades table."""
    async with PolymarketClient(cfg) as client:
        results = await asyncio.gather(*(client.all_trades(w) for w in wallets))
    frames = [trades_to_frame(r) for r in results]
    df = (
        pd.concat(frames, ignore_index=True)
        if any(not f.empty for f in frames)
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    if out_path is not None:
        write_parquet(df, out_path)
    return df


async def ingest_leaderboard(
    limit: int = 200, cfg=None
) -> list[str]:
    async with PolymarketClient(cfg) as client:
        board = await client.leaderboard(limit=limit)
    return extract_wallets(board)
