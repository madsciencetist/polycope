"""Fetch and normalize market metadata + resolution outcomes (the trade labels).

A trade can only be scored once we know whether the bought outcome paid out, so
resolutions are the supervised labels for the whole pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..schema import MARKET_COLUMNS
from .client import PolymarketClient
from .ingest import _first
from .store import write_parquet


def _to_ts(value: Any) -> int:
    """Coerce an int epoch-seconds or ISO-8601 string to an int Unix timestamp."""
    if not value:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).rstrip("Z").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return 0


def _winning_outcome(raw: dict) -> int | float:
    """Infer which outcome index paid out from Gamma's resolution fields.

    Gamma encodes resolution a few different ways across market types; we try the
    common ones and fall back to NaN (treated as unresolved downstream).
    """
    prices = _first(raw, ("outcomePrices", "outcome_prices"))
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            prices = None
    if isinstance(prices, (list, tuple)) and prices:
        floats = [float(p) for p in prices]
        # resolved binary markets settle to ~[1,0] or [0,1]
        if max(floats) >= 0.99:
            return int(max(range(len(floats)), key=lambda i: floats[i]))
    idx = _first(raw, ("winningOutcomeIndex", "winning_outcome", "resolvedOutcomeIndex"))
    if idx is not None:
        return int(idx)
    return float("nan")


def normalize_market(raw: dict) -> dict:
    resolved = bool(_first(raw, ("closed", "resolved", "umaResolutionStatus"), False)) or _first(
        raw, ("closed",), False
    ) is True
    win = _winning_outcome(raw) if resolved else float("nan")
    return {
        "market_id": str(_first(raw, ("conditionId", "condition_id", "id", "marketId"), "")),
        "title": str(_first(raw, ("question", "title", "slug"), "")),
        "created_ts": _to_ts(_first(raw, ("createdTs", "startDate", "created_at", "startTs"), 0)),
        "end_ts": _to_ts(_first(raw, ("endTs", "endDate", "end_at", "closedTime"), 0)),
        "resolved": bool(resolved) and not (isinstance(win, float) and pd.isna(win)),
        "winning_outcome": win,
        "duration_bucket": "",  # filled in by features.trades.add_duration_bucket
    }


def markets_to_frame(raw_markets: list[dict]) -> pd.DataFrame:
    rows = [normalize_market(m) for m in raw_markets]
    return pd.DataFrame(rows, columns=MARKET_COLUMNS).reset_index(drop=True)


async def ingest_markets(limit: int = 1000, cfg=None, out_path=None, **filters: Any) -> pd.DataFrame:
    """Page through Gamma /markets and persist the resolutions table."""
    rows: list[dict] = []
    offset = 0
    async with PolymarketClient(cfg) as client:
        while len(rows) < limit:
            batch = await client.markets(limit=min(100, limit - len(rows)), offset=offset, **filters)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
    df = markets_to_frame(rows)
    if out_path is not None:
        write_parquet(df, out_path)
    return df
