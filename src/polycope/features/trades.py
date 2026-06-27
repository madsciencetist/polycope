"""Turn raw fills into per-(wallet, market, outcome) positions with realized PnL.

A trader who buys 100 shares of an outcome at an average price of 0.30 is making
an implied prediction: "this resolves YES with probability ~0.30, and I think
that's cheap." When the market resolves we learn the truth, which lets us score
that prediction both as PnL and as a calibrated probability (for Brier/skill).

Cashflow accounting (exact realized PnL on resolved markets):
    realized_pnl = cash_out - cash_in + net_shares * payout
where payout = 1.0 if the held outcome won else 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import POSITION_COLUMNS

_EPS = 1e-9

# Bucket boundaries in seconds (market lifetime = end_ts - created_ts).
_HOUR = 3600
_DAY = 24 * _HOUR


def duration_bucket(seconds: float) -> str:
    if seconds <= 0 or not np.isfinite(seconds):
        return "days"  # safe default when timestamps are missing
    if seconds < _DAY:
        return "intraday"
    if seconds < 7 * _DAY:
        return "days"
    if seconds < 60 * _DAY:
        return "weeks"
    return "months"


def add_duration_bucket(markets: pd.DataFrame) -> pd.DataFrame:
    m = markets.copy()
    life = (m["end_ts"].astype(float) - m["created_ts"].astype(float))
    m["duration_bucket"] = life.map(duration_bucket)
    return m


def build_positions(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fills into positions and attach resolution labels."""
    if trades.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)

    t = trades.copy()
    t["notional"] = t["price"] * t["size"]
    is_buy = t["side"].eq("BUY")
    t["buy_shares"] = np.where(is_buy, t["size"], 0.0)
    t["sell_shares"] = np.where(~is_buy, t["size"], 0.0)
    t["cash_in"] = np.where(is_buy, t["notional"], 0.0)
    t["cash_out"] = np.where(~is_buy, t["notional"], 0.0)
    t["buy_ts"] = np.where(is_buy, t["timestamp"], np.inf)

    g = t.groupby(["wallet", "market_id", "outcome_index"], as_index=False).agg(
        buy_shares=("buy_shares", "sum"),
        sell_shares=("sell_shares", "sum"),
        cash_in=("cash_in", "sum"),
        cash_out=("cash_out", "sum"),
        entry_ts=("buy_ts", "min"),
    )
    g = g[g["buy_shares"] > _EPS].copy()  # require an actual long entry
    g["net_shares"] = g["buy_shares"] - g["sell_shares"]
    g["vwap_entry"] = g["cash_in"] / g["buy_shares"]
    g["entry_ts"] = g["entry_ts"].replace(np.inf, 0).astype("int64")

    # Attach resolution labels.
    mk = markets[["market_id", "resolved", "winning_outcome", "duration_bucket"]].copy()
    g = g.merge(mk, on="market_id", how="left")
    g["resolved"] = g["resolved"].fillna(False).astype(bool)
    g["duration_bucket"] = g["duration_bucket"].fillna("days")

    g["won"] = g["resolved"] & (g["winning_outcome"] == g["outcome_index"])
    g["held_to_resolution"] = g["resolved"] & (g["net_shares"] > _EPS)

    payout = np.where(g["won"], 1.0, 0.0)
    realized = g["cash_out"] - g["cash_in"] + g["net_shares"] * payout
    g["realized_pnl"] = np.where(g["resolved"], realized, np.nan)
    g["invested"] = g["cash_in"]
    g["roi"] = np.where(
        g["resolved"] & (g["invested"] > _EPS), g["realized_pnl"] / g["invested"], np.nan
    )

    return g[POSITION_COLUMNS].reset_index(drop=True)
