"""Canonical column names for the normalized data lake.

Every module reads/writes these exact columns so the pipeline stages compose
regardless of whether the source is the live Polymarket API or synthetic fixtures.
"""

from __future__ import annotations

# ---- trades table (one row per fill) ----
TRADE_COLUMNS = [
    "wallet",          # trader address (lowercase hex)
    "market_id",       # condition id
    "outcome_index",   # which outcome token (0/1 for binary markets)
    "side",            # "BUY" | "SELL"
    "price",           # fill price in [0, 1] == market-implied probability at trade time
    "size",            # number of outcome-token shares
    "timestamp",       # unix seconds
    "tx_hash",
]

# ---- markets / resolutions table (one row per market) ----
MARKET_COLUMNS = [
    "market_id",
    "title",
    "created_ts",          # unix seconds, market open
    "end_ts",              # unix seconds, market close/resolution
    "resolved",            # bool
    "winning_outcome",     # int outcome_index that paid out (NaN if unresolved)
    "duration_bucket",     # categorical: see features.trades.duration_bucket
]

# ---- positions table (one row per wallet x market x outcome) ----
POSITION_COLUMNS = [
    "wallet",
    "market_id",
    "outcome_index",
    "buy_shares",
    "sell_shares",
    "net_shares",
    "vwap_entry",          # volume-weighted avg BUY price == the trader's implied prediction
    "cash_in",             # total spent on BUYs
    "cash_out",            # total received on SELLs
    "entry_ts",            # ts of first BUY
    "resolved",
    "won",                 # bool: did the held outcome pay out
    "held_to_resolution",  # bool: net_shares > 0 at resolution
    "realized_pnl",        # cash_out - cash_in + net_shares * payout (resolved markets)
    "invested",            # cash_in (capital at risk)
    "roi",                 # realized_pnl / invested
    "duration_bucket",
]

DURATION_BUCKETS = ["intraday", "days", "weeks", "months"]
