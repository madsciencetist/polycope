"""Position accounting must produce exact realized PnL on hand-computed cases."""

import numpy as np
import pandas as pd

from polycope.features.trades import build_positions, duration_bucket
from polycope.schema import MARKET_COLUMNS, TRADE_COLUMNS


def _markets(rows):
    return pd.DataFrame(rows, columns=MARKET_COLUMNS)


def _trades(rows):
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def test_duration_bucket_boundaries():
    assert duration_bucket(3600) == "intraday"
    assert duration_bucket(3 * 86400) == "days"
    assert duration_bucket(10 * 86400) == "weeks"
    assert duration_bucket(120 * 86400) == "months"


def test_winning_buy_held_to_resolution():
    # Buy 100 shares @ 0.40 of outcome 1, which wins -> payout 100, cost 40, pnl +60.
    trades = _trades([
        {"wallet": "0x1", "market_id": "m", "outcome_index": 1, "side": "BUY",
         "price": 0.40, "size": 100, "timestamp": 10, "tx_hash": "a"},
    ])
    markets = _markets([
        {"market_id": "m", "title": "t", "created_ts": 0, "end_ts": 86400,
         "resolved": True, "winning_outcome": 1, "duration_bucket": "days"},
    ])
    pos = build_positions(trades, markets)
    assert len(pos) == 1
    r = pos.iloc[0]
    assert r["won"]
    assert r["held_to_resolution"]
    assert np.isclose(r["vwap_entry"], 0.40)
    assert np.isclose(r["realized_pnl"], 60.0)
    assert np.isclose(r["roi"], 1.5)


def test_losing_buy():
    trades = _trades([
        {"wallet": "0x1", "market_id": "m", "outcome_index": 0, "side": "BUY",
         "price": 0.60, "size": 50, "timestamp": 10, "tx_hash": "a"},
    ])
    markets = _markets([
        {"market_id": "m", "title": "t", "created_ts": 0, "end_ts": 86400,
         "resolved": True, "winning_outcome": 1, "duration_bucket": "days"},
    ])
    pos = build_positions(trades, markets)
    r = pos.iloc[0]
    assert not r["won"]
    assert np.isclose(r["realized_pnl"], -30.0)  # cost 30, payout 0


def test_partial_close_then_resolution():
    # Buy 100 @ 0.50 (cost 50), sell 40 @ 0.70 (cash_out 28), 60 left, outcome wins (payout 60).
    # realized = 28 - 50 + 60 = 38.
    trades = _trades([
        {"wallet": "0x1", "market_id": "m", "outcome_index": 1, "side": "BUY",
         "price": 0.50, "size": 100, "timestamp": 10, "tx_hash": "a"},
        {"wallet": "0x1", "market_id": "m", "outcome_index": 1, "side": "SELL",
         "price": 0.70, "size": 40, "timestamp": 20, "tx_hash": "b"},
    ])
    markets = _markets([
        {"market_id": "m", "title": "t", "created_ts": 0, "end_ts": 86400,
         "resolved": True, "winning_outcome": 1, "duration_bucket": "days"},
    ])
    pos = build_positions(trades, markets)
    r = pos.iloc[0]
    assert np.isclose(r["net_shares"], 60.0)
    assert np.isclose(r["realized_pnl"], 38.0)


def test_unresolved_market_has_nan_pnl():
    trades = _trades([
        {"wallet": "0x1", "market_id": "m", "outcome_index": 1, "side": "BUY",
         "price": 0.50, "size": 100, "timestamp": 10, "tx_hash": "a"},
    ])
    markets = _markets([
        {"market_id": "m", "title": "t", "created_ts": 0, "end_ts": 86400,
         "resolved": False, "winning_outcome": np.nan, "duration_bucket": "days"},
    ])
    pos = build_positions(trades, markets)
    assert np.isnan(pos.iloc[0]["realized_pnl"])
