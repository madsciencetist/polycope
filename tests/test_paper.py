"""Tests for the forward paper-trading portfolio logic."""

from __future__ import annotations

import pandas as pd

from polycope.paper import PaperConfig, advance_portfolio, new_state


def _markets():
    return pd.DataFrame([
        {"market_id": "m1", "title": "A", "created_ts": 0, "end_ts": 100,
         "resolved": True, "winning_outcome": 1, "duration_bucket": "days"},
        {"market_id": "m2", "title": "B", "created_ts": 0, "end_ts": 200,
         "resolved": True, "winning_outcome": 0, "duration_bucket": "days"},
        {"market_id": "m3", "title": "C", "created_ts": 0, "end_ts": 999,
         "resolved": False, "winning_outcome": None, "duration_bucket": "days"},
    ])


def _trades(rows):
    cols = ["wallet", "market_id", "outcome_index", "side", "price", "size", "timestamp", "tx_hash"]
    return pd.DataFrame(rows, columns=cols)


def test_winner_and_loser_settle_correctly():
    cfg = PaperConfig(initial_bankroll=10_000.0, fee_bps=0.0, latency_slippage=0.0)
    state = new_state(cfg, last_ts=0)
    trades = _trades([
        # buys the winning outcome of m1 (outcome 1) at 0.50 -> doubles the stake
        ("0xaa", "m1", 1, "BUY", 0.50, 100.0, 10, "h1"),
        # buys the losing outcome of m2 (outcome 1; winner is 0) -> total loss
        ("0xaa", "m2", 1, "BUY", 0.50, 100.0, 20, "h2"),
    ])
    state, rep = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)

    assert rep["n_settled"] == 2
    assert rep["open_positions"] == 0
    # m1: stake 200 (2% of 10k) at 0.50 -> 400 shares -> payout 400, pnl +200
    # m2: stake ~196 at 0.50 -> total loss, pnl ~-196
    won = [l for l in state["ledger"] if l["won"]]
    lost = [l for l in state["ledger"] if not l["won"]]
    assert len(won) == 1 and len(lost) == 1
    assert won[0]["pnl"] > 0
    assert lost[0]["payout"] == 0.0


def test_unresolved_market_stays_open():
    cfg = PaperConfig(initial_bankroll=10_000.0)
    state = new_state(cfg, last_ts=0)
    trades = _trades([("0xaa", "m3", 0, "BUY", 0.40, 100.0, 30, "h3")])
    state, rep = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)
    assert rep["open_positions"] == 1
    assert rep["n_settled"] == 0


def test_idempotent_rerun_does_not_double_copy():
    cfg = PaperConfig(initial_bankroll=10_000.0)
    state = new_state(cfg, last_ts=0)
    trades = _trades([("0xaa", "m1", 1, "BUY", 0.50, 100.0, 10, "h1")])
    state, _ = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)
    n_after_first = state["n_settled"] if "n_settled" in state else len(state["ledger"])
    # Re-run on the SAME trade (last_ts reset to before it) — must not copy twice.
    state["last_ts"] = 0
    state, rep = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)
    assert rep["n_settled"] == 1
    assert len(state["ledger"]) == 1


def test_only_listed_wallets_are_copied():
    cfg = PaperConfig(initial_bankroll=10_000.0)
    state = new_state(cfg, last_ts=0)
    trades = _trades([
        ("0xaa", "m1", 1, "BUY", 0.50, 100.0, 10, "h1"),
        ("0xbb", "m1", 1, "BUY", 0.50, 100.0, 11, "h2"),  # not in copy-list
    ])
    state, rep = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)
    assert rep["n_settled"] == 1


def test_sells_are_not_copied():
    cfg = PaperConfig(initial_bankroll=10_000.0)
    state = new_state(cfg, last_ts=0)
    trades = _trades([("0xaa", "m1", 1, "SELL", 0.50, 100.0, 10, "h1")])
    state, rep = advance_portfolio(state, trades, _markets(), ["0xaa"], cfg)
    assert rep["n_settled"] == 0
    assert rep["open_positions"] == 0
