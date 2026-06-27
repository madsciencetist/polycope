"""Backtest accounting must be exact, and copying skilled traders must profit."""

import numpy as np
import pandas as pd

from polycope.backtest.engine import BacktestConfig, run_backtest
from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.model.ranking import rank_traders, top_wallets
from polycope.schema import MARKET_COLUMNS, TRADE_COLUMNS
from polycope.synth import generate


def test_single_winning_copy_accounting():
    # One winning market, no fees/slippage: buy at 0.50, resolves win -> ~double.
    trades = pd.DataFrame([
        {"wallet": "0xw", "market_id": "m", "outcome_index": 1, "side": "BUY",
         "price": 0.50, "size": 100, "timestamp": 100, "tx_hash": "a"},
    ], columns=TRADE_COLUMNS)
    markets = pd.DataFrame([
        {"market_id": "m", "title": "t", "created_ts": 0, "end_ts": 200,
         "resolved": True, "winning_outcome": 1, "duration_bucket": "days"},
    ], columns=MARKET_COLUMNS)
    positions = build_positions(trades, markets)
    cfg = BacktestConfig(initial_bankroll=1000.0, stake_fraction=0.5,
                         latency_slippage=0.0, fee_bps=0.0)
    res = run_backtest(trades, positions, markets, ["0xw"], cfg)
    # Stake 500 @ 0.50 -> 1000 shares -> payout 1000. Cash: 500 left + 1000 = 1500.
    assert np.isclose(res["final_equity"], 1500.0)
    assert res["n_copied"] == 1
    assert res["win_rate"] == 1.0


def test_fees_and_slippage_reduce_return():
    trades, markets, _ = generate(seed=9)
    positions = build_positions(trades, markets)
    ranked = rank_traders(trader_metrics(positions), min_bets=10)
    wallets = top_wallets(ranked, 10)

    frictionless = run_backtest(trades, positions, markets, wallets,
                                BacktestConfig(latency_slippage=0.0, fee_bps=0.0))
    frictiony = run_backtest(trades, positions, markets, wallets,
                             BacktestConfig(latency_slippage=0.03, fee_bps=100.0))
    assert frictiony["total_return"] < frictionless["total_return"]


def test_copying_skilled_beats_copying_random():
    trades, markets, truth = generate(n_traders=80, n_markets=500, seed=4)
    positions = build_positions(trades, markets)
    ranked = rank_traders(trader_metrics(positions), min_bets=10)

    skilled = top_wallets(ranked, 10)
    worst = ranked[ranked["eligible"]].tail(10)["wallet"].tolist()

    cfg = BacktestConfig()
    good = run_backtest(trades, positions, markets, skilled, cfg)
    bad = run_backtest(trades, positions, markets, worst, cfg)
    assert good["total_return"] > bad["total_return"]
