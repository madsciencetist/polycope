"""Out-of-sample validation — the project's go/no-go gate.

Copy-trading only works if trader skill *persists*: traders we identify as good
from past data must keep outperforming on future, unseen data. We test this by
splitting positions in time, ranking on the train window, and measuring the
top cohort's performance on the test window — versus the population average and
versus a naive "rank by raw PnL" leaderboard baseline (which we expect to be
much weaker, because raw PnL is mostly luck + position size).

If the EB cohort does not beat the population OOS, the thesis fails and no capital
should be risked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.skill import trader_metrics
from .ranking import rank_traders, top_wallets


def time_split(
    positions: pd.DataFrame, frac: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split positions into (train, test) with no label leakage.

    Train: positions in markets that *resolved* before the cutoff — their ROI labels
    contain no future information.  end_ts > 0 excludes markets with unknown close times.

    Test: positions *entered* after the cutoff.  Because entry_ts <= end_ts for any
    trade, no market that appears in training can have a test-window entry, so the
    two sets are on completely disjoint markets.
    """
    if positions.empty:
        return positions, positions
    cutoff = int(positions["entry_ts"].quantile(frac))
    # Drop API artifacts where a fill is recorded after the market's scheduled
    # close.  entry_ts > end_ts is physically impossible; these rows would
    # otherwise appear in both train (end_ts <= cutoff) and test (entry_ts > cutoff).
    clean = positions[
        positions["end_ts"].eq(0) | positions["entry_ts"].le(positions["end_ts"])
    ]
    train = clean[
        clean["end_ts"].gt(0) & clean["end_ts"].le(cutoff)
    ].reset_index(drop=True)
    test = clean[clean["entry_ts"].gt(cutoff)].reset_index(drop=True)
    return train, test


def _cohort_test_roi(test: pd.DataFrame, wallets: list[str]) -> dict:
    """Capital-weighted ROI of a set of wallets on the test window."""
    sub = test[test["wallet"].isin(wallets) & test["resolved"]]
    invested = float(sub["invested"].sum())
    pnl = float(sub["realized_pnl"].sum())
    return {
        "n_wallets": len(wallets),
        "n_positions": int(len(sub)),
        "invested": invested,
        "pnl": pnl,
        "pooled_roi": pnl / invested if invested > 0 else np.nan,
    }


def evaluate_oos(
    positions: pd.DataFrame,
    top_k: int = 20,
    min_bets: int = 10,
    frac: float = 0.6,
    metric: str = "roi",
) -> dict:
    """Run the full train/rank/test loop and report cohort comparisons.

    `metric` is passed to rank_traders: "roi" (default) or "irr" (capital velocity).
    """
    cutoff = int(positions["entry_ts"].quantile(frac)) if not positions.empty else 0
    train, test = time_split(positions, frac=frac)

    ranked = rank_traders(trader_metrics(train), min_bets=min_bets, metric=metric)
    eb_cohort = top_wallets(ranked, top_k, require_eligible=True)

    # Naive leaderboard baseline: rank by raw total PnL on the train window.
    train_metrics = trader_metrics(train)
    elig = train_metrics[train_metrics["n_bets"] >= min_bets]
    pnl_cohort = (
        elig.sort_values("total_pnl", ascending=False).head(top_k)["wallet"].tolist()
        if not elig.empty
        else []
    )

    population = test[test["resolved"]]
    pop_invested = float(population["invested"].sum())
    pop_pnl = float(population["realized_pnl"].sum())

    return {
        "n_train_positions": int(len(train)),
        "n_test_positions": int(len(test)),
        "cutoff_ts": cutoff,
        "population_test_roi": pop_pnl / pop_invested if pop_invested > 0 else np.nan,
        "eb_cohort": _cohort_test_roi(test, eb_cohort),
        "pnl_baseline_cohort": _cohort_test_roi(test, pnl_cohort),
        "eb_wallets": eb_cohort,
        "pnl_wallets": pnl_cohort,
    }
