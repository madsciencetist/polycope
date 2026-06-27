"""Synthetic Polymarket data with known ground-truth skill.

Used for tests and offline smoke runs while live API access is unavailable. The
generative model is deliberately simple but captures the core dynamic we care about:

  - Each trader t has a true skill s in [0, 1]. They buy the eventual winner with
    probability q = 0.5 + 0.5*s, at a price drawn near 0.5. So expected per-share
    edge ~= q - 0.5 = 0.5*s: skill maps monotonically to edge, zero skill -> zero edge.
  - Trade *size* varies a lot (lognormal). Some zero-skill traders bet big and, by
    luck, post large total PnL — the "leaderboard whale" trap. A good ranker must
    not be fooled by them, which is exactly what empirical-Bayes shrinkage on ROI
    achieves and what the OOS validation demonstrates.

Outputs match the canonical lake schema, so the same downstream code runs on this
or on real ingested data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features.trades import add_duration_bucket
from .schema import MARKET_COLUMNS, TRADE_COLUMNS

_T0 = 1_700_000_000  # fixed base timestamp (Date.now is unavailable / non-deterministic)
_HOUR = 3600
_DAY = 24 * _HOUR
_BUCKET_SECONDS = {"intraday": 6 * _HOUR, "days": 3 * _DAY, "weeks": 21 * _DAY, "months": 90 * _DAY}


def generate(
    n_traders: int = 60,
    n_markets: int = 400,
    participation: float = 0.35,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (trades_df, markets_df, truth_df).

    truth_df has columns [wallet, skill, size_scale] for validation in tests.
    """
    rng = np.random.default_rng(seed)

    # ---- traders ----
    skills = rng.uniform(0.0, 1.0, n_traders)
    size_scale = rng.lognormal(mean=0.0, sigma=1.2, size=n_traders)
    wallets = [f"0x{i:040x}" for i in range(n_traders)]

    # ---- markets ----
    buckets = list(_BUCKET_SECONDS)
    mk_bucket = rng.choice(buckets, size=n_markets)
    created = _T0 + np.arange(n_markets) * (2 * _HOUR)
    end = created + np.array([_BUCKET_SECONDS[b] for b in mk_bucket])
    winners = rng.integers(0, 2, n_markets)  # winning outcome index per market
    market_ids = [f"m{j:06d}" for j in range(n_markets)]

    markets = pd.DataFrame(
        {
            "market_id": market_ids,
            "title": [f"Synthetic market {j}" for j in range(n_markets)],
            "created_ts": created.astype("int64"),
            "end_ts": end.astype("int64"),
            "resolved": True,
            "winning_outcome": winners.astype(int),
            "duration_bucket": "",
        },
        columns=MARKET_COLUMNS,
    )
    markets = add_duration_bucket(markets)

    # ---- trades (one BUY per bet, held to resolution) ----
    rows = []
    for t in range(n_traders):
        q = 0.5 + 0.5 * skills[t]
        participate = rng.random(n_markets) < participation
        for j in np.flatnonzero(participate):
            winner = int(winners[j])
            bought = winner if rng.random() < q else 1 - winner
            price = float(np.clip(rng.normal(0.5, 0.08), 0.05, 0.95))
            shares = float(max(1.0, rng.lognormal(mean=3.0, sigma=0.7) * size_scale[t]))
            rows.append(
                {
                    "wallet": wallets[t],
                    "market_id": market_ids[j],
                    "outcome_index": bought,
                    "side": "BUY",
                    "price": price,
                    "size": shares,
                    "timestamp": int(created[j]) + 60,
                    "tx_hash": f"0x{t:06d}{j:06d}",
                }
            )

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    truth = pd.DataFrame({"wallet": wallets, "skill": skills, "size_scale": size_scale})
    return trades, markets, truth
