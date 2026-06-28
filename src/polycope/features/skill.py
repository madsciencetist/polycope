"""Per-trader skill metrics computed from resolved positions.

The headline metric is *edge*: realized return on capital (ROI) on resolved bets.
But raw ROI is noisy, so we also compute the ingredients needed to separate skill
from luck downstream (model.ranking):

  - n_bets / wins    -> Beta-Binomial shrinkage of hit rate
  - mean/std roi     -> raw arithmetic mean / std (for reference)
  - mean/std roi_w   -> winsorized at [-1, ROI_WIN_CAP]: removes longshot outliers
                        that inflate within-wallet variance and blind the EB model
  - brier            -> calibration quality of the trader's implied probabilities

ROI_WIN_CAP = 5.0 caps any single trade at a 500% return.  Bets below ~17%
implied probability can pay >5x; these are the noisiest trades and dominate
the arithmetic mean without adding reliable signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cap per-trade ROI at this multiple before computing winsorized statistics.
# Trades at implied probability below ~1/(1+ROI_WIN_CAP) pay above this.
ROI_WIN_CAP = 5.0

METRIC_COLUMNS = [
    "wallet",
    "n_positions",
    "n_bets",
    "wins",
    "hit_rate",
    "mean_roi",
    "std_roi",
    "mean_roi_w",
    "std_roi_w",
    "sharpe",
    "total_invested",
    "total_pnl",
    "pooled_roi",
    "brier",
    "avg_entry_price",
]


def trader_metrics(positions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate resolved positions into one metrics row per wallet."""
    cols = positions[positions["resolved"]].copy()
    if cols.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    out_rows = []
    for wallet, grp in cols.groupby("wallet"):
        roi = grp["roi"].dropna()
        roi_w = roi.clip(-1.0, ROI_WIN_CAP)
        held = grp[grp["held_to_resolution"]]
        wins = int(held["won"].sum())
        n_bets = int(len(held))

        total_invested = float(grp["invested"].sum())
        total_pnl = float(grp["realized_pnl"].sum())

        mean_roi = float(roi.mean()) if len(roi) else 0.0
        std_roi  = float(roi.std(ddof=1)) if len(roi) > 1 else 0.0
        mean_roi_w = float(roi_w.mean()) if len(roi_w) else 0.0
        std_roi_w  = float(roi_w.std(ddof=1)) if len(roi_w) > 1 else 0.0
        sharpe = mean_roi / std_roi * np.sqrt(len(roi)) if std_roi > 0 and len(roi) else 0.0

        if n_bets:
            p = held["vwap_entry"].to_numpy(dtype=float)
            o = held["won"].to_numpy(dtype=float)
            brier = float(np.mean((p - o) ** 2))
            avg_entry = float(np.mean(p))
        else:
            brier = np.nan
            avg_entry = np.nan

        out_rows.append(
            {
                "wallet":         wallet,
                "n_positions":    int(len(grp)),
                "n_bets":         n_bets,
                "wins":           wins,
                "hit_rate":       wins / n_bets if n_bets else np.nan,
                "mean_roi":       mean_roi,
                "std_roi":        std_roi,
                "mean_roi_w":     mean_roi_w,
                "std_roi_w":      std_roi_w,
                "sharpe":         sharpe,
                "total_invested": total_invested,
                "total_pnl":      total_pnl,
                "pooled_roi":     total_pnl / total_invested if total_invested > 0 else np.nan,
                "brier":          brier,
                "avg_entry_price": avg_entry,
            }
        )

    return pd.DataFrame(out_rows, columns=METRIC_COLUMNS).reset_index(drop=True)
