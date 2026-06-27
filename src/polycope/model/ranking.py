"""Rank traders by *skill*, not luck, via empirical-Bayes shrinkage.

The trap: the top of any leaderboard is the luckiest cohort. A trader with +30%
ROI over 10 bets is almost certainly luckier than skilled; one with +8% over 1000
bets is the real deal. Empirical Bayes encodes exactly this by pulling each
trader's noisy estimate toward the population mean in proportion to how little
evidence they have.

We produce two shrunk signals and combine them:
  - eb_edge: James-Stein / Gaussian shrinkage of mean ROI (magnitude of edge)
  - eb_hit:  Beta-Binomial shrinkage of hit rate  (consistency of being right)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eb_gaussian_shrink(
    means: np.ndarray, stds: np.ndarray, ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Hierarchical-normal shrinkage of per-group means.

    Returns (shrunk_means, posterior_se, grand_mean, tau2).
    Each group i has observed mean m_i with sampling variance s_i^2 = sigma_i^2/n_i.
    True means are assumed ~ N(grand_mean, tau2). The posterior mean is a precision-
    weighted blend; low-n groups are pulled hard toward grand_mean.
    """
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    ns = np.asarray(ns, dtype=float)
    ns = np.clip(ns, 1.0, None)

    # Pooled within-group variance as a fallback for groups with std==0 (n==1).
    valid = stds > 0
    pooled_sigma2 = float(np.mean(stds[valid] ** 2)) if valid.any() else float(np.var(means) + 1e-9)
    sigma2 = np.where(stds > 0, stds**2, pooled_sigma2)
    s2 = sigma2 / ns  # sampling variance of each group mean

    grand_mean = float(np.average(means, weights=ns))
    # Between-group variance: total spread minus average sampling noise, floored at 0.
    tau2 = max(0.0, float(np.average((means - grand_mean) ** 2, weights=ns)) - float(np.mean(s2)))

    weight = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(s2)
    shrunk = grand_mean + weight * (means - grand_mean)
    post_se = np.sqrt(weight * s2)
    return shrunk, post_se, grand_mean, tau2


def eb_beta_binomial(wins: np.ndarray, ns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Beta-Binomial shrinkage of hit rates. Returns (posterior_mean, alpha0, beta0).

    The Beta prior is fit to the population of observed hit rates by method of moments.
    """
    wins = np.asarray(wins, dtype=float)
    ns = np.asarray(ns, dtype=float)
    mask = ns > 0
    if not mask.any():
        return np.full_like(wins, np.nan), 1.0, 1.0

    p = wins[mask] / ns[mask]
    mbar = float(np.mean(p))
    vbar = float(np.var(p)) if mask.sum() > 1 else 0.0
    spread = mbar * (1.0 - mbar)
    if 0.0 < vbar < spread:
        s = spread / vbar - 1.0
    else:
        s = 20.0  # weak default prior strength when the population is too small/uniform
    alpha0 = max(1e-3, mbar * s)
    beta0 = max(1e-3, (1.0 - mbar) * s)
    posterior = (wins + alpha0) / (ns + alpha0 + beta0)
    return posterior, alpha0, beta0


def rank_traders(metrics: pd.DataFrame, min_bets: int = 1) -> pd.DataFrame:
    """Attach shrunk skill estimates and a ranking. Sorted best-first.

    `min_bets` flags traders with too little evidence (kept but marked
    `eligible=False` so callers can require a minimum track record).
    """
    if metrics.empty:
        return metrics.assign(eb_edge=[], eb_edge_se=[], eb_hit=[], brier_skill=[], rank=[])

    m = metrics.copy()
    shrunk, se, grand, _ = eb_gaussian_shrink(
        m["mean_roi"].to_numpy(), m["std_roi"].to_numpy(), m["n_positions"].to_numpy()
    )
    m["eb_edge"] = shrunk
    m["eb_edge_se"] = se
    m["grand_mean_roi"] = grand

    hit, _, _ = eb_beta_binomial(m["wins"].to_numpy(), m["n_bets"].to_numpy())
    m["eb_hit"] = hit

    # Brier skill vs the climatology baseline (always predict the base win rate).
    held = m["n_bets"] > 0
    base_rate = float((m.loc[held, "wins"].sum()) / max(1.0, m.loc[held, "n_bets"].sum()))
    brier_base = max(1e-9, base_rate * (1.0 - base_rate))
    m["brier_skill"] = 1.0 - m["brier"] / brier_base

    m["eligible"] = m["n_bets"] >= min_bets
    # Rank ineligible traders last regardless of edge.
    m["_sort"] = np.where(m["eligible"], m["eb_edge"], -np.inf)
    m = m.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    m["rank"] = np.arange(1, len(m) + 1)
    return m


def top_wallets(ranked: pd.DataFrame, k: int, require_eligible: bool = True) -> list[str]:
    sel = ranked[ranked["eligible"]] if require_eligible else ranked
    return sel.head(k)["wallet"].tolist()
