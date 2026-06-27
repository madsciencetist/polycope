"""Empirical-Bayes ranking must recover skill and resist the small-sample trap."""

import numpy as np
from scipy.stats import spearmanr

from polycope.features.skill import trader_metrics
from polycope.features.trades import build_positions
from polycope.model.ranking import eb_beta_binomial, eb_gaussian_shrink, rank_traders
from polycope.synth import generate


def test_gaussian_shrink_pulls_small_n_to_grand_mean():
    # Three groups with the same extreme mean but wildly different sample sizes.
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([1.0, 1.0, 1.0])
    ns = np.array([2.0, 50.0, 5000.0])
    shrunk, se, grand, tau2 = eb_gaussian_shrink(means, stds, ns)
    # Larger n -> estimate stays closer to its own mean -> larger |shrunk - grand|.
    assert abs(shrunk[2] - grand) >= abs(shrunk[1] - grand) >= abs(shrunk[0] - grand)
    assert se[0] >= se[2]  # less data -> more posterior uncertainty


def test_beta_binomial_bounds_and_shrinkage():
    wins = np.array([1.0, 60.0])
    ns = np.array([1.0, 100.0])
    post, a0, b0 = eb_beta_binomial(wins, ns)
    assert np.all((post > 0) & (post < 1))
    # 1/1 should be shrunk well below 1.0 toward the prior.
    assert post[0] < 0.95


def test_ranking_recovers_true_skill():
    trades, markets, truth = generate(n_traders=60, n_markets=400, seed=11)
    positions = build_positions(trades, markets)
    ranked = rank_traders(trader_metrics(positions), min_bets=10)

    merged = ranked.merge(truth, on="wallet")
    rho, _ = spearmanr(merged["eb_edge"], merged["skill"])
    assert rho > 0.6, f"EB edge should track true skill, got rho={rho:.3f}"

    # The top-5 by EB edge should be genuinely high-skill, not lucky whales.
    top5 = ranked[ranked["eligible"]].head(5).merge(truth, on="wallet")
    assert top5["skill"].mean() > merged["skill"].mean()
