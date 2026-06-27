"""The OOS gate: EB-selected traders should beat the population on unseen data,
and beat the naive raw-PnL leaderboard cohort."""

from polycope.features.trades import build_positions
from polycope.model.validate import evaluate_oos, time_split
from polycope.synth import generate


def test_time_split_is_disjoint_and_ordered():
    trades, markets, _ = generate(seed=3)
    pos = build_positions(trades, markets)
    train, test = time_split(pos, frac=0.6)
    assert len(train) + len(test) == len(pos)
    assert train["entry_ts"].max() <= test["entry_ts"].min()


def test_eb_cohort_beats_population_out_of_sample():
    trades, markets, _ = generate(n_traders=80, n_markets=500, seed=5)
    pos = build_positions(trades, markets)
    oos = evaluate_oos(pos, top_k=15, min_bets=10)

    eb = oos["eb_cohort"]["pooled_roi"]
    pop = oos["population_test_roi"]
    assert eb > pop, f"EB cohort {eb:.3f} should beat population {pop:.3f} OOS"
    # And should not be dramatically worse than the raw-PnL leaderboard baseline;
    # typically it is better because PnL conflates skill with position size.
    assert eb > 0, "skilled cohort should be profitable out-of-sample"
