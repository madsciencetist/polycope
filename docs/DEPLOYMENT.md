# Polycope Deployment Runbook

How to take the validated copy-trading model live on a non-custodial hosted
execution service (e.g. Polycopy). The model's edge is in **wallet selection**;
execution is delegated to a third-party service so we never build or run a
key-holding order subsystem.

## The decided strategy

| Axis | Decision | Why |
|---|---|---|
| **Selection signal** | `roi_eb` (EB-shrunk winsorized ROI) | Best *robust* per-bet signal; beats naive PnL/ROI once longshot lottery wins are winsorized away. |
| **min_bets** | 50 | Sweet spot for eligibility; lower admits noise, higher starves the pool. |
| **top_k (cohort size)** | 15 | Edge concentrates in the top ~15; diluting to 30+ roughly halves pooled ROI. |
| **Sizing** | **Fixed / equal allocation** with per-trade + daily caps | The only sizing a hosted venue can do that is *consistent with how we validated selection* (equal-weight). Proportional is mediocre AND mismatched — do not use. |
| **Retrain cadence** | Weekly | Bounds copy-list staleness; cheap to rerun. |
| **Min capital** | $1,000 | Below ~$1k the bankroll binds and capital-recycling drag appears. |

> eb-weighted / wallet-normalized sizing scored marginally higher in our own
> engine (+1–2 pooled-ROI points) but require per-wallet EB edges / notional
> z-scores that a hosted service cannot execute. Set aside until/unless we build
> a custom delegated-trading bot. Their advantage is not worth that build today.

## Go / no-go checklist for the execution service

Confirm ALL before funding:

- [ ] **Non-custodial** — your wallet signs; service never takes custody. Funds
      authorized via a *bounded* USDC approval (revocable), not a key handover.
- [ ] **Fixed / equal-allocation sizing** is configurable (NOT forced proportional).
- [ ] **Per-trade cap** and **daily loss limit** / drawdown stop available.
- [ ] **Custom wallet list** — you can follow arbitrary addresses you supply
      (manual paste of ~15 is fine at weekly cadence; no API required).
- [ ] **Fee schedule understood** — service fees are *on top* of Polymarket's.
      Re-validate edge against the real schedule (see fee note below).
- [ ] Service reputation / volume claims sanity-checked (many are low quality).

## Weekly operating loop

1. **Retrain** — regenerate the ranked cohort from fresh data:
   ```
   python scripts/run_ingest.py --wallets 2000 --min-pnl 100        # refresh lake
   python scripts/run_rank.py --min-bets 50 --top 15               # -> ranked wallets
   ```
   (Use the most recent calendar cutoff = "now"; the cohort is the top-15 eligible
   wallets by `eb_edge`.)

2. **Diff** the new top-15 against last week's list. Expect mostly-stable
   membership; a wallet dropping off usually means its trailing edge faded.

3. **Update the service** — paste/confirm the current top-15 as the followed set.
   Remove any wallet no longer in the list.

4. **Sizing config** — fixed allocation per trade (≈2% of bankroll), per-trade cap
   (≈10% of bankroll), daily loss limit set to your tolerance.

5. **Record** — log the week's copy-list, equity, and realized PnL so live results
   can be compared against the backtest expectation (live WILL be lower; see risks).

## Risk controls (set in the service, not optional)

- **Per-trade cap**: ~10% of bankroll (one bet can't dominate).
- **Daily loss limit / kill switch**: halt copying on a bad day.
- **Slippage filter**: skip fills worse than ~1–2¢ off the target's price
  (favorites barely move; this mainly guards against thin/at-risk markets).
- **Bankroll floor**: keep deployed capital >= $1,000.

## Expectations & known risks

- **Live edge < backtest edge.** Two reasons, both already characterized:
  1. **Survivorship** — the candidate universe came from *today's* leaderboard, so
     the backtest never saw pre-cutoff stars that later went cold. Weekly retrain
     bounds, but does not erase, the resulting optimism.
  2. **Latency** — we react to a target's *public fill*; we always pay a slightly
     worse price. Mitigated because the cohort grinds *favorites* (deep, liquid,
     low impact in the seconds after a fill); sub-second execution is plenty.
- **Fee drag — checked, the edge survives.** Re-running the walk-forward with
  Polycopy's 1% taker (vs our original 60 bps) costs only ~0.5 pp of pooled ROI:

  | Sizing | 60bps | 100bps (Polycopy 1%) | 150bps (conservative) |
  |---|---|---|---|
  | **Fixed (deployable)** | +6.55% | **+6.08%** | +5.49% |
  | eb_weighted (ref only) | +8.41% | +8.00% | +7.49% |
  | wallet_normalized (ref only) | +7.52% | +7.04% | +6.43% |

  The fee is small because we **hold to resolution**: one taker fee on entry, then
  winning shares redeem at $1 with no trade fee. **This assumes we do NOT mirror the
  target's early exits** — if the service copies their sells too, each exit is
  another ~1% taker (the 150bps column models that case; still +5.49%). Configure
  the service to hold to resolution. Re-validate if Polymarket adds its own fees.
  Per-window Fixed @ 1%: Jan +7.96%, Feb -0.12%, Mar +12.96%, Apr +3.50% (Feb is
  the recurring soft window).
- **Strategy style** — the cohort buys favorites (~0.74 avg entry): high hit rate,
  modest ROI per bet, broad diversification (thousands of markets). Edge is *not*
  riding on a few longshot lotteries — which is what makes it copyable.

## Suggested launch sequence

1. **Paper / free-manual mode** on the chosen service for 2–4 weeks: follow the
   top-15, log would-be fills, measure live edge **net of the real fee**.
2. **Small live capital** ($1k) once paper confirms edge survives fees + latency,
   with all risk controls armed.
3. **Scale** only after live results track expectations across several weekly
   retrains. Above $1k results are roughly scale-invariant.
4. (Optional, later) Build a custom non-custodial delegated-trading bot only if we
   want sizing strategies the hosted venue can't express — gated on the marginal
   edge justifying the build + ops.
