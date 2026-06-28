"""Simulate copying selected traders' entries, with the frictions that erode edge.

We copy BUY signals from a chosen set of wallets and hold to resolution. The copy
is *worse* than the original because:
  - latency: we see the trade after a delay, so we pay a worse price (slippage),
  - fees, and
  - bankroll limits / position sizing (we can't always size in).

The sim is event-driven in timestamp order: before acting on each new signal we
settle any positions whose markets have resolved, freeing capital. This makes the
cash constraint realistic rather than letting us copy unlimited trades.

Sizing strategies
-----------------
fixed            : constant stake_fraction of current cash (baseline).
eb_weighted      : stake ∝ wallet's shrunk EB edge, normalized so the cohort average
                   equals the fixed baseline.  Higher-ranked wallets get more capital.
kelly            : fractional Kelly — f* = eb_edge * fill / (1 - fill), capped at
                   max_stake_fraction.  Treats eb_edge (shrunk ROI estimate) as the
                   per-dollar edge at the fill price; naturally bets tiny on long
                   shots and larger on near-even markets.  Skips wallets with no
                   positive edge.
proportional     : mirrors the original wallet's relative bet size within their own
                   history (trade notional / wallet mean notional), scaled to the
                   same average stake as fixed.  Amplifies the wallet's own conviction.
wallet_normalized: z-scores each bet's notional against that wallet's training-period
                   mean and std, then maps z ∈ [-2, +2] → multiplier ∈ [0, 2]× the
                   fixed base stake.  Large-for-this-wallet bets get more capital;
                   small-for-this-wallet bets get less.  Wallet scale differences
                   (one whale vs many small traders) are removed before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_bankroll: float = 10_000.0
    stake_fraction: float = 0.02      # base fraction of current cash per copied entry
    max_stake_fraction: float = 0.10  # ceiling for variable-sizing strategies
    latency_slippage: float = 0.01    # added to fill price (prob units) to model late fills
    fee_bps: float = 60.0             # round-trip-ish fee on buy notional, in basis points
    min_cash: float = 10.0            # don't open a position smaller than this
    max_price: float = 0.98           # skip near-certain outcomes (no edge, dust)
    sizing: str = "fixed"             # "fixed" | "eb_weighted" | "kelly" | "proportional" | "wallet_normalized"
    duration_buckets: tuple[str, ...] = field(
        default_factory=lambda: ("intraday", "days", "weeks", "months")
    )


@dataclass
class _OpenPos:
    market_id: str
    outcome_index: int
    shares: float
    cost: float
    end_ts: int
    won_payout: float  # 1.0 if this outcome resolves as winner else 0.0
    bucket: str


def _signals(trades: pd.DataFrame, wallets: list[str]) -> pd.DataFrame:
    sig = trades[trades["side"].eq("BUY") & trades["wallet"].isin(wallets)].copy()
    return sig.sort_values("timestamp").reset_index(drop=True)


def _build_sizing_tables(
    sig: pd.DataFrame,
    wallets: list[str],
    wallet_metrics: dict[str, dict] | None,
    cfg: BacktestConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Pre-compute per-wallet tables for variable sizing strategies.

    Returns (eb_weights, eb_hits, mean_notional, eb_edges_raw, std_notional).

    mean_notional and std_notional are read from wallet_metrics (training-derived)
    when available, so the sizing signal is not contaminated by test-window data.
    Falls back to computing from sig if the keys are absent.
    """
    # EB-weighted: normalize positive-shifted eb_edge so cohort mean weight = 1.
    eb_edges_raw = {w: float((wallet_metrics or {}).get(w, {}).get("eb_edge", 0.0)) for w in wallets}
    min_edge = min(eb_edges_raw.values(), default=0.0)
    shifted = {w: max(e - min_edge + 1e-9, 1e-9) for w, e in eb_edges_raw.items()}
    mean_shifted = float(np.mean(list(shifted.values()))) or 1.0
    eb_weights = {w: v / mean_shifted for w, v in shifted.items()}

    # Kelly: shrunk hit rate per wallet (retained for reference; Kelly now uses eb_edge).
    eb_hits = {w: float((wallet_metrics or {}).get(w, {}).get("eb_hit", 0.5)) for w in wallets}

    # Notional stats: prefer training-derived values from wallet_metrics.
    wm = wallet_metrics or {}
    if all("mean_notional" in wm.get(w, {}) for w in wallets):
        mean_notional = {w: float(wm[w]["mean_notional"]) for w in wallets}
        std_notional  = {w: float(wm[w].get("std_notional", wm[w]["mean_notional"])) for w in wallets}
    else:
        # Fallback: compute from test signals (mild look-ahead; only used if training
        # stats were not supplied via wallet_metrics).
        if "size" in sig.columns:
            _s = sig.copy()
            _s["_notional"] = _s["price"] * _s["size"]
            grp = _s.groupby("wallet")["_notional"]
            mean_notional = grp.mean().to_dict()
            std_notional  = grp.std(ddof=1).fillna(0).to_dict()
        else:
            mean_notional = {w: 1.0 for w in wallets}
            std_notional  = {w: 1.0 for w in wallets}

    return eb_weights, eb_hits, mean_notional, eb_edges_raw, std_notional


def _stake(
    cfg: BacktestConfig,
    cash: float,
    wallet: str,
    fill: float,
    notional: float,
    eb_weights: dict[str, float],
    eb_hits: dict[str, float],
    mean_notional: dict[str, float],
    eb_edges_raw: dict[str, float] | None = None,
    std_notional: dict[str, float] | None = None,
) -> float:
    base = cfg.stake_fraction * cash
    cap  = cfg.max_stake_fraction * cash

    if cfg.sizing == "eb_weighted":
        return min(eb_weights.get(wallet, 1.0) * base, cap)

    if cfg.sizing == "kelly":
        # ROI-per-odds Kelly: f = edge * fill / (1-fill).
        # Naturally bets tiny on long shots and more on near-even markets,
        # avoiding the leverage explosion of hit-rate Kelly on 1-2 cent bets.
        edge = max(0.0, (eb_edges_raw or {}).get(wallet, 0.0))
        if fill <= 0 or fill >= 1.0 or edge <= 0:
            return 0.0
        f = edge * fill / (1.0 - fill)
        return min(f * cash, cap) if f > 0 else 0.0

    if cfg.sizing == "proportional":
        mn = mean_notional.get(wallet, 1.0) or 1.0
        scale = notional / mn
        return min(scale * base, cap)

    if cfg.sizing == "wallet_normalized":
        mu    = mean_notional.get(wallet, 1.0) or 1.0
        sigma = (std_notional or {}).get(wallet, mu) or mu
        z = (notional - mu) / sigma
        # z ∈ [-2, +2]  →  multiplier ∈ [0, 2]:
        #   mean bet (z=0)   → 1× base (same as fixed)
        #   +1σ bet          → 1.5× base
        #   +2σ or more      → 2× base  (capped further by max_stake_fraction)
        #   −2σ or less      → 0× base  (skip; will be caught by min_cash guard)
        z_clamped  = max(-2.0, min(2.0, z))
        multiplier = max(0.0, 1.0 + 0.5 * z_clamped)
        return min(multiplier * base, cap)

    # default: fixed
    return min(base, cash)


def run_backtest(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    markets: pd.DataFrame,
    wallets: list[str],
    cfg: BacktestConfig | None = None,
    wallet_metrics: dict[str, dict] | None = None,
) -> dict:
    """Replay copied entries and settle at resolution. Returns a results dict.

    `positions` supplies resolution labels (won/end_ts/duration) per market+outcome;
    `trades` supplies the timestamped BUY signals to copy.
    `wallet_metrics` is required for eb_weighted/kelly/proportional/wallet_normalized:
        {wallet: {"eb_edge": float, "eb_hit": float,
                  "mean_notional": float, "std_notional": float}}
    """
    cfg = cfg or BacktestConfig()
    sig = _signals(trades, wallets)

    eb_weights, eb_hits, mean_notional, eb_edges_raw, std_notional = (
        _build_sizing_tables(sig, wallets, wallet_metrics, cfg)
    )

    # market+outcome -> (won, end_ts, bucket); end_ts falls back to market table.
    label  = positions.set_index(["market_id", "outcome_index"]).sort_index()
    mk_end = markets.set_index("market_id")["end_ts"].to_dict()

    cash = cfg.initial_bankroll
    open_pos: list[_OpenPos] = []
    equity_curve: list[tuple[int, float]] = [(int(sig["timestamp"].min()) if len(sig) else 0, cash)]
    ledger: list[dict] = []

    def settle(now: int) -> None:
        nonlocal cash, open_pos
        still_open = []
        settled_any = False
        for p in open_pos:
            if p.end_ts and p.end_ts <= now:
                payout = p.shares * p.won_payout
                cash += payout
                settled_any = True
                ledger.append(
                    {
                        "market_id":     p.market_id,
                        "outcome_index": p.outcome_index,
                        "cost":          p.cost,
                        "payout":        payout,
                        "pnl":           payout - p.cost,
                        "bucket":        p.bucket,
                        "settle_ts":     now,
                    }
                )
            else:
                still_open.append(p)
        open_pos = still_open
        if settled_any:
            equity_curve.append((now, cash + sum(o.cost for o in open_pos)))

    for row in sig.itertuples(index=False):
        now = int(row.timestamp)
        settle(now)

        key = (row.market_id, int(row.outcome_index))
        if key not in label.index:
            continue
        meta = label.loc[key]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        if not bool(meta["resolved"]):
            continue

        fill = min(cfg.max_price, float(row.price) + cfg.latency_slippage)
        if fill <= 0 or fill > cfg.max_price:
            continue

        notional = float(row.price) * float(getattr(row, "size", 1.0))
        stake = _stake(
            cfg, cash, row.wallet, fill, notional,
            eb_weights, eb_hits, mean_notional, eb_edges_raw, std_notional,
        )
        stake = min(stake, cash)
        if stake < cfg.min_cash:
            continue

        fee    = stake * cfg.fee_bps / 10_000.0
        invest = stake - fee
        shares = invest / fill
        cash  -= stake
        end_ts = int(meta["end_ts"]) if "end_ts" in meta and not pd.isna(meta["end_ts"]) else int(
            mk_end.get(row.market_id, now)
        )
        open_pos.append(
            _OpenPos(
                market_id=row.market_id,
                outcome_index=int(row.outcome_index),
                shares=shares,
                cost=stake,
                end_ts=end_ts,
                won_payout=1.0 if bool(meta["won"]) else 0.0,
                bucket=str(meta["duration_bucket"]),
            )
        )

    final_ts = max([p.end_ts for p in open_pos] + [now if len(sig) else 0], default=0)
    settle(final_ts + 1)

    ledger_df    = pd.DataFrame(ledger)
    final_equity = cash
    eq   = np.array([e for _, e in equity_curve], dtype=float)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([cfg.initial_bankroll])
    max_dd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0

    by_bucket: dict = {}
    if not ledger_df.empty:
        for b, grp in ledger_df.groupby("bucket"):
            inv = float(grp["cost"].sum())
            pnl = float(grp["pnl"].sum())
            by_bucket[b] = {
                "n":        int(len(grp)),
                "invested": inv,
                "pnl":      pnl,
                "roi":      pnl / inv if inv > 0 else float("nan"),
            }

    n = len(ledger_df)
    return {
        "initial_bankroll": cfg.initial_bankroll,
        "final_equity":     final_equity,
        "total_return":     final_equity / cfg.initial_bankroll - 1.0,
        "n_copied":         int(n),
        "win_rate":         float((ledger_df["payout"] > 0).mean()) if n else float("nan"),
        "max_drawdown":     max_dd,
        "by_bucket":        by_bucket,
        "ledger":           ledger_df,
        "equity_curve":     equity_curve,
    }
