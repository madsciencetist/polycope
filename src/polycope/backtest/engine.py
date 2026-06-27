"""Simulate copying selected traders' entries, with the frictions that erode edge.

We copy BUY signals from a chosen set of wallets and hold to resolution. The copy
is *worse* than the original because:
  - latency: we see the trade after a delay, so we pay a worse price (slippage),
  - fees, and
  - bankroll limits / position sizing (we can't always size in).

The sim is event-driven in timestamp order: before acting on each new signal we
settle any positions whose markets have resolved, freeing capital. This makes the
cash constraint realistic rather than letting us copy unlimited trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_bankroll: float = 10_000.0
    stake_fraction: float = 0.02      # fraction of current cash per copied entry
    latency_slippage: float = 0.01    # added to fill price (prob units) to model late fills
    fee_bps: float = 60.0             # round-trip-ish fee on buy notional, in basis points
    min_cash: float = 10.0            # don't open a position smaller than this
    max_price: float = 0.98           # skip near-certain outcomes (no edge, dust)
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


def run_backtest(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    markets: pd.DataFrame,
    wallets: list[str],
    cfg: BacktestConfig | None = None,
) -> dict:
    """Replay copied entries and settle at resolution. Returns a results dict.

    `positions` supplies resolution labels (won/end_ts/duration) per market+outcome;
    `trades` supplies the timestamped BUY signals to copy.
    """
    cfg = cfg or BacktestConfig()
    sig = _signals(trades, wallets)

    # market+outcome -> (won, end_ts, bucket); end_ts falls back to market table.
    label = positions.set_index(["market_id", "outcome_index"]).sort_index()
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
                        "market_id": p.market_id,
                        "outcome_index": p.outcome_index,
                        "cost": p.cost,
                        "payout": payout,
                        "pnl": payout - p.cost,
                        "bucket": p.bucket,
                        "settle_ts": now,
                    }
                )
            else:
                still_open.append(p)
        open_pos = still_open
        if settled_any:
            # Mark equity = realized cash + open positions at cost basis (conservative).
            equity_curve.append((now, cash + sum(o.cost for o in open_pos)))

    for row in sig.itertuples(index=False):
        now = int(row.timestamp)
        settle(now)

        key = (row.market_id, int(row.outcome_index))
        if key not in label.index:
            continue  # we can only copy markets we have a resolution label for
        meta = label.loc[key]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        if not bool(meta["resolved"]):
            continue

        fill = min(cfg.max_price, float(row.price) + cfg.latency_slippage)
        if fill <= 0 or fill > cfg.max_price:
            continue
        stake = min(cfg.stake_fraction * cash, cash)
        if stake < cfg.min_cash:
            continue

        fee = stake * cfg.fee_bps / 10_000.0
        invest = stake - fee
        shares = invest / fill
        cash -= stake
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

    # Settle everything left at the latest known time.
    final_ts = max([p.end_ts for p in open_pos] + [now if len(sig) else 0], default=0)
    settle(final_ts + 1)

    ledger_df = pd.DataFrame(ledger)
    final_equity = cash
    eq = np.array([e for _, e in equity_curve], dtype=float)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([cfg.initial_bankroll])
    max_dd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0

    by_bucket = {}
    if not ledger_df.empty:
        for b, grp in ledger_df.groupby("bucket"):
            inv = float(grp["cost"].sum())
            pnl = float(grp["pnl"].sum())
            by_bucket[b] = {
                "n": int(len(grp)),
                "invested": inv,
                "pnl": pnl,
                "roi": pnl / inv if inv > 0 else float("nan"),
            }

    n = len(ledger_df)
    return {
        "initial_bankroll": cfg.initial_bankroll,
        "final_equity": final_equity,
        "total_return": final_equity / cfg.initial_bankroll - 1.0,
        "n_copied": int(n),
        "win_rate": float((ledger_df["payout"] > 0).mean()) if n else float("nan"),
        "max_drawdown": max_dd,
        "by_bucket": by_bucket,
        "ledger": ledger_df,
        "equity_curve": equity_curve,
    }
