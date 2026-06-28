"""Forward paper-trading: a persistent simulated portfolio that copies the cohort.

This is the deployable strategy run *forward* against live data, with no real money
and no orders placed — legal anywhere, since we only read public on-chain trades.

It mirrors the backtest engine's economics (Fixed sizing, fee, latency slippage,
hold-to-resolution, cash recycling) but keeps state between runs so results
accumulate as fresh data is ingested over time:

  state = advance_portfolio(state, trades, markets, wallets, cfg)

Run it repeatedly (after `run_ingest.py` refreshes the lake). Only BUYs newer than
the last processed timestamp are copied; positions settle when their markets resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PaperConfig:
    initial_bankroll: float = 10_000.0
    stake_fraction: float = 0.02       # Fixed sizing — the deployable choice
    max_stake_fraction: float = 0.10   # per-trade cap
    latency_slippage: float = 0.01     # we react to a public fill; pay slightly worse
    fee_bps: float = 100.0             # Polycopy 1% taker on entry notional
    min_cash: float = 1.0
    max_price: float = 0.98


def new_state(cfg: PaperConfig, last_ts: int = 0) -> dict:
    return {
        "cash": float(cfg.initial_bankroll),
        "initial": float(cfg.initial_bankroll),
        "last_ts": int(last_ts),
        "open": [],     # list of open positions
        "ledger": [],   # settled positions
        "seen": [],     # idempotency keys of already-copied trades
    }


def _market_lookup(markets: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in markets.itertuples(index=False):
        mid = str(r.market_id)
        if mid in out:
            continue
        wo = getattr(r, "winning_outcome", None)
        out[mid] = {
            "resolved": bool(getattr(r, "resolved", False)),
            "winning_outcome": None if wo is None or pd.isna(wo) else int(wo),
            "end_ts": int(getattr(r, "end_ts", 0) or 0),
        }
    return out


def _key(t) -> str:
    h = str(getattr(t, "tx_hash", "") or "")
    if h:
        return h
    return f"{t.wallet}|{t.market_id}|{int(t.outcome_index)}|{int(t.timestamp)}|{t.price}|{t.size}"


def _settle(state: dict, mk: dict[str, dict], upto: int) -> None:
    still = []
    for p in state["open"]:
        res = mk.get(p["market_id"])
        if p["end_ts"] and p["end_ts"] <= upto and res and res["resolved"]:
            won = res["winning_outcome"] is not None and res["winning_outcome"] == p["outcome_index"]
            payout = p["shares"] if won else 0.0
            state["cash"] += payout
            state["ledger"].append({
                "market_id": p["market_id"],
                "outcome_index": p["outcome_index"],
                "cost": p["cost"],
                "payout": payout,
                "pnl": payout - p["cost"],
                "entry_ts": p["entry_ts"],
                "settle_ts": p["end_ts"] or upto,
                "won": bool(won),
            })
        else:
            still.append(p)
    state["open"] = still


def advance_portfolio(
    state: dict,
    trades: pd.DataFrame,
    markets: pd.DataFrame,
    wallets: list[str],
    cfg: PaperConfig,
) -> tuple[dict, dict]:
    """Advance the simulated portfolio to the latest data. Returns (state, report).

    Copies BUYs from `wallets` with timestamp > state['last_ts'], settling resolved
    positions chronologically so cash recycling is realistic.  Idempotent: a trade
    already in state['seen'] is never copied twice, so re-running on overlapping data
    is safe.
    """
    mk = _market_lookup(markets)
    wallets_set = set(w.lower() for w in wallets)
    seen = set(state["seen"])

    sig = trades[
        trades["side"].eq("BUY")
        & trades["wallet"].str.lower().isin(wallets_set)
        & trades["timestamp"].gt(state["last_ts"])
    ].sort_values("timestamp")

    now = int(trades["timestamp"].max()) if len(trades) else state["last_ts"]
    now = max([now, state["last_ts"]] + [p["end_ts"] for p in state["open"]])

    for t in sig.itertuples(index=False):
        k = _key(t)
        if k in seen:
            continue
        seen.add(k)

        ts = int(t.timestamp)
        _settle(state, mk, ts)  # free cash from anything resolved by now

        fill = min(cfg.max_price, float(t.price) + cfg.latency_slippage)
        if fill <= 0 or fill > cfg.max_price:
            continue
        stake = min(cfg.stake_fraction * state["cash"], cfg.max_stake_fraction * state["cash"])
        stake = min(stake, state["cash"])
        if stake < cfg.min_cash:
            continue
        fee = stake * cfg.fee_bps / 10_000.0
        invest = stake - fee
        shares = invest / fill
        state["cash"] -= stake

        res = mk.get(str(t.market_id))
        end_ts = res["end_ts"] if res else 0
        state["open"].append({
            "market_id": str(t.market_id),
            "outcome_index": int(t.outcome_index),
            "shares": shares,
            "cost": stake,
            "entry_ts": ts,
            "end_ts": int(end_ts),
        })

    # Final settle: book every position whose market has actually resolved. Use a
    # horizon past all open end_ts so resolution time — not the last trade's
    # timestamp — governs settlement (the `resolved` flag is the real gate).
    final_upto = max([now] + [p["end_ts"] for p in state["open"]])
    _settle(state, mk, final_upto)
    state["last_ts"] = now
    state["seen"] = list(seen)

    return state, report(state)


def report(state: dict) -> dict:
    ledger = state["ledger"]
    open_cost = sum(p["cost"] for p in state["open"])
    realized = sum(l["pnl"] for l in ledger)
    invested = sum(l["cost"] for l in ledger)
    wins = sum(1 for l in ledger if l["payout"] > 0)
    equity = state["cash"] + open_cost
    return {
        "cash": state["cash"],
        "open_positions": len(state["open"]),
        "open_cost": open_cost,
        "equity": equity,
        "total_return": equity / state["initial"] - 1.0 if state["initial"] else 0.0,
        "n_settled": len(ledger),
        "realized_pnl": realized,
        "pooled_roi": realized / invested if invested > 0 else float("nan"),
        "win_rate": wins / len(ledger) if ledger else float("nan"),
        "last_ts": state["last_ts"],
    }
