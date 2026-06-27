# polycope

A research pipeline for **copy-trading skilled Polymarket traders**. Polymarket
activity is public and tied to wallets, so we can analyze trading histories,
identify traders whose edge *persists*, and mirror their trades.

This repo is the **alpha-research half** (milestones 1–6): ingest → score traders →
rank by skill → validate out-of-sample → backtest. Live execution is deferred until
the backtest proves a persistent, copy-able edge (see *Status & roadmap*).

## The core idea (and the trap it avoids)

Ranking traders by raw PnL or the public leaderboard is a trap: the top of any
leaderboard is the **luckiest** cohort, not the most skilled, and PnL is dominated
by position size. So we:

1. Treat each trade as an **implied probability forecast** (buying an outcome at
   30¢ ⇒ "this resolves YES ~30% of the time, and that's cheap").
2. Score forecasts against **actual market resolutions** — as realized ROI and as
   calibration/Brier.
3. Rank traders with **empirical-Bayes shrinkage** (`model/ranking.py`): each
   trader's noisy edge is pulled toward the population mean in proportion to how
   little evidence they have. +8% over 1000 bets beats +30% over 10.
4. **Validate out-of-sample** (`model/validate.py`): rank on older trades, measure
   the top cohort on newer trades. If skill doesn't persist, copy-trading can't
   work — this is the **go/no-go gate**.

## Layout

```
src/polycope/
  config.py            settings (API hosts, data dir, rate limits)
  schema.py            canonical column names for the data lake
  synth.py             synthetic data w/ known ground-truth skill (for offline dev/tests)
  pipeline.py          load_dataset(): synthetic or ingested Parquet
  data/                async API client, normalizers, resolutions, Parquet store
  features/            trades->positions (realized PnL) and per-trader skill metrics
  model/               empirical-Bayes ranking + out-of-sample validation
  backtest/            friction-aware copy-trading simulator + report
scripts/               run_ingest.py, run_rank.py, run_backtest.py
tests/                 unit + statistical recovery tests
```

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

Everything runs offline on synthetic data (no network needed):

```bash
python scripts/run_rank.py --synthetic          # rank synthetic traders by shrunk edge
python scripts/run_backtest.py --synthetic       # OOS validation + friction-aware backtest
pytest -q                                        # 18 tests incl. skill-recovery checks
```

On **real** data (requires outbound access to `*.polymarket.com`):

```bash
python scripts/run_ingest.py --wallets 200 --markets 2000   # -> data/raw/*.parquet
python scripts/run_rank.py                                   # uses the ingested lake
python scripts/run_backtest.py
```

> **Note on network access.** Live ingestion hits Polymarket's public Data API
> (`data-api.polymarket.com`) and Gamma API (`gamma-api.polymarket.com`). Some
> environments' network policies block these hosts (the ingest script reports the
> denial clearly). Open outbound access to those hosts to ingest real data; until
> then the `--synthetic` path exercises the entire pipeline.

## Status & roadmap

- [x] 1. Data client + ingest (leaderboard → wallets → trades; markets/resolutions)
- [x] 2. Positions + realized-PnL accounting
- [x] 3. Per-trader skill metrics (edge, hit rate, Brier)
- [x] 4. Empirical-Bayes ranking
- [x] 5. Out-of-sample validation (go/no-go gate)
- [x] 6. Friction-aware backtester (latency, fees, bankroll; by-duration breakdown)
- [ ] 7. **Execution (deferred):** feed the ranked wallet list to an existing bot
      (OctoBot Prediction Market; its copy-by-wallet feature is WIP upstream, so a
      thin `py-clob-client` fallback that polls target positions is the backup) →
      **paper-trade first**, then live. Only after the backtest demonstrates
      persistent edge on *real* ingested data.

## Caveats

- The synthetic generator produces an intentionally strong signal so the
  statistical machinery is easy to verify; real edges will be far smaller and may
  not survive frictions. The backtester exists precisely to find that out.
- `brier_skill` can be negative even for profitable traders: a trader who buys
  underpriced winners isn't necessarily *calibrated*. ROI/edge is the primary
  signal; Brier is a secondary diagnostic.
