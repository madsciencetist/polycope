#!/usr/bin/env bash
# Weekly paper-trading cycle: refresh the lake -> re-rank the cohort -> advance the
# simulated portfolio. NO real orders are placed; this only reads public Polymarket
# data and updates data/paper_state.json. Safe to run from cron or a systemd timer.
#
#   POLYCOPE_PYTHON=/path/to/python ./scripts/paper_cycle.sh   # override interpreter
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Interpreter: explicit override, else the repo venv, else system python3.
if [[ -n "${POLYCOPE_PYTHON:-}" ]]; then
  PY="$POLYCOPE_PYTHON"
elif [[ -x "$REPO/.venv/bin/python3" ]]; then
  PY="$REPO/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "paper_cycle start  (repo=$REPO  python=$PY)"
log "step 1/3: ingest (refresh leaderboard + trade history + resolutions)"
"$PY" scripts/run_ingest.py --wallets 2000 --min-pnl 100
log "step 2/3: rank (recompute top-15 cohort)"
"$PY" scripts/run_rank.py --min-bets 50 --top 15
log "step 3/3: paper advance (copy new BUYs, settle resolved, persist state)"
"$PY" scripts/run_paper.py
log "paper_cycle done"
