# Scheduling the paper-trading cycle

`scripts/paper_cycle.sh` runs one full cycle: **ingest → rank → paper advance**.
It places no real orders — it only reads public Polymarket data and updates the
simulated portfolio in `data/paper_state.json`. The systemd units here run it
weekly as a **user** service (no root required).

## Install (user-level systemd)

```bash
# from the repo root
mkdir -p ~/.config/systemd/user
cp deploy/polycope-paper.service ~/.config/systemd/user/
cp deploy/polycope-paper.timer   ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now polycope-paper.timer

# so it runs even when you're not logged in (recommended for a scheduled job)
loginctl enable-linger "$USER"
```

## Verify / operate

```bash
systemctl --user list-timers polycope-paper.timer      # next scheduled run
systemctl --user start polycope-paper.service          # run one cycle now
journalctl --user -u polycope-paper.service -f          # follow logs
systemctl --user status polycope-paper.service          # last run result
```

## Notes

- **Paths are hardcoded** to `/home/andrew/local/polycope` in the unit files. If the
  repo moves, edit `WorkingDirectory`, `ExecStart`, and `Documentation` accordingly.
- **Interpreter** defaults to the repo's `.venv/bin/python3` (auto-detected by the
  script). Override with `POLYCOPE_PYTHON` (env, or uncomment the line in the
  `.service`).
- **First run** starts the paper clock "now" and copies only future trades, so
  results accumulate as clean forward out-of-sample evidence. To seed from recent
  history instead, run once manually: `scripts/run_paper.py --replay-days 30 --reset`.
- **Cadence** is set in the `.timer` (`OnCalendar=Mon *-*-* 06:00:00`). Change and
  `systemctl --user daemon-reload && systemctl --user restart polycope-paper.timer`.
- **cron alternative** (if you prefer): `0 6 * * 1 /home/andrew/local/polycope/scripts/paper_cycle.sh >> /home/andrew/local/polycope/data/paper_cycle.log 2>&1`
