"""Human-readable summary of a backtest result dict (no plotting deps required)."""

from __future__ import annotations


def format_report(result: dict) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("BACKTEST REPORT")
    lines.append("=" * 56)
    lines.append(f"Initial bankroll : ${result['initial_bankroll']:,.2f}")
    lines.append(f"Final equity     : ${result['final_equity']:,.2f}")
    lines.append(f"Total return     : {result['total_return'] * 100:+.2f}%")
    lines.append(f"Copied trades    : {result['n_copied']}")
    wr = result["win_rate"]
    lines.append(f"Win rate         : {wr * 100:.1f}%" if wr == wr else "Win rate         : n/a")
    lines.append(f"Max drawdown     : {result['max_drawdown'] * 100:.1f}%")
    lines.append("-" * 56)
    lines.append("By market duration (answers the market-scope question):")
    if result["by_bucket"]:
        for bucket, s in sorted(result["by_bucket"].items()):
            roi = s["roi"]
            roi_s = f"{roi * 100:+.1f}%" if roi == roi else "n/a"
            lines.append(
                f"  {bucket:<9} n={s['n']:<5} invested=${s['invested']:>10,.0f} roi={roi_s}"
            )
    else:
        lines.append("  (no settled trades)")
    lines.append("=" * 56)
    return "\n".join(lines)
