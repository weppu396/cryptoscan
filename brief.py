"""brief.py - deterministic desk note."""

import pandas as pd


def compose(health, macro, top_setups, port_rows, port_sum):
    L = [f"DESK NOTE  {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC", ""]
    L.append(f"MARKET HEALTH {health['score']}/100 ({health['verdict']})")
    for r in health["reasons"]:
        L.append(f"  - {r}")
    L.append("")
    L.append(f"MACRO blackout={'YES' if macro.get('blackout') else 'no'} "
             f"funding={macro.get('funding_annualised_pct')} "
             f"tone={macro.get('sentiment')}")
    for n in macro.get("notes", []):
        L.append(f"  - {n}")
    L.append(f"  size multiplier x{macro.get('multiplier', 1.0)}")
    L.append("")
    L.append(f"PORTFOLIO {port_sum['positions']} positions "
             f"${port_sum['value']:,.0f} P&L {port_sum['pnl_pct']:+.2f}%")
    for r in port_rows:
        if r["severity"] >= 2:
            L.append(f"  EXIT {r['symbol']}: {'; '.join(r['signals'])}")
    L.append("")
    L.append(f"NEW SETUPS {len(top_setups)}")
    for s in top_setups[:5]:
        rt = s.meta.get("rating")
        L.append(f"  {rt.grade if rt else '?'} {s.symbol} {s.direction.upper()} "
                 f"entry {s.entry:.6f} stop {s.stop:.6f} rr {s.rr:.2f}")
    if not top_setups:
        L.append("  none - wait")
    return "\n".join(L)
