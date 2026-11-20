import numpy as np

from core.regime import regime_ok


def score_setup(s, ind_row, zone, cfg) -> float:
    """0-100 ranking score. Higher = cleaner confluence. Ranking beats binary."""
    pts = 0.0
    pts += min(s.rr / 3.0, 1.0) * 25
    pts += 25 if cfg.risk.min_stop_atr <= s.stop_atr <= 1.5 else 10
    pts += min(ind_row["vol_ratio"] / 2.0, 1.0) * 15
    pts += min(ind_row["adx"] / 40.0, 1.0) * 15
    pts += min((zone["touches"] if zone else 1) / 4.0, 1.0) * 20
    pts += min(s.meta.get("candle_q", 0) / 100.0, 1.0) * 10
    return round(min(pts, 100.0), 1)


def size_position(entry, stop, equity, cfg) -> dict:
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return {}
    risk_cash = equity * cfg.risk.risk_per_trade_pct / 100.0
    qty = risk_cash / stop_dist
    return {
        "qty": qty,
        "notional": qty * entry,
        "risk_cash": risk_cash,
        "risk_pct": cfg.risk.risk_per_trade_pct,
        "leverage_needed": round(qty * entry / equity, 2),
    }


def gate(s, ind_row, cfg) -> tuple[bool, list[str]]:
    reasons = []
    if s.rr < cfg.risk.min_rr:
        reasons.append(f"R:R {s.rr:.2f} < {cfg.risk.min_rr}")
    if s.stop_atr > cfg.risk.max_stop_atr:
        reasons.append(f"stop {s.stop_atr:.2f} ATR too wide")
    if s.stop_atr < cfg.risk.min_stop_atr:
        reasons.append(f"stop {s.stop_atr:.2f} ATR too tight (noise)")
    if s.meta.get("candle_q", 0) < 60:
        reasons.append("no valid reversal candle")
    return (len(reasons) == 0), reasons


def filter_setups(setups, ind, zones_up, zones_dn, cfg):
    """One shared filter used by BOTH scanner and backtest. Indicators are
    looked up at the setup's OWN timestamp. Builder meta is preserved."""
    out = []
    for s in setups:
        try:
            row = ind.loc[s.ts]
        except KeyError:
            continue
        ok, why = regime_ok(row, s.direction, cfg)
        if not ok:
            continue
        g_ok, g_why = gate(s, row, cfg)
        if not g_ok:
            continue
        zone = (zones_up[0] if zones_up else None) if s.direction == "long" \
               else (zones_dn[0] if zones_dn else None)
        s.score = score_setup(s, row, zone, cfg)
        keep = dict(s.meta or {})
        s.meta = {**keep, **size_position(s.entry, s.stop, cfg.risk.account_equity, cfg)}
        out.append(s)
    return out
