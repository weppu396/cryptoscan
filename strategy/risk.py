import numpy as np

from core.regime import regime_ok


def plan_rr(entry, stop, tp1, tp2, split=0.5) -> dict:
    """Per-target and weighted R. Direction-agnostic: BUY and SELL both work.

    `split` is the fraction taken at TP1. The weighted figure is the R you
    actually expect if both legs fill, and is the honest headline number.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return {"risk": 0.0, "rr1": 0.0, "rr2": 0.0, "weighted": 0.0}
    direction = 1 if entry > stop else -1
    rr1 = (tp1 - entry) * direction / risk
    rr2 = (tp2 - entry) * direction / risk
    split = min(max(split, 0.0), 1.0)
    return {"risk": risk, "rr1": rr1, "rr2": rr2,
            "weighted": split * rr1 + (1.0 - split) * rr2}


def score_setup(s, ind_row, zone, cfg) -> float:
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


def filter_setups(setups, ind, zones_up, zones_dn, cfg, *, zones_at=None):
    out = []
    for s in setups:
        try:
            row = ind.loc[s.ts]
        except KeyError:
            continue
        ok, why = regime_ok(row, s.direction, cfg)
        # Range reversals deliberately require a low-ADX environment; retain
        # the volatility and RSI safety vetoes, but not the trend-strength veto.
        if not ok and s.meta.get("strategy") == "Range Reversal" and "chop" in why:
            ok = (cfg.regime.atr_pct_min <= row["atr_pct"] <= cfg.regime.atr_pct_max
                  and (s.direction != "long" or row["rsi"] <= cfg.regime.rsi_long_max)
                  and (s.direction != "short" or row["rsi"] >= cfg.regime.rsi_short_min))
        if not ok:
            continue
        g_ok, g_why = gate(s, row, cfg)
        if not g_ok:
            continue
        up, down = zones_at(ind.index.get_loc(s.ts)) if zones_at else (zones_up, zones_dn)
        candidates = up if s.direction == "long" else down
        zone = candidates[0] if candidates else None
        s.score = score_setup(s, row, zone, cfg)
        keep = dict(s.meta or {})
        s.meta = {**keep, **size_position(s.entry, s.stop, cfg.risk.account_equity, cfg)}
        out.append(s)
    return out
