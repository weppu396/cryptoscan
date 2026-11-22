"""strategy/reversal.py - 30m context, 3m trigger.

Nothing here is new maths. It calls the existing engines:
  zones      core.levels.build_zones
  structure  core.structure.build_structure
  candles    core.candles.quality   (the ATR-normalised grader already used
                                    by strategy/setups.py)
"""
import numpy as np

from core.candles import quality as candle_quality
from core.indicators import compute_all
from core.levels import build_zones
from core.structure import build_structure, find_swings

HTF, LTF = "30m", "3m"

APPROACH_ATR = 1.5     # "approaching" band
WATCH_ATR = 6.0        # furthest a coin stays on the radar
BEYOND_ATR = 1.0       # past this, the entry window is gone
MAX_EXT_ATR = 4.0      # travel over the lookback that counts as extended
EXT_LOOKBACK = 12
CONFIRM_WINDOW = 15
MIN_CANDLE_Q = 60

STATUSES = ("APPROACHING", "DEVELOPING", "CONFIRMED", "ACTIVE_ENTRY",
            "MISSED", "INVALIDATED")
RANK = {"ACTIVE_ENTRY": 100, "CONFIRMED": 78, "DEVELOPING": 52,
        "APPROACHING": 34, "MISSED": 12, "INVALIDATED": 0}


def _atr(df, cfg) -> float:
    try:
        v = float(compute_all(df, cfg)["atr"].iloc[-1])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _candidate(zones, price, atr, long):
    best = None
    for z in zones:
        if long and z["high"] > price:
            continue
        if not long and z["low"] < price:
            continue
        dist = ((price - z["high"]) if long else (z["low"] - price)) / atr
        if dist < 0 or dist > WATCH_ATR:
            continue
        if best is None or dist < best[1]:
            best = (z, dist)
    return best


def _targets(zones, entry, risk, long, cap_r):
    pool = [z["mid"] for z in zones]
    if long:
        ahead = sorted(p for p in pool if p > entry + 0.5 * risk)
        tp1 = ahead[0] if ahead else entry + 1.5 * risk
        tp2 = ahead[1] if len(ahead) > 1 else entry + 3.0 * risk
        return tp1, min(tp2, entry + cap_r * risk)
    ahead = sorted((p for p in pool if p < entry - 0.5 * risk), reverse=True)
    tp1 = ahead[0] if ahead else entry - 1.5 * risk
    tp2 = ahead[1] if len(ahead) > 1 else entry - 3.0 * risk
    return tp1, max(tp2, entry - cap_r * risk)


def evaluate(symbol, hdf, ldf, cfg, already: bool = False):
    """One coin. Returns a dict or None when it is not worth a row."""
    if hdf is None or ldf is None or len(hdf) < 60 or len(ldf) < 120:
        return None

    ah, al = _atr(hdf, cfg), _atr(ldf, cfg)
    if ah <= 0 or al <= 0:
        return None

    price = float(ldf["close"].iloc[-1])
    stt_h = build_structure(hdf, cfg.structure)
    bias = str(stt_h["trend"].iloc[-1])

    sw = find_swings(hdf, cfg.structure.swing_left, cfg.structure.swing_right)
    dem = build_zones(hdf, sw, compute_all(hdf, cfg)["atr"], cfg.structure, "sup")
    sup = build_zones(hdf, sw, compute_all(hdf, cfg)["atr"], cfg.structure, "res")

    long_hit = _candidate(dem, price, ah, True)
    short_hit = _candidate(sup, price, ah, False)

    if long_hit and (not short_hit or long_hit[1] <= short_hit[1]):
        long, (zone, dist) = True, long_hit
    elif short_hit:
        long, (zone, dist) = False, short_hit
    else:
        return None

    if long and bias == "up":
        return None          # continuation, not a reversal
    if not long and bias == "down":
        return None

    entry = zone["high"] if long else zone["low"]
    stop = (zone["low"] - cfg.risk.stop_buffer_atr * al if long
            else zone["high"] + cfg.risk.stop_buffer_atr * al)
    risk = abs(entry - stop)
    if risk <= 0 or risk / al > cfg.risk.max_stop_atr:
        return None

    tp1, tp2 = _targets(sup if long else dem, entry, risk, long, cfg.risk.max_tp_r)
    rr = abs(tp1 - entry) / risk

    if long:
        inside = zone["low"] <= price <= zone["high"]
        beyond = price > zone["high"]
        approach = price < zone["low"]
    else:
        inside = zone["low"] <= price <= zone["high"]
        beyond = price < zone["low"]
        approach = price > zone["high"]

    lb = min(EXT_LOOKBACK, len(hdf) - 1)
    travel = abs(price - float(hdf["close"].iloc[-1 - lb])) / ah if lb else 0.0
    extended = travel > MAX_EXT_ATR

    stt_l = build_structure(ldf, cfg.structure)
    win = len(ldf) - 1 - CONFIRM_WINDOW
    pick = ("bos_up", "choch_up") if long else ("bos_dn", "choch_dn")
    sweep = bool(stt_l["choch_" + ("up" if long else "dn")].iloc[win:].any())
    broken = bool(stt_l[pick[0]].iloc[win:].any() or sweep)

    cq = 0
    for i in range(max(2, len(ldf) - 5), len(ldf)):
        cq = max(cq, candle_quality(ldf, i, al, "long" if long else "short")[1])
    confirmed = broken and cq >= MIN_CANDLE_Q

    if (long and price < stop) or (not long and price > stop):
        status = "INVALIDATED"
    elif inside and confirmed:
        status = "ACTIVE_ENTRY" if (rr >= cfg.risk.min_rr and not extended
                                    and not already) else "CONFIRMED"
    elif inside:
        status = "DEVELOPING"
    elif beyond:
        status = "MISSED" if (extended or dist > BEYOND_ATR) else "CONFIRMED"
    else:
        status = "APPROACHING"

    score = RANK[status]
    score += 6.0 * min(int(zone.get("touches", 1)), 4)
    score += min(rr, cfg.risk.max_tp_r) * 4.0
    score += 10.0 if sweep else 0.0
    score -= min(dist, 8.0) * 2.5
    score -= 15.0 if extended else 0.0

    reasons = []
    if sweep:
        reasons.append("3m CHoCH")
    if cq:
        reasons.append(f"candle {cq}")
    if extended:
        reasons.append("extended")
    if rr < cfg.risk.min_rr:
        reasons.append(f"R:R {rr:.2f}")
    if already:
        reasons.append("in book")

    return {"symbol": symbol, "direction": "long" if long else "short",
            "status": status, "zone": (zone["low"], zone["high"]),
            "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "rr": rr,
            "score": round(score, 1), "is_new": not already,
            "eta_min": round(dist * 30.0, 1) if status == "APPROACHING" else None,
            "bias": bias, "dist_atr": round(dist, 2),
            "reasons": reasons, "ts": ldf.index[-1]}


def rank(rows):
    """(live, dead). Approaching first, never an unconfirmed row above a
    confirmed one."""
    live = [r for r in rows if r["status"] in
            ("ACTIVE_ENTRY", "CONFIRMED", "DEVELOPING", "APPROACHING")]
    dead = [r for r in rows if r["status"] in ("MISSED", "INVALIDATED")]
    live.sort(key=lambda r: (-r["score"], r["eta_min"] or 9e9))
    dead.sort(key=lambda r: -r["score"])
    return live, dead


def active_entries(rows):
    return [r for r in rows if r["status"] == "ACTIVE_ENTRY" and r["is_new"]]
