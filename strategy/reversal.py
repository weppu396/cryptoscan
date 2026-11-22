"""strategy/reversal.py - auto-selected pair. HTF 30m/1h, LTF 3m/5m.

The higher timeframe is chosen from 30m or 1h by structure quality, never
slower than 1h. The lower timeframe is chosen from 3m or 5m — never slower than
5m — by running the full assessment on each and keeping the best setup.

30m and 1h are resampled from the loaded 15m frame, so a lower timeframe is
never invented. Confirmation reuses core.candles.quality, the existing ATR
grader.
"""
import numpy as np

from core.candles import quality as candle_quality
from core.indicators import compute_all
from core.levels import build_zones
from core.structure import build_structure, find_swings
from core.tfs import MINUTES, cap_htf, cap_ltf, resample

APPROACH_ATR = 1.5
WATCH_ATR = 6.0
BEYOND_ATR = 1.0
MAX_EXT_ATR = 4.0
EXT_LOOKBACK = 12
CONFIRM_WINDOW = 15
MIN_CANDLE_Q = 60

# loaded frames the caller must supply
BASE = "15m"          # context frame, also the parent for 30m / 1h
FAST = "5m"
SCALP = "3m"

# load budgets. 250 HTF / 400 LTF is a starting point, not a result.
HTF_BARS = 250
LTF_BARS = 400
MIN_HTF_BARS = 60
MIN_LTF_BARS = 120

# HTF capped at 1h, LTF capped at 5m. Nothing else is allowed.
HTF_CHOICES = ("30m", "1h")
LTF_CHOICES = ("3m", "5m")
LTF_FOR = {"30m": LTF_CHOICES, "1h": LTF_CHOICES}
COARSE = ("30m", "1h")

STATUSES = ("APPROACHING", "DEVELOPING", "CONFIRMED", "ACTIVE_ENTRY",
            "MISSED", "INVALIDATED")
FUTURE_STATUSES = ("APPROACHING", "DEVELOPING", "ACTIVE_ENTRY")
RANK = {"ACTIVE_ENTRY": 100, "CONFIRMED": 78, "DEVELOPING": 52,
        "APPROACHING": 34, "MISSED": 12, "INVALIDATED": 0}


def _atr(df, cfg) -> float:
    try:
        v = float(compute_all(df, cfg)["atr"].iloc[-1])
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _frame(src: dict, name: str):
    """Loaded frame, or a resample of 15m when `name` is coarser."""
    if name in src and src[name] is not None:
        return src[name]
    if name in COARSE:
        base = src.get(BASE)
        if base is None or not len(base):
            return None
        try:
            return resample(base, name)
        except Exception:
            return None
    return None


def quality(hdf, ldf, cfg) -> float:
    """Higher = cleaner structure. -1 means unusable."""
    if hdf is None or ldf is None or len(hdf) < MIN_HTF_BARS \
            or len(ldf) < MIN_LTF_BARS:
        return -1.0
    atr = _atr(hdf, cfg)
    if atr <= 0:
        return -1.0
    sw = find_swings(hdf, cfg.structure.swing_left, cfg.structure.swing_right)
    highs = [s for s in sw if s["type"] == "H"]
    lows = [s for s in sw if s["type"] == "L"]
    if len(highs) < 3 or len(lows) < 3:
        return -1.0
    sep = abs(highs[-1]["price"] - lows[-1]["price"]) / atr
    q = min(sep, 12.0) * 2.0
    q += min(len(highs) + len(lows), 12) * 0.6
    px = float(hdf["close"].iloc[-1])
    atr_pct = atr / px * 100 if px else 0.0
    if 0.25 <= atr_pct <= 4.0:
        q += 4.0
    return q


def choose_htf(src: dict, cfg) -> str | None:
    """Highest-scoring HTF from 30m / 1h. Never slower than 1h."""
    best, best_q = None, -1.0
    for htf in HTF_CHOICES:
        htf = cap_htf(htf)
        q = quality(_frame(src, htf), _frame(src, LTF_CHOICES[0]), cfg)
        if q > best_q:
            best, best_q = htf, q
    return best if best_q >= 0 else None


def choose_pair(src: dict, cfg) -> tuple:
    """(htf, ltf, quality). HTF in 30m/1h, LTF in 3m/5m, best structure wins.

    Used for display so a card title can never show 4h on the trigger side.
    """
    best = (HTF_CHOICES[0], LTF_CHOICES[0], -1.0)
    for htf in HTF_CHOICES:
        hdf = _frame(src, cap_htf(htf))
        for ltf in LTF_CHOICES:
            q = quality(hdf, _frame(src, ltf), cfg)
            if q > best[2]:
                best = (cap_htf(htf), ltf, q)
    return best


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


def _assess(symbol, hdf, ldf, htf, ltf, cfg, already: bool, q: float):
    """One HTF/LTF combination. Dict, or None when not worth a row."""
    if hdf is None or ldf is None or len(hdf) < MIN_HTF_BARS \
            or len(ldf) < MIN_LTF_BARS:
        return None
    ah, al = _atr(hdf, cfg), _atr(ldf, cfg)
    if ah <= 0 or al <= 0:
        return None

    price = float(ldf["close"].iloc[-1])
    stt_h = build_structure(hdf, cfg.structure)
    bias = str(stt_h["trend"].iloc[-1])

    ind_h = compute_all(hdf, cfg)
    sw = find_swings(hdf, cfg.structure.swing_left, cfg.structure.swing_right)
    dem = build_zones(hdf, sw, ind_h["atr"], cfg.structure, "sup")
    sup = build_zones(hdf, sw, ind_h["atr"], cfg.structure, "res")

    lh, sh = _candidate(dem, price, ah, True), _candidate(sup, price, ah, False)
    if lh and (not sh or lh[1] <= sh[1]):
        long, (zone, dist) = True, lh
    elif sh:
        long, (zone, dist) = False, sh
    else:
        return None

    if long and bias == "up":
        return None
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

    inside = zone["low"] <= price <= zone["high"]
    beyond = price > zone["high"] if long else price < zone["low"]

    lb = min(EXT_LOOKBACK, len(hdf) - 1)
    travel = abs(price - float(hdf["close"].iloc[-1 - lb])) / ah if lb else 0.0
    extended = travel > MAX_EXT_ATR

    stt_l = build_structure(ldf, cfg.structure)
    win = len(ldf) - 1 - CONFIRM_WINDOW
    chop = "choch_up" if long else "choch_dn"
    bos = "bos_up" if long else "bos_dn"
    sweep = bool(stt_l[chop].iloc[win:].any())
    broken = bool(stt_l[bos].iloc[win:].any() or sweep)

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

    future = (status in FUTURE_STATUSES and rr >= cfg.risk.min_rr
              and not extended and not already)

    score = RANK[status] + 6.0 * min(int(zone.get("touches", 1)), 4)
    score += min(rr, cfg.risk.max_tp_r) * 4.0 + (10.0 if sweep else 0.0)
    score += max(q, 0.0) * 0.5
    score -= min(dist, 8.0) * 2.5 + (15.0 if extended else 0.0)
    if not future:
        score -= 12.0

    reasons = [f"pair {cap_htf(htf)}/{cap_ltf(ltf)}"]
    if sweep:
        reasons.append(f"{ltf} CHoCH")
    if cq:
        reasons.append(f"candle {cq}")
    if extended:
        reasons.append("extended")
    if rr < cfg.risk.min_rr:
        reasons.append(f"R:R {rr:.2f}")
    if already:
        reasons.append("in book")
    reasons.append("executable" if future else "past / played")

    return {"symbol": symbol, "direction": "long" if long else "short",
            "status": status, "zone": (zone["low"], zone["high"]),
            "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2, "rr": rr,
            "score": round(score, 1), "is_new": not already, "future": future,
            "htf": cap_htf(htf), "ltf": cap_ltf(ltf),
            "pair_q": round(max(q, 0.0), 1),
            "eta_min": (round(dist * MINUTES.get(cap_htf(htf), 60), 1)
                        if status == "APPROACHING" else None),
            "bias": bias, "dist_atr": round(dist, 2),
            "reasons": reasons, "ts": ldf.index[-1]}


def evaluate(symbol, src: dict, cfg, already: bool = False):
    """One coin. HTF by structure, then the LTF whose setup scores highest."""
    htf = choose_htf(src, cfg)
    if htf is None:
        return None
    hdf = _frame(src, htf)
    if hdf is None or len(hdf) < MIN_HTF_BARS:
        return None
    q = quality(hdf, _frame(src, LTF_CHOICES[0]), cfg)
    best = None
    for ltf in LTF_CHOICES:
        r = _assess(symbol, hdf, _frame(src, ltf), htf, ltf, cfg, already, q)
        if r is None:
            continue
        if best is None or r["score"] > best["score"]:
            best = r
    return best


def rank(rows):
    """(live, dead). Executable first, never a played setup above a live one."""
    live = [r for r in rows if r["status"] in
            ("ACTIVE_ENTRY", "CONFIRMED", "DEVELOPING", "APPROACHING")]
    dead = [r for r in rows if r["status"] in ("MISSED", "INVALIDATED")]
    live.sort(key=lambda r: (not r.get("future", False),
                             -r["score"], r["eta_min"] or 9e9))
    dead.sort(key=lambda r: -r["score"])
    return live, dead


def future_only(rows):
    """Only setups a limit order could still fill."""
    return [r for r in rows if r.get("future")]


def active_entries(rows):
    return [r for r in rows if r["status"] == "ACTIVE_ENTRY" and r["is_new"]
            and r.get("future")]
