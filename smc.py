"""smc.py - market structure, supply/demand, liquidity, imbalance, playbook."""
import numpy as np
import pandas as pd

BULL_PATTERNS = {"bullish engulfing", "hammer", "bullish pin"}
BEAR_PATTERNS = {"bearish engulfing", "shooting star", "bearish pin"}


def _tr(df) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    pc = np.roll(c, 1)
    if len(pc):
        pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr_series(df, n: int = 14) -> pd.Series:
    return pd.Series(_tr(df), index=df.index).rolling(n).mean()


# ---------------------------------------------------------------- structure
def swings(df, left: int = 3, right: int = 3):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    hi, lo = [], []
    for i in range(left, n - right):
        if h[i] >= h[i - left:i + right + 1].max():
            hi.append(i)
        if l[i] <= l[i - left:i + right + 1].min():
            lo.append(i)
    return hi, lo


def trend_state(df, lookback: int = 400) -> tuple[str, list[str]]:
    """up / down / range from the last confirmed swing sequence."""
    hi, lo = swings(df)
    cut = len(df) - lookback
    hi = [i for i in hi if i >= cut]
    lo = [i for i in lo if i >= cut]
    notes = []
    if len(hi) >= 2:
        if hi[-1] > hi[-2]:
            notes.append("higher highs")
        else:
            notes.append("lower highs")
    if len(lo) >= 2:
        if lo[-1] > lo[-2]:
            notes.append("higher lows")
        else:
            notes.append("lower lows")
    ups = sum(1 for n in notes if n.startswith("higher"))
    downs = len(notes) - ups
    if ups == 2:
        return "up", notes
    if downs == 2:
        return "down", notes
    return "range", notes


def structure_events(df, lookback: int = 400) -> list[dict]:
    """BOS and CHoCH. CHoCH is a break against the prevailing trend."""
    hi, lo = swings(df)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    cut = max(0, n - lookback)
    events = []
    trend = "range"
    hi_i = lo_i = 0
    for i in range(cut, n):
        while hi_i < len(hi) and hi[hi_i] < i - 1:
            last_h = hi[hi_i]
            hi_i += 1
        while lo_i < len(lo) and lo[lo_i] < i - 1:
            last_l = lo[lo_i]
            lo_i += 1
        if "last_h" in dir() and i > 0 and c[i] > h[last_h]:
            kind = "CHoCH" if trend == "down" else "BOS"
            events.append({"kind": kind, "dir": "bull", "idx": i,
                           "level": float(h[last_h]), "age": n - 1 - i})
            trend = "up"
        if "last_l" in dir() and i > 0 and c[i] < l[last_l]:
            kind = "CHoCH" if trend == "up" else "BOS"
            events.append({"kind": kind, "dir": "bear", "idx": i,
                           "level": float(l[last_l]), "age": n - 1 - i})
            trend = "down"
    return [e for e in events if e["age"] <= 40]


# ---------------------------------------------------------------- zones
def supply_demand(df, lookback: int = 300, imp_atr: float = 1.2,
                  max_age: int = 80) -> list[dict]:
    """Base of small candles followed by an impulse."""
    atr = atr_series(df)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    out = {}
    for i in range(max(4, n - lookback), n):
        a = atr.iloc[i]
        if not a or np.isnan(a) or abs(c[i] - o[i]) < imp_atr * a:
            continue
        up = c[i] > o[i]
        for j in range(i - 1, max(-1, i - 5), -1):
            if abs(c[j] - o[j]) < 0.45 * a:
                out[j] = {"dir": "demand" if up else "supply",
                          "top": float(max(h[j], o[j])),
                          "bot": float(min(l[j], o[j])),
                          "idx": j, "age": n - 1 - i}
                break
    return [z for z in out.values() if z["age"] <= max_age]


def order_blocks(df, lookback: int = 300, disp_atr: float = 1.2,
                 max_age: int = 80) -> list[dict]:
    atr = atr_series(df)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    seen = {}
    for i in range(max(5, n - lookback), n):
        a = atr.iloc[i]
        if not a or np.isnan(a) or abs(c[i] - o[i]) < disp_atr * a:
            continue
        up = c[i] > o[i]
        for j in range(i - 1, max(-1, i - 5), -1):
            if up and c[j] < o[j]:
                seen[j] = {"dir": "bull", "top": float(h[j]), "bot": float(l[j]),
                           "idx": j, "age": n - 1 - i}
                break
            if (not up) and c[j] > o[j]:
                seen[j] = {"dir": "bear", "top": float(h[j]), "bot": float(l[j]),
                           "idx": j, "age": n - 1 - i}
                break
    return [b for b in seen.values() if b["age"] <= max_age]


def fair_value_gaps(df, lookback: int = 300, min_atr: float = 0.35,
                    max_age: int = 80) -> list[dict]:
    if len(df) < 5:
        return []
    atr = atr_series(df)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    out = []
    for i in range(max(2, n - lookback), n):
        a = atr.iloc[i]
        if not a or np.isnan(a):
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) > min_atr * a:
            bot = h[i - 2]
            filled = bool(np.any(l[i + 1:] <= bot)) if i + 1 < n else False
            out.append({"dir": "bull", "top": float(l[i]), "bot": float(bot),
                        "idx": i, "age": n - 1 - i, "filled": filled})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) > min_atr * a:
            top = l[i - 2]
            filled = bool(np.any(h[i + 1:] >= top)) if i + 1 < n else False
            out.append({"dir": "bear", "top": float(top), "bot": float(h[i]),
                        "idx": i, "age": n - 1 - i, "filled": filled})
    return [f for f in out if f["age"] <= max_age and not f["filled"]]


def mitigated(zone: dict, df) -> bool:
    """Zone already traded back into."""
    if not zone:
        return False
    after = df.iloc[zone["idx"] + 1:]
    if after.empty:
        return False
    if zone.get("dir") in ("bull", "demand"):
        return bool((after["low"] <= zone["top"]).any())
    return bool((after["high"] >= zone["bot"]).any())


# ---------------------------------------------------------------- liquidity
def equal_levels(df, tol_atr: float = 0.25, lookback: int = 300) -> dict:
    """Clusters of near-identical highs and lows: stop-loss pools."""
    hi, lo = swings(df)
    atr = atr_series(df)
    a = float(atr.iloc[-1] or 0)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    cut = len(df) - lookback
    tol = tol_atr * a if a else 0.0
    def cluster(idxs, arr):
        idxs = [i for i in idxs if i >= cut]
        out = []
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                if abs(arr[idxs[x]] - arr[idxs[y]]) <= tol:
                    out.append({"a": idxs[x], "b": idxs[y],
                                "level": float((arr[idxs[x]] + arr[idxs[y]]) / 2)})
        return out
    return {"highs": cluster(hi, h), "lows": cluster(lo, l)}


def liquidity_sweeps(df, lookback: int = 300, width: int = 20,
                     max_age: int = 30) -> list[dict]:
    if len(df) < width + 3:
        return []
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    out = []
    for i in range(max(width, n - lookback), n):
        ph = h[i - width:i].max()
        pl = l[i - width:i].min()
        if h[i] > ph and c[i] < ph:
            out.append({"dir": "bear", "idx": i, "age": n - 1 - i,
                        "level": float(ph)})
        elif l[i] < pl and c[i] > pl:
            out.append({"dir": "bull", "idx": i, "age": n - 1 - i,
                        "level": float(pl)})
    return [s for s in out if s["age"] <= max_age]


def inducement(df, lookback: int = 200) -> list[dict]:
    """Minor swing sitting just before a bigger one: the trap level."""
    hi, lo = swings(df)
    atr = atr_series(df)
    a = float(atr.iloc[-1] or 0)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    cut = len(df) - lookback
    out = []
    for x in range(1, len(hi)):
        if hi[x] < cut or hi[x - 1] < cut:
            continue
        gap = h[hi[x]] - h[hi[x - 1]]
        if 0 < gap <= 0.6 * a:
            out.append({"dir": "bear", "idx": hi[x - 1], "level": float(h[hi[x - 1]])})
    for x in range(1, len(lo)):
        if lo[x] < cut or lo[x - 1] < cut:
            continue
        gap = l[lo[x - 1]] - l[lo[x]]
        if 0 < gap <= 0.6 * a:
            out.append({"dir": "bull", "idx": lo[x - 1], "level": float(l[lo[x - 1]])})
    return out[-12:]


# ---------------------------------------------------------------- ranges
def premium_discount(df, lookback: int = 120) -> float:
    if len(df) < 5:
        return 50.0
    hi = float(df["high"].tail(lookback).max())
    lo = float(df["low"].tail(lookback).min())
    if hi <= lo:
        return 50.0
    return float((float(df["close"].iloc[-1]) - lo) / (hi - lo) * 100)


def fib_levels(df, lookback: int = 120) -> dict:
    hi = float(df["high"].tail(lookback).max())
    lo = float(df["low"].tail(lookback).min())
    rng = hi - lo
    if rng <= 0:
        return {}
    return {"high": hi, "low": lo,
            "0.382": hi - rng * 0.382, "0.5": hi - rng * 0.5,
            "0.618": hi - rng * 0.618, "0.786": hi - rng * 0.786,
            "ext_1.272": hi + rng * 0.272, "ext_1.618": hi + rng * 0.618}


# ---------------------------------------------------------------- candles
def patterns(df, look: int = 3) -> list[str]:
    if len(df) < 3:
        return []
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    found = []
    for i in range(len(df) - look, len(df)):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        upper = h[i] - max(o[i], c[i])
        lower = min(o[i], c[i]) - l[i]
        if c[i] > o[i] and lower > body * 2:
            found.append("hammer")
        if c[i] < o[i] and upper > body * 2:
            found.append("shooting star")
        if i > 0:
            if c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] > o[i - 1] and o[i] < c[i - 1]:
                found.append("bullish engulfing")
            if c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] < o[i - 1] and o[i] > c[i - 1]:
                found.append("bearish engulfing")
    return sorted(set(found))


def session_info(ts) -> dict:
    try:
        hour = pd.Timestamp(ts).tz_convert("UTC").hour
    except Exception:
        hour = 12
    if 0 <= hour < 7:
        name, vol = "Asia", "low to medium"
    elif 7 <= hour < 12:
        name, vol = "London", "high"
    elif 12 <= hour < 16:
        name, vol = "London / New York overlap", "highest"
    elif 16 <= hour < 21:
        name, vol = "New York", "high"
    else:
        name, vol = "late New York", "low"
    return {"name": name, "vol": vol, "hour": hour}


# ---------------------------------------------------------------- confluence
def confluence(df, direction: str, entry: float, atr: float,
               tol_atr: float = 0.6) -> dict:
    flags, warnings = [], []
    score = 0
    band = tol_atr * atr if atr else 0.0
    trend, tnotes = trend_state(df)
    pd_ = premium_discount(df)
    want = "bull" if direction == "long" else "bear"

    if (direction == "long" and trend == "up") or (direction == "short" and trend == "down"):
        score += 2
        flags.append(f"trend agrees ({trend})")
    elif trend == "range":
        flags.append("range — trend neutral")
    else:
        score -= 1
        warnings.append(f"counter-trend ({trend})")

    ev = [e for e in structure_events(df) if e["dir"] == want]
    if ev:
        e = ev[-1]
        score += 2 if e["kind"] == "CHoCH" else 1
        flags.append(f"{e['kind']} {e['dir']} {e['age']} bars ago")

    zones = [z for z in supply_demand(df) if z["dir"] == ("demand" if want == "bull" else "supply")]
    zone = next((z for z in zones if z["bot"] - band <= entry <= z["top"] + band), None)
    if zone:
        score += 2
        flags.append(f"{zone['dir']} zone at entry")
        if mitigated(zone, df):
            warnings.append("zone already mitigated once")

    ob = [b for b in order_blocks(df) if b["dir"] == want]
    near_ob = next((b for b in ob if b["bot"] - band <= entry <= b["top"] + band), None)
    if near_ob:
        score += 2
        flags.append("order block at entry")

    gaps = [f for f in fair_value_gaps(df) if f["dir"] == want]
    near_fvg = next((f for f in gaps if f["bot"] - band <= entry <= f["top"] + band), None)
    if near_fvg:
        score += 1
        flags.append("unfilled gap at entry")

    sweeps = [s for s in liquidity_sweeps(df) if s["dir"] == want and s["age"] <= 5]
    if sweeps:
        score += 2
        flags.append("recent liquidity sweep")

    eq = equal_levels(df)
    pool = eq["lows"] if want == "bull" else eq["highs"]
    if pool:
        flags.append(f"{len(pool)} equal level cluster(s) nearby")

    ind = [i for i in inducement(df) if i["dir"] == want]
    if ind:
        warnings.append("inducement level overhead")

    if direction == "long":
        if pd_ <= 35:
            score += 1
            flags.append(f"discount {pd_:.0f}%")
        elif pd_ >= 65:
            score -= 1
            warnings.append(f"premium {pd_:.0f}%")
    else:
        if pd_ >= 65:
            score += 1
            flags.append(f"premium {pd_:.0f}%")
        elif pd_ <= 35:
            score -= 1
            warnings.append(f"discount {pd_:.0f}%")

    pats = patterns(df)
    good = BULL_PATTERNS if want == "bull" else BEAR_PATTERNS
    if any(p in good for p in pats):
        score += 1
        flags.append("confirmation candle")

    sess = session_info(df.index[-1])
    if sess["vol"] in ("high", "highest"):
        score += 1
        flags.append(f"{sess['name']} session")

    return {"score": score, "flags": flags, "warnings": warnings,
            "zone": zone, "ob": near_ob, "fvg": near_fvg, "pd": pd_,
            "trend": trend, "trend_notes": tnotes, "sweep": bool(sweeps),
            "session": sess}


# ---------------------------------------------------------------- playbook
def playbook(df, direction: str, equity: float, risk_pct: float) -> dict:
    """Full A to Z plan with a pass/fail checklist."""
    atr = float(atr_series(df).iloc[-1] or 0)
    px = float(df["close"].iloc[-1])
    want = "bull" if direction == "long" else "bear"
    long = direction == "long"

    conf = confluence(df, direction, px, atr)
    fib = fib_levels(df)

    zone = conf["zone"] or conf["ob"]
    if zone:
        entry = zone["top"] if long else zone["bot"]
        sl = zone["bot"] - 0.3 * atr if long else zone["top"] + 0.3 * atr
    else:
        entry = px
        sl = px - 1.5 * atr if long else px + 1.5 * atr

    risk = abs(entry - sl)
    if risk <= 0:
        risk = atr or px * 0.01
        sl = entry - risk if long else entry + risk

    tp1 = fib.get("0.5", entry + risk * 2 if long else entry - risk * 2)
    tp2 = fib.get("ext_1.272", entry + risk * 3 if long else entry - risk * 3)
    if long:
        tp1 = max(tp1, entry + risk)
        tp2 = max(tp2, entry + risk * 2)
    else:
        tp1 = min(tp1, entry - risk)
        tp2 = min(tp2, entry - risk * 2)

    rr1 = abs(tp1 - entry) / risk
    rr2 = abs(tp2 - entry) / risk
    size_usd = equity * risk_pct / 100
    qty = size_usd / risk if risk else 0

    checks = [
        ("Trend agrees", conf["trend"] == ("up" if long else "down")),
        ("Structure break present", bool([e for e in structure_events(df)
                                          if e["dir"] == want])),
        ("In a zone or order block", zone is not None),
        ("Discount/premium correct",
         (conf["pd"] <= 45) if long else (conf["pd"] >= 55)),
        ("Liquidity taken", conf["sweep"]),
        ("R:R at least 2", rr1 >= 2),
        ("Stop inside 3 ATR", abs(entry - sl) <= 3 * atr if atr else True),
    ]
    passed = sum(1 for _, ok in checks if ok)
    verdict = ("A — take it" if passed >= 6 else
               "B — take with smaller size" if passed >= 5 else
               "C — watch only" if passed >= 4 else "D — skip")

    return {"direction": direction, "entry": entry, "sl": sl, "tp1": tp1,
            "tp2": tp2, "risk": risk, "rr1": rr1, "rr2": rr2, "qty": qty,
            "size_usd": size_usd, "checks": checks, "passed": passed,
            "verdict": verdict, "confluence": conf, "atr": atr, "fib": fib}


def ladder(plan: dict, tranches: int = 3) -> list[dict]:
    """Split a spot entry into staged buys into the zone."""
    zone_top = plan["entry"]
    zone_bot = plan["sl"] + (plan["atr"] * 0.3 if plan["atr"] else 0)
    if zone_top <= zone_bot:
        zone_bot = zone_top * 0.985
    weights = [0.4, 0.35, 0.25][:tranches]
    out = []
    for i, w in enumerate(weights):
        price = zone_top - (zone_top - zone_bot) * (i / max(tranches - 1, 1))
        out.append({"tranche": i + 1, "weight": w, "price": price,
                    "qty": plan["qty"] * w})
    return out
