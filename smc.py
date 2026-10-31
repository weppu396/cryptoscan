"""smc.py - smart money confluence: fair value gaps, order blocks, sweeps,
premium/discount. Pure functions over a OHLCV frame."""
import numpy as np
import pandas as pd


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


def fair_value_gaps(df, lookback: int = 250, min_atr: float = 0.35,
                    max_age: int = 60) -> list[dict]:
    """Three-candle imbalances that are still unfilled."""
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


def order_blocks(df, lookback: int = 250, disp_atr: float = 1.2,
                 max_age: int = 60) -> list[dict]:
    """Last opposite candle before a displacement leg."""
    if len(df) < 8:
        return []
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


def liquidity_sweeps(df, lookback: int = 250, width: int = 20,
                     max_age: int = 30) -> list[dict]:
    """Wick through a prior extreme that closes back inside."""
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
            out.append({"dir": "bear", "idx": i, "age": n - 1 - i, "level": float(ph)})
        elif l[i] < pl and c[i] > pl:
            out.append({"dir": "bull", "idx": i, "age": n - 1 - i, "level": float(pl)})
    return [s for s in out if s["age"] <= max_age]


def premium_discount(df, lookback: int = 120) -> float:
    """0 = bottom of range, 100 = top. 50 is equilibrium."""
    if len(df) < 5:
        return 50.0
    hi = float(df["high"].tail(lookback).max())
    lo = float(df["low"].tail(lookback).min())
    if hi <= lo:
        return 50.0
    return float((float(df["close"].iloc[-1]) - lo) / (hi - lo) * 100)


def confluence(df, direction: str, entry: float, atr: float,
               tol_atr: float = 0.6) -> dict:
    """Score how much smart money structure sits behind a setup."""
    flags: list[str] = []
    score = 0
    pd_ = premium_discount(df)

    if direction == "long":
        if pd_ <= 35:
            score += 1
            flags.append(f"discount {pd_:.0f}%")
        elif pd_ >= 65:
            score -= 1
            flags.append(f"premium {pd_:.0f}%")
    else:
        if pd_ >= 65:
            score += 1
            flags.append(f"premium {pd_:.0f}%")
        elif pd_ <= 35:
            score -= 1
            flags.append(f"discount {pd_:.0f}%")

    want = "bull" if direction == "long" else "bear"
    band = tol_atr * atr if atr else 0.0

    fz = [f for f in fair_value_gaps(df) if f["dir"] == want]
    near_f = next((f for f in fz if f["bot"] - band <= entry <= f["top"] + band), None)
    if near_f:
        score += 1
        flags.append("unfilled FVG at entry")

    ob = [b for b in order_blocks(df) if b["dir"] == want]
    near_o = next((b for b in ob if b["bot"] - band <= entry <= b["top"] + band), None)
    if near_o:
        score += 1
        flags.append("order block at entry")

    sw = [s for s in liquidity_sweeps(df) if s["dir"] == want and s["age"] <= 5]
    if sw:
        score += 1
        flags.append("recent liquidity sweep")

    if direction == "long" and near_f and near_f["top"] < entry:
        score -= 1
        flags.append("entry above the gap")

    return {"score": score, "flags": flags, "fvg": near_f, "ob": near_o, "pd": pd_}
