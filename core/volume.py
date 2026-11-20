"""core/volume.py - anchored VWAP and volume profile. Anchor-based, not rolling."""
import numpy as np
import pandas as pd


def anchored_vwap(df, anchor: int) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if anchor is None or not (0 <= anchor < len(df)):
        return out
    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).iloc[anchor:].to_numpy(float)
    v = df["volume"].iloc[anchor:].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out.iloc[anchor:] = np.cumsum(tp * v) / np.cumsum(v)
    return out


def avwap_context(df, direction: str, atr: float, anchor: int | None) -> dict:
    """0-100. A retest entry sits just BELOW the anchor VWAP by design, so
    proximity to value scores higher than being above it."""
    blank = {"score": 0, "flags": [], "warnings": ["no ATR"], "dist": 0.0, "line": None}
    if not atr or atr <= 0:
        return blank
    line = anchored_vwap(df, anchor)
    if not line.notna().any():
        return {**blank, "warnings": ["no anchor"]}
    px = float(df["close"].iloc[-1])
    dist = (px - float(line.iloc[-1])) / atr
    flags, warns, score, flag = [], [], 0, None

    if direction == "long":
        if 0 < dist <= 1.0:
            score, flag = 80, "above AVWAP, close to value"
        elif dist > 1.0:
            score, flag = 45, "extended above AVWAP"
        elif -1.0 <= dist <= 0:
            score, flag = 65, "pullback into AVWAP value"
        elif -2.5 <= dist < -1.0:
            score, flag = 45, "below AVWAP, deep pullback"
        else:
            warns.append("far below AVWAP")
    else:
        if -1.0 <= dist < 0:
            score, flag = 80, "below AVWAP, close to value"
        elif dist < -1.0:
            score, flag = 45, "extended below AVWAP"
        elif 0 <= dist <= 1.0:
            score, flag = 65, "rally into AVWAP value"
        elif 1.0 < dist <= 2.5:
            score, flag = 45, "above AVWAP, deep pullback"
        else:
            warns.append("far above AVWAP")

    if flag:
        flags.append(flag)
    if abs(dist) > 3.0:
        warns.append("stretched from AVWAP")
    return {"score": score, "flags": flags, "warnings": warns,
            "dist": dist, "line": line}


def volume_profile(df, lookback: int = 200, bins: int = 60) -> dict:
    seg = df.tail(lookback)
    if len(seg) < 20:
        return {}
    lo, hi = float(seg["low"].min()), float(seg["high"].max())
    if hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    tp = ((seg["high"] + seg["low"] + seg["close"]) / 3).to_numpy(float)
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    hist = np.bincount(idx, weights=seg["volume"].to_numpy(float), minlength=bins)
    total = hist.sum()
    if total <= 0:
        return {}
    poc_i = int(hist.argmax())
    keep, acc = [], 0.0
    for i in np.argsort(hist)[::-1]:
        acc += hist[i]
        keep.append(int(i))
        if acc >= 0.70 * total:
            break
    return {"poc": float((edges[poc_i] + edges[poc_i + 1]) / 2),
            "val": float(edges[min(keep)]), "vah": float(edges[max(keep) + 1]),
            "node": bool(hist[poc_i] >= 0.15 * total)}
