"""core/volume.py - anchored VWAP and volume profile. Anchor-based, not rolling."""
import numpy as np
import pandas as pd


def anchored_vwap(df, anchor: int) -> pd.Series:
    """VWAP from bar `anchor` forward. NaN before the anchor."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if anchor is None or not (0 <= anchor < len(df)):
        return out
    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).iloc[anchor:].to_numpy(float)
    v = df["volume"].iloc[anchor:].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out.iloc[anchor:] = np.cumsum(tp * v) / np.cumsum(v)
    return out


def avwap_context(df, direction: str, atr: float, anchor: int) -> dict:
    """0-100. Rewards price on the correct side of the anchor VWAP and rewards a
    pullback INTO it. Punishes being stretched away from it."""
    blank = {"score": 0, "flags": [], "warnings": ["no ATR"], "dist": 0.0, "line": None}
    if not atr or atr <= 0:
        return blank
    line = anchored_vwap(df, anchor)
    if not line.notna().any():
        return {**blank, "warnings": ["no anchor"]}
    px = float(df["close"].iloc[-1])
    dist = (px - float(line.iloc[-1])) / atr
    flags, warns, score = [], [], 0

    if direction == "long":
        if dist > 0:
            score += 40
            flags.append(f"{dist:.1f} ATR above anchor VWAP")
        else:
            warns.append(f"{abs(dist):.1f} ATR below anchor VWAP")
        if 0 <= dist <= 1.0:
            score += 30
            flags.append("pullback into AVWAP value")
    else:
        if dist < 0:
            score += 40
            flags.append(f"{abs(dist):.1f} ATR below anchor VWAP")
        else:
            warns.append(f"{dist:.1f} ATR above anchor VWAP")
        if -1.0 <= dist <= 0:
            score += 30
            flags.append("rally into AVWAP value")
    if abs(dist) > 2.5:
        warns.append("stretched from AVWAP")
    return {"score": min(score, 100), "flags": flags, "warnings": warns,
            "dist": dist, "line": line}


def volume_profile(df, lookback: int = 200, bins: int = 60) -> dict:
    """POC and value area. Entry inside the value area = real acceptance."""
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
