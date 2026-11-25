"""core/volume.py - anchored VWAP and volume profile. Anchor-based, not rolling."""
import numpy as np
import pandas as pd

AVWAP_MAX_AGE = 80
AVWAP_MAX_DIST = 25.0


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
    """0-100. usable=False when the anchor is stale or the distance is
    meaningless, so the UI can print n/a instead of -116.61 ATR."""
    blank = {"score": 0, "flags": [], "warnings": ["no ATR"], "dist": 0.0,
             "line": None, "usable": False}
    if not atr or atr <= 0:
        return blank
    if anchor is None or not (0 <= anchor < len(df)):
        return {**blank, "warnings": ["no anchor"]}
    age = len(df) - 1 - anchor
    if age > AVWAP_MAX_AGE:
        return {**blank, "warnings": [f"anchor {age} bars old - too stale"]}
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

    usable = abs(dist) <= AVWAP_MAX_DIST
    if not usable:
        score = 0
        warns.append(f"distance {dist:+.0f} ATR - unreliable")
    if flag:
        flags.append(flag)
    if abs(dist) > 3.0:
        warns.append("stretched from AVWAP")
    return {"score": score, "flags": flags, "warnings": warns,
            "dist": dist, "line": line, "usable": usable}


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


def profile_histogram(df, lookback: int = 200, bins: int = 48,
                      value_area: float = 0.70) -> dict:
    """Horizontal volume-at-price for the chart gutter.

    ESTIMATE. OHLCV carries no trade direction, so each bar's volume is
    spread evenly across the bins its range overlaps. Total volume is
    conserved; distribution inside a bar is a guess. Green and red use candle direction,
    with unchanged candles split evenly.
    """
    if df is None or len(df) < 20:
        return {}
    seg = df.tail(max(20, int(lookback)))
    lo, hi = float(seg["low"].min()), float(seg["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {}

    edges = np.linspace(lo, hi, bins + 1)
    up = np.zeros(bins)
    down = np.zeros(bins)
    candle_move = seg["close"].to_numpy(float) - seg["open"].to_numpy(float)
    up_share = np.where(candle_move > 0, 1.0, np.where(candle_move < 0, 0.0, 0.5))
    lows = seg["low"].to_numpy(float)
    highs = seg["high"].to_numpy(float)
    vols = np.nan_to_num(seg["volume"].to_numpy(float), nan=0.0)

    for i in range(len(seg)):
        v = vols[i]
        if v <= 0:
            continue
        span = highs[i] - lows[i]
        if span <= 0:
            k = min(bins - 1, max(0, int((lows[i] - lo) / (hi - lo) * bins)))
            up[k] += v * up_share[i]
            down[k] += v * (1 - up_share[i])
            continue
        for k in range(bins):
            ov = min(highs[i], edges[k + 1]) - max(lows[i], edges[k])
            if ov > 0:
                portion = v * (ov / span)
                up[k] += portion * up_share[i]
                down[k] += portion * (1 - up_share[i])

    hist = up + down
    total = float(hist.sum())
    if total <= 0:
        return {}

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_i = int(hist.argmax())
    target = total * value_area
    acc, left, right = hist[poc_i], poc_i, poc_i
    while acc < target and (left > 0 or right < bins - 1):
        lv = hist[left - 1] if left > 0 else -1.0
        rv = hist[right + 1] if right < bins - 1 else -1.0
        if rv >= lv:
            right += 1
            acc += hist[right]
        else:
            left -= 1
            acc += hist[left]

    kern = np.array([1, 2, 3, 2, 1], float)
    smooth = np.convolve(hist, kern / kern.sum(), mode="same")
    live = smooth[smooth > 0]
    hvn, lvn = [], []
    if len(live) >= 8:
        hi_cut = float(np.quantile(live, 0.82))
        lo_cut = float(np.quantile(live, 0.18))
        hvn = [float(centers[i]) for i in np.argsort(smooth)[::-1][:6]
               if smooth[i] >= hi_cut]
        lvn = [float(centers[i]) for i in np.argsort(smooth)[:6]
               if smooth[i] <= lo_cut]

    return {"centers": centers, "hist": hist / total,
            "up": up / total, "down": down / total,
            "poc": float(centers[poc_i]),
            "val": float(edges[left]), "vah": float(edges[right + 1]),
            "hvn": hvn, "lvn": lvn, "estimate": True,
            "anchor": seg.index[0], "bars": len(seg), "total": total}
