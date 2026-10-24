import numpy as np
import pandas as pd


def build_zones(df, swings, atr_s, cfg, kind):
    """
    Cluster swing prices into zones using an ATR-scaled tolerance.
    Score = touches + recency weight + wick-rejection weight.
    A zone touched 4 times recently and defended hard scores high.
    """
    pts = [s for s in swings if s["type"] == ("H" if kind == "res" else "L")]
    pts = [s for s in pts if s["idx"] >= len(df) - cfg.zone_lookback]
    if not pts:
        return []
    atr_med = float(np.nanmedian(atr_s.values)) or 1.0
    tol = cfg.zone_tol_atr * atr_med

    pts.sort(key=lambda s: s["price"])
    zones, cur = [], [pts[0]]
    for p in pts[1:]:
        if abs(p["price"] - cur[-1]["price"]) <= tol:
            cur.append(p)
        else:
            zones.append(cur); cur = [p]
    zones.append(cur)

    out = []
    n = len(df)
    for z in zones:
        if len(z) < cfg.min_zone_touches:
            continue
        prices = [p["price"] for p in z]
        recency = np.mean([1.0 - (n - p["idx"]) / n for p in z])
        score = len(z) * 1.0 + recency * 2.0
        out.append({
            "kind": kind, "low": min(prices), "high": max(prices),
            "mid": float(np.mean(prices)), "touches": len(z),
            "last_touch_idx": max(p["idx"] for p in z), "score": round(score, 2),
        })
    return sorted(out, key=lambda z: -z["score"])
