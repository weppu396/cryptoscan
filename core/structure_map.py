"""core/structure_map.py - the few marks worth drawing.

Detection stays core.structure. This ranks, caps and labels:
  * two trend-conforming pivots per timeframe (HH+HL up, LH+LL down)
  * zones with real volume behind them
The final bar is dropped so a forming break never shows as confirmed.
"""
import numpy as np

from core.indicators import compute_all
from core.levels import build_zones
from core.structure import build_structure, find_swings

KEYS = (("bos_up", "BOS", "bull"), ("choch_up", "CHoCH", "bull"),
        ("bos_dn", "BOS", "bear"), ("choch_dn", "CHoCH", "bear"))


def _closed(frame):
    return frame.iloc[:-1] if len(frame) > 2 else frame


def collect(stt, frame, limit: int) -> list[dict]:
    rows = []
    for key, tag, direction in KEYS:
        col = stt[key].to_numpy()
        for t in np.flatnonzero(col)[-limit:]:
            lvl = float(stt["active_high"].iat[t] if direction == "bull"
                        else stt["active_low"].iat[t])
            if not np.isfinite(lvl):
                continue
            rows.append({"ts": frame.index[t], "level": lvl, "kind": tag,
                         "dir": direction})
    rows.sort(key=lambda r: r["ts"])
    return rows


def thin(rows: list[dict], gap: float) -> list[dict]:
    kept = []
    for r in sorted(rows, key=lambda x: x["ts"], reverse=True):
        if all(abs(r["level"] - k["level"]) > gap for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r["ts"])


def _trend(highs, lows) -> str:
    if len(highs) < 2 or len(lows) < 2:
        return "range"
    if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
        return "up"
    if highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
        return "down"
    return "range"


def _tag(df, pivot, trend: str) -> str:
    """BOS if the pivot was broken with the trend, CHoCH if against it."""
    after = df["close"].iloc[pivot["idx"] + 1:]
    if after.empty:
        return ""
    broken = bool((after > pivot["price"]).any()) if pivot["type"] == "H" \
        else bool((after < pivot["price"]).any())
    if not broken:
        return ""
    if trend == "up":
        return "BOS" if pivot["type"] == "H" else "CHoCH"
    if trend == "down":
        return "BOS" if pivot["type"] == "L" else "CHoCH"
    return ""


def structure_lines(df, cfg, tf_label: str, grade: str,
                    max_lines: int = 2) -> list[dict]:
    """Newest trend-conforming pivots. At most two, so four across both frames.

    Up: last HH and last HL. Down: last LH and last LL. Range: last H and L.
    """
    d = _closed(df)
    if len(d) < 40:
        return []
    sw = find_swings(d, cfg.structure.swing_left, cfg.structure.swing_right)
    highs = [s for s in sw if s["type"] == "H"]
    lows = [s for s in sw if s["type"] == "L"]
    if not highs or not lows:
        return []
    trend = _trend(highs, lows)

    out: list[dict] = []

    def add(pivot, label):
        out.append({"ts": d.index[pivot["idx"]], "price": float(pivot["price"]),
                    "label": label, "tag": _tag(d, pivot, trend),
                    "tf": tf_label, "grade": grade})

    if len(highs) >= 2:
        cur, prev = highs[-1], highs[-2]
        if trend == "up" and cur["price"] > prev["price"]:
            add(cur, "HH")
        elif trend == "down" and cur["price"] < prev["price"]:
            add(cur, "LH")
        else:
            add(cur, "H")
    if len(lows) >= 2:
        cur, prev = lows[-1], lows[-2]
        if trend == "up" and cur["price"] > prev["price"]:
            add(cur, "HL")
        elif trend == "down" and cur["price"] < prev["price"]:
            add(cur, "LL")
        else:
            add(cur, "L")

    out.sort(key=lambda r: r["ts"])
    return out[-max_lines:]


def marks(adf, cfg, htf_df, gap: float,
          max_major: int = 2, max_minor: int = 3):
    """(majors, minors) event marks. Kept for callers that want events."""
    max_major, max_minor = max(0, max_major), max(0, max_minor)
    majors, minors = [], []
    if max_major and htf_df is not None and len(htf_df) > 30:
        h = _closed(htf_df)
        try:
            majors = collect(build_structure(h, cfg.structure), h, max_major)
        except Exception:
            majors = []
    m = _closed(adf)
    try:
        if max_minor:
            minors = collect(build_structure(m, cfg.structure), m, max_minor)
    except Exception:
        minors = []
    majors = thin(majors, gap)[-max_major:]
    minors = [x for x in thin(minors, gap)
              if all(abs(x["level"] - k["level"]) > gap for k in majors)]
    return majors, minors[-max_minor:]


def volume_zones(df, cfg, tf_label: str, grade: str, direction: str,
                 min_ratio: float = 1.3, keep: int = 1) -> list[dict]:
    """Unmitigated zones with above-average volume behind the touch.

    Supply for short entries and long exits, demand for the reverse; only the
    side that matters for `direction` is returned.
    """
    d = _closed(df)
    if len(d) < 60:
        return []
    sw = find_swings(d, cfg.structure.swing_left, cfg.structure.swing_right)
    if not sw:
        return []
    try:
        atr_s = compute_all(d, cfg)["atr"]
    except Exception:
        return []
    kind = "sup" if direction == "long" else "res"
    zones = build_zones(d, sw, atr_s, cfg.structure, kind)
    vol = d["volume"].to_numpy(float)
    out = []
    for z in sorted(zones, key=lambda z: -z["last_touch_idx"]):
        i = int(z["last_touch_idx"])
        base = vol[max(0, i - 20):i]
        if not len(base):
            continue
        mean = float(np.nanmean(base))
        if mean <= 0:
            continue
        ratio = float(vol[i]) / mean
        if ratio < min_ratio:
            continue
        after = d.iloc[i + 1:]
        if not after.empty:
            hit = (after["low"] <= z["high"]).any() if kind == "sup" \
                else (after["high"] >= z["low"]).any()
            if hit:
                continue
        out.append({"low": z["low"], "high": z["high"], "ratio": round(ratio, 2),
                    "tf": tf_label, "grade": grade,
                    "side": "demand" if kind == "sup" else "supply",
                    "ts": d.index[i]})
        if len(out) >= keep:
            break
    return out


def live_zones(zones: list[dict], df, direction: str, keep: int = 2) -> list[dict]:
    """Nearest unmitigated zones from a raw build_zones list."""
    want = "sup" if direction == "long" else "res"
    out = []
    for z in sorted([z for z in zones if z["kind"] == want],
                    key=lambda z: -z["last_touch_idx"]):
        after = df.iloc[z["last_touch_idx"] + 1:]
        if not after.empty:
            hit = (after["low"] <= z["high"]).any() if want == "sup" \
                else (after["high"] >= z["low"]).any()
            if hit:
                continue
        out.append(z)
        if len(out) >= keep:
            break
    return out


def chart_zones(df, cfg, keep=1):
    """Full range of the last opposing candle before a directional impulse."""
    if df is None or df.empty or keep <= 0:
        return []
    atr = compute_all(df, cfg)["atr"].to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    zones = {"supply": [], "demand": []}
    for i in range(max(0, len(df) - cfg.structure.zone_lookback), len(df) - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0 or c[i] == o[i] or h[i] <= l[i]:
            continue
        supply = c[i] > o[i]
        for end in range(i + 1, min(i + 4, len(df))):
            opposing = c[end] < o[end] if supply else c[end] > o[end]
            if not opposing:
                break
            displaced = (c[i] - c[end] >= atr[i] and c[end] < l[i]) if supply else (
                c[end] - c[i] >= atr[i] and c[end] > h[i])
            if not displaced:
                continue
            later = c[end + 1:]
            invalid = (later > h[i]).any() if supply else (later < l[i]).any()
            if not invalid:
                side = "supply" if supply else "demand"
                distance = max(l[i] - c[-1], c[-1] - h[i], 0.0)
                zones[side].append(dict(side=side, low=l[i], high=h[i], ts=df.index[i],
                                        distance=distance))
            break
    return [zone for candidates in zones.values()
            for zone in sorted(reversed(candidates), key=lambda z: z["distance"])[:keep]]
