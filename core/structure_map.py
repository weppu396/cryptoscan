"""core/structure_map.py - pick the few structure marks worth drawing.

Detection is core.structure.build_structure. This only ranks, caps and
de-duplicates. The final bar is dropped so a forming break never shows as
confirmed.
"""
import numpy as np

from core.structure import build_structure

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


def marks(adf, cfg, htf_df, gap: float,
          max_major: int = 2, max_minor: int = 3):
    """(majors, minors). Majors from 30m, minors from the trigger frame.
    Minors colliding with a major are dropped."""
    majors, minors = [], []
    if htf_df is not None and len(htf_df) > 30:
        h = _closed(htf_df)
        try:
            majors = collect(build_structure(h, cfg.structure), h, max_major)
        except Exception:
            majors = []
    m = _closed(adf)
    try:
        minors = collect(build_structure(m, cfg.structure), m, max_minor)
    except Exception:
        minors = []
    majors = thin(majors, gap)
    minors = [x for x in thin(minors, gap)
              if all(abs(x["level"] - k["level"]) > gap for k in majors)]
    return majors, minors


def live_zones(zones: list[dict], df, direction: str, keep: int = 2) -> list[dict]:
    """Nearest unmitigated zones in the trade direction, newest first."""
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
