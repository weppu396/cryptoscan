"""core/structure_map.py - pick the few structure marks worth drawing.

Detection is unchanged: it is core.structure.build_structure. This module
only ranks, caps and de-duplicates so the chart stops drowning.
"""
import numpy as np

from core.structure import build_structure

KEYS = (("bos_up", "BOS", "bull"), ("choch_up", "CHoCH", "bull"),
        ("bos_dn", "BOS", "bear"), ("choch_dn", "CHoCH", "bear"))


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
    """Keep the newest of any two marks priced within `gap`."""
    kept = []
    for r in sorted(rows, key=lambda x: x["ts"], reverse=True):
        if all(abs(r["level"] - k["level"]) > gap for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r["ts"])


def marks(adf, cfg, htf_df, gap: float,
          max_major: int = 2, max_minor: int = 3):
    """(majors, minors). Majors from the 30m frame, minors from the trigger
    frame. Minors that collide with a major are dropped."""
    majors, minors = [], []
    try:
        if htf_df is not None and len(htf_df) > 30:
            majors = collect(build_structure(htf_df, cfg.structure),
                             htf_df, max_major)
    except Exception:
        majors = []
    try:
        minors = collect(build_structure(adf, cfg.structure), adf, max_minor)
    except Exception:
        minors = []

    majors = thin(majors, gap)
    minors = [m for m in thin(minors, gap)
              if all(abs(m["level"] - k["level"]) > gap for k in majors)]
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
