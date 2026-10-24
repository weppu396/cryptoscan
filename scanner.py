"""Runs every symbol through the pipeline and returns ranked, rated setups."""
import json
import pathlib

from config import CFG
from data import get_exchange, build_universe, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.setups import scan_bos_retest
from strategy.risk import filter_setups
from strategy.rating import rate


def load_stats():
    p = pathlib.Path("edge_stats.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def setup_status(s, ltf) -> str:
    cur = float(ltf["close"].iat[-1])
    tol = 0.5 * s.atr
    if s.direction == "long":
        if cur <= s.stop: return "INVALID"
        if cur >= s.tp2: return "TARGET_HIT"
        if cur >= s.tp1: return "RUNNING"
        if abs(cur - s.entry) <= tol: return "ACTIONABLE"
        if cur < s.entry: return "BETTER_PRICE"
        return "EXTENDED"
    else:
        if cur >= s.stop: return "INVALID"
        if cur <= s.tp2: return "TARGET_HIT"
        if cur <= s.tp1: return "RUNNING"
        if abs(cur - s.entry) <= tol: return "ACTIONABLE"
        if cur > s.entry: return "BETTER_PRICE"
        return "EXTENDED"


def scan(cfg=CFG, verbose=True, progress_cb=None, max_age_bars=2,
         min_grade=None) -> list:
    ex = get_exchange(cfg)
    universe = build_universe(ex, cfg)
    stats = load_stats()
    if verbose:
        print(f"Scanning {len(universe)} symbols on {cfg.scan.ltf} "
              f"with {cfg.scan.htf} bias filter...\n")

    found, skipped = [], 0
    order = ["A+", "A", "B", "C", "D"]

    for i, sym in enumerate(universe, 1):
        if progress_cb:
            progress_cb(i, len(universe), sym)
        elif verbose:
            print(f"  [{i}/{len(universe)}] {sym}", end="\r")
        try:
            ltf = load(ex, sym, cfg.scan.ltf)
            if len(ltf) < 500:
                skipped += 1
                continue

            ind = compute_all(ltf, cfg)
            st = build_structure(ltf, cfg.structure)
            hb = htf_bias(ltf, cfg)
            sw = find_swings(ltf, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(ltf, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(ltf, sw, ind["atr"], cfg.structure, "sup")

            raw = scan_bos_retest(sym, ltf, hb, ind, st, zu, zd, cfg)
            kept = filter_setups(raw, ind, zu, zd, cfg)

            if max_age_bars is not None and kept:
                cut = ltf.index[max(0, len(ltf) - 1 - max_age_bars)]
                kept = [s for s in kept if s.ts >= cut]

            for s in kept:
                s.meta["status"] = setup_status(s, ltf)
                row = ind.loc[s.ts]
                zone = (zu[0] if zu else None) if s.direction == "long" \
                    else (zd[0] if zd else None)
                rt = rate(s, row, zone, cfg, stats)
                s.meta["rating"] = rt
                s.meta["grade"] = rt.grade
                if min_grade and order.index(rt.grade) > order.index(min_grade):
                    continue
                found.append(s)

        except Exception as e:
            skipped += 1
            if verbose and not progress_cb:
                print(f"\n  [skip] {sym}: {type(e).__name__}: {e}")

    if verbose and not progress_cb:
        print(f"\n\nDone. {len(found)} fresh setups, {skipped} symbols skipped.\n")

    found.sort(key=lambda s: (-s.meta.get("rating").confidence, -s.score))
    return found