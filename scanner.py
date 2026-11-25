"""Runs every symbol through the pipeline and returns ranked, rated setups."""
import json
import pathlib
import pandas as pd
from data import TF_MS

from config import CFG
from data import get_exchange, build_universe, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.multistrategy import scan_seven
from strategy.risk import filter_setups
from strategy.rating import rate
from core.opportunities import assess, current_only


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
    ex = get_exchange(cfg.scan.exchange)
    universe, universe_note = build_universe(ex, cfg)
    universe = universe[:100]
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
            # Last exchange bar may still be forming. Only closed candles can signal.
            if len(ltf) and (pd.Timestamp.now(tz="UTC") - ltf.index[-1]).total_seconds() < TF_MS[cfg.scan.ltf] / 1000:
                ltf = ltf.iloc[:-1]
            if len(ltf) < 500:
                skipped += 1
                continue

            ind = compute_all(ltf, cfg)
            st = build_structure(ltf, cfg.structure)
            hb = htf_bias(ltf, cfg)
            sw = find_swings(ltf, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(ltf, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(ltf, sw, ind["atr"], cfg.structure, "sup")

            raw = scan_seven(sym, ltf, hb, ind, st, zu, zd, cfg, recent=max(3, (max_age_bars or 2) + 1))
            kept = filter_setups(raw, ind, zu, zd, cfg)

            if max_age_bars is not None and kept:
                cut = ltf.index[max(0, len(ltf) - 1 - max_age_bars)]
                kept = [s for s in kept if s.ts >= cut]

            for s in kept:
                s.meta["status"], s.meta["status_reason"] = assess(
                    s, ltf, max_age_bars=max_age_bars,
                    timeframe_seconds=TF_MS[cfg.scan.ltf] / 1000)
                if s.meta["status"] not in ("ACTIONABLE", "BETTER_PRICE"):
                    continue
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

    return current_only(found)