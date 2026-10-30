"""
baserates.py - measured P(TP1 before stop) per setup signature.
Wilson bounds keep small samples honest.
"""
import json
import math
import pathlib

from strategy.setups import scan_bos_retest
from strategy.risk import filter_setups
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from backtest.engine import run as engine_run

OUT = pathlib.Path("baserates.json")
MIN_N = 20


def signature(direction, grade, stop_atr):
    band = "tight" if stop_atr < 0.9 else "normal" if stop_atr < 1.8 else "wide"
    return f"{direction}|{grade}|{band}"


def wilson(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def build(df, cfg, symbol, grader):
    ind = compute_all(df, cfg)
    st = build_structure(df, cfg.structure)
    hb = htf_bias(df, cfg)
    sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
    zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
    zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")

    raw = scan_bos_retest(symbol, df, hb, ind, st, zu, zd, cfg)
    setups = filter_setups(raw, ind, zu, zd, cfg)

    sig_of, signals = {}, {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        g = grader(s) or "C"
        sig_of[loc] = signature(s.direction, g, s.stop_atr)
        signals.setdefault(loc, {"dir": s.direction, "stop": s.stop,
                                 "tp1": s.tp1, "tp2": s.tp2,
                                 "score": s.score, "grade": g})

    _, trades = engine_run(df, signals, cfg, symbol)
    table = {}
    if trades.empty:
        return table

    trades = trades.copy()
    trades["sig"] = [sig_of.get(df.index.get_loc(t), "?") for t in trades["entry_ts"]]
    for sig, g in trades.groupby("sig"):
        n, wins = len(g), int((g["r"] > 0).sum())
        p, lo, hi = wilson(wins, n)
        table[sig] = {"n": n, "wins": wins, "rate": round(p, 3),
                      "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                      "expectancy": round(float(g["r"].mean()), 3)}
    return table


def save(table):
    OUT.write_text(json.dumps(table, indent=2))
    return table


def load():
    if OUT.exists():
        try:
            return json.loads(OUT.read_text())
        except Exception:
            return {}
    return {}


def lookup(table, direction, grade, stop_atr):
    hit = table.get(signature(direction, grade, stop_atr))
    if hit and hit["n"] >= MIN_N:
        return {**hit, "basis": "measured"}
    pooled = [v for k, v in table.items() if k.startswith(direction + "|")]
    n = sum(v["n"] for v in pooled)
    if n >= MIN_N:
        wins = sum(v["wins"] for v in pooled)
        p, lo, hi = wilson(wins, n)
        return {"n": n, "wins": wins, "rate": round(p, 3),
                "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                "expectancy": 0.0, "basis": "direction pool"}
    return None
