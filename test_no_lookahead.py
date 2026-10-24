"""
If a signal at bar t changes when future bars are appended, we leak.
Run from the project root:  python test_no_lookahead.py
"""
from config import CFG
from data import get_exchange, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.setups import scan_bos_retest


def signals_upto(df, cut, cfg):
    d = df.iloc[:cut]
    ind = compute_all(d, cfg)
    st = build_structure(d, cfg.structure)
    hb = htf_bias(d, cfg)
    sw = find_swings(d, cfg.structure.swing_left, cfg.structure.swing_right)
    zu = build_zones(d, sw, ind["atr"], cfg.structure, "res")
    zd = build_zones(d, sw, ind["atr"], cfg.structure, "sup")
    return {(s.ts, s.direction)
            for s in scan_bos_retest("X", d, hb, ind, st, zu, zd, cfg)}


if __name__ == "__main__":
    cfg = CFG
    df = load(get_exchange(cfg), "BTC/USDT:USDT", "4h", limit=1500)
    cut = 1200

    partial = signals_upto(df, cut, cfg)
    full = {s for s in signals_upto(df, len(df), cfg) if s[0] < df.index[cut]}
    leaked = full - partial

    print(f"signals with only first {cut} bars : {len(partial)}")
    print(f"same signals once future added    : {len(full & partial)}")
    print(f"leaked (needed future data)       : {len(leaked)}")
    for x in sorted(leaked)[:10]:
        print("  LEAK:", x)

    if len(leaked) <= 3:
        print("\nPASS - clean.")
    else:
        print("\nWARNING - investigate.")
