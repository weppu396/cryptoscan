import argparse
import json
from dataclasses import replace

from config import CFG
from data import get_exchange, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.setups import scan_bos_retest
from strategy.risk import filter_setups
from backtest.engine import run as engine_run
from backtest.metrics import report, bucket_stats
from backtest.validate import permutation_test, monte_carlo_drawdown, walk_forward


def build_trades(df, cfg, symbol):
    ind = compute_all(df, cfg)
    st = build_structure(df, cfg.structure)
    hb = htf_bias(df, cfg)
    sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
    zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
    zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")

    raw = scan_bos_retest(symbol, df, hb, ind, st, zu, zd, cfg)
    setups = filter_setups(raw, ind, zu, zd, cfg)

    signals = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        if loc not in signals:
            signals[loc] = {"dir": s.direction, "stop": s.stop,
                            "tp1": s.tp1, "tp2": s.tp2, "score": s.score}

    curve, trades = engine_run(df, signals, cfg, symbol)
    return curve, trades, len(raw), len(setups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    a = ap.parse_args()

    cfg = replace(CFG, scan=replace(CFG.scan, ltf=a.tf))
    ex = get_exchange(cfg)
    df = load(ex, a.symbol, a.tf, limit=a.limit, use_cache=not a.no_cache)
    print(f"{len(df)} bars of {a.symbol} {a.tf}")

    curve, trades, n_raw, n_kept = build_trades(df, cfg, a.symbol)
    print(f"{n_raw} raw signals -> {n_kept} after filters")

    stats = report(trades, curve, cfg)
    for k, v in stats.items():
        print(f"  {k:22} {v}")

    if not trades.empty:
        if len(trades) < 30:
            print("\n  WARNING: fewer than 30 trades. Raise --limit or lower min_rr.")
        bs = bucket_stats(trades)
        print("\n  hit rate by score bucket:")
        for lo, hi, wr, n, exp in bs["buckets"]:
            print(f"    score {lo:>3}-{hi:<3}  n={n:<4} win {wr*100:5.1f}%  expectancy {exp:+.2f}R")
        json.dump(bs, open("edge_stats.json", "w"), indent=2)
        print("  -> edge_stats.json written (used by the scanner rating engine)")

        print("\n  permutation:", permutation_test(trades["r"].values))
        print("  monte carlo:", monte_carlo_drawdown(
            trades["r"].values, cfg.risk.risk_per_trade_pct))
        trades.to_csv("trades.csv", index=False)
        print("  trade list -> trades.csv")

    if a.walk_forward:
        res = walk_forward(lambda d, c: build_trades(d, c, a.symbol)[:2], df, cfg)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()