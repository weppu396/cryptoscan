import argparse
import json
from dataclasses import replace

from config import CFG
from data import get_exchange, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import zones_by_bar
from core.regime import htf_bias
from strategy.setups import scan_bos_retest
from strategy.multistrategy import scan_seven, STRATEGIES
from strategy.risk import filter_setups
from backtest.engine import run as engine_run
from backtest.metrics import report, bucket_stats
from backtest.validate import permutation_test, monte_carlo_drawdown, walk_forward


def build_trades(df, cfg, symbol, strategy="BOS Retest"):
    ind = compute_all(df, cfg)
    st = build_structure(df, cfg.structure)
    hb = htf_bias(df, cfg)
    sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
    zones_at = zones_by_bar(df, sw, ind["atr"], cfg.structure)
    zu, zd = [], []

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    # Each backtest evaluates exactly one strategy, avoiding mixed-signal
    # performance claims. Historical candles must be complete before calling.
    if strategy == "BOS Retest":
        raw = scan_bos_retest(symbol, df, hb, ind, st, zu, zd, cfg, zones_at=zones_at)
    else:
        candidates = scan_seven(symbol, df, hb, ind, st, zu, zd, cfg, recent=len(df), zones_at=zones_at)
        raw = [x for x in candidates if x.meta.get("strategy") == strategy]
    setups = filter_setups(raw, ind, zu, zd, cfg, zones_at=zones_at)

    signals = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        if loc not in signals:
            signals[loc] = {"dir": s.direction, "stop": s.stop, "limit": s.entry,
                            "tp1": s.tp1, "tp2": s.tp2, "score": s.score,
                            "strategy": strategy}

    curve, trades = engine_run(df, signals, cfg, symbol)
    return curve, trades, len(raw), len(setups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--strategy", choices=STRATEGIES, default="BOS Retest")
    ap.add_argument("--compare-strategies", action="store_true",
                    help="Backtest each strategy independently on the same historical data")
    a = ap.parse_args()

    cfg = replace(CFG, scan=replace(CFG.scan, ltf=a.tf))
    ex = get_exchange(cfg.scan.exchange)
    df = load(ex, a.symbol, a.tf, limit=a.limit, use_cache=not a.no_cache)
    print(f"{len(df)} bars of {a.symbol} {a.tf}")

    if a.compare_strategies:
        print("\nIndependent strategy comparison (historical, not a profit forecast):")
        for name in STRATEGIES:
            cr, tr, nr, nk = build_trades(df, cfg, a.symbol, strategy=name)
            stats = report(tr, cr, cfg)
            print(f"{name:22} raw={nr:4} kept={nk:4} "
                  f"trades={stats.get('trades', 0):4} "
                  f"expectancy_R={stats.get('expectancy_R', 'N/A')} "
                  f"max_dd_pct={stats.get('max_dd_pct', 'N/A')}")
        return

    curve, trades, n_raw, n_kept = build_trades(df, cfg, a.symbol, strategy=a.strategy)
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
        res = walk_forward(lambda d, c: build_trades(d, c, a.symbol, strategy=a.strategy)[:2], df, cfg)
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()