"""Run the live scanner.
    python run_scan.py --top 60 --recent 1
    python run_scan.py --top 100 --min-grade A
"""
import argparse
from dataclasses import replace

from config import CFG
from scanner import scan
from alerts import notify
from strategy.rating import print_rating

OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}


def pretty(setups):
    if not setups:
        print("No qualifying setups right now.")
        print("This is NORMAL. Most days the correct answer is 'wait'.\n")
        return

    for i, s in enumerate(setups, 1):
        m = s.meta
        stop_dist = abs(s.entry - s.stop)
        flag = "  <-- ACT NOW" if m.get("status") in OK_STATUS else ""
        print("=" * 70)
        print(f" #{i}  {s.symbol}  {s.direction.upper()}  [{m.get('status')}]{flag}")
        print("-" * 70)
        print(print_rating(s.symbol, m["rating"]))
        print("-" * 70)
        print(f" entry         {s.entry:,.4f}")
        print(f" stop-loss     {s.stop:,.4f}   ({s.stop_atr:.2f} ATR)")
        print(f" take-profit 1 {s.tp1:,.4f}   (+{abs(s.tp1 - s.entry) / stop_dist:.1f}R)")
        print(f" take-profit 2 {s.tp2:,.4f}   (+{s.rr:.2f}R)")
        print(f" size          {m.get('qty', 0):,.4f} units   "
              f"notional {m.get('notional', 0):,.0f}   "
              f"risk ${m.get('risk_cash', 0):,.2f}")
        print(f" signal bar    {s.ts}")
    print("=" * 70)
    print("\nGRADE: A+/A = take it. B = take if you have room. C = watch. D = skip.")
    print("STATUS: ACTIONABLE/BETTER_PRICE = act now. RUNNING/EXTENDED/INVALID = info only.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--min-vol", type=float, default=None)
    ap.add_argument("--exchange", default=None)
    ap.add_argument("--recent", type=int, default=2)
    ap.add_argument("--min-grade", default=None, choices=["A+", "A", "B", "C", "D"])
    ap.add_argument("--actionable-only", action="store_true")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--alert", default=None)
    a = ap.parse_args()

    sc = CFG.scan
    if a.min_vol:
        sc = replace(sc, min_quote_vol_24h=a.min_vol)
    if a.exchange:
        sc = replace(sc, exchange=a.exchange)
    cfg = replace(CFG, scan=replace(sc, top_n_symbols=a.top))

    setups = scan(cfg, max_age_bars=a.recent, min_grade=a.min_grade)
    if a.actionable_only:
        setups = [s for s in setups if s.meta.get("status") in OK_STATUS]
    setups = setups[: a.limit]
    pretty(setups)

    if a.alert and setups:
        print(f"Sent {notify(setups, cfg, a.alert)} new alerts.")


if __name__ == "__main__":
    main()