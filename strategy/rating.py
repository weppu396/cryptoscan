"""
Transparent A+ to D grading. Confidence is built from named confluence
factors so you can see exactly WHY a setup is rated the way it is.
Probability = estimated chance of reaching TP1 before the stop.
"""
from dataclasses import dataclass, field


@dataclass
class Rating:
    grade: str
    confidence: int
    verdict: str
    est_win_rate: float
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)
    improve: list = field(default_factory=list)
    basis: str = "heuristic"


BASE_RATE = {"A+": 0.55, "A": 0.47, "B": 0.41, "C": 0.35, "D": 0.29}


def _empirical(score, stats):
    """If we have a backtest, use the REAL historical hit rate for this
    score bucket instead of a guess."""
    if not stats:
        return None
    for lo, hi, wr, n, exp in stats.get("buckets", []):
        if lo <= score < hi and n >= 15:
            return wr
    return None


def rate(s, row, zone, cfg, stats=None) -> Rating:
    pts = 50.0
    pos, neg, imp = [], [], []

    def f(v, d=0.0):
        try:
            v = float(v)
            return v if v == v else d
        except Exception:
            return d

    adx = f(row["adx"])
    if adx >= 30:
        pts += 10; pos.append(f"strong trend, ADX {adx:.0f}")
    elif adx >= 25:
        pts += 5; pos.append(f"decent trend, ADX {adx:.0f}")
    elif adx < 22:
        pts -= 8; neg.append(f"weak trend, ADX {adx:.0f}")
        imp.append("wait for ADX above 25")

    vr = f(row["vol_ratio"], 1.0)
    if vr >= 1.8:
        pts += 10; pos.append(f"volume {vr:.1f}x the 20-bar average")
    elif vr >= 1.3:
        pts += 5; pos.append(f"volume {vr:.1f}x the 20-bar average")
    elif vr < 0.8:
        pts -= 6; neg.append(f"thin volume, only {vr:.1f}x average")
        imp.append("needs volume confirmation")

    sa = s.stop_atr
    if 0.5 <= sa <= 1.2:
        pts += 12; pos.append(f"tight stop at {sa:.2f} ATR - good R multiple math")
    elif sa <= 1.8:
        pts += 5; pos.append(f"stop {sa:.2f} ATR, acceptable")
    elif sa > 2.5:
        pts -= 10; neg.append(f"wide stop at {sa:.2f} ATR")
        imp.append("wait for a tighter structure to form")

    if s.rr >= 3.0:
        pts += 12; pos.append(f"excellent R:R of {s.rr:.2f}")
    elif s.rr >= 2.5:
        pts += 6; pos.append(f"good R:R of {s.rr:.2f}")
    elif s.rr < 2.2:
        pts -= 6; neg.append(f"thin R:R of {s.rr:.2f}")

    t = zone["touches"] if zone else 0
    if t >= 4:
        pts += 10; pos.append(f"level tested {t} times - heavily defended")
    elif t >= 3:
        pts += 6; pos.append(f"level tested {t} times")
    elif t <= 1:
        pts -= 4; neg.append("level barely tested, weak reference")

    r = f(row["rsi"], 50)
    if s.direction == "long":
        if 45 <= r <= 65:
            pts += 8; pos.append(f"RSI {r:.0f} in the sweet spot, room both ways")
        elif r > 72:
            pts -= 5; neg.append(f"RSI {r:.0f} already hot")
            imp.append("wait for RSI to cool below 70")
        elif r < 35:
            pts += 3; pos.append(f"RSI {r:.0f} reset - lots of room to run")
    else:
        if 35 <= r <= 55:
            pts += 8; pos.append(f"RSI {r:.0f} in the sweet spot")
        elif r < 28:
            pts -= 5; neg.append(f"RSI {r:.0f} already oversold")
            imp.append("wait for RSI to bounce above 30")

    ema = f(row["ema_fast"], s.entry)
    ext = abs(s.entry - ema) / s.atr if s.atr else 0
    if ext <= 1.5:
        pts += 6; pos.append(f"not overextended, {ext:.1f} ATR from EMA50")
    elif ext > 3.0:
        pts -= 8; neg.append(f"overextended, {ext:.1f} ATR from EMA50")
        imp.append("wait for a pullback toward EMA50")

    conf = int(max(0, min(100, round(pts))))
    grade = ("A+" if conf >= 88 else "A" if conf >= 78 else
             "B" if conf >= 66 else "C" if conf >= 52 else "D")
    verdict = ("TAKE" if grade in ("A+", "A") else "CONSIDER" if grade == "B"
               else "WATCH" if grade == "C" else "SKIP")

    emp = _empirical(s.score, stats)
    prob = emp if emp is not None else BASE_RATE[grade]
    basis = "empirical backtest" if emp is not None else "heuristic"

    return Rating(grade, conf, verdict, round(prob * 100, 1),
                  pos, neg, imp, basis)


def print_rating(sym, rt):
    lines = [
        f" RATING  {rt.grade}   confidence {rt.confidence}/100   "
        f"est. TP1 hit rate {rt.est_win_rate}%  ({rt.basis})",
        f" verdict: {rt.verdict}",
    ]
    if rt.positives:
        lines.append(" works in favour:")
        lines += [f"   + {x}" for x in rt.positives]
    if rt.negatives:
        lines.append(" works against:")
        lines += [f"   - {x}" for x in rt.negatives]
    if rt.improve:
        lines.append(" to upgrade this setup:")
        lines += [f"   > {x}" for x in rt.improve]
    return "\n".join(lines)
