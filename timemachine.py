"""timemachine.py - Monte Carlo path replay. See previous message for detail."""
import numpy as np

from core.indicators import atr


def _triples(df, idx, lookback=1200):
    sub = df.iloc[max(0, idx - lookback):idx]
    if len(sub) < 120:
        return None
    a = atr(sub, 14).values
    close = sub["close"].values
    high = sub["high"].values
    low = sub["low"].values
    prev = np.roll(close, 1)
    prev[0] = close[0]
    ret = np.log(close / prev)
    up = (high - close) / close
    dn = (close - low) / close
    ok = np.isfinite(a) & (a > 0)
    ret, up, dn, a, close = ret[ok], up[ok], dn[ok], a[ok], close[ok]
    if len(ret) < 100:
        return None
    af = a / close
    t = np.column_stack([ret / af, up / af, dn / af])
    return t[np.isfinite(t).all(axis=1)]


def simulate(df, idx, entry, stop, tp1, tp2, horizon=24, n=10000,
             block=4, fee_bps=5.0, slip_bps=3.0, seed=None):
    t = _triples(df, idx)
    if t is None:
        return {"error": "not enough history"}

    rng = np.random.default_rng(seed)
    risk = abs(entry - stop)
    if risk <= 0:
        return {"error": "bad stop"}
    long = stop < entry
    a_now = float(atr(df.iloc[max(0, idx - 200):idx], 14).iloc[-1])
    if not np.isfinite(a_now) or a_now <= 0:
        return {"error": "atr unavailable"}
    af_now = a_now / entry

    m = len(t)
    starts = rng.integers(0, m - block, size=(n,))
    picks = (starts[:, None] + np.arange(block)[None, :]) % m
    reps = int(np.ceil(horizon / block))
    picks = np.tile(picks, (1, reps))[:, :horizon]
    sim = t[picks]

    ret = sim[:, :, 0] * af_now
    up = np.abs(sim[:, :, 1]) * af_now
    dn = np.abs(sim[:, :, 2]) * af_now

    close = np.full(n, float(entry))
    alive = np.ones(n, dtype=bool)
    tp1_done = np.zeros(n, dtype=bool)
    outcome = np.zeros(n, dtype=np.int8)
    r_mult = np.zeros(n)
    bars = np.zeros(n, dtype=int)

    for k in range(horizon):
        if not alive.any():
            break
        close = np.where(alive, close * np.exp(ret[:, k]), close)
        hi = close * (1 + up[:, k])
        lo = close * (1 - dn[:, k])
        bars = np.where(alive, k + 1, bars)

        if long:
            hs = alive & (lo <= stop)
            r_mult[hs] = np.where(tp1_done[hs], 0.75, -1.0)
            outcome[hs] = 1
            alive &= ~hs
            h1 = alive & ~tp1_done & (hi >= tp1)
            tp1_done |= h1
            h2 = alive & tp1_done & (hi >= tp2)
            r_mult[h2] = 2.25
            outcome[h2] = 3
            alive &= ~h2
        else:
            hs = alive & (hi >= stop)
            r_mult[hs] = np.where(tp1_done[hs], 0.75, -1.0)
            outcome[hs] = 1
            alive &= ~hs
            h1 = alive & ~tp1_done & (lo <= tp1)
            tp1_done |= h1
            h2 = alive & tp1_done & (lo <= tp2)
            r_mult[h2] = 2.25
            outcome[h2] = 3
            alive &= ~h2

    move = (close - entry) / risk * (1 if long else -1)
    r_mult[alive] = np.where(tp1_done[alive], 0.75 + 0.5 * move[alive],
                             0.5 * move[alive])
    outcome[alive] = 4
    bars[alive] = horizon

    cost = (fee_bps * 2 + slip_bps * 2) / 10_000.0
    r_mult = r_mult - cost * entry / risk

    p = float((outcome >= 2).mean())
    se = np.sqrt(p * (1 - p) / n)
    win = r_mult[r_mult > 0]
    lose = r_mult[r_mult <= 0]
    be = (abs(lose.mean()) / (win.mean() + abs(lose.mean()))
          if len(win) and len(lose) else None)

    return {"n": n, "p_tp1_first": round(p, 4),
            "p_tp1_lo": round(max(0.0, p - 1.96 * se), 4),
            "p_tp1_hi": round(min(1.0, p + 1.96 * se), 4),
            "p_stop_first": round(float((outcome == 1).mean()), 4),
            "expectancy_R": round(float(r_mult.mean()), 3),
            "expectancy_lo": round(float(np.percentile(r_mult, 5)), 3),
            "expectancy_hi": round(float(np.percentile(r_mult, 95)), 3),
            "p_profitable": round(float((r_mult > 0).mean()), 4),
            "median_bars": int(np.median(bars)),
            "breakeven_p": round(be, 3) if be else None}


def verdict(sim, hist):
    if sim.get("error"):
        return "UNKNOWN", sim["error"]
    p, exp = sim["p_tp1_first"], sim["expectancy_R"]
    h = hist["rate"] if hist else None
    if p < 0.30:
        return "GEOMETRY_FAIL", f"only {p*100:.0f}% reach TP1 in simulation"
    if exp <= 0:
        return "NEGATIVE", f"simulated expectancy {exp:+.2f}R is negative"
    if h is None:
        return "UNPROVEN", f"sim {p*100:.0f}% / {exp:+.2f}R, no history yet"
    if h >= 0.40 and sim["p_profitable"] >= 0.55:
        return "SUPPORTED", f"sim {p*100:.0f}% and history {h*100:.0f}% agree"
    if h < 0.30:
        return "ENTRY_WEAK", f"geometry fine ({p*100:.0f}%) but history {h*100:.0f}%"
    return "MARGINAL", f"sim {p*100:.0f}%, history {h*100:.0f}%"
