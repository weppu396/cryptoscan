"""core/structure2.py - dual-layer, point-in-time structure engine.

Trend is a STATE VARIABLE. Only the transition table writes it. Swing
comparisons label swings and detect ranging; they never write state.

Pivots: internal 3 left + 3 right, external 5 left + 5 right, both from
cfg.struct so they are tunable. A pivot at i is knowable at i+right and never
earlier. Swings are anchored to the WICK (high for H, low for L), which is the
only honest anchor for a structure line.

Break rules, non-repainting:

* A level is armed only by a pivot whose bar formed AFTER the previous break, so
  a stale pivot can never be re-used.
* A level can only break on a bar strictly AFTER the bar that confirmed it, so
  nothing fires on the confirmation bar itself.
* The armed level ratchets: highs only replace when higher, lows only when
  lower. A bullish BOS therefore originates from the confirmed HH and a bearish
  BOS from the confirmed LL. Continuation levels must extend the prior break.
* A break protects the latest confirmed pullback on the opposite side. A
  close through that swing changes character without requiring a break of
  the entire previous trend's extreme.
* A break requires a CLOSE beyond the level (cfg.struct.close_based).

Events confirm at the break bar's close, invalidation is stamped at the bar it
happens. Nothing reads forward.
"""
import numpy as np
import pandas as pd

from core.models import (Layer, Label, StructureEvent, StructureFrame, Swing,
                         new_id)

DEFAULT = {Layer.EXTERNAL: (5, 5), Layer.INTERNAL: (3, 3)}
TRANSITION = {
    ("unclear", "bull"): (Label.BOS_INIT, "up"),
    ("unclear", "bear"): (Label.BOS_INIT, "down"),
    ("up", "bull"): (Label.BOS, "up"),
    ("up", "bear"): (Label.CHoCH, "down"),
    ("down", "bear"): (Label.BOS, "down"),
    ("down", "bull"): (Label.CHoCH, "up"),
}


def _atr(df, n=14):
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr, index=df.index).rolling(n).mean().to_numpy()


def _swings(df, left, right, layer):
    """Fractal pivots on wicks; ties on the right belong to the first bar.

    This avoids losing pivots when consecutive candles share a high/low
    (common when a candle opens at the previous close). Confirmation
    still requires all right-hand candles to have closed.
    """
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    left = max(1, int(left))
    right = max(1, int(right))
    out = []
    if n < left + right + 1:
        return out
    for i in range(left, n - right):
        if (h[i] > h[i - left:i]).all() and (h[i] >= h[i + 1:i + right + 1]).all():
            out.append(Swing(i, float(h[i]), "H", layer, i + right, "H"))
        if (l[i] < l[i - left:i]).all() and (l[i] <= l[i + 1:i + right + 1]).all():
            out.append(Swing(i, float(l[i]), "L", layer, i + right, "L"))
    return sorted(out, key=lambda s: s.idx)


def _layer(df, sym, tf, layer, left, right, eq_tol, inv_bars, close_based):
    n = len(df)
    atr = _atr(df)
    c = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    up_probe = c if close_based else hi
    dn_probe = c if close_based else lo

    swings = _swings(df, left, right, layer)
    by_conf = {}
    for s in swings:
        by_conf.setdefault(s.confirmed_at, []).append(s)

    state = "unclear"
    state_arr = np.empty(n, dtype=object)
    state_arr[:] = "unclear"
    ranging = np.zeros(n, bool)

    bull = None          # (level, armed_at, swing) — highest confirmed high
    bear = None          # (level, armed_at, swing) — lowest confirmed low
    last_break = -1
    last_level = None
    last_h, last_l = [], []
    events = []

    def tol_at(i):
        a = atr[i]
        return eq_tol * a if a and not np.isnan(a) else 0.0

    for t in range(n):
        for s in by_conf.get(t, []):
            if s.kind == "H":
                last_h = (last_h + [s])[-2:]
                eligible = s.idx > last_break and (state != "up" or s.price > last_level)
                if eligible and (bull is None or (state != "down" and s.price > bull[0])):
                    bull = (s.price, t, s)
            else:
                last_l = (last_l + [s])[-2:]
                eligible = s.idx > last_break and (state != "down" or s.price < last_level)
                if eligible and (bear is None or (state != "up" and s.price < bear[0])):
                    bear = (s.price, t, s)

        if len(last_h) == 2 and len(last_l) == 2:
            et = tol_at(t)
            ranging[t] = (abs(last_h[0].price - last_h[1].price) <= et
                          and abs(last_l[0].price - last_l[1].price) <= et)

        if bull is not None and t > bull[1] and up_probe[t] > bull[0]:
            level, _at, s = bull
            bull = None
            last_break = t
            last_level = level
            label, state = TRANSITION[(state, "bull")]
            if last_l:
                protected = last_l[-1]
                bear = (protected.price, t, protected)
            events.append(StructureEvent(new_id(), sym, tf, layer, label, "bull",
                                         level, s.idx, t, t, None, state))
        elif bear is not None and t > bear[1] and dn_probe[t] < bear[0]:
            level, _at, s = bear
            bear = None
            last_break = t
            last_level = level
            label, state = TRANSITION[(state, "bear")]
            if last_h:
                protected = last_h[-1]
                bull = (protected.price, t, protected)
            events.append(StructureEvent(new_id(), sym, tf, layer, label, "bear",
                                         level, s.idx, t, t, None, state))

        state_arr[t] = state

    for i, e in enumerate(events):
        end = min(e.confirm_idx + 1 + inv_bars, n)
        win = dn_probe[e.confirm_idx + 1:end] if e.direction == "bull" \
            else up_probe[e.confirm_idx + 1:end]
        back = win < e.level if e.direction == "bull" else win > e.level
        idx = np.flatnonzero(back)
        if idx.size:
            events[i] = StructureEvent(e.event_id, e.symbol, e.timeframe, e.layer,
                                       e.label, e.direction, e.level, e.swing_idx,
                                       e.break_idx, e.confirm_idx,
                                       e.confirm_idx + 1 + int(idx[0]), e.trend_after)

    return events, swings, state_arr, ranging


def run(df: pd.DataFrame, symbol: str, tf: str, cfg) -> StructureFrame:
    """cfg is a Config; uses cfg.struct."""
    s = cfg.struct
    ext = (getattr(s, "ext_left", 5), getattr(s, "ext_right", 5))
    inn = (getattr(s, "int_left", 3), getattr(s, "int_right", 3))
    frame = StructureFrame(symbol=symbol, timeframe=tf)
    for layer, (lf, rt) in ((Layer.EXTERNAL, ext), (Layer.INTERNAL, inn)):
        ev, sw, st, rg = _layer(df, symbol, tf, layer, lf, rt, s.eq_tol_atr,
                                s.invalidate_bars, s.close_based)
        frame.events.extend(ev)
        frame.swings.extend(sw)
        frame.state[layer.value] = list(st)
        frame.ranging = list(np.maximum(np.array(frame.ranging or np.zeros(len(df), bool)), rg))
    frame.events.sort(key=lambda e: (e.confirm_idx, e.layer.value))
    return frame
