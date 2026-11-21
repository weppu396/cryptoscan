"""core/structure2.py - dual-layer, point-in-time structure engine.

Trend is a STATE VARIABLE. Only the transition table writes it. Swing
comparisons label swings and detect ranging; they never write state.

Non-repainting: swings confirm at i+right; events confirm at the break bar's
close; invalidation is stamped at the bar it happens.
"""
import numpy as np
import pandas as pd

from core.models import (Layer, Label, StructureEvent, StructureFrame, Swing,
                         new_id)

DEFAULT = {Layer.EXTERNAL: (5, 5), Layer.INTERNAL: (2, 2)}
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
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    out = []
    for i in range(left, n - right):
        if (h[i] > h[i - left:i]).all() and (h[i] > h[i + 1:i + right + 1]).all():
            out.append(Swing(i, float(h[i]), "H", layer, i + right))
        if (l[i] < l[i - left:i]).all() and (l[i] < l[i + 1:i + right + 1]).all():
            out.append(Swing(i, float(l[i]), "L", layer, i + right))
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

    armed_h = armed_l = None
    last_h, last_l = [], []
    events = []

    def tol_at(i):
        a = atr[i]
        return eq_tol * a if a and not np.isnan(a) else 0.0

    for t in range(n):
        for s in by_conf.get(t, []):
            tol = tol_at(t)
            if s.kind == "H":
                last_h = (last_h + [s])[-2:]
                if armed_h is None:
                    armed_h = s
                elif abs(s.price - armed_h.price) <= tol:
                    if s.price > armed_h.price:
                        armed_h = s
                else:
                    armed_h = s
            else:
                last_l = (last_l + [s])[-2:]
                if armed_l is None:
                    armed_l = s
                elif abs(s.price - armed_l.price) <= tol:
                    if s.price < armed_l.price:
                        armed_l = s
                else:
                    armed_l = s

            if len(last_h) == 2:
                last_h[0].__class__  # noop, keeps tag assignment below explicit
        if len(last_h) == 2 and len(last_l) == 2:
            et = tol_at(t)
            ranging[t] = (abs(last_h[0].price - last_h[1].price) <= et
                          and abs(last_l[0].price - last_l[1].price) <= et)

        if armed_h is not None and up_probe[t] > armed_h.price:
            s, armed_h = armed_h, None
            label, state = TRANSITION[(state, "bull")]
            events.append(StructureEvent(new_id(), sym, tf, layer, label, "bull",
                                         s.price, s.idx, t, t, None, state))
        elif armed_l is not None and dn_probe[t] < armed_l.price:
            s, armed_l = armed_l, None
            label, state = TRANSITION[(state, "bear")]
            events.append(StructureEvent(new_id(), sym, tf, layer, label, "bear",
                                         s.price, s.idx, t, t, None, state))

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
    frame = StructureFrame(symbol=symbol, timeframe=tf)
    for layer, (lf, rt) in ((Layer.EXTERNAL, (s.ext_left, s.ext_right)),
                            (Layer.INTERNAL, (s.int_left, s.int_right))):
        ev, sw, st, rg = _layer(df, symbol, tf, layer, lf, rt, s.eq_tol_atr,
                                s.invalidate_bars, s.close_based)
        frame.events.extend(ev)
        frame.swings.extend(sw)
        frame.state[layer.value] = list(st)
        frame.ranging = list(np.maximum(np.array(frame.ranging or np.zeros(len(df), bool)), rg))
    frame.events.sort(key=lambda e: (e.confirm_idx, e.layer.value))
    return frame
