from collections import defaultdict
import numpy as np
import pandas as pd


def find_swings(df: pd.DataFrame, left: int, right: int) -> list[dict]:
    """
    A bar i is a swing high if its high strictly exceeds the `left` bars
    before AND the `right` bars after. It is therefore only KNOWABLE at
    i + right. That is the confirmation lag — the anti-repainting guard.
    """
    h, l = df["high"].values, df["low"].values
    n = len(df)
    swings = []
    for i in range(left, n - right):
        wl, wr = h[i-left:i], h[i+1:i+right+1]
        if (h[i] > wl).all() and (h[i] > wr).all():
            swings.append({"idx": i, "price": float(h[i]), "type": "H",
                           "confirmed_at": i + right})
        wl, wr = l[i-left:i], l[i+1:i+right+1]
        if (l[i] < wl).all() and (l[i] < wr).all():
            swings.append({"idx": i, "price": float(l[i]), "type": "L",
                           "confirmed_at": i + right})
    return sorted(swings, key=lambda s: s["idx"])


def build_structure(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Per-bar trend state + BOS/CHoCH events, using ONLY swings already
    confirmed at bar t. A level is consumed once broken, so a break
    fires once instead of on every subsequent bar.
    """
    n = len(df)
    swings = find_swings(df, cfg.swing_left, cfg.swing_right)
    conf = defaultdict(list)
    for s in swings:
        conf[s["confirmed_at"]].append(s)

    trend = np.empty(n, dtype=object); trend[:] = "unclear"
    bos_up = np.zeros(n, bool); bos_dn = np.zeros(n, bool)
    ch_up  = np.zeros(n, bool); ch_dn  = np.zeros(n, bool)
    act_h  = np.full(n, np.nan); act_l = np.full(n, np.nan)
    idx_h = np.full(n, -1, int); idx_l = np.full(n, -1, int)
    sw_low = np.full(n, np.nan); sw_high = np.full(n, np.nan)

    highs: list[dict] = []
    lows:  list[dict] = []
    state = "unclear"
    broken_high = broken_low = -1
    close = df["close"].values

    for t in range(n):
        for s in conf.get(t, []):
            (highs if s["type"] == "H" else lows).append(s)

        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1]["price"] > highs[-2]["price"]
            hl = lows[-1]["price"]  > lows[-2]["price"]
            lh = highs[-1]["price"] < highs[-2]["price"]
            ll = lows[-1]["price"]  < lows[-2]["price"]
            if hh and hl:      state = "up"
            elif lh and ll:    state = "down"
            elif (hh and ll) or (lh and hl): state = "range"
        trend[t] = state

        if highs:
            act_h[t] = highs[-1]["price"]
            idx_h[t] = highs[-1]["idx"]
        if lows:
            act_l[t] = lows[-1]["price"]
            idx_l[t] = lows[-1]["idx"]
        if len(lows) >= 2:
            sw_low[t] = lows[-2]["price"]
        if len(highs) >= 2:
            sw_high[t] = highs[-2]["price"]

        if highs and highs[-1]["idx"] > broken_high and close[t] > act_h[t]:
            (bos_up if state == "up" else ch_up)[t] = True
            broken_high = highs[-1]["idx"]
        if lows and lows[-1]["idx"] > broken_low and close[t] < act_l[t]:
            (bos_dn if state == "down" else ch_dn)[t] = True
            broken_low = lows[-1]["idx"]

    return pd.DataFrame({
        "trend": trend, "bos_up": bos_up, "bos_dn": bos_dn,
        "choch_up": ch_up, "choch_dn": ch_dn,
        "active_high": act_h, "active_low": act_l,
        "active_high_idx": idx_h, "active_low_idx": idx_l,
        "prev_swing_low": sw_low, "prev_swing_high": sw_high,
    }, index=df.index)
