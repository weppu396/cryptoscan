from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Setup:
    symbol: str
    direction: str
    pattern: str
    ts: pd.Timestamp
    entry: float
    stop: float
    tp1: float
    tp2: float
    atr: float
    rr: float
    score: float
    reasons: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def stop_atr(self) -> float:
        return abs(self.entry - self.stop) / self.atr


def scan_bos_retest(sym, ltf, htf_b, ind, st, zones_up, zones_dn, cfg):
    htf_b = np.asarray(htf_b)

    setups = []
    n = len(ltf)
    pend = None
    o, h, l, c = (ltf[x].values for x in ("open", "high", "low", "close"))
    tol_mult = cfg.structure.zone_tol_atr
    cap = cfg.risk.max_tp_r

    for t in range(n):
        if st["bos_up"].iat[t] and htf_b[t] == "up":
            lvl = st["active_high"].iat[t]
            if not np.isnan(lvl):
                pend = {"dir": "long", "level": float(lvl),
                        "expiry": t + cfg.structure.max_wait_bars}
        elif st["bos_dn"].iat[t] and htf_b[t] == "down":
            lvl = st["active_low"].iat[t]
            if not np.isnan(lvl):
                pend = {"dir": "short", "level": float(lvl),
                        "expiry": t + cfg.structure.max_wait_bars}

        if pend is None:
            continue
        if t > pend["expiry"]:
            pend = None
            continue

        a = float(ind["atr"].iat[t])
        if not np.isfinite(a) or a <= 0:
            continue
        tol = tol_mult * a
        lvl = pend["level"]
        rng = float(h[t] - l[t])
        if rng <= 0:
            continue
        body = abs(float(c[t] - o[t]))

        if pend["dir"] == "long":
            touched = l[t] <= lvl + tol and h[t] >= lvl - tol
            reject = c[t] > o[t] and c[t] > lvl and (body / rng) > 0.5
            if touched and (reject or not cfg.structure.require_rejection_candle):
                entry = float(c[t])
                stop = min(float(l[t]), lvl - tol) - cfg.risk.stop_buffer_atr * a
                risk = entry - stop
                if risk <= 0:
                    pend = None
                    continue
                tp1 = entry + 1.5 * risk
                nxt = [z for z in zones_up if z["mid"] > entry + 0.5 * risk]
                raw_tp2 = max(entry + 3.0 * risk,
                              min(nxt, key=lambda z: z["mid"])["mid"] if nxt else 0.0)
                tp2 = min(raw_tp2, entry + cap * risk)
                setups.append(Setup(
                    sym, "long", "BOS_retest_long", ltf.index[t], entry, stop,
                    tp1, tp2, a, (tp2 - entry) / risk, 0.0,
                    ["HTF bias up",
                     f"4H bullish BOS above {lvl:,.2f}",
                     "retest held with bullish rejection close",
                     f"stop beyond swing + {cfg.risk.stop_buffer_atr} ATR buffer"]))
                pend = None
        else:
            touched = h[t] >= lvl - tol and l[t] <= lvl + tol
            reject = c[t] < o[t] and c[t] < lvl and (body / rng) > 0.5
            if touched and (reject or not cfg.structure.require_rejection_candle):
                entry = float(c[t])
                stop = max(float(h[t]), lvl + tol) + cfg.risk.stop_buffer_atr * a
                risk = stop - entry
                if risk <= 0:
                    pend = None
                    continue
                tp1 = entry - 1.5 * risk
                nxt = [z for z in zones_dn if z["mid"] < entry - 0.5 * risk]
                raw_tp2 = min(entry - 3.0 * risk,
                              max(nxt, key=lambda z: z["mid"])["mid"] if nxt else 1e18)
                tp2 = max(raw_tp2, entry - cap * risk)
                setups.append(Setup(
                    sym, "short", "BOS_retest_short", ltf.index[t], entry, stop,
                    tp1, tp2, a, (entry - tp2) / risk, 0.0,
                    ["HTF bias down",
                     f"4H bearish BOS below {lvl:,.2f}",
                     "retest rejected with bearish close",
                     f"stop beyond swing + {cfg.risk.stop_buffer_atr} ATR buffer"]))
                pend = None

    return setups