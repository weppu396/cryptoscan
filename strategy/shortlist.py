"""Manual research candidates with price-derived targets and closed-bar evidence."""
import math

import numpy as np
import pandas as pd

from core.candles import quality
from core.indicators import compute_all
from core.models import Layer
from core.opportunities import assess, ACTIVE
from core.structure import build_structure, find_swings
from core.structure2 import run
from core.tfs import chart_minutes
from strategy.setups import Setup, scan_bos_retest

PAIRS = (("30m", "3m"), ("1h", "5m"), ("4h", "30m"))


def _channel(df, swings, direction, atr, timestamp):
    """Two trend-side anchors and two independently confirmed far-side touches."""
    long = direction == "long"
    base = [s for s in swings if s["type"] == ("L" if long else "H")]
    opposite = [s for s in swings if s["type"] == ("H" if long else "L")]
    if len(base) < 2:
        return None
    a, b = base[-2:]
    slope = (b["price"] - a["price"]) / (b["idx"] - a["idx"])
    if (slope <= 0 if long else slope >= 0):
        return None
    opposite = [s for s in opposite if s["idx"] >= a["idx"]]
    if len(opposite) < 2:
        return None
    offsets = [(s["price"] - a["price"] - slope * (s["idx"] - a["idx"])) for s in opposite]
    edge = max(offsets) if long else min(offsets)
    if abs(edge) < atr or (edge <= 0 if long else edge >= 0):
        return None
    if sum(abs(value - edge) <= 0.35 * atr for value in offsets) < 2:
        return None
    base_line = a["price"] + slope * (np.arange(a["idx"], len(df)) - a["idx"])
    closes = df["close"].to_numpy()[a["idx"]:]
    if ((closes < base_line + min(0, edge) - 0.25 * atr)
            | (closes > base_line + max(0, edge) + 0.25 * atr)).any():
        return None
    step = df.index.to_series().diff().median()
    projected = a["price"] + slope * ((timestamp - df.index[a["idx"]]) / step)
    return projected, projected + edge, df.index[a["idx"]], df.index[b["idx"]], a["price"], b["price"]


def _net_rr(setup, cfg, target):
    friction = (cfg.costs.fee_bps + cfg.costs.slip_bps) / 10_000.0
    loss = setup.risk + (setup.entry + setup.stop) * friction
    return (abs(target - setup.entry) - (setup.entry + target) * friction) / loss


def scan_pair(symbol, high, low, cfg, htf, ltf, min_net_rr, now):
    if high is None or low is None or len(high) < 60 or len(low) < 60:
        return []
    h_step = pd.Timedelta(minutes=chart_minutes(htf))
    l_step = pd.Timedelta(minutes=chart_minutes(ltf))
    low = low.loc[low.index + l_step <= now]
    if len(low) < 60:
        return []
    ts = low.index[-1]
    high = high.loc[high.index + h_step <= ts + l_step]
    if len(high) < 60 or ts + l_step - (high.index[-1] + h_step) >= 2 * h_step:
        return []
    structure = run(high, symbol, htf, cfg)
    bias = structure.state_at(Layer.EXTERNAL, len(high) - 1)
    if bias not in ("up", "down"):
        return []
    direction = "long" if bias == "up" else "short"
    sign = 1 if direction == "long" else -1
    ind = compute_all(low, cfg)
    atr = float(ind["atr"].iat[-1])
    h_atr = float(compute_all(high, cfg)["atr"].iat[-1])
    if not all(math.isfinite(v) and v > 0 for v in (atr, h_atr)):
        return []
    candle, points = quality(low, len(low) - 1, atr, direction)
    if points < 60:
        return []
    entry = float(low["close"].iat[-1])
    swings = find_swings(high, cfg.struct.ext_left, cfg.struct.ext_right)
    swing_targets = sorted({s["price"] for s in swings
                            if s["type"] == ("H" if sign == 1 else "L")
                            and sign * (s["price"] - entry) > 0}, reverse=sign < 0)

    def price(value):
        return f"{value:,.9g}"

    body = low.iloc[-1]
    vol_ratio = float(ind["vol_ratio"].iat[-1])
    close_stamp = ts + l_step
    options = []
    if len(swing_targets) >= 2:
        history = pd.Series(structure.state[Layer.EXTERNAL.value], index=high.index + h_step)
        aligned = history.reindex(low.index + l_step, method="ffill").fillna("unclear").to_numpy()
        for setup in scan_bos_retest(symbol, low, aligned, ind,
                                     build_structure(low, cfg.structure), [], [], cfg):
            if setup.ts != ts or setup.direction != direction:
                continue
            setup.tp1, setup.tp2 = swing_targets[:2]
            setup.rr = setup.rr2
            setup.meta["strategy"] = "BOS Retest"
            setup.reasons.extend([
                f"Closed retest entry {price(entry)} after reclaiming BOS wick {price(setup.meta['bos_level'])}.",
                f"Stop {price(setup.stop)} beyond the retest extreme or BOS buffer, whichever is farther, plus {cfg.risk.stop_buffer_atr:g} LTF ATR.",
                f"TP1 {price(setup.tp1)} and TP2 {price(setup.tp2)} are the next two confirmed {htf} opposing swing prices.",
            ])
            options.append(setup)

    channel = _channel(high, swings, direction, h_atr, close_stamp)
    if channel:
        near, far, a_time, b_time, a_price, b_price = channel
        wick = float(body["low"] if sign == 1 else body["high"])
        previous_edge = float(low["high"].iat[-2] if sign == 1 else low["low"].iat[-2])
        if (abs(wick - near) <= 0.35 * h_atr and sign * (entry - near) > 0
                and sign * (entry - previous_edge) > 0):
            stop = (min(near, wick) if sign == 1 else max(near, wick)) - sign * cfg.risk.stop_buffer_atr * atr
            mid = (near + far) / 2.0
            # Nearby opposing swings can cap the channel projection.
            targets = sorted({p for p in [mid, far, *swing_targets]
                              if 0 < sign * (p - entry) <= sign * (far - entry)}, reverse=sign < 0)
            if len(targets) >= 2:
                setup = Setup(symbol, direction, "Trend Channel Retest", ts, entry, stop,
                              targets[0], targets[1], atr, 0.0, 0.0,
                              [f"{htf} trend-side anchors {price(a_price)} ({a_time:%d %b %H:%M}) and {price(b_price)} ({b_time:%d %b %H:%M}) UTC. Two confirmed touches on the far side.",
                               f"Projected near boundary {price(near)}, midpoint {price(mid)}, far boundary {price(far)} at confirmation.",
                               f"Retest wick {price(wick)} reached the boundary tolerance, then closed through the prior {ltf} wick at {price(previous_edge)}.",
                               f"Stop {price(stop)} beyond both the channel line and retest wick, plus {cfg.risk.stop_buffer_atr:g} LTF ATR.",
                               *[f"TP{i} {price(p)} · " + ("channel midpoint" if p == mid else "opposite channel boundary" if p == far else f"confirmed {htf} opposing swing")
                                 for i, p in enumerate(targets[:2], 1)]],
                              {"strategy": "Trend Channel Retest"})
                setup.rr = setup.rr2
                options.append(setup)

    accepted = []
    for setup in options:
        if not cfg.risk.min_stop_atr <= setup.stop_atr <= cfg.risk.max_stop_atr:
            continue
        status, reason = assess(setup, low, max_age_bars=0,
                                now=now, timeframe_seconds=l_step.total_seconds())
        net1, net2 = (_net_rr(setup, cfg, target) for target in (setup.tp1, setup.tp2))
        if status not in ACTIVE or net1 < 1.0 or net2 < min_net_rr:
            continue
        setup.meta.update(htf=htf, ltf=ltf, status=status, status_reason=reason,
                          net_tp1=net1, net_tp2=net2, net_weighted=(net1 + net2) / 2,
                          candle=candle, candle_score=points, confirmed_at=close_stamp)
        setup.reasons.insert(0, f"{htf} {bias} structure · {ltf} {candle} ({points}/100), confirmed {close_stamp:%d %b %H:%M} UTC. O {price(body['open'])} / H {price(body['high'])} / L {price(body['low'])} / C {price(body['close'])}. Volume {vol_ratio:.2f}× baseline.")
        accepted.append(setup)
    return accepted


def shortlist(candidates):
    """Reserve up to two swing plans and four intraday plans, one per coin."""
    best = sorted(candidates, key=lambda s: (-s.meta["candle_score"],
                                            -s.meta["net_weighted"], s.symbol))
    chosen, used = [], set()
    for swing, limit in ((True, 2), (False, 4)):
        count = 0
        for setup in best:
            coin = setup.symbol.split("/")[0]
            if (setup.meta["htf"] == "4h") != swing or coin in used:
                continue
            chosen.append(setup)
            used.add(coin)
            count += 1
            if count == limit:
                break
    return chosen
