"""core/candles.py - ATR-normalised reversal candles. One definition, used everywhere."""
import numpy as np


def _geom(df, i):
    o, h, l, c = (float(df[k].iat[i]) for k in ("open", "high", "low", "close"))
    rng = h - l
    body = abs(c - o)
    return o, h, l, c, rng, body, h - max(o, c), min(o, c) - l


def quality(df, i, atr, direction="long"):
    """(name, score 0-100). Score 0 = no valid reversal candle.

    ATR-normalised so the same test means the same thing on BTC and on a 3m
    altcoin. Requires a real range (>= 0.25 ATR) so inside bars never fire.
    """
    if i < 2 or not atr or atr <= 0 or i >= len(df):
        return None, 0
    o, h, l, c, rng, body, upper, lower = _geom(df, i)
    po, ph, pl, pc, _, _, _, _ = _geom(df, i - 1)
    if rng < 0.25 * atr:
        return None, 0
    body_r = body / rng

    if direction == "long":
        if pc < po and c > o and c >= po and o <= pc and body_r >= 0.55:
            return "bullish engulfing", 100
        if lower >= 2.0 * body and lower >= 0.5 * atr and body_r <= 0.35 and c > l + 0.6 * rng:
            return "hammer", 85
        if body_r <= 0.25 and lower >= 0.6 * rng:
            return "bullish pin", 70
        if c > o and body_r >= 0.7 and c >= h - 0.15 * rng:
            return "bullish marubozu", 60
    else:
        if pc > po and c < o and c <= po and o >= pc and body_r >= 0.55:
            return "bearish engulfing", 100
        if upper >= 2.0 * body and upper >= 0.5 * atr and body_r <= 0.35 and c < h - 0.6 * rng:
            return "shooting star", 85
        if body_r <= 0.25 and upper >= 0.6 * rng:
            return "bearish pin", 70
        if c < o and body_r >= 0.7 and c <= l + 0.15 * rng:
            return "bearish marubozu", 60
    return None, 0


def latest(df, atr, direction="long", look=4):
    """Best candle in the last `look` bars: (idx, name, score) or (None, None, 0)."""
    best = (None, None, 0)
    for i in range(max(2, len(df) - look), len(df)):
        name, q = quality(df, i, atr, direction)
        if q > best[2]:
            best = (i, name, q)
    return best
