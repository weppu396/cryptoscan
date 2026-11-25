"""Strict current-opportunity gate. Never display completed or expired trades.

All signal timestamps and price paths refer to *closed* OHLCV bars. A candidate
is only a plan until an order is independently confirmed by an execution system.
"""
from datetime import datetime, timezone
import math
import pandas as pd

ACTIVE = frozenset({'ACTIONABLE', 'BETTER_PRICE'})
GRADES = {'D': 0, 'C': 1, 'B': 2, 'B+': 2.5, 'A-': 2.75, 'A': 3, 'A+': 4}


def assess(s, candles, *, max_age_bars=2, now=None, timeframe_seconds=None,
           max_entry_distance_atr=0.75):
    """Return (status, reason); all rejected states are excluded from ranking.

    Check *every* closed candle after the setup: a TP/SL touch means the
    original opportunity is no longer new, even if price later returns.
    """
    if candles is None or candles.empty or not candles.index.is_monotonic_increasing:
        return 'INVALID', 'No ordered candles'
    try:
        idx = candles.index.get_loc(s.ts)
        if not isinstance(idx, int):
            return 'INVALID', 'Ambiguous signal timestamp'
        age = len(candles) - 1 - idx
        if age < 0 or (max_age_bars is not None and age > max_age_bars):
            return 'EXPIRED', 'Signal outside freshness window'
        entry, stop, tp1, tp2, atr = map(float, (s.entry, s.stop, s.tp1, s.tp2, s.atr))
        if not all(map(math.isfinite, (entry, stop, tp1, tp2, atr))) or atr <= 0 or entry <= 0:
            return 'INVALID', 'Nonfinite or invalid trade levels'
        is_long = s.direction == 'long'
        if not (stop < entry < tp1 <= tp2 if is_long else tp2 <= tp1 < entry < stop):
            return 'INVALID', 'Trade levels on wrong side'
        if timeframe_seconds is not None:
            ts = pd.Timestamp(candles.index[-1]); ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
            now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
            now = now.tz_localize('UTC') if now.tzinfo is None else now
            # A closed bar is indexed by its opening timestamp.
            if (now - ts).total_seconds() > 2 * float(timeframe_seconds):
                return 'STALE_DATA', 'Last market candle is stale'
        future = candles.iloc[idx + 1:]
        if not future.empty:
            if is_long:
                if (future['low'] <= stop).any():
                    return 'INVALID', 'Stop touched after signal'
                if (future['high'] >= tp1).any():
                    return 'PLAYED', 'First target touched after signal'
            else:
                if (future['high'] >= stop).any():
                    return 'INVALID', 'Stop touched after signal'
                if (future['low'] <= tp1).any():
                    return 'PLAYED', 'First target touched after signal'
        cur = float(candles['close'].iloc[-1])
        if not math.isfinite(cur):
            return 'INVALID', 'Invalid latest price'
        if is_long and cur <= stop or not is_long and cur >= stop:
            return 'INVALID', 'Stop breached'
        if is_long and cur >= tp1 or not is_long and cur <= tp1:
            return 'PLAYED', 'Target already reached'
        distance = abs(cur - entry) / atr
        if distance > max_entry_distance_atr:
            return 'EXTENDED', 'Too far from planned entry'
        if (cur < entry if is_long else cur > entry):
            return 'BETTER_PRICE', 'Price improved versus planned entry; reconfirm trigger'
        return 'ACTIONABLE', 'Near planned entry; check live spread and trigger'
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        return 'INVALID', 'Cannot validate setup against candle history'


def rank_key(s):
    """Grade first, then confidence, weighted R and potential; never raw profit alone."""
    meta = s.meta or {}
    rt = meta.get('rating') or meta.get('r') or {}
    grade = rt.get('grade', 'D') if isinstance(rt, dict) else getattr(rt, 'grade', 'D')
    conf = rt.get('confidence', 0) if isinstance(rt, dict) else getattr(rt, 'confidence', 0)
    risk = abs(s.entry - s.stop)
    weighted = getattr(s, 'weighted_rr', 0)
    if not isinstance(weighted, (int, float)) or not math.isfinite(weighted):
        weighted = 0
    potential = abs(s.tp2 / s.entry - 1) if s.entry > 0 else 0
    try:
        confidence = float(conf)
        if not math.isfinite(confidence):
            confidence = 0.0
    except (TypeError, ValueError, OverflowError):
        confidence = 0.0
    return (-GRADES.get(str(grade).strip(), 0), -confidence,
            -float(weighted), -potential, s.symbol)


def current_only(setups):
    return sorted((s for s in setups if s.meta.get('status') in ACTIVE), key=rank_key)
