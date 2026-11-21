"""core/pit.py - point-in-time slicing. Nothing here may see the future.

`bias_at` hands a truncated frame to a bias function so an HTF call at time t
only ever sees bars that had closed by t.
"""
import pandas as pd

VALID = ("up", "down", "unclear", "range")


def _pos(index, ts):
    """Last position at or before ts. None if ts precedes the series."""
    if index is None or len(index) == 0:
        return None
    try:
        target = pd.Timestamp(ts)
        tz = getattr(index, "tz", None)
        if tz is not None and target.tzinfo is None:
            target = target.tz_localize(tz)
        elif tz is None and target.tzinfo is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        pos = int(index.searchsorted(target, side="right")) - 1
    except Exception:
        return None
    if pos < 0 or pos >= len(index):
        return None
    return pos


def slice_to(df, ts):
    """Rows up to and including the bar that contains ts. None if out of range."""
    if df is None or len(df) == 0:
        return None
    pos = _pos(df.index, ts)
    if pos is None:
        return None
    return df.iloc[:pos + 1]


def bias_at(htf, ts, fn, min_bars: int = 60) -> str:
    """fn(slice) -> bias string. Defensive: any failure is 'unclear'."""
    seg = slice_to(htf, ts)
    if seg is None or len(seg) < min_bars:
        return "unclear"
    try:
        out = fn(seg)
    except Exception:
        return "unclear"
    return out if out in VALID else "unclear"


def state_at(frame, layer, ts, index):
    """Convenience: StructureFrame state at ts via the source index."""
    pos = _pos(index, ts)
    if pos is None:
        return "unclear"
    return frame.state_at(layer, pos)
