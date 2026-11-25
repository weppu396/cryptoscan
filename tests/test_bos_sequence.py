"""BOS must come from the confirmed HH / LL and only on a later close beyond it."""
import numpy as np
import pandas as pd
from dataclasses import replace

from config import CFG
from core.models import Layer
import core.structure2 as s2

DENSE = replace(CFG, struct=replace(CFG.struct, ext_left=2, ext_right=2,
                                    int_left=2, int_right=2))


def _df(seq):
    idx = pd.date_range("2024-01-01", periods=len(seq), freq="5min", tz="UTC")
    c = np.array(seq, float)
    o = c.copy()
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.05,
                         "low": np.minimum(o, c) - 0.05, "close": c,
                         "volume": 1.0}, index=idx)


BULL = [100, 101, 102, 103, 104, 103, 102, 103, 104, 105, 106, 105, 104,
        105, 106, 107, 108, 109]
BEAR = [109, 108, 107, 106, 105, 106, 107, 106, 105, 104, 103, 104, 105,
        104, 103, 102, 101, 100]


def _events(seq, direction):
    df = _df(seq)
    frame = s2.run(df, "X", "5m", DENSE)
    return df, [e for e in frame.events
                if e.layer == Layer.INTERNAL and e.direction == direction]


def test_bullish_bos_from_hh_on_later_close():
    df, ev = _events(BULL, "bull")
    assert ev, "no bullish break detected"
    e = ev[0]
    high = float(df["high"].iloc[e.swing_idx])
    assert abs(e.level - high) < 1e-9
    assert e.confirm_idx > e.swing_idx + 2          # never on the pivot bar
    assert float(df["close"].iloc[e.confirm_idx]) > e.level


def test_bearish_bos_from_ll_on_later_close():
    df, ev = _events(BEAR, "bear")
    assert ev, "no bearish break detected"
    e = ev[0]
    low = float(df["low"].iloc[e.swing_idx])
    assert abs(e.level - low) < 1e-9
    assert e.confirm_idx > e.swing_idx + 2
    assert float(df["close"].iloc[e.confirm_idx]) < e.level


def test_level_is_the_extreme_pivot_not_the_last_one():
    """A lower high between two higher highs must not become the BOS level."""
    df, ev = _events(BULL, "bull")
    e = ev[0]
    pivots = [s for s in s2._swings(df, 2, 2, Layer.INTERNAL) if s.kind == "H"]
    armed = [p for p in pivots if p.idx == e.swing_idx][0]
    assert armed.price == max(p.price for p in pivots if p.idx <= e.swing_idx)


def test_no_duplicate_levels():
    df, ev = _events(BULL, "bull")
    levels = [round(e.level, 9) for e in ev]
    assert len(levels) == len(set(levels))
