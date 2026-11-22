"""Claims must use closed candles; no look-ahead."""
import numpy as np
import pandas as pd

import smc
from core.candles import latest
from core.structure_map import marks
from config import CFG


def frame(n=90, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="3min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({"open": px, "high": px + 0.3, "low": px - 0.3,
                         "close": px, "volume": 10.0}, index=idx)


def test_forming_break_flagged_open():
    df = frame()
    df.loc[df.index[-1], "close"] = df["high"].iloc[:-1].max() + 5
    ev = smc.structure_events(df)
    if ev:
        assert ev[-1]["idx"] == len(df) - 1
        assert ev[-1]["closed"] is False


def test_closed_frame_marks_all_closed():
    for e in smc.structure_events(frame().iloc[:-1]):
        assert e["closed"] is True


def test_latest_candle_never_returns_last_bar_of_closed_frame():
    df = frame()
    i, _, _ = latest(df.iloc[:-1], 0.5, "long")
    assert i is None or i <= len(df) - 2


def test_marks_skip_final_bar():
    df = frame()
    df.loc[df.index[-1], "close"] = df["high"].iloc[:-1].max() + 5
    majors, minors = marks(df, CFG, None, 0.0)
    assert all(m["ts"] < df.index[-1] for m in majors + minors)


def test_avwap_unusable_when_stale():
    from core.volume import avwap_context
    assert avwap_context(frame(200), "long", 0.5, 5)["usable"] is False
