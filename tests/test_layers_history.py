"""Regression tests: layer independence, pivot labels, history retention.

Self-contained: builds its own OHLCV so no fixture module is required.
"""
import numpy as np
import pandas as pd

import chart
import chart2
from config import CFG
from core.models import Layer, StructureFrame


def _ohlcv(n=300, seed=0, drift=0.0, start=100.0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, n).cumsum()
    close = start * np.exp(steps)
    high = close * (1 + abs(rng.normal(0, 0.002, n)))
    low = close * (1 - abs(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(10, 100, n)},
                        index=idx)


def test_layers_use_their_own_frames():
    htf = _ohlcv(300, seed=1, drift=0.0012)
    ltf = _ohlcv(400, seed=2, drift=-0.0012)
    frame = chart.merge_frames(htf, ltf, CFG, "1h", "5m")
    layers = {e.layer for e in frame.events}
    assert Layer.EXTERNAL in layers
    assert Layer.INTERNAL in layers
    ext = [s for s in frame.swings if s.layer == Layer.EXTERNAL]
    inn = [s for s in frame.swings if s.layer == Layer.INTERNAL]
    assert ext and inn
    assert max(s.idx for s in ext) < len(htf)


def test_pivot_classification():
    swings = [{"idx": i * 10, "price": p, "kind": k}
              for i, (p, k) in enumerate([(10, "H"), (12, "H"), (11, "H"),
                                          (5, "L"), (4, "L"), (6, "L")])]
    tags = chart2._classify(swings, atr=1.0, eq_tol=0.05)
    assert [tags[id(s)] for s in swings] == ["", "HH", "LH", "", "LL", "HL"]


def test_equal_highs_flagged():
    swings = [{"idx": 0, "price": 10.0, "kind": "H"},
              {"idx": 5, "price": 10.02, "kind": "H"}]
    tags = chart2._classify(swings, atr=1.0, eq_tol=0.1)
    assert tags[id(swings[1])] == "EQH"


def test_no_lookahead():
    df = _ohlcv(200, seed=3)
    frame = chart.merge_frames(df, df, CFG, "1h", "5m")
    for s in frame.swings:
        assert s.confirmed_at > s.idx
    for e in frame.events:
        assert e.confirm_idx >= e.break_idx


def test_history_trace_keeps_whole_frame():
    df = _ohlcv(2500, seed=4)
    fig = chart2.build(df, StructureFrame(), "long", mode="structure", bars=110)
    candles = [t for t in fig.data if t.type == "candlestick"][0]
    assert len(candles.x) == 2500
    x0, x1 = fig.layout.xaxis.range
    assert (x1 - x0) < (df.index[-1] - df.index[0])


def test_uirevision_is_view_scoped():
    df = _ohlcv(400, seed=5)
    a = chart2.build(df, StructureFrame(), "long", mode="structure", chart_tf="5m")
    b = chart2.build(df, StructureFrame(), "long", mode="structure", chart_tf="1h")
    assert a.layout.uirevision != b.layout.uirevision


def test_plan_lines_render_in_analysis_mode():
    df = _ohlcv(300, seed=6)
    plan = {"entry": 100.0, "stop": 98.0, "tp1": 104.0, "tp2": 106.0}
    fig = chart2.build(df, StructureFrame(), "long", plan, mode="analysis")
    ys = {round(s.y0, 6) for s in fig.layout.shapes
          if s.type == "line" and s.y0 == s.y1}
    assert {100.0, 98.0, 104.0, 106.0} <= ys


def test_zone_failure_does_not_block_the_other_side(monkeypatch):
    df = _ohlcv(300, seed=7)
    calls = []

    def boom(f, *a, **k):
        calls.append(1)
        raise RuntimeError("htf down")

    monkeypatch.setattr(chart2, "volume_zones", boom)
    fig = chart2.build(df, StructureFrame(), "long", mode="analysis")
    assert len(calls) == 2
    assert fig is not None
