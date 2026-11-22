"""Chart layout contract."""
import numpy as np
import pandas as pd
import pytest

import chart
import chart2
from config import CFG
from core import structure2

PLAN = {"entry": 0.003625, "sl": 0.00371, "tp1": 0.003497, "tp2": 0.003369}


def frame(n=400, seed=5, freq="5min", px0=0.0036, vol=0.00004):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    px = px0 + np.cumsum(rng.normal(0, vol, n))
    return pd.DataFrame({"open": px, "high": px + vol, "low": px - vol,
                         "close": px, "volume": rng.uniform(1, 30, n)}, index=idx)


def test_candles_own_most_of_the_width():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert tuple(fig.layout.xaxis.domain) == (0.0, 0.82)


def test_profile_is_on_the_right_on_its_own_axis():
    fig = chart.build(frame(), CFG, "short", PLAN)
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof
    for t in prof:
        assert t.xaxis == "x2"
    assert tuple(fig.layout.xaxis2.domain) == (0.845, 1.0)
    assert tuple(fig.layout.xaxis2.range) == (0, 1)
    assert fig.layout.xaxis2.showticklabels is False


def test_default_window_is_100_to_120():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert 100 <= len([t for t in fig.data if t.type == "candlestick"][0].x) <= 120


def test_ohlc_is_complete_and_ordered():
    df = frame(130)
    fig = chart.build(df, CFG, "short", PLAN)
    c = [t for t in fig.data if t.type == "candlestick"][0]
    assert len(c.x) == len(c.open) == len(c.high) == len(c.low) == len(c.close)
    assert list(c.x) == sorted(c.x)
    for o, h, l, cl in zip(c.open, c.high, c.low, c.close):
        assert h >= max(o, cl) and l <= min(o, cl)


def test_price_range_covers_every_wick():
    df = frame(130)
    view = df.tail(110)
    fig = chart.build(df, CFG, "short", PLAN)
    lo, hi = fig.layout.yaxis.range
    assert lo <= float(view["low"].min()) and hi >= float(view["high"].max())


def test_no_bottom_volume_row():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert "yaxis2" not in fig.layout.to_plotly_json()


def test_structure_markers_are_horizontal_and_bounded():
    fig = chart.build(frame(), CFG, "short", PLAN)
    lines = [s for s in fig.layout.shapes
             if s.type == "line" and s.xref == "x" and s.y0 == s.y1]
    assert lines
    for s in lines:
        assert s.x1 > s.x0


@pytest.mark.parametrize("tf", ["1m", "3m", "5m", "15m", "30m", "1H", "4H"])
def test_every_timeframe_renders_candles(tf):
    freq = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1H": "1h", "4H": "4h"}[tf]
    fig = chart.build(frame(200, freq=freq), CFG, "short", PLAN, chart_tf=tf)
    c = [t for t in fig.data if t.type == "candlestick"]
    assert c and len(c[0].x) > 0


def test_display_tf_does_not_change_analysis():
    df = frame(300)
    a = chart.build(df, CFG, "short", PLAN, chart_tf="3m", analysis_df=df)
    b = chart.build(df, CFG, "short", PLAN, chart_tf="4H", analysis_df=df)
    ca = [t for t in a.data if t.type == "candlestick"][0]
    cb = [t for t in b.data if t.type == "candlestick"][0]
    assert len(ca.x) == len(cb.x)


def test_chart2_same_contract():
    df = frame()
    fr = structure2.run(df, "X", "5m", CFG)
    fig = chart2.build(df, fr, "short", PLAN, mode="analysis")
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof and all(t.xaxis == "x2" for t in prof)
    assert tuple(fig.layout.xaxis.domain) == (0.0, 0.82)
