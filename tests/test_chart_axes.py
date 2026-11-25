"""Chart contract: pivots, zones, gutter, zoom."""
import numpy as np
import pandas as pd
import pytest

import chart
from config import CFG
from core.structure_map import structure_lines, volume_zones

PLAN = {"entry": 0.003625, "sl": 0.00371, "tp1": 0.003497, "tp2": 0.003369}


def frame(n=400, seed=5, freq="3min", px0=0.0036, vol=0.00004):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    px = px0 + np.cumsum(rng.normal(0, vol, n))
    return pd.DataFrame({"open": px, "high": px + vol, "low": px - vol,
                         "close": px, "volume": rng.uniform(1, 40, n)}, index=idx)


def test_at_most_two_pivots_per_timeframe():
    df = frame()
    assert len(structure_lines(df, CFG, "3m", "LTF")) <= 2


def test_pivot_labels_are_trend_words():
    df = frame()
    lines = structure_lines(df, CFG, "3m", "LTF")
    assert all(l["label"] in ("HH", "HL", "LH", "LL", "H", "L") for l in lines)


def test_pivots_sit_on_real_wicks():
    df = frame()
    for l in structure_lines(df, CFG, "3m", "LTF"):
        row = df.loc[l["ts"]]
        assert l["price"] == pytest.approx(float(row["high"])) or \
            l["price"] == pytest.approx(float(row["low"]))


def test_zones_require_volume():
    df = frame()
    df.loc[df.index[100], "volume"] = 400.0
    zones = volume_zones(df, CFG, "3m", "LTF", "long", min_ratio=1.3)
    assert all(z["ratio"] >= 1.3 for z in zones)


def test_candles_own_most_of_the_width():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert tuple(fig.layout.xaxis.domain) == (0.0, 0.76)


def test_gutter_is_fenced_and_split():
    fig = chart.build(frame(), CFG, "short", PLAN, pair=("15m", "3m"))
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof
    for t in prof:
        assert t.xaxis == "x2"
    bases = sorted(float(t.base[0] if isinstance(t.base, (list, tuple)) else t.base) for t in prof)
    assert bases[0] < 0.5 and bases[-1] >= 0.5
    assert tuple(fig.layout.xaxis2.domain) == (0.785, 1.0)


def test_both_axes_zoomable_gutter_fixed():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert fig.layout.xaxis.fixedrange is False
    assert fig.layout.yaxis.fixedrange is False
    assert fig.layout.xaxis2.fixedrange is True
    assert fig.layout.dragmode == "pan"


def test_pivot_lines_reach_the_right_edge():
    fig = chart.build(frame(), CFG, "short", PLAN, pair=("15m", "3m"))
    x_hi = pd.Timestamp(fig.layout.xaxis.range[1])
    lines = [s for s in fig.layout.shapes
             if s.type == "line" and s.xref == "x" and s.y0 == s.y1]
    assert lines
    for s in lines:
        assert pd.Timestamp(s.x1) == x_hi


def test_profile_bars_are_thick():
    fig = chart.build(frame(), CFG, "short", PLAN)
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert all(float(t.width[0] if isinstance(t.width, (list, tuple)) else t.width) > 0 for t in prof)


def test_full_history_in_trace_with_recent_initial_window():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert len([t for t in fig.data if t.type == "candlestick"][0].x) == 400
    assert pd.Timestamp(fig.layout.xaxis.range[0]) >= frame().index[0]


def test_ohlc_is_complete_and_ordered():
    df = frame(130)
    fig = chart.build(df, CFG, "short", PLAN)
    c = [t for t in fig.data if t.type == "candlestick"][0]
    assert len(c.x) == len(c.open) == len(c.high) == len(c.low) == len(c.close)
    assert list(c.x) == sorted(c.x)
    for o, h, l, cl in zip(c.open, c.high, c.low, c.close):
        assert h >= max(o, cl) and l <= min(o, cl)


def test_tick_format_follows_timeframe():
    for tf, want in (("1m", "%H:%M"), ("30m", "%H:%M"), ("4H", "%d %b %H:%M")):
        fig = chart.build(frame(), CFG, "short", PLAN, chart_tf=tf)
        assert fig.layout.xaxis.tickformat == want


@pytest.mark.parametrize("tf", ["1m", "3m", "5m", "15m", "30m", "1H", "4H"])
def test_every_timeframe_renders_candles(tf):
    freq = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1H": "1h", "4H": "4h"}[tf]
    fig = chart.build(frame(200, freq=freq), CFG, "short", PLAN, chart_tf=tf)
    c = [t for t in fig.data if t.type == "candlestick"]
    assert c and len(c[0].x) > 0
