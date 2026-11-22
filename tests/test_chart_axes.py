"""Guards the chart axis contract. make_subplots owns x and x2; the profile
gutter must land on a free axis or the price panel collapses."""
import numpy as np
import pandas as pd
import pytest

import chart
import chart2
from config import CFG
from core import structure2
from core.models import Layer


def frame(n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    px = 100 + np.cumsum(rng.normal(0, 0.25, n))
    return pd.DataFrame({"open": px, "high": px + rng.uniform(.05, .4, n),
                         "low": px - rng.uniform(.05, .4, n), "close": px,
                         "volume": rng.uniform(1, 30, n)}, index=idx)


def _axis_of(trace):
    return trace.xaxis or "x"


def test_chart_profile_never_uses_x2():
    df = frame()
    fig = chart.build(df, CFG, "long", {"entry": 100, "sl": 99,
                                        "tp1": 102, "tp2": 104}, bars=150)
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof, "no profile trace drawn"
    for t in prof:
        assert _axis_of(t) != "x2", "profile stole the volume row axis"


def test_chart_candles_on_x_and_volume_on_x2():
    df = frame()
    fig = chart.build(df, CFG, "long", bars=150)
    candles = [t for t in fig.data if t.type == "candlestick"]
    bars = [t for t in fig.data if t.type == "bar" and getattr(t, "orientation", None) != "h"]
    assert candles and _axis_of(candles[0]) == "x"
    assert bars and _axis_of(bars[0]) == "x2"


def test_chart_matched_domains_are_identical():
    df = frame()
    fig = chart.build(df, CFG, "long", bars=150)
    assert tuple(fig.layout.xaxis.domain) == tuple(fig.layout.xaxis2.domain)


def test_chart_gutter_is_isolated_and_normalised():
    df = frame()
    fig = chart.build(df, CFG, "long", bars=150)
    gutter = [k for k in fig.layout.to_plotly_json()
              if k.startswith("xaxis") and k not in ("xaxis", "xaxis2")]
    assert len(gutter) == 1, "expected exactly one gutter axis"
    ax = fig.layout[gutter[0]]
    assert tuple(ax.domain) == (0.0, 0.2)
    assert tuple(ax.range) == (0, 1)
    assert ax.showticklabels is False


def test_chart_price_range_covers_candles():
    df = frame()
    view = df.tail(150)
    fig = chart.build(df, CFG, "long", bars=150)
    lo, hi = fig.layout.yaxis.range
    assert lo <= float(view["low"].min()) and hi >= float(view["high"].max())


def test_chart2_same_contract():
    df = frame()
    fr = structure2.run(df, "X", "5m", CFG)
    fig = chart2.build(df, fr, "long", {"entry": 100, "stop": 99,
                                        "tp1": 102, "tp2": 104}, mode="analysis")
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof
    for t in prof:
        assert _axis_of(t) not in ("x", "x2")
    assert tuple(fig.layout.xaxis.domain) == tuple(fig.layout.xaxis2.domain)


def test_chart2_single_row_has_no_orphan_x2():
    df = frame()
    fr = structure2.run(df, "X", "5m", CFG)
    fig = chart2.build(df, fr, "long", layers={"candles", "vp"})
    assert "xaxis2" not in fig.layout.to_plotly_json()
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof and _axis_of(prof[0]) == "x2"


def test_both_renderers_survive_empty_overlays():
    df = frame(60)
    chart.build(df, CFG, "long", None, bars=40)
    fr = structure2.run(df, "X", "5m", CFG)
    chart2.build(df, fr, "long", None, layers={"candles", "vp"})
