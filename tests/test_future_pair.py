"""HTF ceiling, LTF legality, future-only filtering, zoomable history."""
import numpy as np
import pandas as pd
import pytest

import chart
from config import CFG
from core.tfs import cap_htf, chart_minutes
from strategy import reversal

PLAN = {"entry": 0.003625, "sl": 0.00371, "tp1": 0.003497, "tp2": 0.003369}


def frame(n=900, freq="3min", vol=0.00004, seed=5, px0=0.0036):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    px = px0 + np.cumsum(rng.normal(0, vol, n))
    return pd.DataFrame({"open": px, "high": px + vol, "low": px - vol,
                         "close": px, "volume": rng.uniform(1, 40, n)}, index=idx)


def test_chart_minutes_parses_every_spelling():
    assert chart_minutes("1d") == 1440
    assert chart_minutes("1D") == 1440
    assert chart_minutes("4h") == 240
    assert chart_minutes("1w") == 10080
    assert chart_minutes("1H") == 60
    assert chart_minutes("30m") == 30
    assert chart_minutes(None) == 60


def test_cap_htf_clamps_days_and_lowercase_hours():
    assert cap_htf("1d") == "1h"
    assert cap_htf("1D") == "1h"
    assert cap_htf("4h") == "1h"
    assert cap_htf("4H") == "1h"
    assert cap_htf("15m") == "15m"
    assert cap_htf("30m") == "30m"
    assert cap_htf("1h") == "1h"


def test_htf_choices_are_all_within_the_ceiling():
    assert all(chart_minutes(t) <= 60 for t in reversal.HTF_CHOICES)


def test_ltf_is_legal_and_faster_than_its_htf():
    for htf, ltfs in reversal.LTF_FOR.items():
        assert ltfs
        for ltf in ltfs:
            assert chart_minutes(ltf) < chart_minutes(htf)


def test_ltf_stays_in_the_few_minute_band():
    for ltfs in reversal.LTF_FOR.values():
        for ltf in ltfs:
            assert chart_minutes(ltf) <= 15


def test_evaluate_never_returns_a_coarser_htf():
    src = {"3m": frame(900, "3min"), "5m": frame(900, "5min", 0.00006, 9),
           "15m": frame(900, "15min", 0.0002, 11)}
    r = reversal.evaluate("X", src, CFG)
    if r is not None:
        assert chart_minutes(r["htf"]) <= 60
        assert r["ltf"] in reversal.LTF_FOR[r["htf"]]


def test_played_rows_are_not_future():
    base = dict(zone=(1.0, 2.0), entry=2.0, stop=1.5, tp1=3.0, tp2=4.0,
                rr=3.0, is_new=True, htf="1h", ltf="15m", eta_min=None,
                bias="down", dist_atr=0.2, reasons=[], ts=None, symbol="X",
                direction="long", score=50.0)
    rows = [dict(base, status="MISSED", future=False),
            dict(base, status="ACTIVE_ENTRY", future=True),
            dict(base, status="APPROACHING", future=True)]
    assert [r["status"] for r in reversal.future_only(rows)] == \
        ["ACTIVE_ENTRY", "APPROACHING"]


def test_rank_puts_executable_first():
    base = dict(zone=(1.0, 2.0), entry=2.0, stop=1.5, tp1=3.0, tp2=4.0,
                rr=3.0, is_new=True, htf="1h", ltf="15m", eta_min=None,
                bias="down", dist_atr=0.2, reasons=[], ts=None, symbol="X",
                direction="long", score=50.0)
    live, dead = reversal.rank([dict(base, status="CONFIRMED", future=False),
                                dict(base, status="APPROACHING", future=True)])
    assert live[0]["status"] == "APPROACHING"
    assert dead == []


def test_history_is_loaded_behind_the_window():
    df = frame(900)
    fig = chart.build(df, CFG, "short", PLAN)
    n = len([t for t in fig.data if t.type == "candlestick"][0].x)
    assert n > 500


def test_initial_window_is_short():
    df = frame(900)
    fig = chart.build(df, CFG, "short", PLAN)
    lo, hi = fig.layout.xaxis.range
    assert (pd.Timestamp(hi) - pd.Timestamp(lo)) < (df.index[-1] - df.index[0])


def test_autorange_is_pinned_off():
    fig = chart.build(frame(), CFG, "short", PLAN)
    assert fig.layout.xaxis.autorange is False


def test_title_pair_is_capped():
    fig = chart.build(frame(), CFG, "short", PLAN, pair=("1d", "4h"))
    assert "1h (HTF" in fig.layout.title.text


def test_profile_bars_are_thick():
    fig = chart.build(frame(), CFG, "short", PLAN)
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof
    for t in prof:
        assert float(t.width[0]) > 0
