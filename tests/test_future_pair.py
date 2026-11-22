"""HTF ceiling, LTF ceiling, future-only filtering, zoomable history."""
import numpy as np
import pandas as pd
import pytest

import chart
from config import CFG
from core.tfs import cap_htf, cap_ltf, chart_minutes
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
    assert chart_minutes("4h") == 240
    assert chart_minutes("1s") < 1
    assert chart_minutes("30m") == 30
    assert chart_minutes(None) == 60


def test_cap_htf_clamps_days_and_lowercase_hours():
    assert cap_htf("1d") == "1h"
    assert cap_htf("4h") == "1h"
    assert cap_htf("1h") == "1h"
    assert cap_htf("30m") == "30m"


def test_cap_ltf_clamps_everything_slower_than_five_minutes():
    assert cap_ltf("4h") == "5m"
    assert cap_ltf("1h") == "5m"
    assert cap_ltf("15m") == "5m"
    assert cap_ltf("5m") == "5m"
    assert cap_ltf("3m") == "3m"


def test_htf_choices_are_only_thirty_min_and_one_hour():
    assert set(reversal.HTF_CHOICES) == {"30m", "1h"}


def test_ltf_choices_are_only_three_and_five_minutes():
    assert set(reversal.LTF_CHOICES) == {"3m", "5m"}
    for ltfs in reversal.LTF_FOR.values():
        assert set(ltfs) <= {"3m", "5m"}


def test_evaluate_never_returns_a_slow_pair():
    src = {"3m": frame(900, "3min"), "5m": frame(900, "5min", 0.00006, 9),
           "15m": frame(900, "15min", 0.0002, 11)}
    r = reversal.evaluate("X", src, CFG)
    if r is not None:
        assert chart_minutes(r["htf"]) <= 60
        assert chart_minutes(r["ltf"]) <= 5
        assert r["ltf"] in reversal.LTF_FOR[r["htf"]]


def test_choose_pair_stays_inside_the_bands():
    src = {"3m": frame(900, "3min"), "5m": frame(900, "5min", 0.00006, 3),
           "15m": frame(900, "15min", 0.0002, 4)}
    htf, ltf, _q = reversal.choose_pair(src, CFG)
    assert htf in reversal.HTF_CHOICES
    assert ltf in reversal.LTF_CHOICES


def test_played_rows_are_not_future():
    base = dict(zone=(1.0, 2.0), entry=2.0, stop=1.5, tp1=3.0, tp2=4.0,
                rr=3.0, is_new=True, htf="1h", ltf="5m", eta_min=None,
                bias="down", dist_atr=0.2, reasons=[], ts=None, symbol="X",
                direction="long", score=50.0)
    rows = [dict(base, status="MISSED", future=False),
            dict(base, status="ACTIVE_ENTRY", future=True)]
    assert [r["status"] for r in reversal.future_only(rows)] == ["ACTIVE_ENTRY"]


def test_rank_puts_executable_first():
    base = dict(zone=(1.0, 2.0), entry=2.0, stop=1.5, tp1=3.0, tp2=4.0,
                rr=3.0, is_new=True, htf="1h", ltf="5m", eta_min=None,
                bias="down", dist_atr=0.2, reasons=[], ts=None, symbol="X",
                direction="long", score=50.0)
    live, dead = reversal.rank([dict(base, status="CONFIRMED", future=False),
                                dict(base, status="APPROACHING", future=True)])
    assert live[0]["status"] == "APPROACHING"


def test_title_pair_is_capped_on_both_sides():
    fig = chart.build(frame(), CFG, "short", PLAN, pair=("1d", "4h"))
    assert "1h (HTF" in fig.layout.title.text
    assert "5m (LTF" in fig.layout.title.text


def test_history_is_loaded_behind_the_window():
    df = frame(900)
    fig = chart.build(df, CFG, "short", PLAN)
    assert len([t for t in fig.data if t.type == "candlestick"][0].x) > 500


def test_initial_window_is_short_and_autorange_off():
    df = frame(900)
    fig = chart.build(df, CFG, "short", PLAN)
    lo, hi = fig.layout.xaxis.range
    assert (pd.Timestamp(hi) - pd.Timestamp(lo)) < (df.index[-1] - df.index[0])
    assert fig.layout.xaxis.autorange is False


def test_second_timeframe_uses_seconds_ticks():
    fig = chart.build(frame(300, "1s"), CFG, "short", PLAN, chart_tf="1s")
    assert fig.layout.xaxis.tickformat == "%H:%M:%S"


def test_profile_bars_are_thick():
    fig = chart.build(frame(), CFG, "short", PLAN)
    prof = [t for t in fig.data if getattr(t, "orientation", None) == "h"]
    assert prof and all(float(t.width[0]) > 0 for t in prof)
