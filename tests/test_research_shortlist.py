from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import CFG
from core.models import Layer
from strategy import shortlist as research


def test_shortlist_prioritizes_swing_and_ranks_by_net_reward_to_risk():
    def row(symbol, htf, score):
        return SimpleNamespace(symbol=symbol, meta={"htf": htf, "candle_score": score,
                                                    "net_weighted": 4.0, "net_tp2": score / 25})
    rows = [row(f"S{i}/USDT", "4h", 100 - i) for i in range(4)]
    rows += [row(f"D{i}/USDT", "30m", 95 - i) for i in range(8)]
    rows.append(row("S0/USDT", "1h", 100))
    selected = research.shortlist(rows)
    assert len(selected) == 3
    assert sum(s.meta["htf"] == "4h" for s in selected) == 3
    assert [s.symbol for s in selected] == ["S0/USDT", "S1/USDT", "S2/USDT"]


def test_channel_uses_structural_boundaries_and_rejects_a_broken_channel():
    index = pd.date_range("2026-01-01", periods=16, freq="4h", tz="UTC")
    close = 100 + np.arange(16) + 3.0
    frame = pd.DataFrame({"close": close}, index=index)
    swings = [{"type": "L", "idx": 2, "price": 102},
              {"type": "H", "idx": 4, "price": 110},
              {"type": "L", "idx": 7, "price": 107},
              {"type": "H", "idx": 9, "price": 115}]
    near, far, *_ = research._channel(frame, swings, "long", 2.0, index[-1])
    assert near == 115 and far == 121
    assert research._channel(frame, swings[:-1], "long", 2.0, index[-1]) is None
    frame.iloc[-1, 0] = 113
    assert research._channel(frame, swings, "long", 2.0, index[-1]) is None


def test_shortlist_keeps_valid_structural_targets_below_five_r(monkeypatch):
    close_ts = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    high_ix = pd.date_range(end=close_ts - pd.Timedelta(hours=4), periods=80, freq="4h")
    low_ix = pd.date_range(end=close_ts - pd.Timedelta(minutes=30), periods=80, freq="30min")
    def candles(index):
        return pd.DataFrame({"open": 100, "high": 101.0, "low": 99.8, "close": 100.5,
                             "volume": 1000}, index=index)
    high, low = candles(high_ix), candles(low_ix)
    low.iloc[-2, low.columns.get_loc("high")] = 100.4
    low.iloc[-2, low.columns.get_loc("close")] = 100.2
    monkeypatch.setattr(research, "run", lambda df, *_: SimpleNamespace(
        state_at=lambda *a: "up", state={Layer.EXTERNAL.value: ["up"] * len(df)}))
    monkeypatch.setattr(research, "quality", lambda *a: ("bullish engulfing", 100))
    monkeypatch.setattr(research, "compute_all",
                        lambda df, _: pd.DataFrame({"atr": [1.0] * len(df), "vol_ratio": 1.5},
                                                   index=df.index))
    monkeypatch.setattr(research, "find_swings", lambda *a: [])
    monkeypatch.setattr(research, "_channel",
                        lambda *a: (100, 104, high_ix[1], high_ix[2], 85, 86))
    rows = research.scan_pair("TEST/USDT", high, low, CFG, "4h", "30m", close_ts)
    assert len(rows) == 1
    setup = rows[0]
    assert (setup.entry, setup.stop, setup.tp1, setup.tp2) == (100.5, 99.55, 102, 104)
    assert setup.meta["net_tp2"] < setup.rr2
    assert setup.meta["net_weighted"] == pytest.approx((setup.meta["net_tp1"] + setup.meta["net_tp2"]) / 2)
    assert 1 < setup.meta["net_tp2"] < 5


def test_draws_both_saved_boundaries_at_their_confirmation_prices():
    import plotly.graph_objects as go
    from chart_axes import add_channel_boundaries

    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = start + pd.Timedelta(hours=4)
    figure = go.Figure().update_xaxes(range=[start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=3)])
    prices = add_channel_boundaries(figure, {
        "start": start, "end": end, "lower": (90, 94), "upper": (100, 104),
    })
    assert prices == (102, 104, 92, 94)
    assert [trace.name for trace in figure.data] == ["HTF channel upper", "HTF channel lower"]
    assert all(trace.x[-1] == end for trace in figure.data)
    assert figure.layout.xaxis.range[1] == end


def test_swing_targets_keep_tp1_and_report_levels_before_distinct_tp2():
    assert research._structure_targets([104, 104.1, 104.3, 110], 0.5) == (
        [104, 110], [104.1, 104.3])
    assert research._structure_targets([96, 95.9, 90], 0.5) == ([96, 90], [95.9])
    assert research._structure_targets([104, 104.1], 0.5) == ([104, 104.1], [])


def test_previous_swing_signal_remains_eligible_until_stop_or_target(monkeypatch):
    from strategy.setups import Setup

    now = pd.Timestamp("2026-01-02", tz="UTC")
    low_index = pd.date_range(end=now - pd.Timedelta(minutes=30), periods=80, freq="30min")
    high_index = pd.date_range(end=now - pd.Timedelta(hours=4), periods=80, freq="4h")
    low = pd.DataFrame(dict(open=100.0, high=101.0, low=99.5, close=100.0), index=low_index)
    high = low.set_axis(high_index)
    signal = low.index[-2]
    monkeypatch.setattr(research, "run", lambda *args: SimpleNamespace(state_at=lambda *_: "up"))

    def scan(symbol, high, prefix, *_):
        if prefix.index[-1] == signal:
            return [Setup(symbol, "long", "BOS Retest", signal, 100, 98, 104, 116, 1, 8, 0)]
        return []

    monkeypatch.setattr(research, "_scan_closed_pair", scan)
    assert [s.ts for s in research.scan_pair("X/USDT", high, low, CFG, "4h", "30m", now)] == [signal]
    low.loc[low.index[-1], "high"] = 104
    assert not research.scan_pair("X/USDT", high, low, CFG, "4h", "30m", now)
