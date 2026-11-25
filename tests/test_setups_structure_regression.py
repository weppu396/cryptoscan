from dataclasses import replace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import chart2
from config import CFG
from core.models import Label, Layer
from core.structure2 import run
from strategy import reversal


def test_setups_renders_multiple_cards_for_one_symbol():
    app = AppTest.from_string("""
import pandas as pd
import streamlit as st
from types import SimpleNamespace
from unittest.mock import patch
import board
import plotly.graph_objects as go
from config import CFG

setup = SimpleNamespace(
    symbol="BTC/USDT", direction="long", ts=pd.Timestamp("2026-01-01"),
    entry=100, stop=98, tp1=104, tp2=106,
    rr1=2, rr2=3, weighted_rr=2.5, reasons=[],
    meta={"status": "ACTIONABLE", "r": {"grade": "A", "confidence": 90,
          "positives": [], "negatives": []}},
)
st.session_state["board"] = [setup, setup]
candles = pd.DataFrame({"close": [100, 101, 102]})
with (
    patch.object(board, "_auto_pair", return_value=("30m", "5m", {})),
    patch.object(board.chart, "build", return_value=go.Figure(go.Scatter(y=[1, 2]))),
):
    board.render(None, lambda *a: CFG, lambda *a: (SimpleNamespace(id="test"), ""),
             lambda *a: (candles, setup.symbol), lambda x: x, lambda *a: None,
             None, "test", 25, 0, "5m", 2, 3, 1, 1000, 12, None,
             True, False, False, False)
""").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert not any("chart error" in caption.value for caption in app.caption)
    for label in ("Alert me", "Log to paper"):
        buttons = [button for button in app.button if button.label == label]
        assert len(buttons) == len({button.key for button in buttons}) == 2
    selectors = [widget for widget in app.selectbox if widget.label == "Chart timeframe"]
    assert len(selectors) == len({widget.key for widget in selectors}) == 2


def test_one_hour_uses_only_complete_15_minute_groups():
    index = pd.date_range("2026-01-01 00:15", periods=9, freq="15min", tz="UTC")
    base = pd.DataFrame({key: range(9) for key in ("open", "high", "low", "close", "volume")},
                        index=index)

    hourly = reversal._frame({"15m": base}, "1h")

    assert list(hourly.index) == [index[3]]
    assert hourly.iloc[0][["open", "high", "close"]].tolist() == [3, 6, 6]


@pytest.mark.parametrize("side, sign", [("bull", 1), ("bear", -1)])
def test_break_candle_and_wick_anchor_are_separate(side, sign):
    prices = [7, 10, 8, 11, 14, 11, 13, 11, 13.5, 11, 15, 17, 15, 18]
    prices = [30 + sign * price for price in prices]
    index = pd.date_range("2026-01-01", periods=len(prices), freq="5min", tz="UTC")
    df = pd.DataFrame({"open": prices, "high": [p + .1 for p in prices],
                       "low": [p - .1 for p in prices], "close": prices, "volume": 1},
                      index=index)
    cfg = replace(CFG, struct=replace(CFG.struct, int_left=1, int_right=1,
                                      invalidate_bars=0))
    frame = run(df, "TEST", "5m", cfg)
    events = [event for event in frame.events
              if event.layer == Layer.INTERNAL and event.direction == side]
    assert [event.label for event in events] == [Label.BOS_INIT, Label.BOS, Label.BOS]
    assert [event.swing_idx for event in events] == [1, 4, 11]
    assert [event.break_idx for event in events] == [3, 10, 13]
    assert [event.confirm_idx for event in events] == [3, 10, 13]

    figure = chart2.build(df, frame, "long", layers={"breaks"}, cfg=cfg,
                          frames={Layer.INTERNAL: df})
    lines = [line for line in figure.layout.shapes
             if line.line.color == chart2.LTF_LINE]
    assert {pd.Timestamp(line.x0) for line in lines} == {index[4], index[11]}
    wicks = df.high if side == "bull" else df.low
    assert {line.y0 for line in lines} == {wicks.iloc[4], wicks.iloc[11]}


def test_close_through_protected_higher_low_turns_bullish_layer_bearish():
    from tests.fixtures import price_series

    df = price_series([100, 97, 103, 100, 105, 102, 108, 105, 110, 104])
    cfg = replace(CFG, struct=replace(CFG.struct, int_left=1, int_right=1,
                                      invalidate_bars=0))
    f = run(df, "TEST", "5m", cfg)
    internal = [event for event in f.events if event.layer == Layer.INTERNAL]
    assert [event.label for event in internal] == [Label.BOS_INIT, Label.BOS, Label.BOS, Label.CHoCH]
    shift = internal[-1]
    assert shift.swing_idx == 7
    assert shift.confirm_idx == 9
    assert shift.direction == "bear"
    assert shift.level == df.low.iloc[7]
    assert df.close.iloc[9] > df.low.iloc[3]

    fig = chart2.build(df, f, "short", layers={"breaks"}, cfg=cfg,
                       frames={Layer.INTERNAL: df})
    lines = [s for s in fig.layout.shapes if s.line.color == chart2.LTF_LINE]
    assert len(lines) == 1
    assert lines[0].line.dash == "dash"
    assert lines[0].x0 == df.index[7].tz_localize('UTC')
    assert any(trace.x[0] == df.index[9].tz_localize('UTC')
               for trace in fig.data if trace.type == 'scatter')
    assert any("BEAR CHoCH" in a.text for a in fig.layout.annotations)


@pytest.mark.parametrize("side,stop,tp1,tp2", [
    ("long", 96, 103, 106),
    ("short", 104, 97, 94),
])
def test_trade_boxes_and_both_zone_sides_render(side, stop, tp1, tp2):
    import numpy as np
    from core.models import StructureFrame
    from tests.fixtures import price_series

    df = price_series(100 + 5 * np.sin(np.linspace(0, 20, 240)))
    fig = chart2.build(df, StructureFrame(), side,
                       plan={"entry": 100, "stop": stop, "tp1": tp1, "tp2": tp2},
                       htf_df=df, ltf_df=df, cfg=CFG, layers={"zones", "plan"})
    boxes = {s.name: s for s in fig.layout.shapes if s.type == "rect"}
    for tag in ("HTF 30m", "LTF 3m"):
        assert boxes[f"{tag} supply"].fillcolor == chart2.SUPPLY_FILL
        assert boxes[f"{tag} demand"].fillcolor == chart2.DEMAND_FILL
    risk, reward = boxes["Stop-loss risk"], boxes["Take-profit reward"]
    assert [risk.y0, risk.y1] == sorted((100, stop))
    assert [reward.y0, reward.y1] == sorted((100, tp2))
    assert risk.line.color == chart2.DN
    assert reward.line.color == chart2.UP
    plan = next(a for a in fig.layout.annotations if "TP1 " in a.text)
    assert plan.xref == plan.yref == "paper"
    assert plan.text.count("<br>") == 3
    assert "Entry 100" in plan.text and f"TP2 {tp2}" in plan.text
    assert risk.x0 == reward.x0 > df.index[-1].tz_localize("UTC")
    assert risk.x1 == reward.x1 < fig.layout.xaxis.range[1]
    assert all(s.x1 < risk.x0 for name, s in boxes.items()
               if name.endswith(("supply", "demand")))
    assert len([a for a in fig.layout.annotations if "Supply " in a.text]) == 2
    history = fig.layout.updatemenus[0].buttons[-1]
    assert history.label == "Fit history"


@pytest.mark.parametrize("sign,side", [(1, "bull"), (-1, "bear")])
def test_breakout_candle_can_be_the_next_confirmed_swing(sign, side):
    from tests.fixtures import price_series

    df = price_series([100 + sign * offset for offset in (0, 5, 2, 6, 4, 8)])
    cfg = replace(CFG, struct=replace(CFG.struct, int_left=1, int_right=1,
                                      invalidate_bars=0))
    events = [event for event in run(df, "TEST", "5m", cfg).events
              if event.layer == Layer.INTERNAL and event.direction == side]
    assert [(e.label, e.swing_idx, e.confirm_idx) for e in events] == [
        (Label.BOS_INIT, 1, 3), (Label.BOS, 3, 5),
    ]


def test_finer_chart_marks_source_close_and_matching_wick():
    from tests.fixtures import price_series

    source = price_series([100, 105, 102, 106, 104, 108])
    source.index = pd.date_range("2026-01-01", periods=len(source), freq="30min", tz="UTC")
    display = price_series([100] * 60)
    display.index = pd.date_range(source.index[0], periods=60, freq="3min", tz="UTC")
    fine_wick = source.index[3] + pd.Timedelta(minutes=12)
    display.loc[fine_wick, "high"] = source.high.iloc[3]
    cfg = replace(CFG, struct=replace(CFG.struct, ext_left=1, ext_right=1,
                                      invalidate_bars=0))
    frame = run(source, "TEST", "30m", cfg)
    fig = chart2.build(display, frame, "long", layers={"breaks"}, cfg=cfg,
                       frames={Layer.EXTERNAL: source}, chart_tf="3m")
    bos_line = next(s for s in fig.layout.shapes if s.line.color == chart2.HTF_LINE
                    and s.y0 == source.high.iloc[3])
    assert bos_line.x0 == fine_wick
    assert any(t.x[0] == source.index[5] + pd.Timedelta(minutes=30)
               for t in fig.data if t.type == "scatter" and "BULL BOS " in t.hovertemplate)
    assert any("HTF 30m bullish" in a.text for a in fig.layout.annotations)


def test_zone_poc_tag_uses_the_matching_timeframe(monkeypatch):
    from core.models import StructureFrame
    from tests.fixtures import price_series

    df = price_series([100] * 40)
    zone = {"side": "supply", "low": 105, "high": 107, "ts": df.index[5].tz_localize("UTC")}
    monkeypatch.setattr(chart2, "chart_zones", lambda *a, **k: [zone])
    monkeypatch.setattr(chart2, "_profile", lambda *a: {"poc": 106 if a[-1] == "htf" else 102})
    figure = chart2.build(df, StructureFrame(), "short", htf_df=df, ltf_df=df,
                          layers={"vp", "zones"})
    htf, ltf = (annotation.text for annotation in figure.layout.annotations
                if "Supply " in annotation.text)
    assert "contains POC" in htf and "POC 106" in htf
    assert "contains POC" not in ltf and "POC 102" in ltf
    assert len([s for s in figure.layout.shapes if s.name and s.name.endswith("supply")]) == 2
