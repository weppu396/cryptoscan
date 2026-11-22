"""chart.py - annotated decision chart.

Layout contract: one row. Candles own 0-82% of the width on a datetime axis.
The volume profile owns 84.5-100% on its own numeric axis, so it can never
distort candle time, candle width or price scale. Traces reference the gutter
as 'x2'; update_layout names it 'xaxis2'.

Structure is drawn as short horizontal markers at the broken price, not
full-height verticals. 30m majors are blue and labelled, 3m minors grey.

Read-only: nothing here changes the plan.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.candles import latest as latest_candle
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.structure_map import live_zones, marks
from core.tfs import CHART_TFS, chart_minutes, data_tf, resample
from core.volume import anchored_vwap, profile_histogram

UP, DN, MUT, WARN, TEAL = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6"
BG = "#0f1116"
ANCHOR_BARS = 60
DEFAULT_BARS = 110
MARK_BARS = 6

MAJOR_LINE = "rgba(96,152,255,0.60)"
MINOR_LINE = "rgba(140,148,160,0.32)"
MAJOR_TEXT = "rgba(96,152,255,0.90)"
ZONE_FILL = {"long": "rgba(20,184,166,0.085)", "short": "rgba(240,185,11,0.085)"}
ZONE_EDGE = {"long": "rgba(20,184,166,0.30)", "short": "rgba(240,185,11,0.30)"}
FVG_FILL = {"long": "rgba(22,199,132,0.055)", "short": "rgba(234,57,67,0.055)"}
VP_HTF = "rgba(96,152,255,0.20)"
VP_LTF = "rgba(20,184,166,0.14)"

PRICE_DOMAIN = [0.0, 0.82]
PROFILE_DOMAIN = [0.845, 1.0]


def _future(df, bars: int = 6):
    if len(df) < 2:
        return df.index[-1]
    return df.index[-1] + (df.index[-1] - df.index[-2]) * bars


def _step(df) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(0)
    return df.index[-1] - df.index[-2]


def _anchor(stt, long: bool, span: int = ANCHOR_BARS):
    col = "prev_swing_low" if long else "prev_swing_high"
    try:
        s = stt[col].tail(span).dropna()
        return int(float(s.iloc[-1])) if len(s) else None
    except Exception:
        return None


def _gutter(fig) -> tuple[str, str]:
    """(trace reference, layout property) for a free numeric x axis."""
    used = {k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")}
    i = 2
    while f"xaxis{i}" in used:
        i += 1
    prop = f"xaxis{i}"
    return prop.replace("axis", ""), prop


def _draw_profile(fig, df, bars: int, color: str, axis: str) -> None:
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(
        y=vp["centers"], x=vp["hist"], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=step * 0.9 if step else None,
        hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.data[-1].xaxis = axis

    for lvl, tone, tag, dash in ((vp["poc"], WARN, "POC", "solid"),
                                 (vp["vah"], MUT, "VAH", "dot"),
                                 (vp["val"], MUT, "VAL", "dot")):
        fig.add_shape(type="line", xref=axis, yref="y", x0=0, x1=1,
                      y0=lvl, y1=lvl, line=dict(color=tone, width=1, dash=dash))
        fig.add_annotation(xref="paper", yref="y", x=PRICE_DOMAIN[1], y=lvl,
                           text=tag, showarrow=False, xanchor="right",
                           font=dict(size=7, color=tone), opacity=0.8)

    for lvl in vp["hvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=0, x1=0.45,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.32)", width=1))
    for lvl in vp["lvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=0, x1=0.22,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.18)", width=1))


def _mark(fig, ts, level, step, color, width, label=None, yshift=0):
    fig.add_shape(type="line", row=1, col=1, x0=ts, x1=ts + step * MARK_BARS,
                  y0=level, y1=level, line=dict(color=color, width=width))
    if label:
        fig.add_annotation(x=ts, y=level, row=1, col=1, text=label,
                           showarrow=False, xanchor="left", yshift=yshift,
                           font=dict(size=8, color=color))


def build(df, cfg, direction: str, plan: dict | None = None,
          bars: int = DEFAULT_BARS, chart_tf: str | None = None,
          analysis_df=None, htf_df=None) -> go.Figure:
    adf = analysis_df if analysis_df is not None else df
    view = df.tail(bars)
    step = _step(view)
    ind = compute_all(adf, cfg)
    stt = build_structure(adf, cfg.structure)
    a = ind["atr"].iloc[-1]
    atr = float(a) if a == a else 0.0
    long = direction == "long"
    want = "bull" if long else "bear"
    right = _future(view, 6)
    last_x = view.index[-1]

    if htf_df is None:
        try:
            htf_df = resample(adf, "30m") if len(adf) > 60 else None
        except Exception:
            htf_df = None

    fig = make_subplots(rows=1, cols=1)
    gutter_ref, gutter_prop = _gutter(fig)

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name="price",
        increasing_line_color=UP, increasing_line_width=1.2,
        increasing_fillcolor=UP,
        decreasing_line_color=DN, decreasing_line_width=1.2,
        decreasing_fillcolor=DN, whiskerwidth=0.6,
        showlegend=False), row=1, col=1)

    try:
        sw = find_swings(adf, cfg.structure.swing_left, cfg.structure.swing_right)
        start = view.index[0]
        for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
            pts = [s for s in sw if s["type"] == kind and adf.index[s["idx"]] >= start]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[adf.index[s["idx"]] for s in pts],
                    y=[s["price"] for s in pts], mode="markers",
                    marker=dict(symbol=sym, size=6, color=MUT, opacity=.5),
                    name="swing", showlegend=False), row=1, col=1)
    except Exception:
        pass

    try:
        majors, minors = marks(adf, cfg, htf_df, atr * 0.5)
        for m in minors:
            _mark(fig, m["ts"], m["level"], step, MINOR_LINE, 1)
        for m in majors:
            side = "Bullish" if m["dir"] == "bull" else "Bearish"
            _mark(fig, m["ts"], m["level"], step, MAJOR_LINE, 1.6,
                  f"{side} {m['kind']}", 9 if m["dir"] == "bull" else -9)
    except Exception:
        pass

    try:
        for z in live_zones(smc.supply_demand(adf), adf, direction, keep=2):
            x0 = max(adf.index[z["idx"]], view.index[0])
            fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=right,
                          y0=z["bot"], y1=z["top"],
                          fillcolor=ZONE_FILL[direction],
                          line=dict(width=1, color=ZONE_EDGE[direction]))
    except Exception:
        pass

    try:
        for g in [g for g in smc.fair_value_gaps(adf) if g["dir"] == want][:3]:
            x0 = adf.index[g["idx"]]
            if x0 < view.index[0]:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=right,
                          y0=g["bot"], y1=g["top"],
                          fillcolor=FVG_FILL[direction], line=dict(width=0))
    except Exception:
        pass

    try:
        for s in [s for s in smc.liquidity_sweeps(adf) if s["dir"] == want][:3]:
            if adf.index[s["idx"]] < view.index[0]:
                continue
            fig.add_trace(go.Scatter(
                x=[adf.index[s["idx"]]], y=[s["level"]], mode="markers",
                marker=dict(symbol="x", size=9, color=WARN, opacity=.75),
                name="sweep", showlegend=False), row=1, col=1)
    except Exception:
        pass

    try:
        anchor = _anchor(stt, long)
        if anchor is not None:
            line = anchored_vwap(adf, anchor).tail(bars)
            fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                     line=dict(color=TEAL, width=1.2, dash="dash"),
                                     opacity=.75), row=1, col=1)
    except Exception:
        pass

    try:
        if htf_df is not None and len(htf_df) >= 30:
            _draw_profile(fig, htf_df, bars * 2, VP_HTF, gutter_ref)
    except Exception:
        pass
    try:
        _draw_profile(fig, adf, bars, VP_LTF, gutter_ref)
    except Exception:
        pass

    if plan:
        entry, sl, tp1, tp2 = plan["entry"], plan["sl"], plan["tp1"], plan["tp2"]
        for y, col, txt in ((entry, "#e7eaf0", "ENTRY"), (sl, DN, "STOP"),
                            (tp1, UP, "TP1 50%"), (tp2, UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=last_x, x1=right, y0=y, y1=y,
                          line=dict(color=col, width=1.1))
            fig.add_annotation(x=right, y=y, row=1, col=1, xanchor="left",
                               text=f" {txt} {y:,.6g}", showarrow=False,
                               font=dict(size=9, color=col))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(entry, sl), y1=max(entry, sl),
                      fillcolor="rgba(234,57,67,0.13)", line=dict(width=0))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(entry, tp1), y1=max(entry, tp1),
                      fillcolor="rgba(22,199,132,0.11)", line=dict(width=0))

    try:
        ci, cname, cq = latest_candle(adf, atr, direction)
        if ci is not None and cq >= 60 and adf.index[ci] >= view.index[0]:
            fig.add_annotation(x=adf.index[ci],
                               y=float(adf["low"].iat[ci]) if long
                               else float(adf["high"].iat[ci]),
                               row=1, col=1, text=f" {cname} {cq} ", showarrow=True,
                               arrowhead=2, yshift=-20 if long else 20,
                               font=dict(size=9, color="#0f1116"),
                               bgcolor=UP if long else DN, borderpad=3)
    except Exception:
        pass

    try:
        lo = float(view["low"].min())
        hi = float(view["high"].max())
        pad = (hi - lo) * 0.03 or max(abs(hi), 1.0) * 0.001
        y_range = [lo - pad, hi + pad]
    except Exception:
        y_range = None

    fig.update_layout(
        height=620, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=62, r=10, t=34, b=22), showlegend=False,
        hovermode="x unified", dragmode="pan",
        title=dict(text=f"chart {chart_tf or ''} · structure 30m/3m · "
                        f"profile = OHLCV estimate",
                   x=0, xanchor="left", font=dict(size=10, color=MUT)),
        xaxis=dict(domain=PRICE_DOMAIN, type="date", showgrid=False,
                   rangeslider=dict(visible=False)))
    fig.update_layout(**{gutter_prop: dict(domain=PROFILE_DOMAIN, anchor="y",
                                          range=[0, 1], type="linear",
                                          showgrid=False, showticklabels=False,
                                          zeroline=False, fixedrange=True)})
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range))
    fig.update_xaxes(showgrid=False, gridcolor="#1b1f28")
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="left",
                     tickformat=",.6g")
    return fig


def st_selectbox(key: str, idx: int) -> str:
    import streamlit as st
    return st.selectbox("Chart timeframe", CHART_TFS, index=idx, key=key,
                        help="Display only. Structure and the plan stay on "
                             "30m context / 3m trigger.")


def legend_md() -> str:
    return """
**How to read this chart**

| mark | meaning |
|---|---|
| blue horizontal marker, `Bullish BOS` / `Bearish CHOCH` | **30m major** structural break, at the broken price |
| short grey horizontal marker, no text | **3m minor** structural break |
| grey triangles | confirmed swing high / low |
| teal box | demand zone. Amber box: supply zone |
| faint box, no border | unfilled fair value gap |
| amber ✕ | liquidity sweep — stops taken, closed back inside |
| teal dashed line | anchored VWAP from the origin of the leg |
| right-hand gutter | volume profile — **OHLCV estimate**, 30m blue / 3m teal |
| amber line in gutter | POC — most-traded price |
| grey dotted in gutter | VAH / VAL — 70% of volume sits between them |
| short grey ticks | HVN (longer) and LVN (shorter) |
| green / red boxes | reward to TP1/TP2 and risk to stop |

Markers are 6 bars long so they never cut the chart in half. Only the newest
two 30m and three 3m marks survive, and anything within half an ATR of a newer
one is dropped. Only closed bars produce a mark.

The gutter is an estimate: candles carry no trade direction, so volume is
spread evenly across each bar's range. Total volume is conserved; the split
inside a bar is not exact.
"""
