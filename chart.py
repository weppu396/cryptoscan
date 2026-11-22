"""chart.py - annotated decision chart.

Candles 0-76% on a datetime axis. Profile gutter 78.5-100% in two halves,
HTF left-in-gutter blue, LTF right-in-gutter orange, own numeric axis.

History: the trace carries the WHOLE loaded frame (capped at MAX_TRACE) and only
the initial view is DEFAULT_BARS. Zooming out therefore reveals real earlier
candles instead of empty space.

Structure: exactly two pivots per timeframe, four total. Up = HH + HL, down =
LH + LL, range = H + L. Each line starts on the pivot wick, runs to the right
edge, carries the exact price and a BOS/CHoCH chip. HTF vibrant yellow, LTF
bright cyan.

Zones: only unmitigated zones with above-average volume behind the touch.
Dusty rose supply, mint demand, both low opacity, labelled with timeframe and
multiple.

Drag to zoom: horizontal compresses/expands time, vertical compresses/expands
price. Right-drag pans, double-click resets. Every mark is in data space so it
stays locked while zooming.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.candles import latest as latest_candle
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.structure_map import structure_lines, volume_zones
from core.tfs import CHART_TFS, cap_htf, chart_minutes, data_tf, resample
from core.volume import anchored_vwap, profile_histogram

UP, DN, MUT, WARN, TEAL = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6"
HTF_LINE = "#f5d90a"
LTF_LINE = "#22d3ee"
SUPPLY_FILL = "rgba(190,110,120,0.14)"
SUPPLY_EDGE = "rgba(190,110,120,0.42)"
DEMAND_FILL = "rgba(120,200,160,0.13)"
DEMAND_EDGE = "rgba(120,200,160,0.42)"
BG = "#0f1116"
ANCHOR_BARS = 60
DEFAULT_BARS = 110
MAX_TRACE = 1200

ZONE_FILL = {"long": "rgba(20,184,166,0.075)", "short": "rgba(240,185,11,0.075)"}
FVG_FILL = {"long": "rgba(22,199,132,0.05)", "short": "rgba(234,57,67,0.05)"}
VP_HTF = "rgba(96,152,255,0.30)"
VP_LTF = "rgba(240,140,11,0.28)"

PRICE_DOMAIN = [0.0, 0.76]
PROFILE_DOMAIN = [0.785, 1.0]
HALF = {"htf": (0.52, 1.0), "ltf": (0.0, 0.48)}
OHLC = ("open", "high", "low", "close")


def _clean(df):
    if df is None or not len(df):
        return df
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    elif d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    for c in OHLC:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    keep = [c for c in OHLC if c in d.columns]
    d = d.dropna(subset=keep)
    if keep:
        d = d[np.isfinite(d[keep].to_numpy(float)).all(axis=1)]
    return d


def _step(df) -> pd.Timedelta:
    if len(df) < 3:
        return pd.Timedelta(0)
    d = pd.Series(df.index).diff().dropna()
    d = d[d > pd.Timedelta(0)]
    return d.median() if len(d) else pd.Timedelta(0)


def _future(last, step: pd.Timedelta, bars: int = 6):
    if step <= pd.Timedelta(0):
        return last + pd.Timedelta(minutes=1)
    return last + step * bars


def _anchor(stt, long: bool, span: int = ANCHOR_BARS):
    col = "prev_swing_low" if long else "prev_swing_high"
    try:
        s = stt[col].tail(span).dropna()
        return int(float(s.iloc[-1])) if len(s) else None
    except Exception:
        return None


def _tick(tf: str | None) -> str:
    m = chart_minutes(tf) if tf else 15
    if m < 60:
        return "%H:%M"
    if m < 1440:
        return "%d %b %H:%M"
    return "%d %b"


def _gutter(fig) -> tuple[str, str]:
    used = {k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")}
    i = 2
    while f"xaxis{i}" in used:
        i += 1
    prop = f"xaxis{i}"
    return prop.replace("axis", ""), prop


def _draw_profile(fig, df, bars: int, color: str, axis: str, side: str) -> None:
    """Thick horizontal bars so the profile reads as one block against zones."""
    b0, b1 = HALF[side]
    span = b1 - b0
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(
        y=vp["centers"], x=vp["hist"] * span, base=b0, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=step * 3.0 if step else None,
        hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.data[-1].xaxis = axis

    for lvl, tone, tag, dash in ((vp["poc"], WARN, "POC", "solid"),
                                 (vp["vah"], MUT, "VAH", "dot"),
                                 (vp["val"], MUT, "VAL", "dot")):
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b1,
                      y0=lvl, y1=lvl, line=dict(color=tone, width=1, dash=dash))
        if side == "htf":
            fig.add_annotation(xref="paper", yref="y", x=PROFILE_DOMAIN[1], y=lvl,
                               text=tag, showarrow=False, xanchor="right",
                               font=dict(size=7, color=tone), opacity=0.85)

    for lvl in vp["hvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.5,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(170,170,180,0.40)", width=1))
    for lvl in vp["lvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.25,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(170,170,180,0.24)", width=1))


def _struct(fig, line, x1, color, width):
    """Pivot line from the wick to the right edge, price + tag chip."""
    fig.add_shape(type="line", row=1, col=1, x0=line["ts"], x1=x1,
                  y0=line["price"], y1=line["price"],
                  line=dict(color=color, width=width))
    txt = f"{line['label']} {line['price']:,.6g}"
    if line["tag"]:
        txt += f" {line['tag']}"
    fig.add_annotation(x=line["ts"], y=line["price"], row=1, col=1, text=txt,
                       showarrow=False, xanchor="left", yshift=9,
                       font=dict(size=8, color=color))


def _zone(fig, z, x0, x1):
    fill = SUPPLY_FILL if z["side"] == "supply" else DEMAND_FILL
    edge = SUPPLY_EDGE if z["side"] == "supply" else DEMAND_EDGE
    fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=x1,
                  y0=z["low"], y1=z["high"],
                  fillcolor=fill, line=dict(width=1, color=edge))
    fig.add_annotation(x=x1, y=(z["low"] + z["high"]) / 2, row=1, col=1,
                       text=f" {z['grade']} {z['tf']} {z['side']} "
                            f"{z['ratio']:.1f}x  {z['low']:,.6g}–{z['high']:,.6g} ",
                       showarrow=False, xanchor="right",
                       font=dict(size=7, color=edge), bgcolor="rgba(15,17,22,0.65)")


def build(df, cfg, direction: str, plan: dict | None = None,
          bars: int = DEFAULT_BARS, chart_tf: str | None = None,
          analysis_df=None, htf_df=None, ltf_df=None,
          pair: tuple | None = None) -> go.Figure:
    df = _clean(df)
    adf = _clean(analysis_df) if analysis_df is not None else df
    htf_df = _clean(htf_df) if htf_df is not None else None
    ltf_df = _clean(ltf_df) if ltf_df is not None else None
    if df is None or not len(df):
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG,
                          plot_bgcolor=BG, height=620,
                          annotations=[dict(text="no candles", showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5)])
        return fig

    htf_label, ltf_label = (pair if pair else ("30m", "3m"))
    htf_label = cap_htf(htf_label)
    if htf_df is None:
        try:
            htf_df = resample(adf, htf_label) if len(adf) > 60 else None
        except Exception:
            htf_df = None
    if ltf_df is None:
        ltf_df = adf

    # full history in the trace, short initial window
    hist = df.tail(MAX_TRACE)
    view = hist.tail(bars)
    step = _step(hist)
    ind = compute_all(adf, cfg)
    stt = build_structure(adf, cfg.structure)
    a = ind["atr"].iloc[-1]
    atr = float(a) if a == a else 0.0
    long = direction == "long"
    want = "bull" if long else "bear"
    last_x = hist.index[-1]
    right = _future(last_x, step, 6)
    start = hist.index[0]

    fig = make_subplots(rows=1, cols=1)
    gutter_ref, gutter_prop = _gutter(fig)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"],
        close=hist["close"], name="price",
        increasing_line_color=UP, increasing_line_width=1.2,
        increasing_fillcolor=UP,
        decreasing_line_color=DN, decreasing_line_width=1.2,
        decreasing_fillcolor=DN, whiskerwidth=0.6,
        showlegend=False), row=1, col=1)

    try:
        sw = find_swings(adf, cfg.structure.swing_left, cfg.structure.swing_right)
        for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
            pts = [s for s in sw if s["type"] == kind and adf.index[s["idx"]] >= start]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[adf.index[s["idx"]] for s in pts],
                    y=[s["price"] for s in pts], mode="markers",
                    marker=dict(symbol=sym, size=5, color=MUT, opacity=.4),
                    name="swing", showlegend=False), row=1, col=1)
    except Exception:
        pass

    # exactly two pivots per timeframe, four total
    try:
        for line in structure_lines(htf_df, cfg, htf_label, "HTF"):
            _struct(fig, line, right, HTF_LINE, 1.8)
    except Exception:
        pass
    try:
        for line in structure_lines(ltf_df, cfg, ltf_label, "LTF"):
            _struct(fig, line, right, LTF_LINE, 1.4)
    except Exception:
        pass

    # volume-backed zones only, one per side
    try:
        for z in volume_zones(htf_df, cfg, htf_label, "HTF", direction, keep=1):
            _zone(fig, z, max(z["ts"], start), right)
    except Exception:
        pass
    try:
        for z in volume_zones(ltf_df, cfg, ltf_label, "LTF", direction, keep=1):
            _zone(fig, z, max(z["ts"], start), right)
    except Exception:
        pass

    try:
        for g in [g for g in smc.fair_value_gaps(adf) if g["dir"] == want][:2]:
            x0 = adf.index[g["idx"]]
            if x0 < start:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=right,
                          y0=g["bot"], y1=g["top"],
                          fillcolor=FVG_FILL[direction], line=dict(width=0))
    except Exception:
        pass

    try:
        for s in [s for s in smc.liquidity_sweeps(adf) if s["dir"] == want][:3]:
            if adf.index[s["idx"]] < start:
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
            line = anchored_vwap(adf, anchor).tail(len(hist))
            fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                     line=dict(color=TEAL, width=1.2, dash="dash"),
                                     opacity=.75), row=1, col=1)
    except Exception:
        pass

    try:
        if htf_df is not None and len(htf_df) >= 30:
            _draw_profile(fig, htf_df, bars * 2, VP_HTF, gutter_ref, "htf")
    except Exception:
        pass
    try:
        _draw_profile(fig, ltf_df, bars, VP_LTF, gutter_ref, "ltf")
    except Exception:
        pass

    if plan:
        entry, sl, tp1, tp2 = plan["entry"], plan["sl"], plan["tp1"], plan["tp2"]
        for y, col, txt in ((entry, "#e7eaf0", "ENTRY"), (sl, DN, "STOP"),
                            (tp1, UP, "TP1 50%"), (tp2, UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=last_x, x1=right, y0=y, y1=y,
                          line=dict(color=col, width=1.4))
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
        if ci is not None and cq >= 60 and adf.index[ci] >= start:
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
        y_ref = hist.tail(min(len(hist), max(bars * 4, bars)))
        lo = float(y_ref["low"].min())
        hi = float(y_ref["high"].max())
        pad = (hi - lo) * 0.03 or max(abs(hi), 1.0) * 0.001
        y_range = [lo - pad, hi + pad]
    except Exception:
        y_range = None

    x_lo = view.index[0]
    if x_lo >= right:
        right = x_lo + (step or pd.Timedelta(minutes=1)) * 6

    fig.update_layout(
        height=640, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=62, r=10, t=42, b=22), showlegend=False,
        hovermode="x unified", dragmode="zoom", uirevision="cryptoscan",
        title=dict(text=f"chart {chart_tf or ''} · structure {htf_label} (HTF, "
                        f"yellow) / {ltf_label} (LTF, cyan) · gutter blue = HTF "
                        f"profile, orange = LTF profile (estimate)",
                   x=0, xanchor="left", font=dict(size=10, color=MUT)),
        xaxis=dict(domain=PRICE_DOMAIN, type="date", showgrid=False,
                   range=[x_lo, right], nticks=8, tickformat=_tick(chart_tf),
                   rangeslider=dict(visible=False), fixedrange=False,
                   constrain="range"))
    fig.update_layout(**{gutter_prop: dict(domain=PROFILE_DOMAIN, anchor="y",
                                          range=[0, 1], type="linear",
                                          showgrid=False, showticklabels=False,
                                          zeroline=False, fixedrange=True)})
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range, fixedrange=False))
    fig.update_xaxes(showgrid=False, gridcolor="#1b1f28")
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="left",
                     tickformat=",.6g", fixedrange=False)
    return fig


def st_selectbox(key: str, idx: int) -> str:
    import streamlit as st
    return st.selectbox("Chart timeframe", CHART_TFS, index=idx, key=key,
                        help="Display only. Structure and the plan use the "
                             "auto-selected pair, HTF never above 1h.")


def legend_md() -> str:
    return """
**How to read this chart**

| mark | meaning |
|---|---|
| yellow level + `HH 0.00362 BOS` | **HTF** pivot. HH/HL in an uptrend, LH/LL in a downtrend |
| cyan level + `LL 0.00340 CHoCH` | **LTF** pivot, same labels |
| dusty rose box | **supply** zone, labelled `HTF 1h supply 2.4x 0.00351–0.00356` |
| mint box | **demand** zone, same label format |
| faint box, no border | unfilled fair value gap |
| amber ✕ | liquidity sweep |
| teal dashed line | anchored VWAP |
| right gutter, blue | HTF volume profile (left half of the gutter) |
| right gutter, orange | LTF volume profile (right half) |
| amber line in gutter | POC |
| grey dotted in gutter | VAH / VAL |
| green / red boxes | reward to TP1/TP2 and risk to stop |

The HTF is picked from 15m, 30m or 1h — never above 1h — by how cleanly the
market is structured. The LTF is then picked from the legal triggers for that
HTF by running the full assessment on each and keeping the best setup. Both are
printed in the chart title.

Exactly two pivots per timeframe, four total. The `x` in a zone label is the
touch-bar volume against its 20-bar average; anything under 1.3x is dropped, so
only zones with real participation appear.

**Zoom:** drag horizontally to compress or expand time, drag vertically to
compress or expand price, right-drag to pan, double-click to reset. The whole
loaded history is in the chart, so zooming out reveals earlier candles rather
than empty space.

Only closed bars produce a pivot. The gutter is an OHLCV estimate.
"""
