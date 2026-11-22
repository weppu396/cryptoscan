"""chart.py - annotated decision chart.

Layout contract: one row. Candles own 0-76% of the width on a datetime axis.
The volume profile owns 78.5-100% in two halves: 30m HTF blue on the right
half, 3m LTF orange on the left half. They never overlap and never touch the
candle axis. Traces reference the gutter as 'x2'; update_layout names it
'xaxis2'.

Every structure mark, swing line and active zone runs to the right edge.
Both axes are free for independent zoom; only the initial view is pinned.

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

MAJOR_LINE = "rgba(96,152,255,0.55)"
MINOR_LINE = "rgba(140,148,160,0.28)"
MAJOR_TEXT = "rgba(96,152,255,0.90)"
SWING_LINE = "rgba(140,148,160,0.18)"
ZONE_FILL = {"long": "rgba(20,184,166,0.085)", "short": "rgba(240,185,11,0.085)"}
ZONE_EDGE = {"long": "rgba(20,184,166,0.30)", "short": "rgba(240,185,11,0.30)"}
FVG_FILL = {"long": "rgba(22,199,132,0.055)", "short": "rgba(234,57,67,0.055)"}
VP_HTF = "rgba(96,152,255,0.22)"
VP_LTF = "rgba(240,140,11,0.20)"

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
    """Median positive bar gap. Robust to holes."""
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
    """One profile in its own half of the gutter. No overlap between sides."""
    b0, b1 = HALF[side]
    span = b1 - b0
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(
        y=vp["centers"], x=vp["hist"] * span, base=b0, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=step * 0.9 if step else None,
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
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.45,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.34)", width=1))
    for lvl in vp["lvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.22,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.20)", width=1))


def _mark(fig, x0, x1, level, color, width, label=None, yshift=0):
    """Horizontal structure level running to the right edge."""
    fig.add_shape(type="line", row=1, col=1, x0=x0, x1=x1,
                  y0=level, y1=level, line=dict(color=color, width=width))
    if label:
        fig.add_annotation(x=x0, y=level, row=1, col=1, text=label,
                           showarrow=False, xanchor="left", yshift=yshift,
                           font=dict(size=8, color=color))


def build(df, cfg, direction: str, plan: dict | None = None,
          bars: int = DEFAULT_BARS, chart_tf: str | None = None,
          analysis_df=None, htf_df=None) -> go.Figure:
    df = _clean(df)
    adf = _clean(analysis_df) if analysis_df is not None else df
    htf_df = _clean(htf_df) if htf_df is not None else None
    if df is None or not len(df):
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG,
                          plot_bgcolor=BG, height=620,
                          annotations=[dict(text="no candles", showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5)])
        return fig

    view = df.tail(bars)
    step = _step(view)
    ind = compute_all(adf, cfg)
    stt = build_structure(adf, cfg.structure)
    a = ind["atr"].iloc[-1]
    atr = float(a) if a == a else 0.0
    long = direction == "long"
    want = "bull" if long else "bear"
    last_x = view.index[-1]
    right = _future(last_x, step, 6)

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
            for s in pts:
                fig.add_shape(type="line", row=1, col=1, x0=adf.index[s["idx"]],
                              x1=right, y0=s["price"], y1=s["price"],
                              line=dict(color=SWING_LINE, width=1))
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
            _mark(fig, m["ts"], right, m["level"], MINOR_LINE, 1)
        for m in majors:
            side = "Bullish" if m["dir"] == "bull" else "Bearish"
            _mark(fig, m["ts"], right, m["level"], MAJOR_LINE, 1.6,
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
            _draw_profile(fig, htf_df, bars * 2, VP_HTF, gutter_ref, "htf")
    except Exception:
        pass
    try:
        _draw_profile(fig, adf, bars, VP_LTF, gutter_ref, "ltf")
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

    x_lo = view.index[0]
    if x_lo >= right:
        right = x_lo + (step or pd.Timedelta(minutes=1)) * 6

    fig.update_layout(
        height=640, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=62, r=10, t=40, b=22), showlegend=False,
        hovermode="x unified", dragmode="pan",
        title=dict(text=f"chart {chart_tf or ''} · structure 30m/3m · "
                        f"gutter: blue = 30m HTF profile, orange = 3m LTF "
                        f"profile (OHLCV estimate)",
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
                        help="Display only. Structure and the plan stay on "
                             "30m context / 3m trigger.")


def legend_md() -> str:
    return """
**How to read this chart**

| mark | meaning |
|---|---|
| blue horizontal level, `Bullish BOS` / `Bearish CHOCH` | **30m major** structural break, run to the right edge |
| grey horizontal level, no text | **3m minor** structural break, run to the right edge |
| faint grey horizontal line | swing high / low projection |
| grey triangles | confirmed swing high / low |
| teal box | active demand zone. Amber box: active supply zone |
| faint box, no border | unfilled fair value gap |
| amber ✕ | liquidity sweep — stops taken, closed back inside |
| teal dashed line | anchored VWAP from the origin of the leg |
| right gutter, blue bars | **30m HTF** volume profile, right half |
| right gutter, orange bars | **3m LTF** volume profile, left half |
| amber line in gutter | POC |
| grey dotted in gutter | VAH / VAL |
| short grey ticks | HVN (longer) and LVN (shorter) |
| green / red boxes | reward to TP1/TP2 and risk to stop |

Structure, swings and zones all terminate at the right edge so the levels in
force now are the ones you can see. Only the newest two 30m and three 3m marks
survive, and anything within half an ATR of a newer one is dropped. Only closed
bars produce a mark.

Scroll to zoom. Drag to pan. The X and Y axes scale independently; double-click
the modebar to reset. Bar spacing uses the median gap, so a cache hole cannot
stretch the axis.

The gutter is an estimate: candles carry no trade direction, so volume is
spread evenly across each bar's range.
"""
