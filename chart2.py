"""chart2.py - authoritative renderer. Consumes a StructureFrame, never recomputes it.

Same contract as chart.py: candles 0-76%, gutter 78.5-100% in two halves,
bounded history with a short initial window, all marks to the right edge, both
axes free for independent zoom.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.models import Layer
from core.structure_map import thin, volume_zones
from core.tfs import cap_htf, chart_minutes, resample
from core.volume import anchored_vwap, profile_histogram

UP, DN, MUT, WARN, TEAL, BLUE = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6", "#3b82f6"
HTF_LINE = "#f5d90a"
LTF_LINE = "#22d3ee"
SUPPLY_FILL = "rgba(190,110,120,0.14)"
SUPPLY_EDGE = "rgba(190,110,120,0.42)"
DEMAND_FILL = "rgba(120,200,160,0.13)"
DEMAND_EDGE = "rgba(120,200,160,0.42)"
BG = "#0f1116"
MODES = ("structure", "analysis", "execution")
LAYERS = {
    "structure": {"candles", "ext_swing", "int_swing", "breaks", "tags"},
    "analysis": {"candles", "ext_swing", "breaks", "zones", "fvg", "sweeps",
                 "vp", "avwap"},
    "execution": {"candles", "zone", "plan"},
}
STYLE = {Layer.EXTERNAL: (1.8, HTF_LINE), Layer.INTERNAL: (1.4, LTF_LINE)}
VP_HTF = "rgba(96,152,255,0.30)"
VP_LTF = "rgba(240,140,11,0.28)"
PRICE_DOMAIN = [0.0, 0.76]
PROFILE_DOMAIN = [0.785, 1.0]
HALF = {"htf": (0.52, 1.0), "ltf": (0.0, 0.48)}
MAX_MARKS = {Layer.EXTERNAL: 2, Layer.INTERNAL: 2}
DEFAULT_BARS = 110
MAX_TRACE = 1200
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


def _step(df):
    if len(df) < 3:
        return pd.Timedelta(0)
    d = pd.Series(df.index).diff().dropna()
    d = d[d > pd.Timedelta(0)]
    return d.median() if len(d) else pd.Timedelta(0)


def _future(last, step, bars=6):
    if step is None or step <= pd.Timedelta(0):
        return last + pd.Timedelta(minutes=1)
    return last + step * bars


def _tick(tf):
    m = chart_minutes(tf) if tf else 15
    if m < 60:
        return "%H:%M"
    if m < 1440:
        return "%d %b %H:%M"
    return "%d %b"


def _gutter(fig):
    used = {k for k in fig.layout.to_plotly_json() if k.startswith("xaxis")}
    i = 2
    while f"xaxis{i}" in used:
        i += 1
    prop = f"xaxis{i}"
    return prop.replace("axis", ""), prop


def _profile(fig, df, bars, color, axis, side):
    b0, b1 = HALF[side]
    span = b1 - b0
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(y=vp["centers"], x=vp["hist"] * span, base=b0,
                         orientation="h",
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


def _struct(fig, ts, price, x1, color, width, label, tag):
    fig.add_shape(type="line", row=1, col=1, x0=ts, x1=x1,
                  y0=price, y1=price, line=dict(color=color, width=width))
    txt = f"{label} {price:,.6g}" + (f" {tag}" if tag else "")
    fig.add_annotation(x=ts, y=price, row=1, col=1, text=txt, showarrow=False,
                       xanchor="left", yshift=9, font=dict(size=8, color=color))


def _zone(fig, z, x0, x1, label):
    fill = SUPPLY_FILL if z["side"] == "supply" else DEMAND_FILL
    edge = SUPPLY_EDGE if z["side"] == "supply" else DEMAND_EDGE
    fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=x1,
                  y0=z["low"], y1=z["high"],
                  fillcolor=fill, line=dict(width=1, color=edge))
    fig.add_annotation(x=x1, y=(z["low"] + z["high"]) / 2, row=1, col=1,
                       text=f" {label} {z['side']} {z['ratio']:.1f}x "
                            f"{z['low']:,.6g}–{z['high']:,.6g} ",
                       showarrow=False, xanchor="right",
                       font=dict(size=7, color=edge),
                       bgcolor="rgba(15,17,22,0.65)")


def build(df, frame, direction, plan=None, mode="analysis", layers=None,
          bars=DEFAULT_BARS, chart_tf=None, analysis_df=None,
          htf_df=None, ltf_df=None, pair=None) -> go.Figure:
    if mode not in MODES:
        raise ValueError(mode)
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

    on = set(layers) if layers else LAYERS[mode]
    hist = df.tail(min(MAX_TRACE, max(bars * 8, bars)))
    view = hist.tail(bars)
    step = _step(hist)
    t = len(adf) - 1
    long = direction == "long"
    last_x = hist.index[-1]
    right = _future(last_x, step, 6)
    start = hist.index[0]

    fig = make_subplots(rows=1, cols=1)
    gutter_ref, gutter_prop = _gutter(fig)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"],
        close=hist["close"],
        increasing_line_color=UP, increasing_line_width=1.2, increasing_fillcolor=UP,
        decreasing_line_color=DN, decreasing_line_width=1.2, decreasing_fillcolor=DN,
        whiskerwidth=0.6, showlegend=False), row=1, col=1)

    for layer in (Layer.EXTERNAL, Layer.INTERNAL):
        key = "ext_swing" if layer == Layer.EXTERNAL else "int_swing"
        if key in on:
            pts = [s for s in frame.swings if s.layer == layer
                   and adf.index[s.idx] >= start]
            for s in pts:
                fig.add_trace(go.Scatter(
                    x=[adf.index[s.idx]], y=[s.price], mode="markers",
                    marker=dict(symbol="triangle-down" if s.kind == "H"
                                else "triangle-up", size=5, color=MUT,
                                opacity=.4), showlegend=False), row=1, col=1)

    if "breaks" in on:
        for layer in (Layer.EXTERNAL, Layer.INTERNAL):
            live = [e for e in frame.live(t) if e.layer == layer
                    and adf.index[e.break_idx] >= start
                    and e.confirm_idx <= len(adf) - 2]
            live = thin([{"ts": adf.index[e.break_idx], "level": e.level, "e": e}
                         for e in live], 0.0)
            w, col = STYLE[layer]
            for row in live[-MAX_MARKS[layer]:]:
                e = row["e"]
                c = WARN if (e.label.value == "CHoCH"
                             and layer == Layer.EXTERNAL) else col
                _struct(fig, adf.index[e.break_idx], e.level, right, c, w,
                        f"{e.direction.upper()}", e.label.value)

    if "zones" in on:
        try:
            from config import CFG as _CFG
            for z in volume_zones(htf_df, _CFG, htf_label, "HTF", direction, keep=1):
                _zone(fig, z, max(z["ts"], start), right, "HTF " + htf_label)
            for z in volume_zones(ltf_df, _CFG, ltf_label, "LTF", direction, keep=1):
                _zone(fig, z, max(z["ts"], start), right, "LTF " + ltf_label)
        except Exception:
            pass

    if "fvg" in on:
        for g in [g for g in smc.fair_value_gaps(adf)
                  if g["dir"] == ("bull" if long else "bear")][:2]:
            if adf.index[g["idx"]] < start:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=adf.index[g["idx"]], x1=right,
                          y0=g["bot"], y1=g["top"], line=dict(width=0),
                          fillcolor="rgba(22,199,132,0.05)" if long
                          else "rgba(234,57,67,0.05)")

    if "sweeps" in on:
        for s in [s for s in smc.liquidity_sweeps(adf)
                  if s["dir"] == ("bull" if long else "bear")][:3]:
            if adf.index[s["idx"]] < start:
                continue
            fig.add_trace(go.Scatter(x=[adf.index[s["idx"]]], y=[s.level],
                                     mode="markers",
                                     marker=dict(symbol="x", size=9, color=WARN),
                                     showlegend=False), row=1, col=1)

    if "avwap" in on:
        try:
            pool = [s for s in frame.swings if s.layer == Layer.EXTERNAL
                    and s.kind == ("L" if long else "H") and s.confirmed_at <= t]
            if pool:
                line = anchored_vwap(adf, pool[-1].idx).tail(len(hist))
                fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                         line=dict(color=TEAL, width=1.2, dash="dash"),
                                         opacity=.75), row=1, col=1)
        except Exception:
            pass

    if "vp" in on:
        try:
            if htf_df is not None and len(htf_df) >= 30:
                _profile(fig, htf_df, bars * 2, VP_HTF, gutter_ref, "htf")
        except Exception:
            pass
        try:
            _profile(fig, ltf_df, bars, VP_LTF, gutter_ref, "ltf")
        except Exception:
            pass

    if plan and "plan" in on:
        for y, col, txt in ((plan["entry"], BLUE, "ENTRY"), (plan["stop"], DN, "SL"),
                            (plan["tp1"], UP, "TP1"), (plan["tp2"], UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=last_x, x1=right,
                          y0=y, y1=y, line=dict(color=col, width=1.4))
            fig.add_annotation(x=right, y=y, row=1, col=1, xanchor="left",
                               text=f" {txt} {y:,.6g}", showarrow=False,
                               font=dict(size=8, color=col))

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

    fig.update_layout(height=640, template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=BG, margin=dict(l=62, r=10, t=42, b=22),
                      showlegend=False, hovermode="x unified", dragmode="zoom",
                      uirevision="cryptoscan",
                      title=dict(text=f"chart {chart_tf or ''} · structure "
                                      f"{htf_label} (HTF) / {ltf_label} (LTF) · "
                                      f"gutter blue = HTF, orange = LTF",
                                 x=0, xanchor="left", font=dict(size=10, color=MUT)),
                      xaxis=dict(domain=PRICE_DOMAIN, type="date",
                                 showgrid=False, range=[x_lo, right],
                                 autorange=False, nticks=8,
                                 tickformat=_tick(chart_tf),
                                 rangeslider=dict(visible=False),
                                 fixedrange=False, constrain="range"))
    fig.update_layout(**{gutter_prop: dict(domain=PROFILE_DOMAIN, anchor="y",
                                          range=[0, 1], type="linear",
                                          showgrid=False, showticklabels=False,
                                          zeroline=False, fixedrange=True)})
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range, autorange=False,
                                     fixedrange=False))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="left",
                     tickformat=",.6g", fixedrange=False)
    return fig
