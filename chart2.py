"""chart2.py - authoritative renderer. Consumes a StructureFrame, never recomputes it.

Same layout contract as chart.py: candles 0-82% on a datetime axis, profile
gutter 84.5-100% on its own numeric axis. Traces use 'x2', layout uses 'xaxis2'.

Spacing comes from the median gap, and the x range is pinned, so no tail gap
can collapse the candles into a stripe.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.models import Layer
from core.structure_map import thin
from core.tfs import chart_minutes, resample
from core.volume import anchored_vwap, profile_histogram

UP, DN, MUT, WARN, TEAL, BLUE = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6", "#3b82f6"
BG = "#0f1116"
MODES = ("structure", "analysis", "execution")
LAYERS = {
    "structure": {"candles", "ext_swing", "int_swing", "breaks", "tags"},
    "analysis": {"candles", "ext_swing", "breaks", "zones", "fvg", "sweeps",
                 "vp", "avwap"},
    "execution": {"candles", "zone", "plan"},
}
STYLE = {Layer.EXTERNAL: (1.6, "rgba(96,152,255,0.60)"),
         Layer.INTERNAL: (1.0, "rgba(140,148,160,0.32)")}
VP_HTF = "rgba(96,152,255,0.20)"
VP_LTF = "rgba(20,184,166,0.14)"
PRICE_DOMAIN = [0.0, 0.82]
PROFILE_DOMAIN = [0.845, 1.0]
MAX_MARKS = {Layer.EXTERNAL: 2, Layer.INTERNAL: 3}
DEFAULT_BARS = 110
MARK_BARS = 6
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
    """Median positive bar gap."""
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


def _profile(fig, df, bars, color, axis):
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(y=vp["centers"], x=vp["hist"], orientation="h",
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


def build(df, frame, direction, plan=None, mode="analysis", layers=None,
          bars=DEFAULT_BARS, chart_tf=None, analysis_df=None,
          htf_df=None) -> go.Figure:
    if mode not in MODES:
        raise ValueError(mode)
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

    on = set(layers) if layers else LAYERS[mode]
    view = df.tail(bars)
    step = _step(view)
    t = len(adf) - 1
    long = direction == "long"
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
        close=view["close"],
        increasing_line_color=UP, increasing_line_width=1.2, increasing_fillcolor=UP,
        decreasing_line_color=DN, decreasing_line_width=1.2, decreasing_fillcolor=DN,
        whiskerwidth=0.6, showlegend=False), row=1, col=1)

    for layer in (Layer.EXTERNAL, Layer.INTERNAL):
        key = "ext_swing" if layer == Layer.EXTERNAL else "int_swing"
        if key in on:
            pts = [s for s in frame.swings if s.layer == layer
                   and adf.index[s.idx] >= view.index[0]]
            for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
                sub = [s for s in pts if s.kind == kind]
                if sub:
                    fig.add_trace(go.Scatter(
                        x=[adf.index[s.idx] for s in sub],
                        y=[s.price for s in sub], mode="markers",
                        marker=dict(symbol=sym, size=6, color=MUT, opacity=.5),
                        showlegend=False), row=1, col=1)

    if "breaks" in on:
        for layer in (Layer.EXTERNAL, Layer.INTERNAL):
            live = [e for e in frame.live(t) if e.layer == layer
                    and adf.index[e.break_idx] >= view.index[0]
                    and e.confirm_idx <= len(adf) - 2]
            live = thin([{"ts": adf.index[e.break_idx], "level": e.level, "e": e}
                         for e in live], 0.0)
            for row in live[-MAX_MARKS[layer]:]:
                e = row["e"]
                w, col = STYLE[layer]
                if e.label.value == "CHoCH":
                    col = WARN if layer == Layer.EXTERNAL \
                        else "rgba(240,185,11,0.30)"
                label = None
                if layer == Layer.EXTERNAL:
                    side = "Bullish" if e.direction == "bull" else "Bearish"
                    label = f"{side} {e.label.value}"
                _mark(fig, adf.index[e.break_idx], e.level, step, col, w, label,
                      9 if e.direction == "bull" else -9)

    if "zones" in on:
        for z in [z for z in smc.supply_demand(adf)
                  if z["dir"] == ("demand" if long else "supply")][:2]:
            x0 = max(adf.index[z["idx"]], view.index[0])
            fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=right,
                          y0=z["bot"], y1=z["top"], line=dict(width=1),
                          line_color="rgba(20,184,166,0.30)" if long
                          else "rgba(240,185,11,0.30)",
                          fillcolor="rgba(20,184,166,0.085)" if long
                          else "rgba(240,185,11,0.085)")

    if "fvg" in on:
        for g in [g for g in smc.fair_value_gaps(adf)
                  if g["dir"] == ("bull" if long else "bear")][:3]:
            if adf.index[g["idx"]] < view.index[0]:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=adf.index[g["idx"]], x1=right,
                          y0=g["bot"], y1=g["top"], line=dict(width=0),
                          fillcolor="rgba(22,199,132,0.055)" if long
                          else "rgba(234,57,67,0.055)")

    if "sweeps" in on:
        for s in [s for s in smc.liquidity_sweeps(adf)
                  if s["dir"] == ("bull" if long else "bear")][:3]:
            if adf.index[s["idx"]] < view.index[0]:
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
                line = anchored_vwap(adf, pool[-1].idx).tail(bars)
                fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                         line=dict(color=TEAL, width=1.2, dash="dash"),
                                         opacity=.75), row=1, col=1)
        except Exception:
            pass

    if "vp" in on:
        try:
            if htf_df is not None and len(htf_df) >= 30:
                _profile(fig, htf_df, bars * 2, VP_HTF, gutter_ref)
        except Exception:
            pass
        try:
            _profile(fig, adf, bars, VP_LTF, gutter_ref)
        except Exception:
            pass

    if plan and "plan" in on:
        for y, col, txt in ((plan["entry"], BLUE, "ENTRY"), (plan["stop"], DN, "SL"),
                            (plan["tp1"], UP, "TP1"), (plan["tp2"], UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=last_x, x1=right,
                          y0=y, y1=y, line=dict(color=col, width=1.1))
            fig.add_annotation(x=right, y=y, row=1, col=1, xanchor="left",
                               text=f" {txt} {y:,.6g}", showarrow=False,
                               font=dict(size=8, color=col))

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

    fig.update_layout(height=620, template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=BG, margin=dict(l=62, r=10, t=34, b=22),
                      showlegend=False, hovermode="x unified", dragmode="pan",
                      title=dict(text=f"chart {chart_tf or ''} · profile = "
                                      f"OHLCV estimate", x=0, xanchor="left",
                                 font=dict(size=10, color=MUT)),
                      xaxis=dict(domain=PRICE_DOMAIN, type="date",
                                 showgrid=False, range=[x_lo, right], nticks=8,
                                 tickformat=_tick(chart_tf),
                                 rangeslider=dict(visible=False)))
    fig.update_layout(**{gutter_prop: dict(domain=PROFILE_DOMAIN, anchor="y",
                                          range=[0, 1], type="linear",
                                          showgrid=False, showticklabels=False,
                                          zeroline=False, fixedrange=True)})
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="left",
                     tickformat=",.6g")
    return fig
