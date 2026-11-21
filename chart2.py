"""chart2.py - authoritative renderer. Consumes a StructureFrame, never recomputes it."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.models import Layer
from core.volume import anchored_vwap, volume_profile

UP, DN, MUT, WARN, TEAL, BLUE = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6", "#3b82f6"
BG = "#0f1116"
MODES = ("structure", "analysis", "execution")
LAYERS = {
    "structure": {"candles", "ext_swing", "int_swing", "breaks", "tags"},
    "analysis": {"candles", "ext_swing", "breaks", "zones", "fvg", "sweeps",
                 "volume", "vp", "avwap"},
    "execution": {"candles", "zone", "plan"},
}
STYLE = {Layer.EXTERNAL: (1.6, "solid", "#e7eaf0"),
         Layer.INTERNAL: (1.1, "dash", "#8b93a3")}


def _future(df, bars=14):
    if len(df) < 2:
        return df.index[-1]
    return df.index[-1] + (df.index[-1] - df.index[-2]) * bars


def build(df, frame, direction, plan=None, mode="analysis", layers=None,
          bars=260) -> go.Figure:
    if mode not in MODES:
        raise ValueError(mode)
    on = set(layers) if layers else LAYERS[mode]
    view = df.tail(bars)
    t = len(df) - 1
    long = direction == "long"
    right = _future(view)

    rows = 2 if "volume" in on else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22] if rows == 2 else [1.0],
                        vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], increasing_line_color=UP, decreasing_line_color=DN,
        increasing_fillcolor=UP, decreasing_fillcolor=DN, showlegend=False),
        row=1, col=1)

    if "volume" in on:
        cols = [UP if c >= o else DN for o, c in zip(view["open"], view["close"])]
        fig.add_trace(go.Bar(x=view.index, y=view["volume"], marker_color=cols,
                             opacity=.6, showlegend=False), row=2, col=1)

    for layer in (Layer.EXTERNAL, Layer.INTERNAL):
        key = "ext_swing" if layer == Layer.EXTERNAL else "int_swing"
        if key in on:
            pts = [s for s in frame.swings if s.layer == layer
                   and s.idx >= len(df) - bars]
            for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
                sub = [s for s in pts if s.kind == kind]
                if sub:
                    fig.add_trace(go.Scatter(
                        x=[df.index[s.idx] for s in sub],
                        y=[s.price for s in sub], mode="markers",
                        marker=dict(symbol=sym, size=7, color=MUT, opacity=.75),
                        showlegend=False), row=1, col=1)

    if "breaks" in on:
        for e in frame.live(t):
            if e.break_idx < len(df) - bars:
                continue
            w, dash, col = STYLE[e.layer]
            if e.label.value == "CHoCH":
                col = WARN
            fig.add_vline(x=df.index[e.break_idx], row=1, col=1,
                          line=dict(color=col, width=w, dash=dash))
            fig.add_annotation(x=df.index[e.break_idx], y=e.level, row=1, col=1,
                               text=f"{e.label.value} {e.level:,.6g}",
                               showarrow=False, xanchor="right",
                               yshift=14 if e.direction == "bull" else -14,
                               font=dict(size=8, color=col))

    if "zones" in on:
        for z in [z for z in smc.supply_demand(df)
                  if z["dir"] == ("demand" if long else "supply")][:6]:
            if z["idx"] < len(df) - bars:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=df.index[z["idx"]], x1=right,
                          y0=z["bot"], y1=z["top"],
                          fillcolor="rgba(20,184,166,.16)" if long else "rgba(240,185,11,.16)",
                          line=dict(width=1, color=TEAL if long else WARN))

    if "fvg" in on:
        for g in [g for g in smc.fair_value_gaps(df)
                  if g["dir"] == ("bull" if long else "bear")][:4]:
            if g["idx"] < len(df) - bars:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=df.index[g["idx"]], x1=right,
                          y0=g["bot"], y1=g["top"], line=dict(width=0),
                          fillcolor="rgba(22,199,132,.09)" if long else "rgba(234,57,67,.09)")

    if "sweeps" in on:
        for s in [s for s in smc.liquidity_sweeps(df)
                  if s["dir"] == ("bull" if long else "bear")][:5]:
            if s["idx"] < len(df) - bars:
                continue
            fig.add_trace(go.Scatter(x=[df.index[s["idx"]]], y=[s.level],
                                     mode="markers",
                                     marker=dict(symbol="x", size=10, color=WARN),
                                     showlegend=False), row=1, col=1)

    if "avwap" in on:
        try:
            lows = [s for s in frame.swings if s.layer == Layer.EXTERNAL
                    and s.kind == "L" and s.confirmed_at <= t]
            highs = [s for s in frame.swings if s.layer == Layer.EXTERNAL
                     and s.kind == "H" and s.confirmed_at <= t]
            pool = lows if long else highs
            if pool:
                line = anchored_vwap(df, pool[-1].idx).tail(bars)
                fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                         line=dict(color=TEAL, width=1.4, dash="dash")),
                              row=1, col=1)
        except Exception:
            pass

    if "vp" in on:
        try:
            vp = volume_profile(df)
            if vp:
                fig.add_hline(y=vp["poc"], row=1, col=1, line=dict(color=WARN, width=1.2))
                for lvl in (vp["vah"], vp["val"]):
                    fig.add_hline(y=lvl, row=1, col=1,
                                  line=dict(color=MUT, width=1, dash="dot"))
        except Exception:
            pass

    if plan and "plan" in on:
        for y, col, txt in ((plan["entry"], BLUE, "ENTRY"), (plan["stop"], DN, "SL"),
                            (plan["tp1"], UP, "TP1"), (plan["tp2"], UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=view.index[-1], x1=right,
                          y0=y, y1=y, line=dict(color=col, width=1.4))
            fig.add_annotation(x=right, y=y, row=1, col=1, xanchor="left",
                               text=f" {txt} {y:,.6g}", showarrow=False,
                               font=dict(size=8, color=col))

    fig.update_layout(height=600, template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=BG, margin=dict(l=6, r=90, t=24, b=6),
                      xaxis_rangeslider_visible=False, hovermode="x unified",
                      dragmode="pan", showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="right")
    return fig
