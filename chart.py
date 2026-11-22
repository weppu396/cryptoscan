"""chart.py - annotated decision chart.

Every mark is drawn from the engine's own state, never hand-placed:

  swing high/low   find_swings           where structure pivots
  BOS / CHoCH      build_structure       where structure broke, and which way
  zone rects       smc.supply_demand     where the order imbalance sits
  gap rects        smc.fair_value_gaps   unfilled imbalance, a magnet
  sweep crosses    smc.liquidity_sweeps  where stops were harvested
  AVWAP line       volume.anchored_vwap  fair value of the current leg
  POC / VAH / VAL  volume.profile_histogram  where volume actually transacted
  volume bars      raw volume            conviction behind each bar
  entry/stop/TP    smc.playbook          the trade, with risk and reward shaded

Clutter rules, added 2026-11: at most two 30m majors and three 3m minors are
drawn, anything priced within half an ATR of a newer mark is dropped, minors
carry no text, and every fill is under 0.20 alpha. Major marks are blue,
minor marks grey, so a 30m structural break can never be mistaken for a 3m
twitch.

Green = reward, red = risk. Read-only: nothing here changes the plan.
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

MAJOR_LINE = "rgba(96,152,255,0.30)"
MINOR_LINE = "rgba(140,148,160,0.16)"
MAJOR_TEXT = "rgba(96,152,255,0.75)"
ZONE_FILL = {"long": "rgba(20,184,166,0.085)", "short": "rgba(240,185,11,0.085)"}
ZONE_EDGE = {"long": "rgba(20,184,166,0.30)", "short": "rgba(240,185,11,0.30)"}
FVG_FILL = {"long": "rgba(22,199,132,0.055)", "short": "rgba(234,57,67,0.055)"}
VP_HTF = "rgba(96,152,255,0.17)"
VP_LTF = "rgba(20,184,166,0.12)"
PROFILE_DOMAIN = [0.0, 0.20]
PRICE_DOMAIN = [0.235, 1.0]


def _future(df, bars: int = 14):
    if len(df) < 2:
        return df.index[-1]
    return df.index[-1] + (df.index[-1] - df.index[-2]) * bars


def _anchor(stt, long: bool, span: int = ANCHOR_BARS):
    """Last confirmed opposite swing, bounded to `span` bars so the VWAP stays
    relevant to the current leg instead of the start of history."""
    col = "prev_swing_low" if long else "prev_swing_high"
    try:
        s = stt[col].tail(span).dropna()
        return int(float(s.iloc[-1])) if len(s) else None
    except Exception:
        return None


def _draw_profile(fig, df, bars: int, color: str, right: object) -> None:
    """Horizontal volume profile in the left gutter. Estimate, not tick data."""
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(
        y=vp["centers"], x=vp["hist"], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=step * 0.9 if step else None,
        hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.data[-1].xaxis = "x2"

    for lvl, tone, tag in ((vp["poc"], WARN, "POC"),
                           (vp["vah"], MUT, "VAH"),
                           (vp["val"], MUT, "VAL")):
        fig.add_shape(type="line", xref="x2", yref="y", x0=0, x1=1,
                      y0=lvl, y1=lvl,
                      line=dict(color=tone, width=1 if tag == "POC" else 1,
                                dash="solid" if tag == "POC" else "dot"))
        fig.add_annotation(xref="paper", yref="y", x=PRICE_DOMAIN[0], y=lvl,
                           text=tag, showarrow=False, xanchor="left",
                           font=dict(size=7, color=tone), opacity=0.75)

    for lvl in vp["hvn"][:2]:
        fig.add_shape(type="line", xref="x2", yref="y", x0=0, x1=0.45,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.30)", width=1))
    for lvl in vp["lvn"][:2]:
        fig.add_shape(type="line", xref="x2", yref="y", x0=0, x1=0.22,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(150,150,160,0.18)", width=1))


def build(df, cfg, direction: str, plan: dict | None = None, bars: int = 260,
          chart_tf: str | None = None, analysis_df=None,
          htf_df=None) -> go.Figure:
    """`df` is what gets drawn. `analysis_df` is what the engine read — the
    plan passed in was built from it. Changing `chart_tf` therefore cannot
    change the trade."""
    adf = analysis_df if analysis_df is not None else df
    view = df.tail(bars)
    ind = compute_all(adf, cfg)
    stt = build_structure(adf, cfg.structure)
    a = ind["atr"].iloc[-1]
    atr = float(a) if a == a else 0.0
    long = direction == "long"
    want = "bull" if long else "bear"
    right = _future(view)
    last_x = view.index[-1]

    if htf_df is None:
        try:
            htf_df = resample(adf, "30m") if len(adf) > 60 else None
        except Exception:
            htf_df = None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name="price",
        increasing_line_color=UP, decreasing_line_color=DN,
        increasing_fillcolor=UP, decreasing_fillcolor=DN,
        showlegend=False), row=1, col=1)

    cols = [UP if c >= o else DN for o, c in zip(view["open"], view["close"])]
    fig.add_trace(go.Bar(x=view.index, y=view["volume"], marker_color=cols,
                         marker_line_width=0, opacity=.65, name="volume",
                         showlegend=False), row=2, col=1)
    vma = view["volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=view.index, y=vma, name="vol avg",
                             line=dict(color=MUT, width=1), showlegend=False),
                  row=2, col=1)
    lv = vma.iloc[-1]
    thr = float(lv) * 1.5 if lv == lv else 0.0
    if thr > 0:
        fig.add_hline(y=thr, row=2, col=1, line=dict(color=WARN, width=1, dash="dot"))
        fig.add_annotation(x=view.index[min(3, len(view) - 1)], y=thr, row=2, col=1,
                           yshift=9, text="1.5x = conviction", showarrow=False,
                           font=dict(size=9, color=WARN))

    try:
        sw = find_swings(adf, cfg.structure.swing_left, cfg.structure.swing_right)
        start = adf.index[max(0, len(adf) - bars)] if len(adf) > bars else adf.index[0]
        for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
            pts = [s for s in sw if s["type"] == kind and adf.index[s["idx"]] >= start]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[adf.index[s["idx"]] for s in pts],
                    y=[s["price"] for s in pts], mode="markers",
                    marker=dict(symbol=sym, size=7, color=MUT, opacity=.55),
                    name="swing", showlegend=False), row=1, col=1)
    except Exception:
        pass

    # structure: two 30m majors labelled, three 3m minors as bare lines
    try:
        gap = atr * 0.5
        majors, minors = marks(adf, cfg, htf_df, gap)
        for m in minors:
            fig.add_vline(x=m["ts"], row=1, col=1,
                          line=dict(color=MINOR_LINE, width=1, dash="dot"))
        for m in majors:
            fig.add_vline(x=m["ts"], row=1, col=1,
                          line=dict(color=MAJOR_LINE, width=1, dash="dot"))
            fig.add_annotation(x=m["ts"], y=m["level"], row=1, col=1,
                               text=m["kind"], showarrow=False, xanchor="right",
                               yshift=15 if m["dir"] == "bull" else -15,
                               font=dict(size=8, color=MAJOR_TEXT))
    except Exception:
        pass

    try:
        for z in live_zones(smc.supply_demand(adf),
                            adf, direction, keep=2):
            x0 = adf.index[z["idx"]]
            if x0 < start:
                x0 = start
            fig.add_shape(type="rect", row=1, col=1, x0=x0, x1=right,
                          y0=z["bot"], y1=z["top"],
                          fillcolor=ZONE_FILL[direction],
                          line=dict(width=1, color=ZONE_EDGE[direction]))
    except Exception:
        pass

    try:
        for g in [g for g in smc.fair_value_gaps(adf) if g["dir"] == want][:3]:
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
                marker=dict(symbol="x", size=10, color=WARN, opacity=.8),
                name="sweep", showlegend=False), row=1, col=1)
    except Exception:
        pass

    try:
        anchor = _anchor(stt, long)
        if anchor is not None:
            line = anchored_vwap(adf, anchor).tail(bars)
            fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                     line=dict(color=TEAL, width=1.4, dash="dash"),
                                     opacity=.8), row=1, col=1)
    except Exception:
        pass

    # two profiles, two colours, both labelled as estimates
    try:
        if htf_df is not None and len(htf_df) >= 30:
            _draw_profile(fig, htf_df, bars * 2, VP_HTF, right)
    except Exception:
        pass
    try:
        _draw_profile(fig, adf, bars, VP_LTF, right)
    except Exception:
        pass

    if plan:
        entry, sl, tp1, tp2 = plan["entry"], plan["sl"], plan["tp1"], plan["tp2"]
        for y, col, txt in ((entry, "#e7eaf0", "ENTRY"), (sl, DN, "STOP"),
                            (tp1, UP, "TP1 50%"), (tp2, UP, "TP2")):
            fig.add_shape(type="line", row=1, col=1, x0=last_x, x1=right, y0=y, y1=y,
                          line=dict(color=col, width=1.2))
            fig.add_annotation(x=right, y=y, row=1, col=1, xanchor="left",
                               text=f" {txt} {y:,.6g}", showarrow=False,
                               font=dict(size=9, color=col))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(entry, sl), y1=max(entry, sl),
                      fillcolor="rgba(234,57,67,0.15)", line=dict(width=0))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(entry, tp1), y1=max(entry, tp1),
                      fillcolor="rgba(22,199,132,0.13)", line=dict(width=0))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(tp1, tp2), y1=max(tp1, tp2),
                      fillcolor="rgba(22,199,132,0.07)", line=dict(width=0))

    try:
        ci, cname, cq = latest_candle(adf, atr, direction)
        if ci is not None and cq >= 60 and adf.index[ci] >= start:
            fig.add_annotation(x=adf.index[ci],
                               y=float(adf["low"].iat[ci]) if long
                               else float(adf["high"].iat[ci]),
                               row=1, col=1, text=f" {cname} {cq} ", showarrow=True,
                               arrowhead=2, yshift=-22 if long else 22,
                               font=dict(size=9, color="#0f1116"),
                               bgcolor=UP if long else DN, borderpad=3)
    except Exception:
        pass

    shown_tf = chart_tf or ""
    fig.update_layout(
        height=640, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=6, r=76, t=34, b=6), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=9)),
        hovermode="x unified", dragmode="pan",
        title=dict(text=f"chart {shown_tf} · structure 30m/3m · "
                        f"profile = OHLCV estimate",
                   x=0, xanchor="left", font=dict(size=10, color=MUT)),
        xaxis=dict(domain=PRICE_DOMAIN),
        xaxis2=dict(domain=PROFILE_DOMAIN, anchor="y", range=[0, 1],
                    showgrid=False, showticklabels=False, zeroline=False))
    fig.update_xaxes(showgrid=False, gridcolor="#1b1f28")
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="right")
    return fig


def timeframe_control(key: str, current: str, default: str = "15m") -> str:
    """One selectbox. Display only — the engine keeps its own pair."""
    idx = CHART_TFS.index(current) if current in CHART_TFS else \
        (CHART_TFS.index(default) if default in CHART_TFS else 3)
    return st_selectbox(key, idx)


def st_selectbox(key: str, idx: int) -> str:
    import streamlit as st
    return st.selectbox("Chart timeframe", CHART_TFS, index=idx, key=key,
                        help="Display only. Structure and the plan stay on "
                             "30m context / 3m trigger.")


def legend_md() -> str:
    return """
| mark | meaning |
|---|---|
| blue dotted vertical + `BOS` / `CHoCH` | **30m** structural break — major, labelled |
| grey dotted vertical, no text | **3m** structural break — minor, unlabelled on purpose |
| grey triangles | confirmed swing high / low |
| teal box | demand zone (long side). Amber box: supply zone (short side) |
| faint box, no border | unfilled fair value gap — a magnet, not an entry |
| amber ✕ | liquidity sweep — stops harvested, then closed back inside |
| teal dashed line | anchored VWAP from the origin of the leg |
| left gutter bars | volume profile — **OHLCV estimate**, 30m blue / 3m teal |
| amber line in gutter | POC — most-traded price |
| grey dotted in gutter | VAH / VAL — 70% of volume sits between them |
| short grey ticks | HVN (longer) and LVN (shorter) |
| green box | reward to TP1 / TP2 |
| red box | risk to stop |
| candle callout | the reversal candle and its ATR-normalised score |

Only the newest two 30m and three 3m marks survive, and anything within half
an ATR of a newer mark is dropped. The gutter is an estimate: candles carry
no trade direction, so volume is spread evenly across each bar's range.
"""
