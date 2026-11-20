"""chart.py - annotated decision chart.

Every mark is drawn from the engine's own state, never hand-placed:

  swing high/low   find_swings           where structure pivots
  BOS / CHoCH      build_structure       where structure broke, and which way
  zone rects       smc.supply_demand     where the order imbalance sits
  gap rects        smc.fair_value_gaps   unfilled imbalance, a magnet
  sweep crosses    smc.liquidity_sweeps  where stops were harvested
  AVWAP line       volume.anchored_vwap  fair value of the current leg
  POC / VAH / VAL  volume.volume_profile where volume actually transacted
  volume bars      raw volume            conviction behind each bar
  entry/stop/TP    smc.playbook          the trade, with risk and reward shaded

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
from core.volume import anchored_vwap, volume_profile

UP, DN, MUT, WARN, TEAL = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6"
BG = "#0f1116"
ANCHOR_BARS = 60


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


def build(df, cfg, direction: str, plan: dict | None = None,
          bars: int = 260) -> go.Figure:
    view = df.tail(bars)
    ind = compute_all(df, cfg)
    stt = build_structure(df, cfg.structure)
    a = ind["atr"].iloc[-1]
    atr = float(a) if a == a else 0.0
    long = direction == "long"
    want = "bull" if long else "bear"
    right = _future(view)
    last_x = view.index[-1]

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
        sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
        for kind, sym in (("H", "triangle-down"), ("L", "triangle-up")):
            pts = [s for s in sw if s["type"] == kind and s["idx"] >= len(df) - bars]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[df.index[s["idx"]] for s in pts],
                    y=[s["price"] for s in pts], mode="markers",
                    marker=dict(symbol=sym, size=8, color=MUT, opacity=.8),
                    name="swing", showlegend=False), row=1, col=1)
    except Exception:
        pass

    try:
        events = []
        for key, tag in (("bos_up", "BOS"), ("choch_up", "CHoCH"),
                         ("bos_dn", "BOS"), ("choch_dn", "CHoCH")):
            for t in np.flatnonzero(stt[key].to_numpy())[-4:]:
                if t >= len(df) - bars:
                    events.append((int(t), tag, key.endswith("up")))
        for t, tag, bull in sorted(events)[-8:]:
            lvl = float(stt["active_high"].iat[t] if bull else stt["active_low"].iat[t])
            if not np.isfinite(lvl):
                continue
            fig.add_vline(x=df.index[t], row=1, col=1,
                          line=dict(color=WARN if tag == "CHoCH" else "#e7eaf0",
                                    width=1, dash="dot"))
            fig.add_annotation(x=df.index[t], y=lvl, row=1, col=1, text=tag,
                               showarrow=False, yshift=16 if bull else -16,
                               font=dict(size=9, color=WARN if tag == "CHoCH" else "#e7eaf0"))
    except Exception:
        pass

    try:
        for z in [z for z in smc.supply_demand(df)
                  if z["dir"] == ("demand" if long else "supply")][:6]:
            if z["idx"] < len(df) - bars:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=df.index[z["idx"]], x1=right,
                          y0=z["bot"], y1=z["top"],
                          fillcolor="rgba(22,199,132,.16)" if long else "rgba(240,185,11,.16)",
                          line=dict(width=1, color=TEAL if long else WARN))
    except Exception:
        pass

    try:
        for g in [g for g in smc.fair_value_gaps(df) if g["dir"] == want][:4]:
            if g["idx"] < len(df) - bars:
                continue
            fig.add_shape(type="rect", row=1, col=1, x0=df.index[g["idx"]], x1=right,
                          y0=g["bot"], y1=g["top"],
                          fillcolor="rgba(22,199,132,.10)" if long else "rgba(234,57,67,.10)",
                          line=dict(width=0))
    except Exception:
        pass

    try:
        for s in [s for s in smc.liquidity_sweeps(df) if s["dir"] == want][:5]:
            if s["idx"] < len(df) - bars:
                continue
            fig.add_trace(go.Scatter(
                x=[df.index[s["idx"]]], y=[s["level"]], mode="markers",
                marker=dict(symbol="x", size=11, color=WARN, line=dict(width=2)),
                name="sweep", showlegend=False), row=1, col=1)
    except Exception:
        pass

    try:
        anchor = _anchor(stt, long)
        if anchor is not None:
            line = anchored_vwap(df, anchor).tail(bars)
            fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                     line=dict(color=TEAL, width=1.6, dash="dash")),
                          row=1, col=1)
    except Exception:
        pass

    try:
        vp = volume_profile(df)
        if vp:
            fig.add_hline(y=vp["poc"], row=1, col=1, line=dict(color=WARN, width=1.4))
            for lvl in (vp["vah"], vp["val"]):
                fig.add_hline(y=lvl, row=1, col=1, line=dict(color=MUT, width=1, dash="dot"))
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
                      fillcolor="rgba(234,57,67,.18)", line=dict(width=0))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(entry, tp1), y1=max(entry, tp1),
                      fillcolor="rgba(22,199,132,.16)", line=dict(width=0))
        fig.add_shape(type="rect", row=1, col=1, x0=last_x, x1=right,
                      y0=min(tp1, tp2), y1=max(tp1, tp2),
                      fillcolor="rgba(22,199,132,.08)", line=dict(width=0))

    try:
        ci, cname, cq = latest_candle(df, atr, direction)
        if ci is not None and cq >= 60:
            fig.add_annotation(x=df.index[ci],
                               y=float(df["low"].iat[ci]) if long else float(df["high"].iat[ci]),
                               row=1, col=1, text=f" {cname} {cq} ", showarrow=True,
                               arrowhead=2, yshift=-22 if long else 22,
                               font=dict(size=9, color="#0f1116"),
                               bgcolor=UP if long else DN, borderpad=3)
    except Exception:
        pass

    fig.update_layout(
        height=620, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=6, r=86, t=28, b=6), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9)),
        hovermode="x unified", dragmode="pan")
    fig.update_xaxes(showgrid=False, gridcolor="#1b1f28")
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="right")
    return fig


def legend_md() -> str:
    return """
| mark | meaning |
|---|---|
| white dotted vertical + `BOS` | structure broke in trend direction — continuation |
| amber dotted vertical + `CHoCH` | structure broke against the trend — possible reversal |
| grey triangles | confirmed swing high / low, the pivots structure is built from |
| teal box | demand zone (long side). Amber box: supply zone (short side) |
| faint box, no border | unfilled fair value gap — a magnet, not an entry |
| amber ✕ | liquidity sweep — stops harvested, then closed back inside |
| teal dashed line | anchored VWAP from the origin of the leg |
| amber solid line | point of control — most-traded price in 200 bars |
| grey dotted lines | value area high / low — 70% of volume sits between them |
| volume pane | raw volume, grey 20-bar average, amber 1.5x conviction line |
| green box | reward to TP1 / TP2 |
| red box | risk to stop |
| candle callout | the reversal candle and its ATR-normalised score |
"""
