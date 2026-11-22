"""live.py - one second tape. Display only; the strategy never reads it.

A venue ticker is polled once per second inside a fragment and appended to a
bounded session buffer. The buffer is seeded from real 1s candles where the
venue serves them, and simply starts empty where it does not, so nothing is
invented. Streamlit re-runs only the fragment, so the rest of the page keeps
its state.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MAX_POINTS = 240
WINDOW = 90
UP, DN, BG = "#16c784", "#ea3943", "#0f1116"


def _tape(sym):
    return st.session_state.setdefault(f"tape::{sym}", [])


def _seed(ex, sym, flag):
    if st.session_state.get(flag) or _tape(sym):
        return
    st.session_state[flag] = True
    try:
        rows = ex.fetch_ohlcv(sym, "1s", limit=120)
    except Exception:
        rows = []
    tape = _tape(sym)
    for r in rows or []:
        try:
            tape.append((pd.Timestamp(int(r[0]), unit="ms", tz="UTC"),
                         float(r[4])))
        except Exception:
            continue
    del tape[:-MAX_POINTS]


def _price(ex, sym):
    try:
        t = ex.fetch_ticker(sym) or {}
    except Exception:
        return None
    for k in ("last", "close", "bid", "ask"):
        try:
            if t.get(k):
                return float(t[k])
        except Exception:
            continue
    return None


def _figure(view, side):
    xs = [p[0] for p in view]
    ys = [p[1] for p in view]
    col = UP if side == "long" else DN
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.15 or max(abs(hi), 1.0) * 0.0005
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=col, width=1.6, shape="hv"),
        fill="tozeroy",
        fillcolor="rgba(22,199,132,0.07)" if side == "long"
        else "rgba(234,57,67,0.07)",
        hovertemplate="%{x|%H:%M:%S}  %{y:,.6g}<extra></extra>"))
    fig.add_annotation(x=xs[-1], y=ys[-1], text=f" {ys[-1]:,.6g} ",
                       showarrow=False, xanchor="left",
                       font=dict(size=10, color=col),
                       bgcolor="rgba(15,17,22,0.7)")
    fig.update_layout(height=260, template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=BG, showlegend=False, dragmode="pan",
                      margin=dict(l=58, r=56, t=8, b=18), uirevision="live",
                      xaxis=dict(type="date", showgrid=False,
                                 tickformat="%H:%M:%S", nticks=6,
                                 range=[xs[0], xs[-1] + pd.Timedelta(seconds=4)],
                                 autorange=False, fixedrange=False),
                      yaxis=dict(range=[lo - pad, hi + pad], autorange=False,
                                 side="left", gridcolor="#1b1f28",
                                 tickformat=",.6g", fixedrange=False))
    return fig


def _body(ex, sym, side, flag):
    _seed(ex, sym, flag)
    px = _price(ex, sym)
    now = pd.Timestamp.now(tz="UTC")
    tape = _tape(sym)
    if px:
        tape.append((now, px))
    del tape[:-MAX_POINTS]
    if len(tape) < 2:
        st.caption(f"live tape · waiting for ticks · {now:%H:%M:%S} UTC")
        return
    cutoff = now - pd.Timedelta(seconds=WINDOW)
    view = [p for p in tape if p[0] >= cutoff] or tape[-2:]
    first = view[0][1]
    chg = (view[-1][1] / first - 1) * 100 if first else 0.0
    st.plotly_chart(_figure(view, side), use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})
    st.caption(f"live 1s tape · {view[-1][1]:,.6g} · {chg:+.3f}% over {WINDOW}s · "
               f"{len(view)} ticks · {now:%H:%M:%S} UTC · polled once a second, "
               f"not a websocket. Display only.")


def render(ex, sym, side="long", flag="live_seeded"):
    if hasattr(st, "fragment"):
        st.fragment(run_every="1s")(_body)(ex, sym, side, flag)
    else:
        _body(ex, sym, side, flag)
