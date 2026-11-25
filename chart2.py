"""chart2.py - authoritative renderer. Consumes a StructureFrame, never recomputes it.

Each layer is placed on its own source frame: external events are indexed in the
HTF frame, internal events in the LTF frame. Feeding both layers from one frame
is what made an opposing LTF impossible to draw.

Structure lines start at the broken pivot wick. After a change of character,
the current trend's BOS lines and the dashed CHoCH level show that layer's
direction and invalidation context.

Window: the visible span is derived from the structure timeframes, not from a
fixed bar count, so a 30m pivot (which needs right+1 HTF bars to confirm) is
always on screen at any display timeframe without zooming out. HTF_VISIBLE and
LTF_VISIBLE set the guaranteed bars of each.

History: the trace carries the whole loaded frame (up to MAX_TRACE bars), so
panning left always finds candles. The initial window is only an initial range,
and uirevision is keyed to the view so zoom survives reruns but not a timeframe
switch.

Swings, events and sweeps arrive in three shapes: Swing objects, StructureEvent
objects, and plain dicts from smc. Every read goes through _sv().
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import smc
from core.models import Layer
from core.structure_map import chart_zones, thin
from core.tfs import cap_htf, cap_ltf, chart_minutes, resample
from core.volume import anchored_vwap, profile_histogram

UP, DN, MUT, WARN, TEAL, BLUE = "#16c784", "#ea3943", "#7f8798", "#f0b90b", "#14b8a6", "#3b82f6"
HTF_LINE = "#f5d90a"
LTF_LINE = "#22d3ee"
SL_DARK = "#a61b1b"
TP_DARK = "#1a7f37"
ENTRY_DARK = "#1d4ed8"
SUPPLY_FILL = "rgba(234,57,67,0.23)"
SUPPLY_EDGE = "rgba(214,116,128,0.85)"
DEMAND_FILL = "rgba(22,199,132,0.23)"
DEMAND_EDGE = "rgba(88,206,150,0.85)"
BG = "#0f1116"
MODES = ("structure", "analysis", "execution")
LAYERS = {
    "structure": {"candles", "ext_swing", "int_swing", "breaks", "tags", "zones", "plan"},
    "analysis": {"candles", "ext_swing", "int_swing", "breaks", "zones", "fvg", "sweeps",
                 "vp", "avwap", "plan"},
    "execution": {"candles", "zones", "plan"},
}
STYLE = {Layer.EXTERNAL: (2.2, HTF_LINE), Layer.INTERNAL: (1.8, LTF_LINE)}
VP_HTF = "rgba(96,152,255,0.34)"
VP_LTF = "rgba(240,140,11,0.30)"
PRICE_DOMAIN = [0.0, 0.76]
PROFILE_DOMAIN = [0.785, 1.0]
HALF = {"htf": (0.52, 1.0), "ltf": (0.0, 0.48)}
MAX_MARKS = 2
LABEL_KEEP = 4
DEFAULT_BARS = 70
HTF_VISIBLE = 45
LTF_VISIBLE = 60
WINDOW_CAP = 900
MAX_TRACE = 6000
OHLC = ("open", "high", "low", "close")
EQ_TOL_ATR = 0.25
BOS_LABELS = ("BOS", "BOS_INIT")


def _sv(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        v = obj.get(key, default)
    else:
        v = getattr(obj, key, default)
    return default if v is None else v


def _lay(obj):
    v = _sv(obj, "layer", "")
    return getattr(v, "value", v)


def _lab(obj):
    v = _sv(obj, "label", "")
    return getattr(v, "value", v)


def _swings(frame):
    return list(getattr(frame, "swings", None) or [])


def _live(frame, t):
    fn = getattr(frame, "live", None)
    if callable(fn):
        try:
            return list(fn(t))
        except Exception:
            pass
    out = []
    for e in getattr(frame, "events", None) or []:
        if _sv(e, "confirm_idx", 10 ** 9) > t:
            continue
        inv = _sv(e, "invalidated_at")
        if inv is not None and t >= inv:
            continue
        out.append(e)
    return out


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
    if m < 1:
        return "%H:%M:%S"
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


def _classify(swings, atr, eq_tol):
    """HH / HL / LH / LL against the previous pivot of the same kind, plus EQ."""
    out = {}
    last_h = last_l = None
    tol = max(eq_tol, 0.0)
    for s in sorted(swings, key=lambda x: int(_sv(x, "idx", 0))):
        price = float(_sv(s, "price"))
        if _sv(s, "kind") == "H":
            if last_h is None:
                tag = ""
            elif abs(price - last_h) <= tol:
                tag = "EQH"
            else:
                tag = "HH" if price > last_h else "LH"
            last_h = price
        else:
            if last_l is None:
                tag = ""
            elif abs(price - last_l) <= tol:
                tag = "EQL"
            else:
                tag = "HL" if price > last_l else "LL"
            last_l = price
        out[id(s)] = tag
    return out


def _profile(fig, df, bars, color, axis, side):
    """Proportional bars in the dedicated gutter half. Thicker than the bar step
    so they read as one band, never touching the candles."""
    b0, b1 = HALF[side]
    span = b1 - b0
    vp = profile_histogram(df, lookback=min(len(df), max(bars, 140)))
    if not vp or not len(vp["centers"]):
        return
    step = float(vp["centers"][1] - vp["centers"][0]) if len(vp["centers"]) > 1 else 0.0
    fig.add_trace(go.Bar(y=vp["centers"], x=vp["hist"] * span, base=b0,
                         orientation="h",
                         marker=dict(color=color, line=dict(width=0)),
                         width=step * 6.0 if step else None,
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
                               yanchor="middle", font=dict(size=8, color=tone),
                               opacity=0.9)
    for lvl in vp["hvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.6,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(180,180,190,0.45)", width=1))
    for lvl in vp["lvn"][:2]:
        fig.add_shape(type="line", xref=axis, yref="y", x0=b0, x1=b0 + span * 0.3,
                      y0=lvl, y1=lvl,
                      line=dict(color="rgba(180,180,190,0.26)", width=1))
    return vp


def _struct(fig, ts, price, x1, color, width, dash):
    """Start the level at its source wick and end at the latest candle."""
    fig.add_shape(type="line", xref="x", yref="y", x0=ts, x1=x1,
                  y0=price, y1=price, line=dict(color=color, width=width, dash=dash))


def _pivot_label(fig, ts, price, tag, kind, color):
    fig.add_annotation(x=ts, xref="x", y=price, yref="y", text=tag,
                       showarrow=False, xanchor="center",
                       yanchor="bottom" if kind == "H" else "top",
                       yshift=6 if kind == "H" else -6,
                       font=dict(size=8, color=color))


def _zone(fig, z, x0, x1, label, tf):
    supply = z["side"] == "supply"
    fill = SUPPLY_FILL if supply else DEMAND_FILL
    edge = SUPPLY_EDGE if supply else DEMAND_EDGE
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1,
                  y0=z["low"], y1=z["high"], fillcolor=fill,
                  line=dict(width=1.5, color=edge), layer="below",
                  name=f"{label} {tf} {z['side']}")
    return f'<span style="color:{edge}">{z["side"].title()} {z["low"]:,.6g}–{z["high"]:,.6g}</span>'


def _legend_row(fig, text, index):
    fig.add_annotation(x=0, y=0, xref="paper", yref="paper",
                       xanchor="left", yanchor="top", yshift=-37 - 17 * index,
                       text=text, showarrow=False, align="left",
                       font=dict(size=11, color="#c7cfda"))


def build(df, frame, direction, plan=None, mode="analysis", layers=None,
          bars=DEFAULT_BARS, chart_tf=None, analysis_df=None,
          htf_df=None, ltf_df=None, pair=None, cfg=None,
          frames=None, symbol=None) -> go.Figure:
    """frames maps Layer -> the frame that layer's indices belong to."""
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

    keep = MAX_MARKS
    zone_keep = 1
    eq_tol = 0.0
    if cfg is not None:
        keep = max(1, int(getattr(cfg.structure, "line_keep", MAX_MARKS)))
        zone_keep = max(1, int(getattr(cfg.structure, "zone_keep", 1)))

    if pair:
        htf_label, ltf_label = cap_htf(pair[0]), cap_ltf(pair[1])
    else:
        source_tf = getattr(frame, "timeframe", "")
        htf_label, ltf_label = (source_tf, source_tf) if source_tf else ("30m", "3m")
    if htf_df is None:
        try:
            htf_df = resample(adf, htf_label) if len(adf) > 60 else None
        except Exception:
            htf_df = None
    if ltf_df is None:
        ltf_df = adf

    src = {layer: _clean(source) for layer, source in (frames or {}).items()}
    if htf_df is not None and Layer.EXTERNAL not in src:
        src[Layer.EXTERNAL] = htf_df
    if Layer.INTERNAL not in src:
        src[Layer.INTERNAL] = ltf_df if ltf_df is not None else adf

    def layer_frame(layer):
        f = src.get(layer)
        if f is None or not len(f):
            return adf
        return f

    on = set(layers) if layers else LAYERS[mode]
    hist = df  # retain every candle the caller loaded; never truncate chart history
    step = _step(hist)
    step_min = step.total_seconds() / 60.0 if step and step > pd.Timedelta(0) else 1.0
    need = max(chart_minutes(htf_label) * HTF_VISIBLE,
               chart_minutes(ltf_label) * LTF_VISIBLE)
    bars = int(min(WINDOW_CAP, max(int(bars), round(need / max(step_min, 1e-4)))))
    bars = max(20, min(bars, len(hist)))
    view = hist.tail(bars)
    long = direction == "long"
    last_x = hist.index[-1]
    has_plan = bool(plan and "plan" in on)
    plot_step = step if step > pd.Timedelta(0) else pd.Timedelta(minutes=1)
    plan_start = last_x + plot_step
    if has_plan:
        # Reserve a fixed fraction of the initial viewport, including for sparse markets.
        right = last_x + max(last_x - view.index[0], plot_step * 4) * 0.30
        plan_end = right - (right - last_x) * 0.06
    else:
        right = _future(last_x, step, 6)
    structure_end = plan_start - plot_step * 0.10 if has_plan else right
    start = hist.index[0]
    legend_rows = []

    fig = make_subplots(rows=1, cols=1)
    gutter_ref, gutter_prop = _gutter(fig)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"],
        close=hist["close"],
        increasing_line_color=UP, increasing_line_width=1.2, increasing_fillcolor=UP,
        decreasing_line_color=DN, decreasing_line_width=1.2, decreasing_fillcolor=DN,
        whiskerwidth=0.6, showlegend=False), row=1, col=1)

    swings = _swings(frame)
    pts_by_layer = {}
    tags_by_layer = {}
    idx_by_layer = {}
    for layer in (Layer.EXTERNAL, Layer.INTERNAL):
        pts = [s for s in swings if _lay(s) == layer.value]
        pts_by_layer[layer] = pts
        source = layer_frame(layer)
        atr_ref = float((source["high"] - source["low"]).tail(50).median())
        eq_ratio = cfg.struct.eq_tol_atr if cfg is not None else EQ_TOL_ATR
        tags_by_layer[layer] = _classify(pts, atr_ref, eq_ratio * atr_ref)
        idx_by_layer[layer] = {(int(_sv(s, "idx", -1)), _sv(s, "kind")): s
                               for s in pts}

    for layer in (Layer.EXTERNAL, Layer.INTERNAL):
        key = "ext_swing" if layer == Layer.EXTERNAL else "int_swing"
        f = layer_frame(layer)
        tags = tags_by_layer[layer]
        if key not in on:
            continue
        if layer == Layer.INTERNAL and chart_tf and chart_minutes(chart_tf) > chart_minutes(ltf_label):
            continue
        visible = [(f.index[idx], swing) for swing in pts_by_layer[layer]
                   if 0 <= (idx := int(_sv(swing, "idx", -1))) < len(f)
                   and view.index[0] <= f.index[idx] <= last_x]
        for ts, swing in visible[-LABEL_KEEP:]:
            tag = tags.get(id(swing), "")
            if tag:
                _pivot_label(fig, ts, float(_sv(swing, "price")), tag,
                             _sv(swing, "kind"), MUT)

    if "breaks" in on:
        for layer in (Layer.EXTERNAL, Layer.INTERNAL):
            f = layer_frame(layer)
            t_l = len(f) - 1
            layer_events = [e for e in getattr(frame, "events", [])
                            if _lay(e) == layer.value
                            and 0 <= int(_sv(e, "confirm_idx", -1)) <= t_l]
            shift = max((int(_sv(e, "confirm_idx")) for e in layer_events
                         if _lab(e) == "CHoCH"), default=-1)
            visible_events = [e for e in _live(frame, t_l) if _lay(e) == layer.value
                              and 0 <= int(_sv(e, "swing_idx", -1)) < len(f)
                              and 0 <= int(_sv(e, "break_idx", -1)) < len(f)
                              and start <= f.index[int(_sv(e, "break_idx"))] <= structure_end]
            bos = [e for e in visible_events if _lab(e) in BOS_LABELS
                   and int(_sv(e, "confirm_idx")) > shift]
            bos = thin([{"ts": f.index[int(_sv(e, "break_idx", 0))],
                         "level": float(_sv(e, "level")), "e": e} for e in bos],
                       0.0)
            w, col_line = STYLE[layer]
            tags = tags_by_layer[layer]
            by_idx = idx_by_layer[layer]
            marks = bos[-keep:]
            marks += [{"e": e, "level": float(_sv(e, "level"))} for e in visible_events
                      if _lab(e) == "CHoCH" and int(_sv(e, "confirm_idx")) == shift]
            labels = []
            tf = htf_label if layer == Layer.EXTERNAL else ltf_label
            layer_name = "HTF" if layer == Layer.EXTERNAL else "LTF"
            source_minutes = chart_minutes(tf)
            source_period = pd.Timedelta(minutes=source_minutes)
            finer_display = chart_tf is not None and step_min < source_minutes
            bars_per_source = source_minutes / step_min
            aligned_bars = finer_display and np.isclose(bars_per_source, round(bars_per_source))
            for row in marks:
                e = row["e"]
                break_side = _sv(e, "direction", "")
                swing_idx = int(_sv(e, "swing_idx"))
                pivot = by_idx.get((swing_idx, "H" if break_side == "bull" else "L"))
                tag = tags.get(id(pivot), "") if pivot is not None else ""
                text = f"{break_side.upper()} {_lab(e)} {row['level']:,.6g}"
                if tag:
                    text += f" {tag}"
                labels.append(text)
                wick_time = f.index[swing_idx]
                wick_column = "high" if break_side == "bull" else "low"
                if aligned_bars:
                    first, end = hist.index.searchsorted([wick_time, wick_time + source_period])
                    matching = hist.iloc[first:end]
                    exact = np.isclose(matching[wick_column], row["level"], rtol=1e-10, atol=0)
                    if exact.any():
                        wick_time = matching.index[np.flatnonzero(exact)[0]]
                _struct(fig, wick_time, row["level"], structure_end, col_line, w,
                        dash="dash" if _lab(e) == "CHoCH" else "solid")
                if wick_time in hist.index and np.isclose(
                        hist.at[wick_time, wick_column], row["level"], rtol=1e-10, atol=0):
                    fig.add_trace(go.Scatter(
                        x=[wick_time], y=[row["level"]], mode="markers",
                        marker=dict(symbol="circle-open", size=7, color=col_line),
                        hovertemplate=(f"{layer_name} {tf} swing wick: "
                                       f"{row['level']:,.6g}<extra></extra>"),
                        showlegend=False), row=1, col=1)
                confirmed = int(_sv(e, "confirm_idx"))
                source_start = f.index[confirmed]
                close_time = source_start + source_period
                fig.add_trace(go.Scatter(
                    x=[close_time if finer_display else source_start],
                    y=[row["level"]], mode="markers",
                    marker=dict(symbol="diamond", size=6, color=col_line),
                    hovertemplate=(f"{layer_name} {tf} · {text}<br>"
                                   f"Broken swing wick: {row['level']:,.6g}<br>"
                                   f"{close_time.tz_convert('UTC'):%d %b %H:%M:%S} UTC close: "
                                   f"{f['close'].iat[confirmed]:,.6g}<extra></extra>"),
                    showlegend=False), row=1, col=1)
            state = frame.state_at(layer, t_l) if hasattr(frame, "state_at") else "unclear"
            trend = {"up": "bullish", "down": "bearish"}.get(state, "unconfirmed")
            if labels or state in ("up", "down"):
                legend_rows.append(f'<span style="color:{col_line}">{layer_name} {tf} {trend}</span>'
                                   + (f' · {" &nbsp; | &nbsp; ".join(labels)}' if labels else ""))

    profiles = {}
    if "vp" in on:
        for source, window, color, side in (
            (htf_df, bars * 2, VP_HTF, "htf"),
            (ltf_df, bars, VP_LTF, "ltf"),
        ):
            if source is None or (side == "htf" and len(source) < 30):
                continue
            try:
                profiles[side] = _profile(fig, source, window, color, gutter_ref, side) or {}
            except Exception:
                continue

    zone_levels = []
    if "zones" in on:
        from config import CFG

        zone_cfg = cfg if cfg is not None else CFG
        for source, tf, label, left, right_bars in (
            (htf_df, htf_label, "HTF", zone_cfg.struct.ext_left, zone_cfg.struct.ext_right),
            (ltf_df, ltf_label, "LTF", zone_cfg.struct.int_left, zone_cfg.struct.int_right),
        ):
            try:
                zones = chart_zones(source, zone_cfg, left, right_bars, keep=zone_keep)
            except Exception:
                continue
            labels = []
            poc = profiles.get(label.lower(), {}).get("poc")
            for zone in zones:
                if zone["ts"] >= structure_end:
                    continue
                text = _zone(fig, zone, max(zone["ts"], start), structure_end, label, tf)
                if poc is not None and zone["low"] <= poc <= zone["high"]:
                    text += " (contains POC)"
                labels.append(text)
                zone_levels.extend((zone["low"], zone["high"]))
            if labels:
                poc_label = f" · POC {poc:,.6g}" if poc is not None else ""
                legend_rows.append(f'{label} {tf} · {" &nbsp; | &nbsp; ".join(labels)}'
                                   f'{poc_label}')

    if "fvg" in on:
        for g in [g for g in smc.fair_value_gaps(adf)
                  if _sv(g, "dir") == ("bull" if long else "bear")][:2]:
            idx = int(_sv(g, "idx", 0))
            if adf.index[idx] < start:
                continue
            fig.add_shape(type="rect", xref="x", yref="y", x0=adf.index[idx],
                          x1=structure_end, y0=float(_sv(g, "bot")), y1=float(_sv(g, "top")),
                          line=dict(width=0),
                          fillcolor="rgba(22,199,132,0.05)" if long
                          else "rgba(234,57,67,0.05)")

    if "sweeps" in on:
        for s in [s for s in smc.liquidity_sweeps(adf)
                  if _sv(s, "dir") == ("bull" if long else "bear")][:3]:
            idx = int(_sv(s, "idx", 0))
            if adf.index[idx] < start:
                continue
            fig.add_trace(go.Scatter(x=[adf.index[idx]],
                                     y=[float(_sv(s, "level"))],
                                     mode="markers",
                                     marker=dict(symbol="x", size=9, color=WARN),
                                     showlegend=False), row=1, col=1)

    if "avwap" in on:
        try:
            pool = [s for s in swings if _lay(s) == Layer.EXTERNAL.value
                    and _sv(s, "kind") == ("L" if long else "H")
                    and int(_sv(s, "confirmed_at", 0)) <= len(adf) - 1]
            if pool:
                line = anchored_vwap(adf, int(_sv(pool[-1], "idx"))).tail(len(hist))
                fig.add_trace(go.Scatter(x=line.index, y=line, name="AVWAP",
                                         line=dict(color=TEAL, width=1.2, dash="dash"),
                                         opacity=.75), row=1, col=1)
        except Exception:
            pass

    plan_levels = []
    if has_plan:
        entry = plan.get("entry")
        stop = plan.get("stop")
        targets = [target for target in (plan.get("tp1"), plan.get("tp2"))
                   if target is not None and np.isfinite(target) and entry is not None
                   and (target > entry if long else target < entry)]
        target = (max(targets) if long else min(targets)) if targets else None
        for level, fill, edge, name in (
            (stop, "rgba(234,57,67,0.36)", DN, "Stop-loss risk"),
            (target, "rgba(22,199,132,0.36)", UP, "Take-profit reward"),
        ):
            if (entry is None or level is None or not np.isfinite(entry)
                    or not np.isfinite(level) or entry == level):
                continue
            if name == "Stop-loss risk" and (level > entry if long else level < entry):
                continue
            lo, hi = sorted((entry, level))
            fig.add_shape(type="rect", xref="x", yref="y", x0=plan_start, x1=plan_end,
                          y0=lo, y1=hi, fillcolor=fill, line=dict(color=edge, width=1.5),
                          name=name, layer="below")
        plan_labels = []
        for y, col, txt in ((plan.get("entry"), "#93b7ff", "Entry"),
                            (plan.get("stop"), "#ff8a90", "SL"),
                            (plan.get("tp1"), "#70e6ae", "TP1"),
                            (plan.get("tp2"), "#70e6ae", "TP2")):
            if y is None or not np.isfinite(y):
                continue
            plan_levels.append(y)
            fig.add_shape(type="line", xref="x", yref="y", x0=plan_start, x1=plan_end,
                          y0=y, y1=y, line=dict(color=col, width=1.2))
            plan_labels.append(f'<span style="color:{col}">{txt} {y:,.6g}</span>')
        if plan_labels:
            fig.add_annotation(x=PRICE_DOMAIN[1] - 0.012, y=0.98,
                               xref="paper", yref="paper", xanchor="right", yanchor="top",
                               text="<br>".join(plan_labels), align="left", showarrow=False,
                               font=dict(size=11), bgcolor=BG, bordercolor="#38445a",
                               borderwidth=1, borderpad=6)

    # Keep the history in the trace. Start planned trades at a readable price
    # scale and offer a one-click fit for older candles.
    y_ref = view if has_plan else hist
    lo = min([float(y_ref["low"].min()), *zone_levels, *plan_levels])
    hi = max([float(y_ref["high"].max()), *zone_levels, *plan_levels])
    pad = (hi - lo) * 0.055 or max(abs(hi), 1.0) * 0.001
    y_range = [lo - pad, hi + pad]
    history_y = [min(y_range[0], float(hist["low"].min()) - pad),
                 max(y_range[1], float(hist["high"].max()) + pad)]
    x_lo = view.index[0]

    for index, text in enumerate(legend_rows):
        _legend_row(fig, text, index)

    fig.update_layout(height=640 + 17 * len(legend_rows), template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=BG, margin=dict(l=62, r=10, t=100, b=45 + 17 * len(legend_rows)),
                      showlegend=False, hovermode="x unified", dragmode="pan",
                      uirevision=f"layout-2|{symbol or 'unknown'}|{chart_tf or ''}|{htf_label}|{ltf_label}",
                      title=dict(text=f"chart {chart_tf or ''} · structure "
                                      f"{htf_label} (HTF) / {ltf_label} (LTF) · "
                                      f"gutter blue = HTF, orange = LTF",
                                 x=0, xanchor="left", font=dict(size=10, color=MUT)),
                      xaxis=dict(domain=PRICE_DOMAIN, type="date",
                                 showgrid=False, range=[x_lo, right],
                                 autorange=False, nticks=8,
                                 tickformat=_tick(chart_tf),
                                 rangeslider=dict(visible=False),
                                 fixedrange=False, constrain="range",
                                 showspikes=True, spikemode="across",
                                 spikethickness=1, spikedash="dot",
                                 spikecolor=MUT))
    fig.update_layout(**{gutter_prop: dict(domain=PROFILE_DOMAIN, anchor="y",
                                          range=[0, 1], type="linear",
                                          showgrid=False, showticklabels=False,
                                          zeroline=False, fixedrange=True)})
    if y_range:
        fig.update_layout(yaxis=dict(range=y_range, autorange=False,
                                     fixedrange=False))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#1b1f28", side="left",
                     tickformat=",.6g", fixedrange=False, showspikes=True,
                     spikemode="across", spikethickness=1, spikedash="dot",
                     spikecolor=MUT)
    from chart_axes import add_independent_axis_controls
    if y_range is not None:
        add_independent_axis_controls(fig, [x_lo, right], y_range,
                                      history=([hist.index[0], right], history_y)
                                      if has_plan else None)
    return fig
