"""chart.py - renderer shim. One frame per layer, then defers to chart2.

The HTF frame feeds EXTERNAL only and the LTF frame feeds INTERNAL only, so an
opposing higher and lower timeframe can coexist on one chart. `cfg` is the
second positional argument; `plan` may use "sl" or "stop".
"""

import chart2
import core.structure2 as s2
from core.models import Layer, StructureFrame
from core.tfs import CHART_TFS

DEFAULT = "30m"


def st_selectbox(key: str, index: int = 0, default: str = DEFAULT) -> str:
    idx = index if 0 <= index < len(CHART_TFS) else CHART_TFS.index(DEFAULT)
    import streamlit as st
    return st.selectbox("Chart timeframe", CHART_TFS, index=idx, key=key,
                        help="Display only. Structure uses the auto pair.")


def legend_md() -> str:
    return """
**Reading the map**

- Yellow lines: higher-timeframe structure, from the pivot wick to the right edge.
- Cyan lines: lower-timeframe structure, same anchoring. The two run independently.
- `BOS` continues that layer's trend. `CHoCH` is the first break against it.
- `HH` / `HL` / `LH` / `LL` label each confirmed pivot; `EQH` / `EQL` mark equals.
- Rose zones are supply, mint zones are demand, labelled with timeframe,
  multiple, price band and traded volume.
- Blue profile is HTF context, orange is LTF trigger. Both live in the right
  gutter and never overlap the candles.
- Drag to pan, scroll or drag a box to zoom, double-click to reset. X and Y zoom
  independently.
"""


def _layer_frame(df, cfg, tf):
    if df is None or not len(df):
        return None
    try:
        return s2.run(df, "", tf, cfg)
    except Exception:
        return None


def merge_frames(ext_df, int_df, cfg, htf_tf: str, ltf_tf: str) -> StructureFrame:
    """One frame, two independent sources. External never leaks into internal."""
    frame = StructureFrame()
    ext = _layer_frame(ext_df, cfg, htf_tf)
    inn = _layer_frame(int_df, cfg, ltf_tf)
    for f, layer in ((ext, Layer.EXTERNAL), (inn, Layer.INTERNAL)):
        if f is None:
            continue
        frame.events.extend(e for e in f.events if e.layer == layer)
        frame.swings.extend(s for s in f.swings if s.layer == layer)
        frame.state[layer.value] = f.state.get(layer.value, [])
    base = inn or ext
    if base is not None:
        frame.ranging = list(base.ranging)
    frame.events.sort(key=lambda e: (e.confirm_idx, e.layer.value))
    return frame


def build(df, cfg, direction, plan=None, chart_tf=None, analysis_df=None,
          htf_df=None, ltf_df=None, pair=None, bars=chart2.DEFAULT_BARS, symbol=None):
    adf = analysis_df if analysis_df is not None else df
    htf_tf, ltf_tf = pair if pair else ("30m", "3m")
    frame = merge_frames(htf_df, ltf_df, cfg, htf_tf, ltf_tf)
    frames = {}
    if htf_df is not None and len(htf_df):
        frames[Layer.EXTERNAL] = htf_df
    if ltf_df is not None and len(ltf_df):
        frames[Layer.INTERNAL] = ltf_df
    p = None
    if plan:
        p = {"entry": plan.get("entry"),
             "stop": plan.get("stop", plan.get("sl")),
             "tp1": plan.get("tp1"), "tp2": plan.get("tp2")}
    return chart2.build(df, frame, direction, p, mode="analysis", bars=bars,
                        chart_tf=chart_tf, analysis_df=adf, htf_df=htf_df,
                        ltf_df=ltf_df, pair=pair, cfg=cfg, frames=frames, symbol=symbol)
