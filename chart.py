"""chart.py - renderer shim. Builds the frame from cfg, then defers to chart2.

Kept so every existing call site (SMC tab, setups board) renders the same chart
with the same pan/zoom behaviour, structure lines and zones. `cfg` is the second
positional argument, `plan` may use "sl" or "stop".
"""
import streamlit as st

import chart2
import core.structure2 as s2
from core.models import StructureFrame
from core.tfs import CHART_TFS

DEFAULT = "30m"


def st_selectbox(key: str, index: int = 0, default: str = DEFAULT) -> str:
    idx = index if 0 <= index < len(CHART_TFS) else CHART_TFS.index(DEFAULT)
    return st.selectbox("Chart timeframe", CHART_TFS, index=idx, key=key,
                        help="Display only. Structure uses the auto pair.")


def legend_md() -> str:
    return """
**Reading the map**

- Yellow lines: higher-timeframe structure, from the pivot wick to the right edge.
- Cyan lines: lower-timeframe structure, same anchoring.
- `BOS` continues the trend. `CHoCH` is the first break against it.
- Rose zones are supply, mint zones are demand, both labelled with timeframe,
  multiple, price band and traded volume.
- Blue profile is HTF context, orange is LTF trigger. Both live in the right
  gutter and never overlap the candles.
- Drag to pan, scroll or drag a box to zoom, double-click to reset. X and Y zoom
  independently.
"""


def build(df, cfg, direction, plan=None, chart_tf=None, analysis_df=None,
          htf_df=None, ltf_df=None, pair=None, bars=chart2.DEFAULT_BARS):
    adf = analysis_df if analysis_df is not None else df
    frame = StructureFrame()
    try:
        frame = s2.run(adf, "", str(chart_tf or ""), cfg)
    except Exception:
        pass
    p = None
    if plan:
        p = {"entry": plan.get("entry"),
             "stop": plan.get("stop", plan.get("sl")),
             "tp1": plan.get("tp1"), "tp2": plan.get("tp2")}
    return chart2.build(df, frame, direction, p, mode="analysis", bars=bars,
                        chart_tf=chart_tf, analysis_df=adf, htf_df=htf_df,
                        ltf_df=ltf_df, pair=pair, cfg=cfg)
