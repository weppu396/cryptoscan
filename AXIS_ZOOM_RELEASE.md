# Independent X/Y zoom and candle history

- Both detail and structure charts now have **Time +**, **Time −**, **Price +**, **Price −**, and **Reset** controls.
- Time controls change only X axis. Price controls change only Y axis. Both axes permit direct Plotly axis dragging; pan remains the default gesture and the Plotly modebar offers box zoom.
- The full loaded candle trace is retained; the initial X viewport shows the recent section, so panning or zooming out reveals older loaded candles.
- The Y-axis range covers all loaded candle highs/lows to prevent old candles from becoming invisible after horizontal panning. Use Price + to inspect a tighter price region.
- Buttons use the **initial chart range** as their reference, rather than cumulative zoom; direct dragging supports continuous adjustments.
- The chart is built with Plotly, not the TradingView engine; its gestures and appearance are not identical to TradingView.
- **No automatic older-history fetch** exists beyond candles already downloaded from the exchange. Chart history depth remains constrained by exchange API and data loader limits.

## Offline verification
`python -m pytest -q tests/test_independent_axis_zoom.py tests/test_chart_axes.py tests/test_layers_history.py` -> 29 passed.
Python compilation passed. Cloud UI and exchange behavior remain unverified.
