# CryptoScan chart history / pan / zoom refinement

## Fixed
- The main chart (`chart2.py`) no longer truncates loaded OHLCV to the last 6,000 bars. Candlestick traces contain the full history provided to the renderer.
- Both chart views use the price range of the loaded history rather than a recent-only slice, so earlier candles are not clipped vertically when panning.
- The coin-detail chart no longer truncates loaded data to 1,200 candles; its gap detection inspects the full loaded history.
- Coin-detail history requests increased from 1,500 to 3,000 bars, subject to exchange availability and request limits.
- The analysis chart attempts to fetch up to 3,000 bars for the display timeframe rather than 800, without changing the structure-analysis frames.
- Plotly `uirevision` is stable across reruns for the same symbol/timeframe, but different for a different coin/timeframe, so panning/zooming survives refreshes without leaking another coin's viewport.
- `chart.py` lazily imports Streamlit only for its UI widget, making chart logic testable without Streamlit installed.
- Updated obsolete chart tests that assumed candles were truncated or that drag mode was zoom rather than pan.

## Verification
- `python -m pytest -q tests/test_chart_axes.py tests/test_chart_history_navigation.py tests/test_layers_history.py`: **30 passed**.
- Python compilation of modified chart modules passed.
- Full Streamlit UI and exchange-data tests require cloud deployment / dependencies. No live exchange validation was performed.

## Limitations
- Panning only reveals history **actually fetched**. Exchanges may not provide 3,000 bars for every timeframe or symbol; the UI does not dynamically fetch older candles when reaching the left edge.
- Price-axis scaling covers the loaded history to prevent old candles disappearing; a very wide historical price range can compress the currently visible candles. Manual Y-axis zoom remains available.
- More loaded candles increase memory, transfer time and browser rendering cost.
