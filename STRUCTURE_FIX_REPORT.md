# CryptoScan pre-deployment BOS/CHoCH and fixture fixes

## Changes
- `core/structure2.py`: confirm first-in-plateau wick pivots using strict left-side and non-strict right-side comparisons. Fixes missed bullish/bearish BOS on consecutive candles sharing the same wick high/low; right-side confirmation delay remains enforced.
- `tests/test_structure2.py`: replaced impossible flat/list fixture expectations with DataFrame OHLC series that actually form confirmed swings and breaks. Added meaningful BOS_INIT, CHoCH, invalidation, consumed-level and layer-isolation checks. No artificial forced trend state.
- `tests/test_alerts2.py`: retry test now advances a mocked clock past the configured backoff instead of expecting an immediate retry.
- `data.py`: CCXT imported only when opening an exchange, allowing non-network timeframe tests to run without CCXT installed.

## Test evidence (this environment)
- `pytest -q tests/test_bos_sequence.py tests/test_structure2.py tests/test_alerts2.py`: **16 passed**.
- `pytest -q --continue-on-collection-errors`: **74 passed, 2 skipped, 4 collection errors**.
- All four collection errors are `ModuleNotFoundError: streamlit` from chart/spot tests. No remaining failures in tests that were collected.
- Python 3.13 was used locally; GitHub Actions should run the project's Python 3.11 dependency install and complete suite.

## Important limitations
- These are synthetic regression tests, not a guarantee of BOS/CHoCH accuracy on live markets. The plateau rule chooses the earliest qualifying bar and needs validation against real chart examples.
- Streamlit UI tests, exchange connectivity, full seven-strategy out-of-sample backtests, scanning latency and cloud persistence are not verified here.
- Keep real-money automation disabled. This is a pre-deployment validation build, not a 100% certified live-trading system.
