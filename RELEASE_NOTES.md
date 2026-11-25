# CryptoScan — Seven-strategy integrated development build

## Delivered
- Retained original Spot/Bot Streamlit desks and existing chart, portfolio, paper trading, risk, SMC, alerts and backtest modules.
- Added `strategy/multistrategy.py` with seven distinct rule-based candidate strategies: BOS Retest, Trend Pullback, Momentum Breakout, Range Reversal, Liquidity Sweep, EMA Reclaim, Volatility Squeeze.
- Integrated all seven with the Bot's `cached_scan` and CLI `scanner.scan` pipeline, preserving rating, SMC, AVWAP, risk and MTF filters.
- Added Bot opportunity comparison table: symbol, strategy, side, status, grade, entry, stop, targets, weighted R:R, gross TP2 potential percentage; and a strategy filter.
- Added targeted unit tests in `tests/test_multistrategy.py` and low-ADX exception for range-reversal regime checks.
- Carried forward security, lifecycle, backtest partial-exit, cache, zone and BOS retest fixes from the earlier partial build.

## Test evidence (this environment)
- `python -m compileall -q .`: PASS.
- `pytest -q tests/test_multistrategy.py tests/test_audit_regressions.py tests/test_plan_math.py`: **16 passed**.
- Broader targeted test run: **15 passed, 8 failed** (pre-existing BOS/structure2 logic and faulty list-based fixtures). See `TEST_RESULTS_FINAL.txt`.
- Full suite cannot collect `tests/test_chart_axes.py` because `streamlit` is not installed in this environment. `ccxt` is also unavailable; live exchange/UI tests not executed.

## IMPORTANT: NOT PRODUCTION / LIVE-TRADING READY
- The original request's exact seven strategy specifications were not available in the ZIP. The seven named rules above are explicit provisional implementations, not a verified reconstruction of previously agreed strategies.
- Seven strategies have rule-based detection and scanner integration; **they do not yet have independently validated historical performance**. The existing backtester is still primarily BOS-based; do not interpret the table's gross target potential as expected profit.
- Remaining known failures: BOS/structure2 tests; full Streamlit/UI and exchange integration untested; candle cache/pagination exchange-specific behavior; chart history pan issue not end-to-end verified; backtest intrabar ordering and unrealized equity modeling.
- No automatic real-money execution enabled. Options deliberately deferred.

## Run
Install `pip install -r requirements.txt -r requirements-test.txt` and set `ADMIN_PASS` securely, then `python -m pytest -q` and `streamlit run app.py`. Do not connect live capital without completing the above acceptance gates.
