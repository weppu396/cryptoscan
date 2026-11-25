# CryptoScan completion status — 2026-11-25

This is an updated development build, **not a 100% complete or live-capital-ready trading system**.

## Changes in this iteration
- `run_backtest.py`: `--strategy` supports each of the seven strategy names; `--compare-strategies` independently evaluates all seven on the same history and prints trade counts, expectancy R, and drawdown.
- `backtest/engine.py`: retains the strategy name on each completed trade, including exported trade lists.
- Existing Spot and Bot functionality retained; options deliberately excluded.

## Verified locally
- `python -m compileall -q .` — passed.
- `python -m pytest -q tests/test_multistrategy.py tests/test_audit_regressions.py tests/test_plan_math.py` — 16 passed.
- Broader tests without UI dependencies in prior build — 15 passed, 8 failed. Failures include BOS sequence detection and structure2 fixtures/expectations.
- Full test collection blocked by missing `streamlit`. Also missing `ccxt` and `pyarrow`; live exchange and UI validation unavailable.

## Unfinished / safety gates
- Fix and validate BOS/CHoCH event logic against independent hand-verified fixtures, rather than modifying tests merely to make them green.
- Full Streamlit chart pan/zoom/history integration testing.
- Exchange-specific OHLCV pagination, stale data handling and real-world closed-candle checks.
- Independent historical out-of-sample/walk-forward validation for each of seven strategies. The added comparison is in-sample and not proof of edge.
- Slippage, spread, funding, fees, intrabar ordering, mark-to-market equity, portfolio exposure and execution failure simulations.
- Paper trading and monitored deployment before any real-money automation.

## How to run after dependencies are installed
```
pip install -r requirements.txt -r requirements-test.txt
python -m pytest -q
python run_backtest.py --symbol BTC/USDT:USDT --tf 5m --strategy 'Trend Pullback' --limit 6000
python run_backtest.py --symbol BTC/USDT:USDT --tf 5m --compare-strategies --limit 6000
streamlit run app.py
```
Set a strong `ADMIN_PASS` before starting the web UI. Never interpret gross TP potential as expected profit.
