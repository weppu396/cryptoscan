# CryptoScan audited working copy — partial implementation

## Implemented fixes
- `auth.py`: removed predictable default admin password; first deployment requires ADMIN_PASS.
- `core/lifecycle.py`: transactionally consistent order/position/event changes; rejects overfills, invalid quantities, duplicate order IDs, cancel of filled orders, and over-closes; weighted average fill entry.
- `strategy/setups.py`: BOS retest must happen after breakout candle.
- `strategy/risk.py`: R:R respects long/short direction; wrong-side targets do not earn positive R.
- `strategy/reversal.py` and `core/reversal.py`: consider inside/approaching zones, no unconfirmed CONFIRMED classification; basic missing-data handling.
- `core/structure_map.py`: bound event marker count.
- `backtest/engine.py`: reported trade R includes partial TP1 realization and entry/exit fees; exposes net_pnl.
- `data.py`: reject stale cached OHLCV and trim cache to requested limit.
- `core/pit.py`: optional bar_duration enables filtering out open-timestamp HTF candles; callers must pass it for correct behavior.
- `test_no_lookahead.py`: leaks now cause nonzero exit.
- `tests/test_audit_regressions.py`: six new deterministic regression checks.

## Not completed / do not trade live based on this build
- Seven independent strategies and a unified strategy-selection dashboard are **not yet implemented**. Exact agreed seven-strategy specification is not in this ZIP. Current scanner still primarily uses BOS retest.
- No live exchange test, Streamlit UI test, or full test suite pass. Streamlit and CCXT are not installed in the test environment.
- Known pre-existing failures remain in BOS/structure tests and fixture tests. See `AUDIT_TEST_RESULTS.txt`.
- OHLCV pagination still requires exchange-specific verification; no live market validation.
- Chart pan/history issue not verified end-to-end.
- Backtest still has intrabar ordering and unrealized-equity modeling limitations.
- Options deferred, as requested. Live real-money execution not authorized.

## Run locally
`pip install -r requirements.txt -r requirements-test.txt`
`python -m pytest -q`
`streamlit run app.py`
Set ADMIN_PASS before first launch. Do not use real funds before full validation.
