# CryptoScan — deployment candidate preflight

## Local checks
- `python -m compileall -q .`: passed.
- Offline-collectable regression suite: **66 passed, 1 skipped**.
- Full test suite cannot be verified here because Streamlit is not installed. No live exchange or cloud deployment test has been performed.

## Additional fixes in this release
- `data.py`: reject invalid/NaN/infinite quote volume and non-spot/non-swap markets in liquidity universe; cap at 100; raise on later-page OHLCV fetch failure rather than silently returning truncated history.
- `app.py`: apply safe ticker parsing in interactive scanner, preserve already-closed historical candles, default Actionable only to checked.
- `tests/test_final_predeploy.py`: regression coverage for top-100 selection and incomplete candle pagination.

## Important deployment conditions
- Run GitHub Actions `CryptoScan cloud quality checks` with Python 3.11 and all dependencies. It must pass before claiming full automated validation.
- Streamlit Cloud entrypoint is `app.py`. Exchange access from its cloud region may be restricted.
- Cloud local files are **ephemeral**. SQLite order/alert state and local account/portfolio data are not reliable across restarts. Production use needs an external durable database.
- `config.py` currently defaults to `AUTH_ENABLED = False`: public deployment is unauthenticated unless enabled. Enable authentication and configure `ADMIN_PASS` in Streamlit secrets before exposing sensitive data. Never commit secrets.
- Scanner grades and target profit percentages are heuristic estimates, not proven outcomes. Verify live spread/depth, run paper trading and out-of-sample backtests before real-money use.
- Do not enable automated real-money execution based on this release.

## Cloud smoke-test order
1. Push source (not local state/secrets) to GitHub; confirm Actions is green.
2. Deploy `app.py` on Streamlit Cloud, Python 3.11.
3. Check exchange connectivity and scan 10 symbols, then 100.
4. Verify stale/expired setups are hidden, rank order and chart pan/history persist.
5. Restart the app and confirm persistence requirements before relying on accounts/alerts.
