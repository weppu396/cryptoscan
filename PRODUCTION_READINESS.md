# CryptoScan production-readiness review — 2026-11-25

## Changes in this revision
- **Seven independent backtests in Streamlit:** choose a named strategy or compare all seven. The underlying `run_backtest.build_trades` uses strategy-specific candidates, so results are not silently pooled.
- **Anchored walk-forward:** expanding train slices, untouched test intervals, prior-bar indicator warmup, and out-of-sample trade filtering by `entry_ts`. This is a structural improvement, **not** evidence of profitability.
- **Backtest integrity:** invalid stop direction rejected rather than using `abs(stop-entry)`; partial exit fraction bounded to [0,1].
- **State durability groundwork:** `CRYPTOSCAN_STATE_DIR` can point to a mounted persistent volume on a host that supports one. Account JSON, paper trades, portfolio, and SQLite are stored under this directory; paper and portfolio writes use atomic replace.
- Added tests for independent strategies, stop validity, walk-forward time boundaries, and atomic local JSON persistence.

## Local verification
- Python source compilation: PASS.
- Offline-testable suite: 92 passed, 1 skipped (some modules require Streamlit and CCXT).
- The full dependency-enabled test suite and real Streamlit UI **have not been verified** in this environment.

## Known release blockers
1. **Streamlit Community Cloud has ephemeral local filesystem.** `CRYPTOSCAN_STATE_DIR` does not make it persistent there. Durable multi-user accounts, paper trades, alerts, and portfolio require an external database/object store with authentication, migrations and concurrency testing. This is NOT implemented. Do not put valuable user data in the cloud app until addressed.
2. **Live exchange connectivity and 100-symbol scan** have not been verified from the deployed region; venue access, rate limits, scan duration and ticker coverage can differ.
3. **Seven-strategy performance** has not been validated on sufficient out-of-sample symbols/regimes with fees, funding, slippage and realistic execution. A+ is a heuristic score, not a measured profit probability.
4. **Streamlit interactions** (axis drag zoom, chart history retention across reruns, cloud memory use, authenticated sessions) require real-browser end-to-end checks.
5. **Security:** `AUTH_ENABLED` is disabled by default. Public access to sensitive data must not be allowed without a durable identity system, secret management and access controls.
6. **Execution:** real-money automated order placement is not validated and must remain disabled.

## Safe cloud testing rollout
1. Upload project contents to the same GitHub branch used by the existing Streamlit app. Do not commit `state/`, `.env`, `.streamlit/secrets.toml`, local portfolios, or exchange keys.
2. Wait for GitHub Actions full pytest workflow to pass. A failure blocks promotion.
3. Test on a separate staging Streamlit app before updating the production URL.
4. Verify a single market, 10 markets, then max 100 liquid markets. Confirm no stale/expired signals and that chart X/Y axes and historical navigation work.
5. Run independent backtests and walk-forward across several regimes. Treat results as research, not live-ready profitability proof.
6. Add external persistent storage and authentication before enabling user accounts or persistent alerts on Community Cloud.

## Run instructions
- Entry point: `app.py`
- Dependencies: `requirements.txt`; Python 3.11 recommended.
- CLI independent strategy comparison: `python run_backtest.py --compare-strategies --symbol BTC/USDT:USDT --tf 5m --limit 6000`
- CLI OOS check: `python run_backtest.py --strategy 'Trend Pullback' --walk-forward --symbol BTC/USDT:USDT --tf 5m --limit 6000`

This is a **staging/deployment-test candidate**, not a verified production trading system.
