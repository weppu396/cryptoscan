# CryptoScan pre-deployment hardening — 2026-11-25

## New fixes in this ZIP
- `data.py`: CCXT historical OHLCV pagination now walks forward from a bounded start timestamp, deduplicates progress and surfaces initial exchange errors rather than treating an outage as no market data.
- `auth.py`: atomic write for local user database, with corrupted JSON failing closed instead of silently resetting all accounts.
- `.gitignore`: exclude `.streamlit/secrets.toml`, `.env*`, virtual environments and SQLite databases from new Git commits.
- `tests/test_predeploy_data.py`: regression coverage for forward pagination and API outages.

## Local verification
- `python -m compileall -q .`: PASS.
- `pytest -q tests/test_predeploy_data.py`: 2 passed.
- `pytest -q --continue-on-collection-errors`: 76 passed, 2 skipped, 4 collection errors because Streamlit is not installed in this environment.
- Full installed-dependency Streamlit UI and exchange end-to-end checks: NOT RUN.

## Before Streamlit Cloud deployment
1. Commit this project to GitHub, including `.github/workflows/quality.yml` and `requirements.txt`.
2. Check GitHub Actions full test run is green. The workflow installs requirements on Python 3.11.
3. Deploy Streamlit Community Cloud with entry point `app.py`.
4. If login is enabled, set `ADMIN_USER`, `ADMIN_PASS` and `AUTH_ENABLED` in Streamlit app Secrets; never commit `secrets.toml`.
5. Verify exchange availability from the cloud region, a 100-coin scan, signal freshness, chart history and alerts. Some exchanges block datacenter IPs.
6. Cloud local filesystem is ephemeral: accounts, alerts and trade journals require external durable storage for reliable production use.
7. Do not enable real-money auto trading before paper-trading and live-data verification.

This build is NOT 100% production-validated and no profit is guaranteed.
