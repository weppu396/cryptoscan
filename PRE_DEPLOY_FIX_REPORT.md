# Pre-deployment fixes and verification

## Code changes
- `core/alerts2.py`: SQLite `BEGIN IMMEDIATE` atomic alert claim, prevents concurrent duplicate sends; PENDING is not re-eligible until stale recovery; retries respect backoff; explicit `now=0` is honored.
- `tests/test_cloud_safety.py`: regression coverage for duplicate claims, backoff, and top-100 liquidity cap (liquidity test runs when CCXT is installed).

## Verification
- Python compilation: PASS.
- Targeted tests: 11 passed, 1 skipped (CCXT unavailable).
- Full available-dependency suite: see `test_output.txt` in ZIP.

## Unresolved blockers (not concealed)
- BOS structure sequence fixtures and algorithm disagree; must review semantics rather than fabricate break events.
- `tests/fixtures.py` `flat()` returns a list while four tests treat it as a DataFrame; tests need correction to represent intended market behavior.
- Existing alerts test expects immediate retry, contrary to implemented backoff policy.
- Streamlit and CCXT are not installed in this execution environment, so complete UI/exchange tests cannot be run.
- End-to-end cloud deployment, market-data freshness, walk-forward results, and persistent cloud state remain unverified.

**Do not enable real-money automated orders.**
