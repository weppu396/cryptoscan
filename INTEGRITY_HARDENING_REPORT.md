# CryptoScan — Integrity Hardening Build

## Changes in this build
- `core/lifecycle.py`: atomic SQLite order/position updates using `BEGIN IMMEDIATE`, commit/rollback, and transition validation in the same transaction.
- Reject negative, zero, nonfinite, overfilled, and overclosed quantities.
- Reject cancel of filled/terminal orders without corrupting state.
- Prevent replacing existing order IDs and compute weighted average position entry across partial fills.
- `tests/test_lifecycle_integrity.py`: four additional regression tests.

## Verification performed
- `python -m compileall -q .`: PASS.
- `pytest -q tests/test_lifecycle_integrity.py tests/test_lifecycle.py`: 9 passed.
- Broad available-dependency suite: 62 passed, 10 failed, 1 skipped.
- Four other test modules cannot be collected: Streamlit is not installed.
- CCXT and PyArrow are also unavailable; installing from PyPI failed because this environment cannot resolve PyPI.
- No live-exchange, production deployment, or end-to-end Streamlit test was performed.

## Remaining known failures
- `tests/test_bos_sequence.py`: 3 BOS cases fail to produce expected events.
- `tests/test_structure2.py`: 1 trend state failure; 4 tests call `.loc` or `.iloc` on list-valued fixtures.
- `tests/test_alerts2.py`: retry-backoff implementation conflicts with a test expecting immediate retry.
- `tests/test_reversal.py`: CCXT import unavailable.
- Four Streamlit-dependent test modules fail during collection.

## Trading caveats
The scanner identifies potential setups, not guaranteed profit. Seven strategy implementations and current-opportunity filtering are inherited from the prior development build, but they have not passed comprehensive historical walk-forward and live market validation. Automatic real-money trading must remain disabled. This build is NOT 100% complete or production-ready.

## Run with dependencies available
`pip install -r requirements.txt`
`streamlit run app.py`
`pytest -q`
