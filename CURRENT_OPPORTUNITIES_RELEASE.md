# CryptoScan — Current Opportunities Release

## What this release actually implements

- Bot scans all seven configured strategy families; `Scan every eligible symbol` selects the complete eligible exchange universe rather than the default limited scan. This may take a long time and exchange rate limits apply. Not all worldwide coins are listed on one exchange.
- Bot scanner ranks by A+/A/B/C/D **first**, then heuristic confluence score, weighted R:R and gross upside. `Show only the best strategy per coin` is enabled by default.
- Bot only displays **ACTIONABLE** and **BETTER_PRICE** candidates. Historical/completed/expired, stop-touched, TP1-touched, extended and invalid candidates are excluded. The entire closed-candle path after the signal is checked, not only the last close.
- Scans must have recent market candles; cached scanner results older than two LTF bars are hidden until rescanned.
- Every candidate retains strategy, direction, planned entry, stop, TP1, TP2, weighted R:R, grade, heuristic score and potential *gross* target percentage. The latter is not expected profit or a probability.
- CLI: `python run_scan.py --top 0 --recent 2 --actionable-only` (all eligible markets; slow). `--top 100` limits to 100.
- Spot remains a separate desk; options are intentionally not included. Automatic real-money order placement has not been enabled.

## Verification performed in this environment

- `python -m compileall -q .` — PASS.
- `pytest -q tests/test_opportunities.py tests/test_plan_math.py tests/test_lifecycle.py tests/test_readiness.py` — **25 passed**.
- `pytest -q tests/test_bos_sequence.py tests/test_structure2.py tests/test_plan_math.py` — **8 failed, 9 passed**. Existing BOS/structure tests require further reconciliation. Some structure tests misuse `tests.fixtures.flat` (returns a list, not a DataFrame); other failures are actual structure expectations.
- `pytest -q tests/test_opportunities.py tests/test_plan_math.py tests/test_lifecycle.py tests/test_alerts2.py` — **1 failed, 23 passed**. Alert retry test expects immediate retry but code implements retry backoff; needs explicit contract decision.
- Streamlit, CCXT and PyArrow unavailable in the current environment; full UI and exchange integration tests could not be run.

## Important limitations / not 100% verified

1. This is not a live exchange-tested, production-ready or proven-profitable trading system. No claim of 100% correctness.
2. The seven strategies are deterministic rule hypotheses; each needs out-of-sample validation, slippage, fee and liquidity checks.
3. Scores are heuristic, not independently calibrated win probabilities. Ranking by potential gross target percentage alone would be misleading.
4. No system can guarantee future profits or predict the best strategy with certainty.
5. A candle-close signal may cease to be executable between exchange refreshes. Verify live price, order book, spread, volume and position risk before any trade.
6. Historical structure, chart-history/panning and alert failures remain outside this release's verified scope.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Choose **Bot > Scan**, choose **Scan every eligible symbol** to cover the exchange's eligible universe, click **Run scan**, and leave **Show only the best strategy per coin** checked. Refresh the scan before using the result. No setup is displayed when none passes strict entry checks.
