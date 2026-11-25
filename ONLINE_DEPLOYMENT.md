# CryptoScan — Online-only Streamlit deployment

## Status before deployment
This package is **not certified production-ready**. The available offline test environment
reported 62 passed, 10 failed, 1 skipped, and 4 modules unable to collect because Streamlit
is not installed. The new GitHub Actions workflow runs the **full** suite with dependencies
installed and will correctly show red until defects are fixed. Do not interpret a working
Streamlit page as evidence of a correct trading system.

## Deploy without installing anything on your computer
1. Upload the **contents of `cryptoscan-main`** to your GitHub repository root (not the
   parent folder), or push them using GitHub's web interface. Keep `.github/workflows/quality.yml`.
2. Open the GitHub repository's **Actions** tab and run/check `CryptoScan cloud quality checks`.
   Do not ignore failed checks; copy their failure output for debugging.
3. In Streamlit Community Cloud, choose **Create app**, select the GitHub repository,
   branch `main`, and main file path `app.py`, then deploy.
4. Streamlit Cloud installs Python dependencies automatically from `requirements.txt`.
   No local Python/Streamlit/venv installation is required.
5. Check the Streamlit **Manage app / Logs** page if deployment fails; provide redacted logs.

## Security and limitations
- Do not commit API secrets, exchange keys, `.streamlit/secrets.toml`, `.env`, user files,
  `state/`, local SQLite databases, or portfolio/trade journals to a public repository.
- No ChatGPT connection is required for public market-data scans. An LLM integration is
  optional and should not be trusted to determine order safety.
- Streamlit Community Cloud may restart/sleep. Its local filesystem is not reliable durable
  storage. Persisting order state, alerts, or user data needs a proper external database;
  an in-memory UI/session or local SQLite file is not sufficient for unattended automation.
- Cloud hosting does not guarantee exchange connectivity. Geoblocking, API rate limits,
  ticker availability, and scan runtime must be tested from the deployed instance.
- Cap the universe at 100 eligible USDT symbols ranked by 24h volume. Volume alone does
  not guarantee low spreads or deep order books.
- Display only actionable opportunities after candle-close confirmation, freshness,
  invalidation and distance-to-entry checks. A+ is a model grade, not a guaranteed profit.
- Keep real-money auto-execution **disabled** until full tests, backtesting, monitoring,
  risk controls and explicit user approval are in place.
