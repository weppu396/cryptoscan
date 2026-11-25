# Liquidity-limited scanning

- Scanner maximum: 100 exchange-eligible USDT markets. Default 100.
- Rank by 24-hour quote trading volume (or base volume x last price if quote volume absent).
- Enforce selected minimum 24-hour volume; fewer than 100 is acceptable.
- If ticker volume is unavailable, show no unverified opportunities instead of guessing liquidity.
- Volume is not an order-book depth/spread guarantee.
- API scan can be slow; market data and spread checks require real exchange access.

## Local installation (Windows PowerShell)

1. Install Python 3.11 or 3.12 from python.org and enable Add Python to PATH.
2. Open PowerShell in the extracted `cryptoscan-main` folder.
3. `py -3.11 -m venv .venv` (or replace 3.11 with your installed version).
4. `.\.venv\Scripts\python.exe -m pip install --upgrade pip`
5. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
6. `.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt`
7. `.\.venv\Scripts\python.exe -m pytest -q`
8. `.\.venv\Scripts\python.exe -m streamlit run app.py`

Send the complete pytest output and any installation error (redact passwords, API keys and tokens).
No ChatGPT integration is required to run the scanner; exchange public-market-data access is required.
This build is not certified production-ready and must not be used for unattended real-money trading.
