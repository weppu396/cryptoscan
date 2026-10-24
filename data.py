"""
Fetches OHLCV candles from the exchange and caches them to disk.
Delete the data_cache folder any time to force a fresh download.
"""
import time
import pathlib
import ccxt
import pandas as pd

CACHE = pathlib.Path("./data_cache")
CACHE.mkdir(exist_ok=True)

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
}


def get_exchange(cfg):
    klass = getattr(ccxt, cfg.scan.exchange)
    return klass({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        "timeout": 30_000,
    })


def build_universe(ex, cfg) -> list:
    ex.load_markets()
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        print(f"[warn] tickers failed ({e}); falling back to top 20")
        tickers = {}

    rows = []
    for sym, m in ex.markets.items():
        if not m.get("active") or m.get("quote") != cfg.scan.quote:
            continue
        if m.get("base") in cfg.scan.exclude_suffixes:
            continue
        if not (m.get("swap") or m.get("spot")):
            continue
        t = tickers.get(sym) or {}
        qv = float(t.get("quoteVolume") or 0.0)
        if qv == 0.0 and t.get("baseVolume") and t.get("last"):
            qv = float(t["baseVolume"]) * float(t["last"])
        rows.append((sym, qv))

    rows = [r for r in rows if r[1] >= cfg.scan.min_quote_vol_24h]
    rows.sort(key=lambda r: -r[1])
    return [s for s, _ in rows][: cfg.scan.top_n_symbols]


def load(ex, symbol, timeframe, limit=1500, use_cache=True) -> pd.DataFrame:
    """Clean OHLCV frame indexed by UTC timestamp. Cache is re-fetched if it
    holds fewer bars than requested - this was silently truncating backtests."""
    safe = symbol.replace("/", "_").replace(":", "_")
    key = CACHE / f"{ex.id}_{safe}_{timeframe}.parquet"

    if use_cache and key.exists():
        try:
            cached = pd.read_parquet(key)
            if len(cached) >= limit:
                return cached
        except Exception:
            pass

    step = TF_MS[timeframe]
    since = ex.milliseconds() - step * limit
    rows = []

    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    if not df.empty:
        df.to_parquet(key)
    return df