"""
Candles + cache. Optional Cloudflare Worker proxy so Binance is reachable
from US datacentre IPs.

ROUTE THAT WORKS: binance SPOT via the proxy (api-gcp.binance.com).
Symbols look like BTC/USDT. Futures is attempted first and falls back
to spot inside the worker.
"""
from dataclasses import replace
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

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

BINANCE_HOSTS = sorted([
    "https://api-gcp.binance.com",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
    "https://fapi-gcp.binance.com",
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
], key=len, reverse=True)

# binance (spot) FIRST - it is the route that actually works through the proxy
CANDIDATES = [
    "binance",        # spot   -> BTC/USDT
    "binanceusdm",    # futures -> BTC/USDT:USDT
    "bybit",
    "okx",
    "bitget",
    "gateio",
    "kucoinfutures",
    "mexc",
    "htx",
]

DEFAULT_TYPE = {
    "binance": "spot",
    "binanceusdm": "future",
    "bybit": "swap",
    "okx": "swap",
    "bitget": "swap",
    "gateio": "swap",
    "kucoinfutures": "swap",
    "mexc": "swap",
    "htx": "swap",
}

BINANCE_CLASSES = ("binance", "binanceusdm", "binanceus", "binancecoinm")
_RESOLVED = {}


def _apply_proxy(urls, proxy):
    base = proxy.rstrip("/")
    for k, v in list(urls.items()):
        if isinstance(v, str):
            for h in BINANCE_HOSTS:
                if v.startswith(h):
                    urls[k] = base + v[len(h):]
                    break
        elif isinstance(v, dict):
            _apply_proxy(v, proxy)


def get_exchange(name, proxy=None):
    klass = getattr(ccxt, name)
    ex = klass({
        "enableRateLimit": True,
        "timeout": 30_000,
        "options": {"defaultType": DEFAULT_TYPE.get(name, "swap")},
        "headers": {"User-Agent": BROWSER_UA,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9"},
    })
    if proxy and name in BINANCE_CLASSES:
        _apply_proxy(ex.urls, proxy)
    return ex


def probe_exchanges(proxy=None):
    out = {}
    for name in CANDIDATES:
        t0 = time.time()
        try:
            ex = get_exchange(name, proxy)
            ex.load_markets()
            out[name] = {"ok": len(ex.markets) > 50,
                         "markets": len(ex.markets),
                         "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            out[name] = {"ok": False, "error": type(e).__name__,
                         "ms": int((time.time() - t0) * 1000)}
    return out


def probe_proxy(proxy):
    import requests
    try:
        return requests.get(proxy.rstrip("/") + "/__health", timeout=25).json()
    except Exception as e:
        return {"error": str(e)}


def resolve_exchange(cfg, preferred=None, proxy=None, verbose=False):
    key = f"{preferred}|{proxy}"
    if key in _RESOLVED:
        name = _RESOLVED[key]
        try:
            ex = get_exchange(name, proxy)
            ex.load_markets()
            if len(ex.markets) > 50:
                return ex, name
        except Exception:
            _RESOLVED.pop(key, None)

    order = [preferred] if preferred and preferred != "auto" else []
    order += [c for c in CANDIDATES if c not in order]

    last_err = None
    for name in order:
        try:
            ex = get_exchange(name, proxy)
            ex.load_markets()
            if len(ex.markets) > 50:
                _RESOLVED[key] = name
                if verbose:
                    print(f"[exchange] using {name}")
                return ex, name
        except Exception as e:
            last_err = e

    raise RuntimeError(f"no reachable exchange. last error: {last_err}")


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
    batch_limit = int(ex.options.get("fetchOHLCVLimit") or 1000)
    batch_limit = max(50, min(batch_limit, 1000))

    since = ex.milliseconds() - step * limit
    rows = []

    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step
        if len(batch) < batch_limit:
            break
        time.sleep(ex.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    if not df.empty:
        df.to_parquet(key)
    return df
