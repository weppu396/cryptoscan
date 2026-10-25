"""
Candles + cache, with optional Cloudflare Worker proxy for Binance.

Binance geo-blocks US datacentre IPs. Only api-gcp.binance.com is reachable
through Cloudflare. The proxy override is explicit, and symbols are
auto-mapped onto whatever the resolved exchange actually lists.
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

HTF_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
               "1h": 24, "2h": 12, "4h": 6, "6h": 4, "12h": 2, "1d": 1}

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

CANDIDATES = [
    "binance", "binanceusdm", "bybit", "okx", "bitget",
    "gateio", "kucoinfutures", "mexc", "htx",
]

DEFAULT_TYPE = {
    "binance": "spot", "binanceusdm": "future", "bybit": "swap",
    "okx": "swap", "bitget": "swap", "gateio": "swap",
    "kucoinfutures": "swap", "mexc": "swap", "htx": "swap",
}

BINANCE_CLASSES = ("binance", "binanceusdm", "binanceus", "binancecoinm")
_RESOLVED = {}
LAST_ERRORS = {}


def min_bars_needed(cfg, htf_need=230):
    return int(HTF_PER_DAY.get(cfg.scan.htf, 6) * htf_need) + 600


def _override_binance_urls(ex, proxy):
    base = proxy.rstrip("/")
    ex.urls["api"] = {
        "public": f"{base}/api/v3",
        "private": f"{base}/api/v3",
        "v3": f"{base}/api/v3",
        "sapi": f"{base}/sapi/v1",
        "fapiPublic": f"{base}/fapi/v1",
        "fapiPrivate": f"{base}/fapi/v1",
        "fapiPublicV2": f"{base}/fapi/v2",
        "fapiPrivateV2": f"{base}/fapi/v2",
        "fapiPublicV3": f"{base}/fapi/v3",
        "fapiPrivateV3": f"{base}/fapi/v3",
        "dapiPublic": f"{base}/dapi/v1",
        "dapiPrivate": f"{base}/dapi/v1",
    }


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
        _override_binance_urls(ex, proxy)
    return ex


def resolve_symbol(ex, wanted: str) -> str:
    """
    Map a loose symbol like BTC/USDT onto something this exchange lists.
    BTC/USDT <-> BTC/USDT:USDT are interchangeable across venues.
    """
    ex.load_markets()
    if wanted in ex.markets:
        return wanted

    base = wanted.split(":")[0]
    for cand in (f"{base}:USDT", f"{base}:USD", base):
        if cand in ex.markets:
            return cand

    b, _, q = base.partition("/")
    matches = [s for s, m in ex.markets.items()
               if m.get("base") == b and m.get("quote") == q and m.get("active")]
    if matches:
        matches.sort(key=lambda s: ":" not in s)
        return matches[0]

    raise KeyError(f"'{wanted}' not listed on {ex.id}")


def diagnose_binance(proxy):
    import requests
    out = {}
    if not proxy:
        return {"error": "no proxy URL set"}
    base = proxy.rstrip("/")

    for label, url in [
        ("worker_health", f"{base}/__health"),
        ("api_ping", f"{base}/api/v3/ping"),
        ("api_exchangeInfo", f"{base}/api/v3/exchangeInfo"),
    ]:
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": BROWSER_UA})
            out[label] = {"status": r.status_code, "bytes": len(r.text),
                          "head": r.text[:150]}
        except Exception as e:
            out[label] = {"error": f"{type(e).__name__}: {e}"}

    try:
        ex = get_exchange("binance", proxy)
        out["ccxt_url_public"] = ex.urls["api"].get("public")
        mk = ex.load_markets()
        out["ccxt_markets_loaded"] = len(mk)
        out["ccxt_symbol_sample"] = list(mk.keys())[:5]
    except Exception as e:
        out["ccxt_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    return out


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
    LAST_ERRORS.clear()

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
            LAST_ERRORS[name] = f"{type(e).__name__}: {str(e)[:200]}"

    raise RuntimeError("no reachable exchange. see LAST_ERRORS")


def build_universe(ex, cfg) -> list:
    ex.load_markets()
    try:
        tickers = ex.fetch_tickers()
    except Exception:
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


def list_symbols(ex, quote="USDT", limit=800):
    ex.load_markets()
    syms = sorted(s for s, m in ex.markets.items()
                  if m.get("active") and m.get("quote") == quote
                  and (m.get("swap") or m.get("spot")))
    return syms[:limit]


def load(ex, symbol, timeframe, limit=1500, use_cache=True) -> pd.DataFrame:
    symbol = resolve_symbol(ex, symbol)

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
    guard = 0

    while len(rows) < limit and guard < 40:
        guard += 1
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
