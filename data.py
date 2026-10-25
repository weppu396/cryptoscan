"""
Candles + cache. Multiple Binance routes, tested live.

Binance geo-blocks US datacentre IPs. Cloudflare Workers do not help because
they run in the colo nearest the CALLER. So we test direct routes instead.
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

FALLBACK_MAJORS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "LINK",
    "DOT", "LTC", "ATOM", "NEAR", "APT", "ARB", "OP", "SUI", "INJ",
]

CANDIDATES = [
    "okx", "binanceus", "binance_vision", "binance", "binanceusdm",
    "mexc", "kucoinfutures", "gateio", "bitget", "bybit", "htx",
]

ROUTES = {
    "binance_vision": ("binance", "https://data-api.binance.vision"),
    "binanceus": ("binanceus", None),
    "binance": ("binance", None),
    "binanceusdm": ("binanceusdm", None),
}

DEFAULT_TYPE = {
    "binance": "spot", "binance_vision": "spot", "binanceus": "spot",
    "binanceusdm": "future", "bybit": "swap", "okx": "swap",
    "bitget": "swap", "gateio": "swap", "kucoinfutures": "swap",
    "mexc": "swap", "htx": "swap",
}

BINANCE_CLASSES = ("binance", "binanceusdm", "binanceus", "binancecoinm")
_RESOLVED = {}
LAST_ERRORS = {}


def min_bars_needed(cfg, htf_need=230):
    return int(HTF_PER_DAY.get(cfg.scan.htf, 6) * htf_need) + 600


def _set_api_base(ex, base):
    b = base.rstrip("/")
    ex.urls["api"] = {
        "public": f"{b}/api/v3", "private": f"{b}/api/v3", "v3": f"{b}/api/v3",
        "sapi": f"{b}/sapi/v1",
        "fapiPublic": f"{b}/fapi/v1", "fapiPrivate": f"{b}/fapi/v1",
        "fapiPublicV2": f"{b}/fapi/v2", "fapiPrivateV2": f"{b}/fapi/v2",
        "dapiPublic": f"{b}/dapi/v1", "dapiPrivate": f"{b}/dapi/v1",
    }


def get_exchange(name, proxy=None):
    real, base = ROUTES.get(name, (name, None))
    klass = getattr(ccxt, real)
    ex = klass({
        "enableRateLimit": True,
        "timeout": 30_000,
        "options": {"defaultType": DEFAULT_TYPE.get(name, "swap")},
        "headers": {"User-Agent": BROWSER_UA,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9"},
    })
    if base and not proxy:
        _set_api_base(ex, base)
    if proxy and real in BINANCE_CLASSES:
        _set_api_base(ex, proxy)
    return ex


def probe_binance_routes(proxy=None):
    import requests
    tests = [
        ("direct api.binance.com", "https://api.binance.com/api/v3/ping"),
        ("data-api.binance.vision", "https://data-api.binance.vision/api/v3/ping"),
        ("api.binance.us", "https://api.binance.us/api/v3/ping"),
        ("fapi.binance.com", "https://fapi.binance.com/fapi/v1/ping"),
        ("bybit.com", "https://api.bybit.com/v5/market/time"),
        ("okx.com", "https://www.okx.com/api/v5/public/time"),
    ]
    if proxy:
        tests.append(("your proxy", proxy.rstrip("/") + "/api/v3/ping"))
    out = {}
    for label, url in tests:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": BROWSER_UA})
            out[label] = {"status": r.status_code, "body": r.text[:70]}
        except Exception as e:
            out[label] = {"error": type(e).__name__}
    return out


def diagnose_binance(proxy=None):
    out = {"routes": probe_binance_routes(proxy)}
    for name in ("binanceus", "binance_vision", "binance"):
        try:
            ex = get_exchange(name, proxy)
            mk = ex.load_markets()
            out[name] = {"markets": len(mk), "url": ex.urls["api"].get("public")}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
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
            LAST_ERRORS[name] = f"{type(e).__name__}: {str(e)[:160]}"

    raise RuntimeError("no reachable exchange. see LAST_ERRORS")


def resolve_symbol(ex, wanted: str) -> str:
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


def build_universe(ex, cfg) -> tuple:
    """
    Returns (symbols, note). Progressively relaxes the volume filter and
    falls back to majors if tickers are unavailable - otherwise a thin
    venue like Binance US yields an empty universe with no error.
    """
    ex.load_markets()
    tickers = {}
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        pass

    def qv_of(sym):
        t = tickers.get(sym) or {}
        q = float(t.get("quoteVolume") or 0.0)
        if q == 0.0 and t.get("baseVolume") and t.get("last"):
            q = float(t["baseVolume"]) * float(t["last"])
        return q

    cands = []
    for sym, m in ex.markets.items():
        if not m.get("active") or m.get("quote") != cfg.scan.quote:
            continue
        if m.get("base") in cfg.scan.exclude_suffixes:
            continue
        if not (m.get("swap") or m.get("spot")):
            continue
        cands.append((sym, qv_of(sym)))

    if not cands:
        return [], "no USDT markets found"

    if all(q == 0 for _, q in cands):
        syms = sorted(s for s, m in ex.markets.items()
                      if m.get("base") in FALLBACK_MAJORS
                      and m.get("quote") == cfg.scan.quote and m.get("active"))
        return syms[:cfg.scan.top_n_symbols], "tickers unavailable, using majors"

    note = f"volume filter ${cfg.scan.min_quote_vol_24h/1e6:.0f}M"
    for thr in (cfg.scan.min_quote_vol_24h, 20e6, 10e6, 5e6, 1e6, 0.0):
        keep = [c for c in cands if c[1] >= thr]
        if len(keep) >= 15 or thr == 0.0:
            if thr != cfg.scan.min_quote_vol_24h:
                note = f"volume filter relaxed to ${thr/1e6:.0f}M"
            break

    keep.sort(key=lambda x: -x[1])
    return [s for s, _ in keep][:cfg.scan.top_n_symbols], note


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
