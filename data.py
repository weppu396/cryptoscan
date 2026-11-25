"""
Candles + cache. Multiple Binance routes, tested live.

Binance geo-blocks US datacentre IPs. Cloudflare Workers do not help because
they run in the colo nearest the CALLER. So we test direct routes instead.
"""
from dataclasses import replace
import time
import math
import pathlib
import os
import tempfile
import pandas as pd

CACHE = pathlib.Path("./data_cache")
CACHE.mkdir(exist_ok=True)

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
}

HTF_PER_DAY = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
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
    import ccxt  # Optional until an exchange connection is actually requested.
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
        try:
            quote = float(t.get("quoteVolume") or 0.0)
            if not math.isfinite(quote) or quote <= 0:
                base = float(t.get("baseVolume") or 0.0)
                last = float(t.get("last") or 0.0)
                quote = base * last
            return quote if math.isfinite(quote) and quote > 0 else 0.0
        except (TypeError, ValueError, OverflowError):
            return 0.0

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

    limit = min(100, max(1, int(cfg.scan.top_n_symbols or 100)))
    if all(q == 0 for _, q in cands):
        return [], "24h volume unavailable; cannot verify liquidity"

    keep = [c for c in cands if c[1] >= cfg.scan.min_quote_vol_24h]
    keep.sort(key=lambda x: (-x[1], x[0]))
    return [s for s, _ in keep[:limit]], (
        f"24h quote volume >= ${cfg.scan.min_quote_vol_24h/1e6:g}M; "
        f"top {limit} max"
    )


def list_symbols(ex, quote="USDT", limit=800):
    ex.load_markets()
    syms = sorted(s for s, m in ex.markets.items()
                  if m.get("active") and m.get("quote") == quote
                  and (m.get("swap") or m.get("spot")))
    return syms[:limit]


def load(ex, symbol, timeframe, limit=1500, use_cache=True, max_cache_age_bars=2) -> pd.DataFrame:
    symbol = resolve_symbol(ex, symbol)

    safe = symbol.replace("/", "_").replace(":", "_")
    key = CACHE / f"{ex.id}_{safe}_{timeframe}.parquet"

    if use_cache and key.exists():
        try:
            cached = pd.read_parquet(key)
            if len(cached) >= limit and not cached.empty:
                latest = pd.Timestamp(cached.index[-1])
                if latest.tzinfo is None:
                    latest = latest.tz_localize("UTC")
                age = pd.Timestamp.now(tz="UTC") - latest
                if pd.Timedelta(0) <= age <= pd.Timedelta(milliseconds=TF_MS[timeframe] * max_cache_age_bars):
                    return cached.tail(limit)
        except Exception:
            pass

    step = TF_MS[timeframe]
    batch_limit = int(ex.options.get("fetchOHLCVLimit") or 1000)
    batch_limit = max(50, min(batch_limit, 1000))

    # CCXT `since` is a FORWARD pagination cursor. Walking it backwards
    # can repeatedly fetch the same candles and silently truncate history.
    now_ms = int(time.time() * 1000)
    since = now_ms - (limit + 3) * step
    rows = []
    guard = 0
    last_ts = None
    while len(rows) < limit + 3 and guard < 60:
        guard += 1
        remaining = limit + 3 - len(rows)
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=since,
                                   limit=min(batch_limit, remaining))
        except Exception:
            # Never silently accept a partial history after a pagination failure.
            # Indicators and structure would otherwise be calculated on truncated data.
            raise
        if not batch:
            break
        batch = sorted(batch, key=lambda r: r[0])
        fresh = [r for r in batch if r[0] >= since and
                 (last_ts is None or r[0] > last_ts)]
        if not fresh:
            break  # Exchange ignored the cursor or returned duplicate data.
        rows.extend(fresh)
        last_ts = fresh[-1][0]
        since = last_ts + step
        if since > now_ms:
            break
        time.sleep(max(0, getattr(ex, "rateLimit", 0)) / 1000)

    # Never silently treat a short or discontinuous response as a complete
    # requested history: missing candles can invalidate pivots and indicators.
    if len(rows) < limit:
        raise RuntimeError(
            f"Incomplete OHLCV history for {symbol} {timeframe}: "
            f"received {len(rows)} of {limit} requested bars"
        )
    rows = rows[-limit:]
    if any(rows[i][0] - rows[i - 1][0] != step for i in range(1, len(rows))):
        raise RuntimeError(f"Gapped OHLCV history for {symbol} {timeframe}")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index().astype(float)
    if not df.empty:
        # Parquet cache writes must be atomic: concurrent cloud sessions can
        # otherwise read a partially-written cache file.
        fd, temp_path = tempfile.mkstemp(prefix="candles-", suffix=".parquet", dir=CACHE)
        os.close(fd)
        try:
            df.to_parquet(temp_path)
            os.replace(temp_path, key)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return df
