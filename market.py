"""market.py - public market context. No API key, 2-minute cache."""
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; CryptoScan)"}
TTL = 120
_cache: dict[str, tuple[float, object]] = {}


def _get(url: str, timeout: int = 12):
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    _cache[url] = (time.time(), data)
    return data


def overview() -> dict:
    """Global market stats. Missing keys stay None rather than raising."""
    out = {"total_mcap": None, "mcap_change_24h": None, "btc_dom": None,
           "eth_dom": None, "stable_mcap": None, "fng": None, "fng_label": None}
    try:
        g = _get("https://api.coingecko.com/api/v3/global")["data"]
        out["total_mcap"] = g["total_market_cap"].get("usd")
        out["mcap_change_24h"] = g.get("market_cap_change_percentage_24h_usd")
        out["btc_dom"] = g["market_cap_percentage"].get("btc")
        out["eth_dom"] = g["market_cap_percentage"].get("eth")
        out["stable_mcap"] = g.get("total_stablecoin_market_cap", {}).get("usd")
    except Exception:
        pass
    try:
        f = _get("https://api.alternative.me/fng/?limit=1")["data"][0]
        out["fng"] = int(f["value"])
        out["fng_label"] = f["value_classification"]
    except Exception:
        pass
    return out


def movers(limit: int = 8) -> list[dict]:
    """Top 100 by cap, sorted by absolute 24h move."""
    try:
        rows = _get("https://api.coingecko.com/api/v3/coins/markets"
                    "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
                    "&price_change_percentage=24h")
    except Exception:
        return []
    rows = [r for r in rows if r.get("price_change_percentage_24h") is not None]
    rows.sort(key=lambda r: abs(r["price_change_percentage_24h"]), reverse=True)
    return [{
        "symbol": r["symbol"].upper(),
        "name": r["name"],
        "price": r["current_price"],
        "chg": r["price_change_percentage_24h"],
        "vol": r.get("total_volume") or 0,
    } for r in rows[:limit]]


def money(x) -> str:
    if x is None:
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"${x / div:.2f}{unit}"
    return f"${x:.2f}"


def pct(x) -> str:
    return "—" if x is None else f"{x:+.2f}%"


def regime_label(fng) -> tuple[str, str]:
    """Return (label, tone) from Fear & Greed."""
    if fng is None:
        return "unknown", "mut"
    if fng <= 20:
        return "extreme fear", "dn"
    if fng <= 40:
        return "fear", "warn"
    if fng <= 60:
        return "neutral", "mut"
    if fng <= 80:
        return "greed", "up"
    return "extreme greed", "dn"
