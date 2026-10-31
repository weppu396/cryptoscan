"""market.py - public market context. No API key, 2-minute cache."""
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; CryptoScan)"}
TTL = 120
_cache: dict[str, tuple[float, object]] = {}


def _get(url: str, timeout: int = 15):
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    _cache[url] = (time.time(), data)
    return data


def overview() -> dict:
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


def btc_history(days: int = 90) -> tuple[list, list]:
    """Daily BTC market cap and price series."""
    try:
        d = _get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
                 f"?vs_currency=usd&days={days}&interval=daily", timeout=20)
        return d.get("market_caps", []), d.get("prices", [])
    except Exception:
        return [], []


def dominance_split(ov: dict) -> tuple[list, list]:
    """BTC / ETH / rest as market cap slices."""
    total = ov.get("total_mcap")
    btc, eth = ov.get("btc_dom"), ov.get("eth_dom")
    if not total or btc is None:
        return [], []
    btc_v = total * btc / 100
    eth_v = total * (eth or 0) / 100
    rest = max(total - btc_v - eth_v, 0)
    return ["BTC", "ETH", "Others"], [btc_v, eth_v, rest]


def stable_ratio(ov: dict):
    """Stablecoin cap as a share of total. Higher means more dry powder."""
    if not ov.get("stable_mcap") or not ov.get("total_mcap"):
        return None
    return ov["stable_mcap"] / ov["total_mcap"] * 100


def movers(limit: int = 8) -> list[dict]:
    try:
        rows = _get("https://api.coingecko.com/api/v3/coins/markets"
                    "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
                    "&price_change_percentage=24h")
    except Exception:
        return []
    rows = [r for r in rows if r.get("price_change_percentage_24h") is not None]
    rows.sort(key=lambda r: abs(r["price_change_percentage_24h"]), reverse=True)
    return [{"symbol": r["symbol"].upper(), "name": r["name"],
             "price": r["current_price"], "chg": r["price_change_percentage_24h"],
             "vol": r.get("total_volume") or 0} for r in rows[:limit]]


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


def stance(fng, btc_dom, stab) -> tuple[str, str, list[str]]:
    """Turn the numbers into a plain position stance."""
    notes = []
    if fng is None:
        return "unknown", "mut", ["Fear & Greed unavailable."]
    if fng <= 25:
        notes.append("Crowd is fearful. Accumulation is cheaper here, but entries still need structure.")
    elif fng >= 75:
        notes.append("Crowd is greedy. Good for trimming, poor for chasing.")
    else:
        notes.append("Sentiment is mid-range. Take signals on their own merit.")
    if btc_dom and btc_dom > 58:
        notes.append(f"BTC dominance at {btc_dom:.1f}% — money is hiding in BTC, alts bleed.")
    elif btc_dom and btc_dom < 45:
        notes.append(f"BTC dominance at {btc_dom:.1f}% — capital is rotating into alts.")
    if stab is not None:
        notes.append(f"Stablecoins are {stab:.1f}% of the market — that is unspent buying power.")
    score = 0
    score += 1 if fng <= 40 else -1 if fng >= 70 else 0
    score += 1 if (btc_dom or 50) < 52 else 0
    label = "risk on" if score >= 2 else "risk off" if score <= -1 else "selective"
    tone = "up" if label == "risk on" else "dn" if label == "risk off" else "warn"
    return label, tone, notes
