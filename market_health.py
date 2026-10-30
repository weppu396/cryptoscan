"""
market_health.py - risk-on / risk-off context from BTC dominance and
stablecoin supply. Snapshots stored locally so trends can be computed.
Free source: CoinGecko public endpoints.
"""
import json
import pathlib
import requests
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (compatible; cryptoscan/1.0)"}
HIST = pathlib.Path("health_history.json")
CG = "https://api.coingecko.com/api/v3"
STABLES = "tether,usd-coin,dai,first-digital-usd,true-usd"


def _get(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def snapshot():
    g = _get(f"{CG}/global")
    st = _get(f"{CG}/coins/markets",
              {"vs_currency": "usd", "ids": STABLES, "per_page": 10})
    if not g or "data" not in g:
        return None

    dom = g["data"].get("market_cap_percentage", {})
    total = g["data"].get("total_market_cap", {})
    snap = {
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "btc_dominance": round(float(dom.get("btc") or 0), 3),
        "total_mcap_usd": float(total.get("usd") or 0),
        "stablecoin_mcap_usd": sum(float(c.get("market_cap") or 0) for c in (st or [])),
        "mcap_24h_change_pct": round(
            float(g["data"].get("market_cap_change_percentage_24h_usd") or 0), 3),
    }
    hist = []
    if HIST.exists():
        try:
            hist = json.loads(HIST.read_text())
        except Exception:
            hist = []
    hist.append(snap)
    HIST.write_text(json.dumps(hist[-400:]))
    return snap


def _trend(hist, key, days=7):
    if len(hist) < 2:
        return 0.0
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    pts = [h for h in hist if pd.Timestamp(h["ts"]) >= cutoff and h.get(key) is not None]
    if len(pts) < 2:
        return 0.0
    return float(pts[-1][key]) - float(pts[0][key])


def health():
    hist = []
    if HIST.exists():
        try:
            hist = json.loads(HIST.read_text())
        except Exception:
            hist = []

    snap = snapshot()
    if snap is None:
        return {"score": 50, "verdict": "unknown",
                "reasons": ["CoinGecko unreachable"], "snapshot": {}}

    score = 50.0
    reasons = []
    dom_7d = _trend(hist, "btc_dominance")
    stable_7d = _trend(hist, "stablecoin_mcap_usd")
    mcap_24h = snap["mcap_24h_change_pct"]

    if dom_7d > 0.6:
        score -= 12
        reasons.append(f"BTC dominance +{dom_7d:.2f}pt/7d - money rotating to BTC, alts bleed")
    elif dom_7d < -0.6:
        score += 10
        reasons.append(f"BTC dominance {dom_7d:.2f}pt/7d - altcoin tilt")

    if stable_7d < -2e8:
        score -= 10
        reasons.append(f"stablecoin supply ${stable_7d/1e6:.0f}M/7d - liquidity draining")
    elif stable_7d > 2e8:
        score += 10
        reasons.append(f"stablecoin supply +${stable_7d/1e6:.0f}M/7d - fresh buying power")

    if mcap_24h < -3:
        score -= 8
        reasons.append(f"total market cap {mcap_24h:.1f}% in 24h")
    elif mcap_24h > 3:
        score += 6
        reasons.append(f"total market cap +{mcap_24h:.1f}% in 24h")

    score = max(0, min(100, round(score)))
    verdict = ("hostile" if score < 35 else "cautious" if score < 50
               else "neutral" if score < 65 else "supportive")
    return {"score": score, "verdict": verdict,
            "reasons": reasons or ["no extreme readings"],
            "snapshot": snap, "dom_7d": round(dom_7d, 3),
            "stable_7d_usd": round(stable_7d)}


def size_multiplier(h):
    return {"hostile": 0.5, "cautious": 0.75}.get(h.get("verdict"), 1.0)
