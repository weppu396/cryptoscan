"""
signals.py - fear/greed, 200-day value zones, pump detection, and
entry/exit signals with written reasons.

Fear & Greed: alternative.me (free, no key).
Value zones: 200-day SMA and EMA on daily closes.
"""
import requests
import pandas as pd

from core.indicators import atr, rsi, ema

UA = {"User-Agent": "Mozilla/5.0 (compatible; cryptoscan/1.0)"}
FNG = "https://api.alternative.me/fng/?limit=30&format=json"


def fear_greed():
    try:
        d = requests.get(FNG, headers=UA, timeout=20).json()["data"]
        vals = [int(x["value"]) for x in d]
        now = vals[0]
        week_ago = vals[7] if len(vals) > 7 else vals[-1]
        return {"value": now, "label": d[0]["value_classification"],
                "change_7d": now - week_ago, "series": vals[:14]}
    except Exception:
        return {"value": None, "label": "unavailable", "change_7d": 0, "series": []}


def daily_frame(df):
    o = df.resample("1d").agg({"open": "first", "high": "max",
                               "low": "min", "close": "last",
                               "volume": "sum"}).dropna()
    o["sma200"] = o["close"].rolling(200).mean()
    o["ema200"] = ema(o["close"], 200)
    o["rsi"] = rsi(o["close"], 14)
    o["atr"] = atr(o, 14)
    o["vol_ma"] = o["volume"].rolling(30).mean()
    return o


def value_zone(df):
    """Where price sits against the 200-day lines."""
    d = daily_frame(df)
    if d.empty or d["sma200"].isna().all():
        return {"zone": "unknown", "dist_sma": None, "dist_ema": None,
                "rsi": None, "vol_ratio": None}
    last = float(d["close"].iat[-1])
    sma = float(d["sma200"].iat[-1])
    em = float(d["ema200"].iat[-1])
    r = float(d["rsi"].iat[-1])
    vm = float(d["vol_ma"].iat[-1]) or 1.0
    vr = float(d["volume"].iat[-1]) / vm

    dist_sma = (last / sma - 1) * 100 if sma else None
    dist_ema = (last / em - 1) * 100 if em else None

    if dist_sma is None:
        zone = "unknown"
    elif dist_sma <= -25:
        zone = "capitulation"
    elif dist_sma <= -10:
        zone = "deep_value"
    elif dist_sma <= 0:
        zone = "value"
    elif dist_sma <= 25:
        zone = "fair"
    elif dist_sma <= 60:
        zone = "extended"
    else:
        zone = "euphoric"

    return {"zone": zone, "dist_sma": round(dist_sma, 2) if dist_sma is not None else None,
            "dist_ema": round(dist_ema, 2) if dist_ema is not None else None,
            "rsi": round(r, 1) if r == r else None,
            "vol_ratio": round(vr, 2), "sma200": round(sma, 8) if sma else None,
            "ema200": round(em, 8) if em else None, "last": last}


def pump_score(df):
    """How hard has price just run? 0-100."""
    if len(df) < 40:
        return {"score": 0, "ret_3": 0, "ret_7": 0, "vol_ratio": 1.0}
    c = df["close"]
    ret3 = (float(c.iat[-1]) / float(c.iat[-4]) - 1) * 100
    ret7 = (float(c.iat[-1]) / float(c.iat[-8]) - 1) * 100
    vm = float(df["volume"].rolling(20).mean().iat[-1]) or 1.0
    vr = float(df["volume"].iat[-1]) / vm

    score = 0.0
    score += min(max(ret3, 0) * 4, 40)
    score += min(max(ret7, 0) * 1.5, 30)
    score += min(max(vr - 1, 0) * 15, 30)
    return {"score": round(min(score, 100)), "ret_3": round(ret3, 2),
            "ret_7": round(ret7, 2), "vol_ratio": round(vr, 2)}


def exit_signals(h, df, fng):
    """Reasons to reduce or exit a holding."""
    sig, sev = [], 0
    vz = value_zone(df)
    p = pump_score(df)
    d = daily_frame(df)
    r = float(d["rsi"].iat[-1]) if not d.empty and d["rsi"].notna().any() else 50

    if vz["zone"] == "euphoric":
        sig.append(f"price {vz['dist_sma']:+.0f}% above the 200-day average - euphoric")
        sev = max(sev, 2)
    elif vz["zone"] == "extended":
        sig.append(f"price {vz['dist_sma']:+.0f}% above the 200-day average - extended")
        sev = max(sev, 1)

    if p["score"] >= 55:
        sig.append(f"pump score {p['score']}/100 "
                   f"(+{p['ret_3']:.1f}% in 3 bars, volume {p['vol_ratio']:.1f}x)")
        sev = max(sev, 2)

    if r >= 78:
        sig.append(f"daily RSI {r:.0f} - overbought")
        sev = max(sev, 2)

    if fng["value"] is not None:
        if fng["value"] >= 80:
            sig.append(f"Fear & Greed {fng['value']} ({fng['label']}) - extreme greed")
            sev = max(sev, 2)
        if fng["change_7d"] >= 15 and fng["value"] >= 65:
            sig.append(f"greed jumped {fng['change_7d']:+d} points in a week")
            sev = max(sev, 2)

    gain = (vz["last"] / h["avg"] - 1) * 100 if h.get("avg") else 0
    if gain >= 100:
        sig.append(f"position up {gain:.0f}% - consider taking the original stake")
        sev = max(sev, 2)

    return sig, sev, vz, p


def entry_signals(df, fng):
    """Reasons this coin is a good spot entry."""
    sig, sev = [], 0
    vz = value_zone(df)
    d = daily_frame(df)
    r = float(d["rsi"].iat[-1]) if not d.empty and d["rsi"].notna().any() else 50
    vr = vz["vol_ratio"] or 1.0

    if vz["zone"] == "capitulation":
        sig.append(f"price {vz['dist_sma']:+.0f}% below the 200-day average - capitulation")
        sev = max(sev, 3)
    elif vz["zone"] == "deep_value":
        sig.append(f"price {vz['dist_sma']:+.0f}% below the 200-day average - deep value")
        sev = max(sev, 2)
    elif vz["zone"] == "value":
        sig.append(f"price {vz['dist_sma']:+.0f}% below the 200-day average")
        sev = max(sev, 1)

    if r <= 25:
        sig.append(f"daily RSI {r:.0f} - oversold")
        sev = max(sev, 2)
    elif r <= 35:
        sig.append(f"daily RSI {r:.0f} - washed out")
        sev = max(sev, 1)

    if vr < 0.7:
        sig.append(f"volume dry-up ({vr:.1f}x average) - sellers exhausted")
        sev = max(sev, 1)

    if fng["value"] is not None:
        if fng["value"] <= 20:
            sig.append(f"Fear & Greed {fng['value']} ({fng['label']}) - extreme fear")
            sev = max(sev, 2)
        elif fng["value"] <= 35:
            sig.append(f"Fear & Greed {fng['value']} - fear")
            sev = max(sev, 1)

    return sig, sev, vz
