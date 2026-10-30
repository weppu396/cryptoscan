"""
macro.py - economic calendar, funding rate, and news risk.

Free sources, no API keys:
  - ForexFactory weekly calendar mirror (nfs.faireconomy.media)
  - OKX public funding-rate endpoint
  - CoinDesk RSS for headline sentiment
"""
import re
import requests
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (compatible; cryptoscan/1.0)"}
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

BULL_WORDS = ("surge", "rally", "soar", "record", "approve", "etf inflow",
              "bullish", "gain", "jump", "adopt")
BEAR_WORDS = ("crash", "plunge", "selloff", "sell-off", "hack", "lawsuit",
              "bearish", "drop", "fall", "ban", "delay", "reject")


def _now():
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fetch_calendar(timeout=20) -> pd.DataFrame:
    try:
        r = requests.get(CAL_URL, headers=UA, timeout=timeout)
        data = r.json()
    except Exception:
        return pd.DataFrame()

    rows = []
    for ev in data or []:
        try:
            when = pd.to_datetime(ev.get("date"), utc=True).tz_localize(None)
        except Exception:
            continue
        rows.append({
            "when": when,
            "currency": (ev.get("country") or "").upper(),
            "title": ev.get("title") or "",
            "impact": (ev.get("impact") or "").lower(),
            "forecast": ev.get("forecast") or "",
            "previous": ev.get("previous") or "",
        })
    df = pd.DataFrame(rows)
    return df.sort_values("when") if not df.empty else df


def upcoming_events(hours=48, currency="USD", impacts=("high",)) -> pd.DataFrame:
    df = fetch_calendar()
    if df.empty:
        return df
    now, end = _now(), _now() + pd.Timedelta(hours=hours)
    d = df[(df["when"] >= now) & (df["when"] <= end)]
    if currency:
        d = d[d["currency"] == currency]
    if impacts:
        d = d[d["impact"].isin(list(impacts))]
    return d.reset_index(drop=True)


def blackout_status(hours_before=3, hours_after=1):
    """(in_blackout, reason) when a high-impact USD event is near."""
    df = fetch_calendar()
    if df.empty:
        return False, "calendar unavailable"
    now = _now()
    for _, ev in df.iterrows():
        if ev["currency"] != "USD" or ev["impact"] != "high":
            continue
        delta_h = (ev["when"] - now).total_seconds() / 3600.0
        if -hours_after <= delta_h <= hours_before:
            return True, f"{ev['title']} at {ev['when']:%d %b %H:%M} UTC"
    return False, "clear"


def fetch_funding(symbol="BTC/USDT:USDT"):
    """OKX public funding, annualised %. Positive = longs pay shorts."""
    inst = symbol.replace("/", "-").replace(":USDT", "-SWAP")
    url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}"
    try:
        r = requests.get(url, headers=UA, timeout=15).json()
        d = (r.get("data") or [{}])[0]
        cur = float(d.get("fundingRate") or 0)
        nxt = float(d.get("nextFundingRate") or 0)
        return cur * 3 * 365 * 100, nxt * 3 * 365 * 100
    except Exception:
        return None, None


def news_sentiment(limit=40):
    """Rough headline tone, -1..1, from CoinDesk RSS."""
    url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
        if not titles:
            titles = re.findall(r"<title>(.*?)</title>", r.text)
    except Exception:
        return 0.0, 0

    titles = [t.lower() for t in titles[:limit]]
    if not titles:
        return 0.0, 0
    bull = sum(any(w in t for w in BULL_WORDS) for t in titles)
    bear = sum(any(w in t for w in BEAR_WORDS) for t in titles)
    total = bull + bear
    if total == 0:
        return 0.0, len(titles)
    return round((bull - bear) / total, 3), len(titles)


def macro_state(symbol="BTC/USDT:USDT"):
    blackout, reason = blackout_status()
    funding, _ = fetch_funding(symbol)
    sentiment, n = news_sentiment()

    multiplier = 1.0
    notes = []
    if blackout:
        multiplier *= 0.5
        notes.append(f"BLACKOUT: {reason} - size halved")
    if funding is not None and abs(funding) > 40:
        multiplier *= 0.75
        side = "longs" if funding > 0 else "shorts"
        notes.append(f"funding extreme ({funding:+.0f}% annualised), {side} crowded")
    if abs(sentiment) > 0.5:
        notes.append(f"news tone {'bullish' if sentiment > 0 else 'bearish'} "
                     f"({sentiment:+.2f} over {n} headlines)")

    return {"blackout": blackout, "blackout_reason": reason,
            "funding_annualised_pct": round(funding, 1) if funding is not None else None,
            "sentiment": sentiment, "headlines": n,
            "multiplier": round(multiplier, 2), "notes": notes}


def macro_report(symbol="BTC/USDT:USDT"):
    state = macro_state(symbol)
    cal = upcoming_events(hours=72)
    return {"state": state,
            "calendar": cal.to_dict("records") if not cal.empty else []}
