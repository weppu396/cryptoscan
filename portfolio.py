"""portfolio.py - holdings with edit, remove, export, optional Gist persistence."""
import json
import pathlib

import pandas as pd
import requests

from data import load
from core.indicators import compute_all
from core.structure import build_structure
from core.regime import htf_bias

P = pathlib.Path("portfolio.json")
MAJORS = {"BTC", "ETH", "SOL"}
GIST_API = "https://api.github.com/gists"


def load_pf():
    if P.exists():
        try:
            return json.loads(P.read_text())
        except Exception:
            return {}
    return {}


def save_pf(pf):
    P.write_text(json.dumps(pf, indent=2))


def add(symbol, qty, price, stop=None, tp=None, note=""):
    pf = load_pf()
    cur = pf.get(symbol, {"qty": 0.0, "avg": 0.0, "stop": None, "tp": None,
                          "opened": None, "notes": []})
    total = cur["qty"] + qty
    if total > 0:
        cur["avg"] = (cur["avg"] * cur["qty"] + price * qty) / total
    cur["qty"] = total
    if stop is not None:
        cur["stop"] = stop
    if tp is not None:
        cur["tp"] = tp
    cur["opened"] = cur.get("opened") or pd.Timestamp.now(tz="UTC").isoformat()
    if note:
        cur["notes"].append(note)
    pf[symbol] = cur
    save_pf(pf)


def edit(symbol, qty=None, avg=None, stop=None, tp=None):
    pf = load_pf()
    if symbol not in pf:
        return
    if qty is not None:
        pf[symbol]["qty"] = qty
    if avg is not None:
        pf[symbol]["avg"] = avg
    if stop is not None:
        pf[symbol]["stop"] = None if stop == 0 else stop
    if tp is not None:
        pf[symbol]["tp"] = None if tp == 0 else tp
    save_pf(pf)


def remove(symbol, qty=None):
    pf = load_pf()
    if symbol not in pf:
        return
    if qty is None:
        pf.pop(symbol)
    else:
        pf[symbol]["qty"] -= qty
        if pf[symbol]["qty"] <= 1e-12:
            pf.pop(symbol)
    save_pf(pf)


def export_json() -> str:
    return json.dumps(load_pf(), indent=2)


def import_json(text) -> int:
    try:
        data = json.loads(text)
    except Exception:
        return -1
    if not isinstance(data, dict):
        return -1
    save_pf(data)
    return len(data)


def gist_push(token, gist_id=None):
    payload = {"description": "cryptoscan portfolio", "public": False,
               "files": {"portfolio.json": {"content": export_json()}}}
    h = {"Authorization": f"Bearer {token}",
         "Accept": "application/vnd.github+json"}
    try:
        if gist_id:
            r = requests.patch(f"{GIST_API}/{gist_id}", json=payload,
                               headers=h, timeout=25)
        else:
            r = requests.post(GIST_API, json=payload, headers=h, timeout=25)
        return r.json().get("id") if r.status_code < 300 else None
    except Exception:
        return None


def gist_pull(token, gist_id):
    h = {"Authorization": f"Bearer {token}",
         "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(f"{GIST_API}/{gist_id}", headers=h, timeout=25)
        if r.status_code >= 300:
            return -1
        return import_json(r.json()["files"]["portfolio.json"]["content"])
    except Exception:
        return -1


def majors_held():
    return len({s.split("/")[0] for s in load_pf() if s.split("/")[0] in MAJORS})


def open_risk_pct(equity):
    total = 0.0
    for _, h in load_pf().items():
        if h.get("stop") and h["qty"] > 0:
            total += abs(h["avg"] - h["stop"]) * h["qty"]
    return round(total / equity * 100, 2) if equity else 0.0


def health_check(ex, cfg, equity):
    out = []
    for sym, h in load_pf().items():
        row = {"symbol": sym, "qty": h["qty"], "avg": h["avg"],
               "stop": h.get("stop"), "tp": h.get("tp"), "signals": [],
               "severity": 0, "last": None, "pnl_pct": None,
               "value": 0.0, "pnl_usd": 0.0, "ret_pct": 0.0}
        try:
            df = load(ex, sym, cfg.scan.ltf, limit=1500)
            if len(df) < 300:
                out.append(row)
                continue
            ind = compute_all(df, cfg)
            st = build_structure(df, cfg.structure)
            hb = htf_bias(df, cfg)
            last = float(df["close"].iat[-1])
            row["last"] = last
            row["pnl_pct"] = round((last / h["avg"] - 1) * 100, 2) if h["avg"] else None
            row["ret_pct"] = row["pnl_pct"] or 0.0
            row["value"] = last * h["qty"]
            row["pnl_usd"] = (last - h["avg"]) * h["qty"]

            if h.get("stop") and last <= h["stop"]:
                row["signals"].append("STOP HIT - exit now")
                row["severity"] = 3
            elif h.get("stop"):
                atr = float(ind["atr"].iat[-1])
                if atr > 0 and (last - h["stop"]) / atr < 0.5:
                    row["signals"].append("within 0.5 ATR of stop")
                    row["severity"] = max(row["severity"], 2)
            if h.get("tp") and last >= h["tp"]:
                row["signals"].append("target reached - take profit")
                row["severity"] = max(row["severity"], 2)
            if st["choch_dn"].tail(3).any():
                row["signals"].append("bearish CHoCH on 4H")
                row["severity"] = max(row["severity"], 2)
            if last < float(ind["ema_fast"].iat[-1]):
                row["signals"].append("below EMA50")
                row["severity"] = max(row["severity"], 1)
            if float(ind["rsi"].iat[-1]) > 78:
                row["signals"].append(f"RSI {float(ind['rsi'].iat[-1]):.0f} overbought")
                row["severity"] = max(row["severity"], 1)
            if hb.iat[-1] == "down" and not h.get("tp"):
                row["signals"].append("daily bias flipped down")
                row["severity"] = max(row["severity"], 1)
        except Exception as e:
            row["signals"].append(f"data error: {type(e).__name__}")
        out.append(row)
    return sorted(out, key=lambda r: -r["severity"])


def summary(rows, equity):
    val = sum(r.get("value") or 0 for r in rows)
    pnl = sum(r.get("pnl_usd") or 0 for r in rows)
    return {"positions": len(rows), "value": round(val, 2),
            "pnl_usd": round(pnl, 2),
            "pnl_pct": round(pnl / equity * 100, 2) if equity else 0.0,
            "weight_pct": round(val / equity * 100, 1) if equity else 0.0,
            "exits": sum(1 for r in rows if r["severity"] >= 2)}
