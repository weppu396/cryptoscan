"""ticket.py - one bracket object per trade, requiring explicit human yes."""
import json
import pathlib
import uuid

import pandas as pd

OUT = pathlib.Path("tickets.json")


def build(setup, size, mult, gate_result, reasons):
    verdict, blocks, warns, _ = gate_result
    qty = size.get("qty", 0) * mult
    return {
        "id": uuid.uuid4().hex[:10],
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": setup.symbol,
        "side": "BUY" if setup.direction == "long" else "SELL",
        "type": "LIMIT",
        "entry": round(setup.entry, 8), "stop": round(setup.stop, 8),
        "tp1": round(setup.tp1, 8), "tp2": round(setup.tp2, 8),
        "qty": round(qty, 8),
        "qty_tp1": round(qty * 0.5, 8), "qty_tp2": round(qty * 0.5, 8),
        "risk_usd": round(size.get("risk_cash", 0) * mult, 2),
        "rr": round(setup.rr, 2),
        "grade": setup.meta.get("rating").grade if setup.meta.get("rating") else "?",
        "verdict": verdict, "blocks": blocks, "warns": warns,
        "reasons": reasons, "time_stop_bars": 24,
        "state": "PENDING_APPROVAL" if verdict == "PASS" else "BLOCKED",
        "approved": False,
    }


def save(ticket):
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    rows.append(ticket)
    OUT.write_text(json.dumps(rows, indent=2))
    return ticket


def all_tickets():
    return json.loads(OUT.read_text()) if OUT.exists() else []


def approve(ticket_id):
    rows = all_tickets()
    for r in rows:
        if r["id"] == ticket_id:
            r["approved"] = True
            r["state"] = "READY"
    OUT.write_text(json.dumps(rows, indent=2))


def as_text(t):
    L = [f"{t['side']} {t['symbol']}  LIMIT {t['entry']}",
         f"  size {t['qty']}  risk ${t['risk_usd']}  R:R {t['rr']}  grade {t['grade']}",
         f"  stop {t['stop']}  tp1 {t['tp1']} (50%)  tp2 {t['tp2']} (50%)",
         f"  time stop {t['time_stop_bars']} bars"]
    for b in t["blocks"]:
        L.append(f"  BLOCKED: {b}")
    for w in t["warns"]:
        L.append(f"  ! {w}")
    for r in t["reasons"]:
        L.append(f"  + {r}")
    return "\n".join(L)
