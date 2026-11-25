"""paper.py - forward test. Same tickets, no money."""
import json
import os
import tempfile
import pathlib
import uuid

import pandas as pd

BOOK = pathlib.Path(os.environ.get("CRYPTOSCAN_STATE_DIR", "./state")) / "paper.json"


def _load() -> list:
    return json.loads(BOOK.read_text()) if BOOK.exists() else []


def _save(rows: list) -> None:
    BOOK.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="paper-", suffix=".tmp", dir=BOOK.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, BOOK)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def open_trade(symbol, direction, entry, stop, tp1, tp2, qty, equity, reason=""):
    rows = _load()
    rows.append({"id": uuid.uuid4().hex[:8], "symbol": symbol, "dir": direction,
                 "entry": entry, "stop": stop, "tp1": tp1, "tp2": tp2,
                 "qty": qty, "equity0": equity, "reason": reason,
                 "ts": pd.Timestamp.utcnow().isoformat(), "state": "OPEN",
                 "exit": None, "r": None})
    _save(rows)


def close_trade(tid, price, tag):
    rows = _load()
    for r in rows:
        if r["id"] == tid and r["state"] == "OPEN":
            sign = 1 if r["dir"] == "long" else -1
            risk = abs(r["entry"] - r["stop"]) or 1e-9
            r["exit"], r["state"], r["tag"] = price, "CLOSED", tag
            r["r"] = round((price - r["entry"]) * sign / risk, 3)
    _save(rows)


def book() -> list:
    return _load()


def open_rows() -> list:
    return [r for r in _load() if r["state"] == "OPEN"]


def stats() -> dict:
    done = [r for r in _load() if r["state"] == "CLOSED" and r["r"] is not None]
    if not done:
        return {"n": 0}
    rs = [r["r"] for r in done]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gl = abs(sum(losses))
    return {"n": len(rs), "win_rate": round(len(wins) / len(rs) * 100, 1),
            "avg_r": round(sum(rs) / len(rs), 3),
            "pf": round(sum(wins) / gl, 2) if gl else None,
            "best": max(rs), "worst": min(rs),
            "expectancy": round(sum(rs) / len(rs), 3)}


def gate() -> tuple[bool, str]:
    """Promotion gate. 30 trades minimum, profit factor at least 1.0."""
    s = stats()
    if s["n"] < 30:
        return False, f"{s['n']}/30 trades — keep going"
    if (s.get("pf") or 0) < 1.0:
        return False, f"profit factor {s.get('pf')} under 1.0"
    if (s.get("avg_r") or 0) <= 0:
        return False, f"expectancy {s.get('avg_r')}R is not positive"
    return True, "cleared — eligible for live at 0.25% risk"
