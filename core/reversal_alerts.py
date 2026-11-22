"""core/reversal_alerts.py - early warnings and entry alerts.

alerts.py dedupes once per calendar day; useless inside a 5-10 minute window.
This uses a TTL instead. Nothing here places an order.
"""
import hashlib
from datetime import datetime, timedelta, timezone

EARLY_MIN, EARLY_MAX = 5.0, 10.0


class State:
    def __init__(self, ttl_min: float = 45.0):
        self.ttl = timedelta(minutes=ttl_min)
        self.seen: dict[str, datetime] = {}

    @staticmethod
    def _key(r, bucket: str) -> str:
        raw = f"{r['symbol']}|{r['direction']}|{r['zone'][0]:.10g}|{bucket}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _fresh(self, key: str, now: datetime) -> bool:
        last = self.seen.get(key)
        if last and now - last < self.ttl:
            return False
        self.seen[key] = now
        return True

    @staticmethod
    def message(r, kind: str) -> str:
        head = "EARLY WARNING" if kind == "early" else "CONFIRMED REVERSAL"
        lines = [f"[{head}] {r['symbol']} {r['direction'].upper()}",
                 f"status {r['status']}",
                 f"zone {r['zone'][0]:,.6g} - {r['zone'][1]:,.6g}",
                 f"entry {r['entry']:,.6g}  stop {r['stop']:,.6g}",
                 f"tp1 {r['tp1']:,.6g} ({abs(r['tp1'] - r['entry']) / max(abs(r['entry'] - r['stop']), 1e-12):.2f}R)"
                 f"  tp2 {r['tp2']:,.6g} ({abs(r['tp2'] - r['entry']) / max(abs(r['entry'] - r['stop']), 1e-12):.2f}R)",
                 f"R:R {r['rr']:.2f}"]
        if r.get("eta_min") is not None:
            lines.append(f"ETA ~{r['eta_min']:.0f} min (estimate, not guaranteed)")
        lines.append("30m context / 3m trigger. Paper only, no auto-execution.")
        return "\n".join(lines)

    def due(self, rows, min_rr: float, now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        out = []
        for r in rows:
            if r["status"] == "APPROACHING" and r.get("eta_min") is not None:
                if not (EARLY_MIN <= r["eta_min"] <= EARLY_MAX):
                    continue
                k = self._key(r, f"eta{int(r['eta_min'] // 2) * 2}")
                if self._fresh(k, now):
                    out.append((r, "early", self.message(r, "early")))
            elif r["status"] == "ACTIVE_ENTRY" and r["is_new"]:
                if r["rr"] < min_rr:
                    continue
                k = self._key(r, "entry")
                if self._fresh(k, now):
                    out.append((r, "entry", self.message(r, "entry")))
        return out

    def prune(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        cut = now - self.ttl * 4
        self.seen = {k: v for k, v in self.seen.items() if v >= cut}
