"""core/alerts2.py - eligibility separated from delivery.

PENDING is written before the HTTP call. SENT only after the poster accepts.
A failure can never suppress a later attempt. Config reads are defensive, and
the poster may return (ok, retry) or a bare bool, with retry as int/float/str.
"""
import time

import store
from core.models import Delivery, Stage

COOLDOWN_DEFAULT = {"APPROACHING": 900, "IN_ZONE": 600, "READY": 300}
BACKOFF_DEFAULT = (30, 60, 120, 300, 600)
_ATTR = {"APPROACHING": "cooldown_approaching",
         "IN_ZONE": "cooldown_in_zone",
         "READY": "cooldown_ready"}


def _alert_cfg():
    try:
        from config import CFG
        return getattr(CFG, "alert", None)
    except Exception:
        return None


def _cooldown(stage: Stage) -> int:
    c = _alert_cfg()
    attr = _ATTR.get(stage.value)
    fallback = COOLDOWN_DEFAULT.get(stage.value, 1800)
    if c is not None and attr:
        return int(getattr(c, attr, fallback))
    return int(fallback)


def _max_attempts() -> int:
    c = _alert_cfg()
    return int(getattr(c, "max_attempts", 5)) if c is not None else 5


def _backoff(attempts: int) -> float:
    c = _alert_cfg()
    seq = getattr(c, "backoff", BACKOFF_DEFAULT) if c is not None else BACKOFF_DEFAULT
    seq = tuple(seq) or BACKOFF_DEFAULT
    return float(seq[min(max(attempts - 1, 0), len(seq) - 1)])


def _retry_after(v):
    """Posters vary. Normalise to float seconds or None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _row(key: str):
    with store.conn() as c:
        return c.execute("SELECT status, attempts, next_at FROM alerts WHERE key=?",
                         (key,)).fetchone()


def _set(key: str, setup_id: str, stage: Stage, gen: int, status: Delivery,
         attempts: int, next_at: float):
    with store.conn() as c:
        c.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?)",
                  (key, setup_id, stage.value, gen, status.value, attempts,
                   next_at, time.time()))


def eligible(setup_id: str, stage: Stage, generation: int = 0,
             now: float | None = None) -> bool:
    now = now or time.time()
    row = _row(f"{setup_id}:{stage.value}:{generation}")
    if row is None:
        return True
    status, attempts, next_at = row
    if status in (Delivery.SENT.value, Delivery.DEAD.value):
        return False
    if attempts >= _max_attempts():
        return False
    return now >= (next_at or 0)


def claim(setup_id: str, stage: Stage, generation: int = 0) -> str:
    key = f"{setup_id}:{stage.value}:{generation}"
    row = _row(key)
    attempts = (row[1] if row else 0) + 1
    _set(key, setup_id, stage, generation, Delivery.PENDING, attempts, 0.0)
    return key


def sent(key: str, setup_id: str, stage: Stage, gen: int):
    _set(key, setup_id, stage, gen, Delivery.SENT, 1, 0.0)


def failed(key: str, setup_id: str, stage: Stage, gen: int,
           retry_after=None):
    row = _row(key)
    attempts = row[1] if row else 1
    if attempts >= _max_attempts():
        _set(key, setup_id, stage, gen, Delivery.DEAD, attempts, 0.0)
        return
    delay = _retry_after(retry_after)
    if delay is None:
        delay = _backoff(attempts)
    _set(key, setup_id, stage, gen, Delivery.RETRY, attempts, time.time() + delay)


def rearm(setup_id: str, stage: Stage, generation: int):
    _set(f"{setup_id}:{stage.value}:{generation}", setup_id, stage, generation,
         Delivery.ARMED, 0, 0.0)


def kill(setup_id: str, stages: tuple = ()):
    with store.conn() as c:
        if stages:
            for s in stages:
                c.execute("UPDATE alerts SET status=? WHERE setup_id=? AND stage=?",
                          (Delivery.DEAD.value, setup_id, s.value))
        else:
            c.execute("UPDATE alerts SET status=? WHERE setup_id=?",
                      (Delivery.DEAD.value, setup_id))


def replay_stale(age: float = 120.0):
    """After a restart, PENDING older than `age` becomes RETRY."""
    cutoff = time.time() - age
    with store.conn() as c:
        c.execute("UPDATE alerts SET status=?, next_at=? WHERE status=? AND updated<?",
                  (Delivery.RETRY.value, time.time(), Delivery.PENDING.value, cutoff))


def send(setup_id: str, stage: Stage, text: str, generation: int = 0,
         poster=None) -> str:
    """poster(text) -> (ok, retry_after) or bool. Returns final status."""
    if not eligible(setup_id, stage, generation):
        return "skipped"
    key = claim(setup_id, stage, generation)
    if poster is None:
        try:
            from app_alerts import post as poster
        except Exception:
            def poster(_t):
                return False, None
    try:
        result = poster(text)
    except Exception:
        result = (False, None)
    if isinstance(result, tuple):
        ok, retry = (result + (None,))[:2]
    else:
        ok, retry = result, None
    if ok:
        sent(key, setup_id, stage, generation)
        return Delivery.SENT.value
    failed(key, setup_id, stage, generation, retry)
    row = _row(key)
    return row[0] if row else Delivery.FAILED.value


def message(stage: Stage, symbol: str, direction: str, tf: str, setup_id: str,
            grade: str, zone: tuple | None, price: float, bias: str,
            missing: tuple, invalidate: float | None, action: str) -> str:
    lines = [stage.value, "", f"Symbol: {symbol}",
             f"Direction: {direction.upper()}", f"Timeframe: {tf}",
             f"Setup: {setup_id}", f"Grade: {grade}", f"Bias: {bias}",
             f"Price: {price:,.6g}"]
    if zone:
        lines.append(f"Zone: {zone[0]:,.6g} - {zone[1]:,.6g}")
    if invalidate is not None:
        lines.append(f"Invalidation: {invalidate:,.6g}")
    if missing:
        lines += ["", "Missing:"] + [f"- {m}" for m in missing]
    lines += ["", f"NEXT ACTION: {action}"]
    return "\n".join(lines)
