"""core/lifecycle.py - three machines: setup, order, position. Append-only."""
import json

import store
from core.models import OrderState, PositionState, State

SETUP = {
    State.FORMING: {State.READY, State.INVALIDATED, State.EXPIRED},
    State.READY: {State.TRIGGERED, State.INVALIDATED, State.EXPIRED},
    State.TRIGGERED: {State.HANDED_OFF, State.INVALIDATED},
    State.HANDED_OFF: {State.INVALIDATED},
    State.INVALIDATED: set(),
    State.EXPIRED: set(),
}

ORDER = {
    OrderState.NONE: {OrderState.PENDING},
    OrderState.PENDING: {OrderState.PARTIALLY_FILLED, OrderState.FILLED,
                         OrderState.CANCELLED, OrderState.REJECTED,
                         OrderState.EXPIRED},
    OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}

POSITION = {
    PositionState.NONE: {PositionState.OPEN},
    PositionState.OPEN: {PositionState.PARTIALLY_CLOSED, PositionState.CLOSED},
    PositionState.PARTIALLY_CLOSED: {PositionState.CLOSED},
    PositionState.CLOSED: set(),
}


class TransitionError(Exception):
    pass


def _table(machine: str) -> dict:
    return {"setup": SETUP, "order": ORDER, "position": POSITION}[machine]


def current(setup_id: str, machine: str):
    with store.conn() as c:
        row = c.execute("SELECT state FROM events WHERE setup_id=? AND machine=? "
                        "ORDER BY seq DESC LIMIT 1", (setup_id, machine)).fetchone()
    return row[0] if row else None


def transition(setup_id: str, machine: str, to: str, ts: str,
               price: float | None = None, note: str = "") -> bool:
    now = current(setup_id, machine)
    if now == to:
        return False
    allowed = _table(machine)
    key = now if now is not None else list(allowed)[0]
    if to not in allowed.get(key, set()) and now is not None:
        raise TransitionError(f"{machine}: {now} -> {to}")
    with store.conn() as c:
        c.execute("INSERT INTO events(setup_id, machine, state, ts, price, note) "
                  "VALUES (?,?,?,?,?,?)", (setup_id, machine, to, ts, price, note))
    return True


def history(setup_id: str, machine: str | None = None) -> list:
    q = "SELECT machine, state, ts, price, note FROM events WHERE setup_id=?"
    args = [setup_id]
    if machine:
        q += " AND machine=?"
        args.append(machine)
    with store.conn() as c:
        rows = c.execute(q + " ORDER BY seq", args).fetchall()
    return [dict(zip(("machine", "state", "ts", "price", "note"), r)) for r in rows]


# ---- order accounting: a fill requires an execution event, never a touch

def open_order(order_id: str, setup_id: str, qty: float, price: float, ts: str):
    with store.conn() as c:
        c.execute("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?)",
                  (order_id, setup_id, OrderState.PENDING.value, qty, 0.0, price, ts))
    transition(setup_id, "order", OrderState.PENDING.value, ts, price)


def execute(order_id: str, filled_qty: float, price: float, ts: str) -> str:
    with store.conn() as c:
        row = c.execute("SELECT setup_id, qty, filled, state FROM orders WHERE order_id=?",
                        (order_id,)).fetchone()
        if not row:
            raise TransitionError(f"unknown order {order_id}")
        setup_id, qty, filled, state = row
        if state in (OrderState.CANCELLED.value, OrderState.REJECTED.value,
                     OrderState.EXPIRED.value):
            raise TransitionError(f"order {order_id} is {state}, needs a new order")
        total = round(filled + filled_qty, 12)
        new = OrderState.FILLED.value if total >= qty - 1e-12 \
            else OrderState.PARTIALLY_FILLED.value
        c.execute("UPDATE orders SET filled=?, state=?, updated=? WHERE order_id=?",
                  (total, new, ts, order_id))
    transition(setup_id, "order", new, ts, price, f"exec {filled_qty}")
    if current(setup_id, "position") is None:
        with store.conn() as c:
            c.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?)",
                      (setup_id, PositionState.OPEN.value, total, price, 0.0, 0.0, ts))
        transition(setup_id, "position", PositionState.OPEN.value, ts, price)
    else:
        with store.conn() as c:
            c.execute("UPDATE positions SET qty=qty+? WHERE setup_id=?", (filled_qty, setup_id))
    return new


def cancel(order_id: str, ts: str, reason: str = "manual"):
    with store.conn() as c:
        row = c.execute("SELECT setup_id, state FROM orders WHERE order_id=?",
                        (order_id,)).fetchone()
        if not row:
            raise TransitionError("unknown order")
        setup_id, state = row
        c.execute("UPDATE orders SET state=?, updated=? WHERE order_id=?",
                  (OrderState.CANCELLED.value, ts, order_id))
    transition(setup_id, "order", OrderState.CANCELLED.value, ts, None, reason)


def close_position(setup_id: str, qty: float, price: float, ts: str) -> str:
    with store.conn() as c:
        row = c.execute("SELECT qty, closed, state FROM positions WHERE setup_id=?",
                        (setup_id,)).fetchone()
        if not row:
            raise TransitionError("no position")
        qty_total, closed, state = row
        if state == PositionState.CLOSED.value:
            raise TransitionError("position already closed")
        done = round(closed + qty, 12)
        new = PositionState.CLOSED.value if done >= qty_total - 1e-12 \
            else PositionState.PARTIALLY_CLOSED.value
        c.execute("UPDATE positions SET closed=?, state=?, updated=? WHERE setup_id=?",
                  (done, new, ts, setup_id))
    transition(setup_id, "position", new, ts, price)
    return new
