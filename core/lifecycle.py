"""core/lifecycle.py - three machines: setup, order, position. Append-only."""
import json
import math

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


def _transition_in_tx(c, setup_id: str, machine: str, to: str, ts: str,
                      price=None, note="") -> bool:
    row = c.execute("SELECT state FROM events WHERE setup_id=? AND machine=? "
                    "ORDER BY seq DESC LIMIT 1", (setup_id, machine)).fetchone()
    now = row[0] if row else None
    if now == to:
        return False
    allowed = _table(machine)
    initial = next(iter(allowed))
    if not (now is None and to == initial) and to not in allowed.get(now if now is not None else initial, set()):
        raise TransitionError(f"{machine}: {now or initial} -> {to}")
    c.execute("INSERT INTO events(setup_id, machine, state, ts, price, note) "
              "VALUES (?,?,?,?,?,?)", (setup_id, machine, to, ts, price, note))
    return True


def transition(setup_id: str, machine: str, to: str, ts: str,
               price: float | None = None, note: str = "") -> bool:
    with store.conn() as c:
        c.execute("BEGIN IMMEDIATE")
        return _transition_in_tx(c, setup_id, machine, to, ts, price, note)


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

def _positive(value, name):
    import math
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise TransitionError(f"{name} must be positive and finite")


def _positive(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TransitionError(f"{name} must be finite and positive")


def _record_transition(c, setup_id, machine, to, ts, price=None, note=""):
    row = c.execute(
        "SELECT state FROM events WHERE setup_id=? AND machine=? ORDER BY seq DESC LIMIT 1",
        (setup_id, machine),
    ).fetchone()
    now = row[0] if row else None
    if now == to:
        return False
    allowed = _table(machine)
    key = now if now is not None else next(iter(allowed))
    if to not in allowed.get(key, set()):
        raise TransitionError(f"{machine}: {now} -> {to}")
    c.execute(
        "INSERT INTO events(setup_id,machine,state,ts,price,note) VALUES (?,?,?,?,?,?)",
        (setup_id, machine, to, ts, price, note),
    )
    return True


def open_order(order_id: str, setup_id: str, qty: float, price: float, ts: str):
    _positive(qty, "qty")
    _positive(price, "price")
    with store.conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            if c.execute("SELECT 1 FROM orders WHERE order_id=?", (order_id,)).fetchone():
                raise TransitionError("order id already exists")
            _record_transition(c, setup_id, "order", OrderState.PENDING.value, ts, price)
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                      (order_id, setup_id, OrderState.PENDING.value, qty, 0.0, price, ts))
            c.commit()
        except Exception:
            c.rollback()
            raise


def execute(order_id: str, filled_qty: float, price: float, ts: str) -> str:
    _positive(filled_qty, "filled_qty")
    _positive(price, "price")
    with store.conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT setup_id,qty,filled,state FROM orders WHERE order_id=?",
                            (order_id,)).fetchone()
            if row is None:
                raise TransitionError(f"unknown order {order_id}")
            setup_id, qty, filled, state = row
            if state not in (OrderState.PENDING.value, OrderState.PARTIALLY_FILLED.value):
                raise TransitionError(f"order {order_id} is {state}")
            total = round(filled + filled_qty, 12)
            if total > qty + 1e-12:
                raise TransitionError("fill exceeds remaining order quantity")
            new = (OrderState.FILLED.value if total >= qty - 1e-12
                   else OrderState.PARTIALLY_FILLED.value)
            _record_transition(c, setup_id, "order", new, ts, price, f"exec {filled_qty}")
            c.execute("UPDATE orders SET filled=?,state=?,updated=? WHERE order_id=?",
                      (total, new, ts, order_id))
            position = c.execute(
                "SELECT state,qty,entry FROM positions WHERE setup_id=?", (setup_id,)
            ).fetchone()
            if position is None:
                _record_transition(c, setup_id, "position", PositionState.OPEN.value, ts, price)
                c.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?)",
                          (setup_id, PositionState.OPEN.value, filled_qty, price, 0.0, 0.0, ts))
            else:
                state_pos, old_qty, old_entry = position
                if state_pos != PositionState.OPEN.value:
                    raise TransitionError("cannot fill into a non-open position")
                new_qty = old_qty + filled_qty
                avg = (old_qty * old_entry + filled_qty * price) / new_qty
                c.execute("UPDATE positions SET qty=?,entry=?,updated=? WHERE setup_id=?",
                          (new_qty, avg, ts, setup_id))
            c.commit()
            return new
        except Exception:
            c.rollback()
            raise


def cancel(order_id: str, ts: str, reason: str = "manual"):
    with store.conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT setup_id,state FROM orders WHERE order_id=?",
                            (order_id,)).fetchone()
            if row is None:
                raise TransitionError("unknown order")
            setup_id, state = row
            if state not in (OrderState.PENDING.value, OrderState.PARTIALLY_FILLED.value):
                raise TransitionError(f"cannot cancel order in {state}")
            _record_transition(c, setup_id, "order", OrderState.CANCELLED.value,
                               ts, None, reason)
            c.execute("UPDATE orders SET state=?,updated=? WHERE order_id=?",
                      (OrderState.CANCELLED.value, ts, order_id))
            c.commit()
        except Exception:
            c.rollback()
            raise


def close_position(setup_id: str, qty: float, price: float, ts: str) -> str:
    _positive(qty, "qty")
    _positive(price, "price")
    with store.conn() as c:
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute("SELECT qty,closed,state FROM positions WHERE setup_id=?",
                            (setup_id,)).fetchone()
            if row is None:
                raise TransitionError("no position")
            qty_total, closed, state = row
            if state == PositionState.CLOSED.value:
                raise TransitionError("position already closed")
            done = round(closed + qty, 12)
            if done > qty_total + 1e-12:
                raise TransitionError("close exceeds remaining position")
            new = (PositionState.CLOSED.value if done >= qty_total - 1e-12
                   else PositionState.PARTIALLY_CLOSED.value)
            _record_transition(c, setup_id, "position", new, ts, price)
            c.execute("UPDATE positions SET closed=?,state=?,updated=? WHERE setup_id=?",
                      (done, new, ts, setup_id))
            c.commit()
            return new
        except Exception:
            c.rollback()
            raise
