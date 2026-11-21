"""core/models.py - frozen contracts. One vocabulary, one source of truth."""
from dataclasses import dataclass, field
from enum import Enum
import uuid


class Layer(str, Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class Label(str, Enum):
    BOS_INIT = "BOS_INIT"
    BOS = "BOS"
    CHoCH = "CHoCH"


class State(str, Enum):
    FORMING = "FORMING"
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    HANDED_OFF = "HANDED_OFF"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class OrderState(str, Enum):
    NONE = "NONE"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionState(str, Enum):
    NONE = "NONE"
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


class Decision(str, Enum):
    WAIT = "WAIT"
    PREPARE = "PREPARE"
    CHECK_HTF = "CHECK HTF"
    CHECK_SETUP_TF = "CHECK SETUP TF"
    CHECK_ENTRY_TF = "CHECK ENTRY TF"
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    REJECT = "REJECT"


class Stage(str, Enum):
    NEW_SETUP = "NEW_SETUP"
    APPROACHING = "APPROACHING"
    IN_ZONE = "IN_ZONE"
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    SL_HIT = "SL_HIT"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class Delivery(str, Enum):
    ARMED = "ARMED"
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    RETRY = "RETRY"
    DEAD = "DEAD"


@dataclass(frozen=True)
class Swing:
    idx: int
    price: float
    kind: str
    layer: Layer
    confirmed_at: int
    tag: str = ""


@dataclass(frozen=True)
class StructureEvent:
    event_id: str
    symbol: str
    timeframe: str
    layer: Layer
    label: Label
    direction: str
    level: float
    swing_idx: int
    break_idx: int
    confirm_idx: int
    invalidated_at: int | None = None
    trend_after: str = "unclear"

    @property
    def live_from(self) -> int:
        return self.confirm_idx


@dataclass
class StructureFrame:
    symbol: str = ""
    timeframe: str = ""
    events: list = field(default_factory=list)
    swings: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    ranging: list = field(default_factory=list)

    def live(self, t: int, layer: Layer | None = None) -> list:
        out = []
        for e in self.events:
            if e.confirm_idx > t:
                continue
            if e.invalidated_at is not None and t >= e.invalidated_at:
                continue
            if layer and e.layer != layer:
                continue
            out.append(e)
        return out

    def state_at(self, layer: Layer, t: int) -> str:
        arr = self.state.get(layer.value)
        if not arr or t >= len(arr):
            return "unclear"
        return arr[t]

    def last(self, layer: Layer, direction: str, t: int):
        for e in reversed(self.live(t, layer)):
            if e.direction == direction:
                return e
        return None


@dataclass(frozen=True)
class ReadinessInput:
    valid_setup: bool
    bias_aligned: bool
    bias_reason: str
    distance_atr: float
    approach_atr: float
    in_zone: bool
    structure_confirmed: bool
    candle_ok: bool
    risk_ok: bool
    invalidated: bool
    expired: bool
    trigger_fired: bool
    optional: dict = field(default_factory=dict)


@dataclass
class Plan:
    setup_id: str
    symbol: str
    direction: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    qty: float
    risk_cash: float
    risk_pct: float
    rr1: float
    rr2: float
    weighted_rr: float
    fees: float
    invalidate_price: float
    expire_bars: int
    manual: dict = field(default_factory=dict)


def new_id() -> str:
    return uuid.uuid4().hex[:12]
