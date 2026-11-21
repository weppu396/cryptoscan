"""core/nextaction.py - decision to instruction. Never invents a confirmation."""
from core.models import Decision

ACTION = {
    Decision.WAIT: "Wait. Price has not reached the zone.",
    Decision.PREPARE: "Prepare. Price is approaching the zone.",
    Decision.CHECK_HTF: "Check the higher timeframe.",
    Decision.CHECK_SETUP_TF: "Check the setup timeframe.",
    Decision.CHECK_ENTRY_TF: "Check the entry timeframe for confirmation.",
    Decision.READY: "All enabled conditions met. Place the bracket.",
    Decision.TRIGGERED: "Trigger fired. Submit the order.",
    Decision.REJECT: "Stand down.",
}


def build(decision: Decision, symbol: str, direction: str, price: float,
          level: float | None, zone: tuple | None, missing: tuple, tf: str,
          invalidate: float | None) -> dict:
    return {
        "decision": decision.value,
        "action": ACTION[decision],
        "symbol": symbol,
        "direction": direction,
        "price": price,
        "level": level,
        "zone": zone,
        "missing": missing,
        "timeframe": tf,
        "invalidate": invalidate,
        "actionable": decision in (Decision.READY, Decision.TRIGGERED),
    }
