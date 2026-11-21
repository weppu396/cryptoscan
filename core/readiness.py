"""core/readiness.py - readiness only. Quality lives elsewhere."""
from core.models import Decision, ReadinessInput

MANDATORY = ("valid_setup", "bias_aligned", "in_zone", "structure_confirmed",
             "candle_ok", "risk_ok")


def evaluate(r: ReadinessInput) -> tuple:
    """(Decision, missing). READY implies missing == ()."""
    if not r.valid_setup:
        return Decision.REJECT, ("setup no longer valid",)
    if r.invalidated:
        return Decision.REJECT, ("setup invalidated",)
    if r.expired:
        return Decision.REJECT, ("entry window expired",)
    if not r.bias_aligned:
        return Decision.REJECT, (r.bias_reason,)

    if not r.in_zone:
        if r.distance_atr > r.approach_atr:
            return Decision.WAIT, ("price not near the zone",)
        return Decision.PREPARE, ("approaching the zone, not in it",)

    if not r.structure_confirmed:
        return Decision.CHECK_ENTRY_TF, ("structure break not confirmed",)
    if not r.candle_ok:
        return Decision.CHECK_ENTRY_TF, ("no valid confirmation candle",)
    if not r.risk_ok:
        return Decision.REJECT, ("risk gate failed",)

    failed = tuple(k for k, v in r.optional.items() if not v)
    if failed:
        return Decision.CHECK_ENTRY_TF, failed
    if r.trigger_fired:
        return Decision.TRIGGERED, ()
    return Decision.READY, ()


def actionable(d: Decision) -> bool:
    return d in (Decision.READY, Decision.TRIGGERED)
