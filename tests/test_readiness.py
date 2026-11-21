from core.models import Decision, ReadinessInput
from core import readiness


def base(**kw):
    d = dict(valid_setup=True, bias_aligned=True, bias_reason="ok",
             distance_atr=0.0, approach_atr=1.0, in_zone=True,
             structure_confirmed=True, candle_ok=True, risk_ok=True,
             invalidated=False, expired=False, trigger_fired=False, optional={})
    d.update(kw)
    return ReadinessInput(**d)


def test_ready_has_empty_missing():
    dec, missing = readiness.evaluate(base())
    assert dec is Decision.READY
    assert missing == ()


def test_ready_requires_all_mandatory():
    for k in ("bias_aligned", "in_zone", "structure_confirmed", "candle_ok", "risk_ok"):
        dec, _ = readiness.evaluate(base(**{k: False}))
        assert dec is not Decision.READY


def test_optional_failure_blocks_ready():
    dec, missing = readiness.evaluate(base(optional={"volume missing": False}))
    assert dec is Decision.CHECK_ENTRY_TF
    assert missing == ("volume missing",)


def test_triggered_only_when_all_pass():
    dec, _ = readiness.evaluate(base(trigger_fired=True))
    assert dec is Decision.TRIGGERED
    dec2, _ = readiness.evaluate(base(trigger_fired=True, candle_ok=False))
    assert dec2 is not Decision.TRIGGERED


def test_reject_paths():
    assert readiness.evaluate(base(invalidated=True))[0] is Decision.REJECT
    assert readiness.evaluate(base(expired=True))[0] is Decision.REJECT
    assert readiness.evaluate(base(bias_aligned=False))[0] is Decision.REJECT


def test_wait_and_prepare():
    assert readiness.evaluate(base(in_zone=False, distance_atr=5))[0] is Decision.WAIT
    assert readiness.evaluate(base(in_zone=False, distance_atr=0.5))[0] is Decision.PREPARE
