import pandas as pd

from config import CFG
from core import structure2 as s2
from core.models import Layer
from tests.fixtures import down_then_internal_bull, equal_highs
from tests.test_bos_sequence import _df, BULL, BEAR, DENSE


def test_internal_bull_does_not_move_external():
    from dataclasses import replace
    # Short swing is confirmed, but external (5/5) pivot is not confirmed yet.
    cfg = replace(DENSE, struct=replace(DENSE.struct, ext_left=5, ext_right=5))
    df = _df(BULL[:10])
    f = s2.run(df, "X", "5m", cfg)
    t = len(df) - 1
    assert f.state_at(Layer.EXTERNAL, t) == "unclear"
    assert f.state_at(Layer.INTERNAL, t) == "up"
    assert all(e.layer is not Layer.EXTERNAL for e in f.live(t)
               if e.label.value == "CHoCH" and e.direction == "bull")


def test_first_break_is_bos_init():
    # A flat series has no confirmed swing to break; use real pivots.
    f = s2.run(_df(BULL[:10]), "X", "5m", DENSE)
    labels = [e.label.value for e in f.events]
    assert "BOS_INIT" in labels
    assert "CHoCH" not in labels


def test_choch_after_established_trend():
    f = s2.run(_df(BULL + BEAR), "X", "5m", DENSE)
    assert any(e.label.value == "BOS_INIT" and e.direction == "bull" for e in f.events)
    assert any(e.label.value == "CHoCH" and e.direction == "bear" for e in f.events)


def test_equal_highs_fire_once():
    df = equal_highs()
    f = s2.run(df, "X", "1h", CFG)
    bulls = [e for e in f.events if e.direction == "bull"]
    assert len(bulls) <= 2  # external + internal, never two on the same level
    levels = [round(e.level, 1) for e in bulls]
    assert len(levels) == len(set(levels))


def test_no_repaint_before_confirmation():
    df = down_then_internal_bull()
    f = s2.run(df, "X", "1h", CFG)
    for e in f.events:
        assert e.confirm_idx == e.break_idx
        assert all(x.confirm_idx <= e.confirm_idx for x in f.events
                   if x is not e and x.break_idx < e.break_idx)


def test_invalidation_uses_real_bar():
    f = s2.run(_df(BULL + BEAR), "X", "5m", DENSE)
    inv = [e for e in f.events if e.invalidated_at is not None]
    assert inv, "fixture should produce a broken level and later invalidation"
    for e in inv:
        assert e.invalidated_at > e.confirm_idx
        assert e not in f.live(e.invalidated_at)
        assert e in f.live(e.invalidated_at - 1)


def test_consumed_level_does_not_refire():
    f = s2.run(_df(BULL + [109, 108, 107, 108, 109, 110, 111]), "X", "5m", DENSE)
    levels = [(e.layer, round(e.level, 2)) for e in f.events]
    assert len(levels) == len(set(levels))
