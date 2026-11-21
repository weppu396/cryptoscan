import pandas as pd

from config import CFG
from core import structure2 as s2
from core.models import Layer
from tests.fixtures import down_then_internal_bull, equal_highs, flat


def test_internal_bull_does_not_move_external():
    df = down_then_internal_bull()
    f = s2.run(df, "X", "1h", CFG)
    t = len(df) - 1
    assert f.state_at(Layer.EXTERNAL, t) == "down"
    assert f.state_at(Layer.INTERNAL, t) == "up"
    assert all(e.layer is not Layer.EXTERNAL for e in f.live(t)
               if e.label.value == "CHoCH" and e.direction == "bull")


def test_first_break_is_bos_init():
    df = flat(30)
    df.loc[df.index[30]:, "close"] = 101.0
    df.loc[df.index[30]:, "high"] = 101.5
    f = s2.run(df, "X", "1h", CFG)
    labels = [e.label.value for e in f.events]
    assert "BOS_INIT" in labels
    assert "CHoCH" not in labels


def test_choch_after_established_trend():
    df = flat(30)
    for i in range(30, 60):
        df.iloc[i, [0, 1, 2, 3]] = [101, 101.5, 100.8, 101.4]
    for i in range(60, 80):
        df.iloc[i, [0, 1, 2, 3]] = [99, 99.2, 98.2, 98.4]
    f = s2.run(df, "X", "1h", CFG)
    assert any(e.label.value == "CHoCH" for e in f.events)


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
    df = flat(40)
    df.iloc[30, [1, 3]] = [101.5, 101.4]
    for i in range(31, 40):
        df.iloc[i, [0, 1, 2, 3]] = [99, 99.2, 98.8, 98.9]
    f = s2.run(df, "X", "1h", CFG)
    inv = [e for e in f.events if e.invalidated_at is not None]
    for e in inv:
        assert e.invalidated_at > e.confirm_idx
        assert e not in f.live(e.invalidated_at)
        assert e in f.live(e.invalidated_at - 1)


def test_consumed_level_does_not_refire():
    df = flat(30)
    for i in range(30, 50):
        df.iloc[i, [0, 1, 2, 3]] = [101, 101.4, 100.6, 101.2]
    f = s2.run(df, "X", "1h", CFG)
    levels = [(e.layer, round(e.level, 2)) for e in f.events]
    assert len(levels) == len(set(levels))
