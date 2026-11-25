from config import CFG
from core import structure2 as s2
from core.models import Label, Layer
from tests.fixtures import down_then_internal_bull, equal_highs, price_series


def test_internal_bull_does_not_move_external():
    df = down_then_internal_bull()
    f = s2.run(df, "X", "1h", CFG)
    t = len(df) - 1
    assert f.state_at(Layer.EXTERNAL, t) == "down"
    assert f.state_at(Layer.INTERNAL, t) == "up"
    reversals = [e for e in f.events if e.label == Label.CHoCH and e.direction == "bull"]
    assert [(e.layer, e.swing_idx, e.confirm_idx) for e in reversals] == [
        (Layer.INTERNAL, 20, 24),
    ]


def first_bull_prices():
    return [100] * 6 + [101, 95] + [100] * 5 + [102]


def test_first_break_is_bos_init():
    df = price_series(first_bull_prices())
    f = s2.run(df, "X", "1h", CFG)
    assert {e.layer for e in f.events} == {Layer.INTERNAL, Layer.EXTERNAL}
    assert all(e.label == Label.BOS_INIT and e.direction == "bull"
               and e.swing_idx == 6 and e.confirm_idx == 13 for e in f.events)


def test_choch_after_established_trend():
    df = price_series(first_bull_prices() + [94, 93])
    f = s2.run(df, "X", "1h", CFG)
    reversals = [e for e in f.events if e.label == Label.CHoCH]
    assert {e.layer for e in reversals} == {Layer.INTERNAL, Layer.EXTERNAL}
    assert all(e.direction == "bear" and e.swing_idx == 7 and e.confirm_idx == 14
               for e in reversals)


def test_equal_highs_fire_once():
    df = equal_highs()
    f = s2.run(df, "X", "1h", CFG)
    bulls = [e for e in f.events if e.direction == "bull"]
    assert len(bulls) == 2
    assert {e.layer for e in bulls} == {Layer.INTERNAL, Layer.EXTERNAL}
    assert all(e.level == 101.5 and e.confirm_idx == 21 for e in bulls)


def test_no_repaint_before_confirmation():
    df = down_then_internal_bull()
    full = s2.run(df, "X", "1h", CFG)

    def points(frame):
        return [(e.layer, e.label, e.level, e.swing_idx, e.confirm_idx)
                for e in frame.events]

    assert full.events
    for stop in range(1, len(df) + 1):
        prefix = s2.run(df.iloc[:stop], "X", "1h", CFG)
        assert points(prefix) == [p for p in points(full) if p[-1] < stop]
        assert all(e.confirm_idx == e.break_idx for e in prefix.events)


def test_invalidation_uses_real_bar():
    df = price_series(first_bull_prices() + [101])
    f = s2.run(df, "X", "1h", CFG)
    invalidated = [e for e in f.events if e.invalidated_at is not None]
    assert {e.layer for e in invalidated} == {Layer.INTERNAL, Layer.EXTERNAL}
    for e in invalidated:
        assert e.invalidated_at == 14
        assert e not in f.live(e.invalidated_at)
        assert e in f.live(e.invalidated_at - 1)


def test_consumed_level_does_not_refire():
    df = price_series(first_bull_prices() + [102, 101, 102, 101, 102])
    f = s2.run(df, "X", "1h", CFG)
    levels = [(e.layer, e.level) for e in f.events if e.direction == "bull"]
    assert len(levels) == len(set(levels)) == 2
