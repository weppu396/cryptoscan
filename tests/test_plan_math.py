import pytest

from strategy.risk import plan_rr, size_position
from config import CFG


def test_short_example_matches_hand_calculation():
    r = plan_rr(0.003625, 0.0037101956, 0.0034972066, 0.0033694132)
    assert r["rr1"] == pytest.approx(1.5, rel=1e-6)
    assert r["rr2"] == pytest.approx(3.0, rel=1e-6)
    assert r["weighted"] == pytest.approx(2.25, rel=1e-6)


def test_long_mirrors_short():
    r = plan_rr(100.0, 99.0, 101.5, 103.0)
    assert (r["rr1"], r["rr2"], r["weighted"]) == pytest.approx((1.5, 3.0, 2.25))


def test_targets_are_not_both_the_far_r():
    r = plan_rr(0.003625, 0.0037101956, 0.0034972066, 0.0033694132)
    assert r["rr1"] != r["rr2"]


def test_split_weighting():
    r = plan_rr(100.0, 99.0, 101.5, 103.0, split=0.25)
    assert r["weighted"] == pytest.approx(0.25 * 1.5 + 0.75 * 3.0)


def test_zero_risk_is_safe():
    assert plan_rr(100.0, 100.0, 101.0, 102.0)["rr1"] == 0.0


def test_position_size_matches_hand_calculation():
    s = size_position(0.003625, 0.0037101956, 10_000.0, CFG)
    assert s["qty"] == pytest.approx(586884.87, rel=1e-4)
    assert s["risk_cash"] == pytest.approx(50.0)
