"""Ranking must be deterministic and robust to missing/nonfinite ratings."""
from types import SimpleNamespace
from core.opportunities import current_only


def setup(grade, confidence, symbol, status="ACTIONABLE"):
    return SimpleNamespace(symbol=symbol, entry=100., stop=95., tp2=115.,
        weighted_rr=2., meta={"status": status,
                              "r": {"grade": grade, "confidence": confidence}})


def test_ranking_grade_then_confidence_and_excludes_expired():
    items = [setup("A-", 99, "A"), setup("A+", 20, "B"),
             setup("A", 80, "C"), setup("A+", 99, "D", "EXPIRED")]
    assert [x.symbol for x in current_only(items)] == ["B", "C", "A"]


def test_missing_or_nan_confidence_does_not_crash():
    items = [setup("B", float("nan"), "A"), setup("B", None, "B"),
             setup("B", 90, "C")]
    assert [x.symbol for x in current_only(items)][0] == "C"
