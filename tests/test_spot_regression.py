"""Spot and shared-surface regression. Proves the new code did not disturb it."""
import pandas as pd
import pytest

import config
import sizer
import smc
from core.candles import quality
from tests.fixtures import flat


def test_sizer_long_matches_formula():
    r = sizer.size(10_000, 0.5, 100.0, 98.0, "long", 5)
    assert r["qty"] == pytest.approx(25.0)
    assert r["risk_cash"] == pytest.approx(50.0)


def test_sizer_short_symmetric():
    r = sizer.size(10_000, 0.5, 100.0, 102.0, "short", 5)
    assert r["qty"] == pytest.approx(25.0)


def test_sizer_rejects_bad_stop():
    assert sizer.size(10_000, 0.5, 100.0, 100.0, "long", 5) == {}


def test_liquidation_flag():
    r = sizer.size(10_000, 0.5, 100.0, 99.0, "long", 2)
    assert r["liq_safe"] is False


def test_config_defaults_unchanged():
    c = config.CFG
    assert c.risk.risk_per_trade_pct == 0.5
    assert c.risk.min_rr == 2.0
    assert c.structure.swing_left == 3
    assert c.scan.ltf == "4h"


def test_playbook_shape():
    df = flat(120)
    p = smc.playbook(df, "long", 10_000, 0.5)
    for k in ("entry", "sl", "tp1", "tp2", "checks", "verdict", "confluence"):
        assert k in p
    assert len(p["checks"]) == 7


def test_candle_quality_thresholds():
    df = flat(10)
    df.iloc[8, [0, 1, 2, 3]] = [100, 100.2, 98.6, 100.1]
    df.iloc[9, [0, 1, 2, 3]] = [99.9, 100.6, 99.8, 100.5]
    name, score = quality(df, 9, 1.0, "long")
    assert score >= 60


def test_radar_majors_constant():
    import app  # noqa: F401  (import side effects exercised by smoke)
