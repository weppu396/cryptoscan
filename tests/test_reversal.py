"""Regression guards for the reversal layer and the chart clutter rules."""
import numpy as np
import pandas as pd
import pytest

from core import reversal_alerts, structure_map, tfs, volume
from core.indicators import compute_all
from core.structure import build_structure
from config import CFG
from strategy import reversal


def frame(n=600, step="3min", seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=step, tz="UTC")
    px = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": px, "high": px + rng.uniform(.05, .5, n),
                         "low": px - rng.uniform(.05, .5, n), "close": px,
                         "volume": rng.uniform(1, 40, n)}, index=idx)


def test_3m_key_exists():
    from data import TF_MS
    assert "3m" in TF_MS


def test_resample_is_causal():
    df = frame(120)
    up = tfs.resample(df, "15m")
    assert up.index[0] >= df.index[0]
    for ts, row in up.iterrows():
        span = df.loc[ts:ts + pd.Timedelta("15min") - pd.Timedelta("1ns")]
        if len(span):
            assert row["high"] == pytest.approx(span["high"].max())


def test_structure_marks_are_capped():
    df = frame()
    htf = tfs.resample(df, "30m")
    majors, minors = structure_map.marks(df, CFG, htf, gap=0.0)
    assert len(majors) <= 2
    assert len(minors) <= 3


def test_thin_drops_duplicate_prices():
    rows = [{"ts": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=i),
             "level": 100 + i * 0.01, "kind": "BOS", "dir": "bull"}
            for i in range(6)]
    assert len(structure_map.thin(rows, 0.5)) == 1


def test_profile_conserves_volume():
    df = frame(200)
    vp = volume.profile_histogram(df, lookback=120)
    assert vp["estimate"] is True
    assert sum(vp["hist"]) == pytest.approx(1.0, rel=1e-6)
    assert vp["val"] <= vp["poc"] <= vp["vah"]


def test_lookahead_free_structure():
    df = frame()
    full = build_structure(df, CFG.structure)["bos_up"].to_numpy()
    cut = 80
    part = build_structure(df.iloc[:-cut], CFG.structure)["bos_up"].to_numpy()
    assert (full[:len(part)] == part).all()


def test_evaluate_returns_none_without_data():
    assert reversal.evaluate("X", None, None, CFG) is None


def test_active_entry_requires_rr_and_new():
    hdf = frame(400, "30min", seed=11)
    ldf = frame(400, "3min", seed=12)
    row = reversal.evaluate("X", hdf, ldf, CFG)
    if row is None:
        pytest.skip("synthetic series produced no candidate zone")
    assert row["status"] in reversal.STATUSES
    if row["status"] == "ACTIVE_ENTRY":
        assert row["rr"] >= CFG.risk.min_rr
        assert row["is_new"] is True


def test_rank_puts_live_above_dead():
    def mk(status):
        return {"status": status, "score": 1.0, "eta_min": None,
                "zone": (1.0, 2.0), "symbol": "X", "direction": "long",
                "entry": 2.0, "stop": 1.5, "tp1": 3.0, "tp2": 4.0, "rr": 2.0,
                "is_new": True, "dist_atr": 0.1, "reasons": [], "bias": "down"}
    live, dead = reversal.rank([mk("MISSED"), mk("APPROACHING"),
                                mk("ACTIVE_ENTRY")])
    assert [r["status"] for r in dead] == ["MISSED"]
    assert live[0]["status"] == "ACTIVE_ENTRY"


def test_alert_dedupe_suppresses_repeat():
    st = reversal_alerts.State()
    row = {"symbol": "X", "direction": "long", "zone": (1.0, 2.0),
           "status": "APPROACHING", "eta_min": 7.0, "rr": 2.0, "is_new": True,
           "entry": 2.0, "stop": 1.5, "tp1": 3.0, "tp2": 4.0}
    assert len(st.due([row], 2.0)) == 1
    assert len(st.due([row], 2.0)) == 0


def test_alert_ignores_far_eta():
    st = reversal_alerts.State()
    row = {"symbol": "X", "direction": "long", "zone": (1.0, 2.0),
           "status": "APPROACHING", "eta_min": 25.0, "rr": 2.0, "is_new": True,
           "entry": 2.0, "stop": 1.5, "tp1": 3.0, "tp2": 4.0}
    assert st.due([row], 2.0) == []


def test_alert_blocks_low_rr_entry():
    st = reversal_alerts.State()
    row = {"symbol": "X", "direction": "long", "zone": (1.0, 2.0),
           "status": "ACTIVE_ENTRY", "eta_min": None, "rr": 1.1, "is_new": True,
           "entry": 2.0, "stop": 1.5, "tp1": 3.0, "tp2": 4.0}
    assert st.due([row], 2.0) == []


def test_compute_all_still_works_after_change():
    df = frame(300)
    ind = compute_all(df, CFG)
    assert {"atr", "rsi", "adx"} <= set(ind.columns)
