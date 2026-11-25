"""Regression checks for security-independent execution and trade accounting."""
from types import SimpleNamespace

import pandas as pd
import pytest

import store
from core import lifecycle as lc
from strategy.risk import plan_rr
from strategy.reversal import _candidate, evaluate
from backtest.engine import run


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "orders.db")


def test_filled_order_cannot_cancel_or_change_record(db):
    lc.open_order("o", "s", 1, 100, "t")
    lc.execute("o", 1, 100, "t")
    with pytest.raises(lc.TransitionError):
        lc.cancel("o", "t")
    with store.conn() as c:
        assert c.execute("SELECT state FROM orders WHERE order_id='o'").fetchone()[0] == "FILLED"


def test_invalid_fill_rolls_back_everything(db):
    lc.open_order("o", "s", 1, 100, "t")
    for qty in (0, -1, 2):
        with pytest.raises(lc.TransitionError):
            lc.execute("o", qty, 100, "t")
    with store.conn() as c:
        assert c.execute("SELECT filled FROM orders WHERE order_id='o'").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0


def test_close_cannot_exceed_position(db):
    lc.open_order("o", "s", 1, 100, "t")
    lc.execute("o", 1, 100, "t")
    with pytest.raises(lc.TransitionError):
        lc.close_position("s", 2, 101, "t")
    with store.conn() as c:
        assert c.execute("SELECT closed FROM positions WHERE setup_id='s'").fetchone()[0] == 0


def test_wrong_side_targets_are_not_profitable():
    assert plan_rr(100, 99, 98, 97)["weighted"] < 0
    assert plan_rr(100, 101, 102, 103)["weighted"] < 0


def test_zone_inside_is_reachable():
    zone = {"low": 99, "high": 101}
    assert _candidate([zone], 100, 1, True)[1] == 0
    assert _candidate([zone], 100, 1, False)[1] == 0
    assert evaluate("X", None, None) is None


def test_partial_take_profit_is_in_trade_r():
    idx = pd.date_range("2026-01-01", periods=3, freq="1h")
    df = pd.DataFrame({"open": [100, 100, 102], "high": [100, 102, 103],
                       "low": [100, 100, 102], "close": [100, 102, 103]}, index=idx)
    risk = SimpleNamespace(account_equity=1000, risk_per_trade_pct=1,
                           partial_size=0.5, move_stop_to_be_at_r=1,
                           time_stop_bars=20)
    cfg = SimpleNamespace(costs=SimpleNamespace(fee_bps=0, slip_bps=0), risk=risk)
    signals = {0: {"dir": "long", "stop": 99, "tp1": 102, "tp2": 103}}
    _, trades = run(df, signals, cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["r"] == pytest.approx(2.5)
    assert trades.iloc[0]["net_pnl"] == pytest.approx(25)
