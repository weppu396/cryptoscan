"""Regression checks for order/position accounting and rollback."""
import pytest
import store
from core import lifecycle as lc

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DB', tmp_path / 'integrity.db')

def test_overfill_rolls_back_everything():
    lc.open_order('o', 's', 2, 100, 't0')
    with pytest.raises(lc.TransitionError):
        lc.execute('o', 3, 101, 't1')
    with store.conn() as c:
        assert c.execute('SELECT filled,state FROM orders').fetchone() == (0.0, 'PENDING')
        assert c.execute('SELECT COUNT(*) FROM positions').fetchone()[0] == 0

def test_cancel_filled_does_not_mutate():
    lc.open_order('o', 's', 1, 100, 't0')
    lc.execute('o', 1, 101, 't1')
    with pytest.raises(lc.TransitionError):
        lc.cancel('o', 't2')
    with store.conn() as c:
        assert c.execute('SELECT state FROM orders').fetchone()[0] == 'FILLED'

def test_negative_and_overclose_rejected():
    lc.open_order('o', 's', 1, 100, 't0')
    lc.execute('o', 1, 101, 't1')
    for qty in (-1, 0, 2):
        with pytest.raises(lc.TransitionError):
            lc.close_position('s', qty, 99, 't2')
    with store.conn() as c:
        assert c.execute('SELECT closed FROM positions').fetchone()[0] == 0

def test_weighted_average_on_partial_fill():
    lc.open_order('o', 's', 2, 100, 't0')
    lc.execute('o', 1, 100, 't1')
    lc.execute('o', 1, 110, 't2')
    with store.conn() as c:
        assert c.execute('SELECT qty,entry FROM positions').fetchone() == (2.0, 105.0)
