import pytest

import core.lifecycle as lc
import store


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "t.db")
    yield


def test_illegal_transition_raises():
    lc.transition("s1", "setup", "FORMING", "t")
    with pytest.raises(lc.TransitionError):
        lc.transition("s1", "setup", "TRIGGERED", "t")


def test_duplicate_state_is_noop():
    lc.transition("s2", "setup", "FORMING", "t")
    assert lc.transition("s2", "setup", "FORMING", "t") is False


def test_cancel_then_fill_needs_new_order():
    lc.open_order("o1", "s3", 1.0, 100.0, "t")
    lc.cancel("o1", "t")
    with pytest.raises(lc.TransitionError):
        lc.execute("o1", 1.0, 100.0, "t")


def test_partial_fill_accounting():
    lc.open_order("o2", "s4", 1.0, 100.0, "t")
    assert lc.execute("o2", 0.4, 100.0, "t") == "PARTIALLY_FILLED"
    assert lc.execute("o2", 0.6, 100.0, "t") == "FILLED"
    with store.conn() as c:
        assert c.execute("SELECT qty FROM positions WHERE setup_id='s4'").fetchone()[0] == 1.0


def test_position_partial_close():
    lc.open_order("o3", "s5", 1.0, 100.0, "t")
    lc.execute("o3", 1.0, 100.0, "t")
    assert lc.close_position("s5", 0.5, 105.0, "t") == "PARTIALLY_CLOSED"
    assert lc.close_position("s5", 0.5, 106.0, "t") == "CLOSED"
