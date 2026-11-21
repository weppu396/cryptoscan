import pytest

import core.alerts2 as a2
import store
from core.models import Stage


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "a.db")
    yield


def test_failure_does_not_suppress():
    calls = []

    def poster(text):
        calls.append(text)
        return (len(calls) > 1), None

    assert a2.send("s1", Stage.APPROACHING, "x", 0, poster) == "RETRY"
    assert a2.send("s1", Stage.APPROACHING, "x", 0, poster) == "SENT"
    assert len(calls) == 2


def test_sent_blocks_duplicate():
    a2.send("s2", Stage.READY, "x", 0, lambda t: (True, None))
    assert a2.send("s2", Stage.READY, "x", 0, lambda t: (True, None)) == "skipped"


def test_new_generation_rearms():
    a2.send("s3", Stage.IN_ZONE, "x", 0, lambda t: (True, None))
    assert a2.send("s3", Stage.IN_ZONE, "x", 1, lambda t: (True, None)) == "SENT"


def test_429_retry_after_respected():
    a2.send("s4", Stage.READY, "x", 0, lambda t: (False, 7.0))
    assert a2.eligible("s4", Stage.READY, 0) is False


def test_kill_blocks_retry():
    a2.send("s5", Stage.READY, "x", 0, lambda t: (False, None))
    a2.kill("s5")
    assert a2.eligible("s5", Stage.READY, 0) is False
