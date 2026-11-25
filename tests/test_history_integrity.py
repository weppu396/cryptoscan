"""Fail-closed checks for incomplete and gapped exchange candle history."""
import time
import pytest
from data import TF_MS, load
from tests.test_predeploy_data import FakeExchange


def _bars(count):
    step = TF_MS['1m']
    now = int(time.time() * 1000)
    return [[now - (count-i)*step, 1, 2, .5, 1.5, 10] for i in range(count)]


def test_short_history_rejected():
    with pytest.raises(RuntimeError, match='Incomplete OHLCV'):
        load(FakeExchange(_bars(19)), 'BTC/USDT', '1m', limit=30, use_cache=False)


def test_gap_in_history_rejected():
    bars = _bars(40)
    del bars[20]
    with pytest.raises(RuntimeError, match='Gapped OHLCV'):
        load(FakeExchange(bars), 'BTC/USDT', '1m', limit=30, use_cache=False)
