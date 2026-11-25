"""Exchange pagination and cloud safety regression tests."""
import pandas as pd
import pytest
from data import load


class FakeExchange:
    id = "mock"
    options = {"fetchOHLCVLimit": 1000}
    rateLimit = 0

    def __init__(self, rows, fail=False):
        self.markets = {"BTC/USDT": {"active": True}}
        self.rows = rows
        self.fail = fail
        self.calls = []

    def load_markets(self):
        return self.markets

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls.append(since)
        if self.fail:
            raise RuntimeError("API unavailable")
        return [r for r in self.rows if r[0] >= since][:limit]


def test_forward_pagination_returns_ordered_unique_rows(monkeypatch):
    from data import TF_MS
    import time
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: None)
    step = TF_MS["1m"]
    now = int(time.time() * 1000)
    rows = [[now - (1100 - i) * step, 1, 2, 0.5, 1.5, 10] for i in range(1100)]
    ex = FakeExchange(rows)
    result = load(ex, "BTC/USDT", "1m", limit=1050, use_cache=False)
    assert len(result) == 1050
    assert result.index.is_monotonic_increasing
    assert all(b > a for a, b in zip(ex.calls, ex.calls[1:]))


def test_api_failure_is_not_misrepresented_as_empty_market_data():
    ex = FakeExchange([], fail=True)
    with pytest.raises(RuntimeError, match="API unavailable"):
        load(ex, "BTC/USDT", "1m", limit=20, use_cache=False)
