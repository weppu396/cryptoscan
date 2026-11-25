"""Deployment-critical liquidity and data integrity checks (offline)."""
from types import SimpleNamespace
import pytest
from data import build_universe, load

class Venue:
    id = 'mock'
    options = {'fetchOHLCVLimit': 1000}
    rateLimit = 0
    def __init__(self):
        self.markets = {f'C{i}/USDT': {'active': True, 'quote': 'USDT', 'base': f'C{i}', 'spot': True} for i in range(130)}
        self.markets['BAD/USDT'] = {'active': True, 'quote': 'USDT', 'base': 'BAD', 'spot': True}
        self.markets['FAKE/USDT'] = {'active': True, 'quote': 'USDT', 'base': 'FAKE'}
    def load_markets(self): return self.markets
    def fetch_tickers(self):
        out = {f'C{i}/USDT': {'quoteVolume': (i + 1) * 1_000_000} for i in range(130)}
        out['BAD/USDT'] = {'quoteVolume': float('nan')}
        out['FAKE/USDT'] = {'quoteVolume': 900_000_000}
        return out

def test_universe_is_liquid_and_capped():
    cfg = SimpleNamespace(scan=SimpleNamespace(quote='USDT', exclude_suffixes=(), top_n_symbols=1000, min_quote_vol_24h=40_000_000))
    selected, _ = build_universe(Venue(), cfg)
    assert len(selected) == 91
    assert selected[0] == 'C129/USDT'
    assert 'BAD/USDT' not in selected and 'FAKE/USDT' not in selected

def test_failed_later_page_does_not_return_partial_history():
    import time
    from data import TF_MS
    ex = Venue()
    ex.markets = {'C1/USDT': {'active': True, 'quote': 'USDT', 'spot': True}}
    now = int(time.time() * 1000)
    step = TF_MS['1m']
    calls = []
    def fetch(symbol, timeframe, since=None, limit=None):
        calls.append(since)
        if len(calls) > 1:
            raise RuntimeError('network lost')
        return [[now - (1100-i)*step, 1, 2, .5, 1.5, 10] for i in range(1000)]
    ex.fetch_ohlcv = fetch
    with pytest.raises(RuntimeError, match='network lost'):
        load(ex, 'C1/USDT', '1m', limit=1050, use_cache=False)
