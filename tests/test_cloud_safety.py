"""Cloud-safe selection and alert claiming regressions."""
from types import SimpleNamespace
import pytest
import store
from core.models import Stage
from core import alerts2


class Exchange:
    id = 'test'
    def __init__(self):
        self.markets = {f'C{i}/USDT': {'active': True, 'quote': 'USDT', 'base': f'C{i}', 'spot': True} for i in range(150)}
    def load_markets(self): pass
    def fetch_tickers(self):
        return {s: {'quoteVolume': (i+1)*1_000_000} for i,s in enumerate(self.markets)}


def test_liquid_top100_never_exceeds_limit():
    cfg = SimpleNamespace(scan=SimpleNamespace(quote='USDT', exclude_suffixes=[], top_n_symbols=1000, min_quote_vol_24h=60_000_000))
    pytest.importorskip("ccxt")
    from data import build_universe
    selected, _ = build_universe(Exchange(), cfg)
    assert len(selected) == 91
    assert selected[0] == 'C149/USDT'


def test_claim_cannot_be_stolen(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DB', tmp_path/'test.db')
    key = alerts2.claim('setup', Stage.READY)
    assert key
    assert alerts2.claim('setup', Stage.READY) is None
    alerts2.sent(key, 'setup', Stage.READY, 0)
    assert alerts2.claim('setup', Stage.READY) is None


def test_retry_waits_for_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DB', tmp_path/'test.db')
    assert alerts2.send('setup', Stage.READY, 'x', poster=lambda t: (False, 5)) == 'RETRY'
    assert alerts2.send('setup', Stage.READY, 'x', poster=lambda t: (True, None)) == 'skipped'
