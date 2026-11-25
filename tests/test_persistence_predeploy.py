"""Local persistence contracts; a mounted persistent volume is still required in cloud."""
import json
from pathlib import Path
import paper
import portfolio


def test_paper_atomic_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, 'BOOK', tmp_path / 'paper.json')
    rows = [{'id': 'abc', 'state': 'OPEN', 'entry': 100}]
    paper._save(rows)
    assert paper._load() == rows
    assert not list(tmp_path.glob('paper-*.tmp'))


def test_portfolio_atomic_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(portfolio, 'P', tmp_path / 'portfolio.json')
    rows = {'BTC/USDT': {'qty': 1.5}}
    portfolio.save_pf(rows)
    assert portfolio.load_pf() == rows
    assert not list(tmp_path.glob('portfolio-*.tmp'))
