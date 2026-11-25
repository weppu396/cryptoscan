"""Causal signal math and risk contracts for the seven-strategy engine."""
import numpy as np
import pandas as pd
from config import CFG
from core.indicators import compute_all
from core.structure import build_structure
from strategy.multistrategy import STRATEGIES, _candidate, scan_seven


def fixture(n=350):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.04, 0.3, n))
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame({'open': close-0.1, 'high': close+0.3,
                         'low': close-0.3, 'close': close,
                         'volume': rng.uniform(100, 200, n)}, index=idx)


def test_all_seven_unique_named_strategies():
    assert len(STRATEGIES) == 7 and len(set(STRATEGIES)) == 7


def test_candidate_math_both_sides():
    df = fixture(); ind = compute_all(df, CFG)
    for side in ('long', 'short'):
        s = _candidate('X', df, ind, CFG, len(df)-1, side, 'Test', 'unit check',
                       stop=float(df.close.iloc[-1]) + (-1 if side=='long' else 1)*float(ind.atr.iloc[-1]))
        assert s is not None
        assert s.rr >= CFG.risk.min_rr
        assert (s.stop < s.entry < s.tp1 < s.tp2) if side == 'long' else (s.tp2 < s.tp1 < s.entry < s.stop)


def test_seven_scan_never_returns_future_timestamp():
    df = fixture(); ind = compute_all(df, CFG)
    st = build_structure(df, CFG.structure)
    out = scan_seven('X', df, ['unclear']*len(df), ind, st, [], [], CFG, 3)
    assert all(df.index[-3] <= s.ts <= df.index[-1] for s in out)
    assert all(s.meta.get('strategy') in STRATEGIES for s in out)


def test_candidate_rejects_wrong_side_stop():
    df=fixture(); ind=compute_all(df,CFG); i=len(df)-1; p=float(df.close.iloc[-1]);
    assert _candidate('X',df,ind,CFG,i,'long','Test','invalid',stop=p+1) is None
    assert _candidate('X',df,ind,CFG,i,'short','Test','invalid',stop=p-1) is None
