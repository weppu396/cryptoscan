"""The Bot list must never show old, played, invalid or far-away entries."""
from types import SimpleNamespace
import pandas as pd
from core.opportunities import assess, current_only


def candles(closes=None):
    closes = closes or [100, 100.1, 100.2]
    idx = pd.date_range('2026-01-01', periods=len(closes), freq='5min', tz='UTC')
    return pd.DataFrame({'open': closes, 'high': [x+.2 for x in closes],
                         'low': [x-.2 for x in closes], 'close': closes}, index=idx)


def setup(df, direction='long', idx=-2, **kwargs):
    d = dict(ts=df.index[idx], direction=direction, entry=100, stop=98, tp1=103,
             tp2=106, atr=2, meta={'status': 'ACTIONABLE', 'rating': {'grade': 'A', 'confidence': 80}},
             symbol='ETH/USDT', weighted_rr=2.4)
    d.update(kwargs)
    return SimpleNamespace(**d)


def test_near_entry_is_actionable():
    df = candles(); assert assess(setup(df), df)[0] == 'ACTIONABLE'


def test_expired_is_hidden():
    df = candles([100]*8); assert assess(setup(df, idx=0), df)[0] == 'EXPIRED'


def test_old_target_hit_never_reappears():
    df = candles([100, 103, 100]); df.loc[df.index[1], 'high'] = 103.5
    assert assess(setup(df, idx=0), df)[0] == 'PLAYED'


def test_old_stop_hit_never_reappears():
    df = candles([100, 97, 100]); df.loc[df.index[1], 'low'] = 97
    assert assess(setup(df, idx=0), df)[0] == 'INVALID'


def test_wrong_side_target_rejected():
    df = candles(); assert assess(setup(df, tp1=99), df)[0] == 'INVALID'


def test_extended_rejected():
    df = candles([100, 100, 104]); assert assess(setup(df), df)[0] == 'PLAYED'
    df = candles([100, 100, 102]); assert assess(setup(df), df)[0] == 'EXTENDED'


def test_stale_exchange_data_rejected():
    df = candles(); assert assess(setup(df), df, timeframe_seconds=300,
                                  now=pd.Timestamp('2026-11-24', tz='UTC'))[0] == 'STALE_DATA'


def test_grade_first_then_score_and_no_played():
    df = candles()
    a = setup(df, symbol='A', meta={'status': 'ACTIONABLE', 'rating': {'grade': 'A+', 'confidence': 80}})
    b = setup(df, symbol='B', meta={'status': 'ACTIONABLE', 'rating': {'grade': 'A', 'confidence': 99}})
    c = setup(df, symbol='C', meta={'status': 'PLAYED', 'rating': {'grade': 'A+', 'confidence': 100}})
    assert [s.symbol for s in current_only([b,c,a])] == ['A','B']
