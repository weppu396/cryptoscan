"""Historical candle navigation: old candles must remain in the trace."""
import numpy as np
import pandas as pd
import chart2
from core.models import StructureFrame


def candles(n=7200):
    index = pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC')
    close = np.linspace(100, 200, n)
    return pd.DataFrame({'open': close, 'high': close + 2,
                         'low': close - 2, 'close': close,
                         'volume': np.ones(n)}, index=index)


def test_chart_retains_all_loaded_candles_and_initial_view_is_recent():
    df = candles()
    fig = chart2.build(df, StructureFrame(), 'long', mode='structure', bars=120)
    trace = next(t for t in fig.data if t.type == 'candlestick')
    assert len(trace.x) == len(df)
    assert trace.x[0] == df.index[0]
    assert pd.Timestamp(fig.layout.xaxis.range[0]) > df.index[0]
    assert fig.layout.xaxis.fixedrange is False


def test_old_price_levels_are_inside_chart_y_range():
    df = candles()
    fig = chart2.build(df, StructureFrame(), 'long', mode='structure', bars=120)
    lo, hi = fig.layout.yaxis.range
    assert lo < df.low.min()
    assert hi > df.high.max()


def test_uirevision_stable_on_refresh_but_distinct_per_symbol():
    df = candles(300)
    a = chart2.build(df, StructureFrame(), 'long', mode='structure', symbol='BTC/USDT')
    b = chart2.build(df.iloc[1:], StructureFrame(), 'long', mode='structure', symbol='BTC/USDT')
    c = chart2.build(df, StructureFrame(), 'long', mode='structure', symbol='ETH/USDT')
    assert a.layout.uirevision == b.layout.uirevision
    assert a.layout.uirevision != c.layout.uirevision
