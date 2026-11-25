"""Regression checks for independent strategy backtesting and causal OOS windows."""
from dataclasses import replace
import numpy as np
import pandas as pd
from config import CFG
from backtest.engine import run
from backtest.validate import walk_forward
from strategy.multistrategy import STRATEGIES
from run_backtest import build_trades


def candles(n=520):
    rng = np.random.default_rng(19)
    close = 100 + np.cumsum(rng.normal(0, 0.25, n))
    return pd.DataFrame({"open": close, "high": close + 0.35,
                         "low": close - 0.35, "close": close,
                         "volume": rng.uniform(100, 200, n)},
                        index=pd.date_range('2025-01-01', periods=n, freq='5min', tz='UTC'))


def test_each_strategy_backtests_independently():
    df = candles()
    for name in STRATEGIES:
        curve, trades, raw, kept = build_trades(df, CFG, 'X/USDT', strategy=name)
        assert len(curve) == len(df)
        assert raw >= kept >= 0
        if not trades.empty:
            assert set(trades['strategy']) == {name}


def test_engine_rejects_wrong_side_stop():
    df = candles(8)
    signals = {0: {'dir': 'long', 'stop': 1000, 'tp1': 1020, 'tp2': 1030}}
    _, trades = run(df, signals, CFG)
    assert trades.empty


def test_walk_forward_training_expands_and_oos_never_enters_before_boundary():
    df = candles(1500)
    def pipe(segment, cfg):
        idx = segment.index
        # Always return a trade at the beginning and the end of the provided
        # slice. The OOS filter must discard the initial in-sample trade.
        tr = pd.DataFrame({'entry_ts': [idx[0], idx[-2]], 'r': [999.0, 1.0]})
        return pd.DataFrame({'equity': [1.0]}, index=idx[:1]), tr
    out = walk_forward(pipe, df, CFG, folds=3, grid=[{'min_rr': 2.0, 'max_stop_atr': 3.5}])
    assert len(out) == 3
    assert list(out.train_bars) == sorted(out.train_bars)
    assert all(out.oos_total_R == 1.0)
