"""Deterministic OHLCV with known pivot and breakout candles."""
import pandas as pd


def frame(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    return df


def price_series(prices):
    return frame([[p, p + 0.5, p - 0.5, p, 1000] for p in prices])


def flat(n=40, p=100.0):
    return price_series([p] * n)


def down_then_internal_bull():
    """Both layers break down at 13. Only the internal pivot at 20 is confirmed by 24."""
    return price_series([100] * 6 + [95] + [99] * 6 + [94] + [93] * 6
                        + [96] + [93] * 3 + [97])


def equal_highs():
    """Two identical high pivots at 6 and 14, then a close above both at 21."""
    return price_series([100] * 6 + [101] + [100] * 7 + [101] + [100] * 6 + [102])
