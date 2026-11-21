"""Hand-built OHLC. Each fixture asserts an exact expected event list."""
import pandas as pd


def frame(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="1h")
    return df


def flat(n=40, p=100.0):
    return [[p, p + 0.5, p - 0.5, p, 1000] for _ in range(n)]


def down_then_internal_bull():
    """E1: external down, internal bull break. External must not flip."""
    rows = []
    for i in range(20):
        rows.append([100 - i, 101 - i, 99.5 - i, 99.8 - i, 1000])
    for i in range(20):
        rows.append([80 - i * 0.2, 80.5 - i * 0.2, 79.6 - i * 0.2, 79.9 - i * 0.2, 1000])
    rows += [[79.9, 80.2, 79.7, 79.8, 1000],
             [79.8, 80.6, 79.75, 80.5, 1000],
             [80.5, 81.6, 80.4, 81.5, 1000]]
    return frame(rows)


def equal_highs():
    """Two confirmed highs within tolerance, then one break."""
    rows = flat(30)
    rows += [[100, 100.6, 99.9, 100.2, 1000],
             [100, 100.4, 99.8, 100.0, 1000],
             [100, 100.9, 99.9, 100.4, 1000],
             [100, 100.3, 99.9, 100.0, 1000],
             [100, 101.2, 99.9, 101.0, 1000]]
    return frame(rows)
