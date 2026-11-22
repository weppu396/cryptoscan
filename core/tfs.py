"""core/tfs.py - chart timeframes. DISPLAY ONLY.

The strategy always reads 30m context and 3m triggers. Changing the chart
timeframe never changes what the engine looks at.
"""

CHART_TFS = ["1m", "3m", "5m", "15m", "30m", "1H", "4H"]
DATA_TF = {"1H": "1h", "4H": "4h"}
MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240}

STRAT_HTF = "30m"
STRAT_LTF = "3m"

_RULE = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
         "30m": "30min", "1H": "1h", "4H": "4h"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def data_tf(tf: str) -> str:
    """Chart label -> ccxt timeframe string."""
    return DATA_TF.get(tf, tf)


def chart_minutes(tf: str) -> int:
    return MINUTES.get(tf, 60)


def can_resample_to(src_tf: str, dst_tf: str) -> bool:
    """Only upwards. You cannot invent 1m candles from 15m ones."""
    return chart_minutes(dst_tf) >= chart_minutes(src_tf)


def resample(df, tf: str):
    """Aggregate candles up to a coarser display timeframe. Left closed and
    left labelled so no bar sees the future."""
    if df is None or not len(df) or tf not in _RULE:
        return df
    agg = {k: v for k, v in _AGG.items() if k in df.columns}
    out = df.resample(_RULE[tf], label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
