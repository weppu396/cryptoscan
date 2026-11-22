"""core/tfs.py - display timeframes. Display only; the strategy owns its pair.

The higher timeframe is never allowed above the 1h ceiling. Changing the chart
timeframe never changes what the engine reads.
"""

CHART_TFS = ["1m", "3m", "5m", "15m", "30m", "1H", "4H"]
DATA_TF = {"1H": "1h", "4H": "4h"}
MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240}

HTF_CEILING_MIN = 60

# Every other spelling the rest of the app uses, including the lowercase and
# daily/weekly forms that htf_for() returns. Without these, chart_minutes()
# fell back to its 60 default and cap_htf() waved "1d" and "4h" straight
# through, which is exactly how 1d/4h ended up on screen.
_UNIT = {"m": 1, "h": 60, "d": 1440, "w": 10080}
_ALIAS = {"1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
          "1d": 1440, "1w": 10080, "1M": 43200}

_RULE = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
         "30m": "30min", "1H": "1h", "4H": "4h"}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

# data.py has no 3m entry. Register it here rather than editing that file.
try:
    import data as _data
    _data.TF_MS.setdefault("3m", 180_000)
    _data.HTF_PER_DAY.setdefault("3m", 480)
except Exception:
    pass


def data_tf(tf: str) -> str:
    return DATA_TF.get(tf, tf)


def chart_minutes(tf: str) -> int:
    """Minutes per bar for any spelling: 1m, 4h, 1d, 1w, 1H, 1D. Never raises."""
    if tf is None:
        return 60
    t = str(tf).strip()
    if not t:
        return 60
    if t in MINUTES:
        return MINUTES[t]
    lo = t.lower()
    if lo in _ALIAS:
        return _ALIAS[lo]
    if lo in MINUTES:
        return MINUTES[lo]
    unit = lo[-1:]
    head = lo[:-1]
    if head.isdigit() and unit in _UNIT:
        return int(head) * _UNIT[unit]
    return 60


def cap_htf(tf: str) -> str:
    """Never return a higher timeframe above the 1h ceiling."""
    return "1h" if chart_minutes(tf) > HTF_CEILING_MIN else (tf or "1h")


def can_resample_to(src: str, dst: str) -> bool:
    return chart_minutes(dst) >= chart_minutes(src)


def resample(df, tf: str):
    """Aggregate upwards only. Left closed, left labelled: no future data."""
    if df is None or not len(df) or tf not in _RULE:
        return df
    agg = {k: v for k, v in _AGG.items() if k in df.columns}
    out = df.resample(_RULE[tf], label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
