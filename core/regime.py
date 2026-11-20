import numpy as np
import pandas as pd
from .indicators import ema


def htf_bias(ltf: pd.DataFrame, cfg) -> pd.Series:
    """
    Resample to the PAIRED htf, compute the bias, then shift by ONE htf bar.
    Only completed htf candles may inform an ltf decision.

    If the resampled series is too short for the configured slow EMA, the EMA
    pair is shortened so the bias still resolves instead of going all-NaN.
    """
    try:
        o = ltf.resample(cfg.scan.htf).agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    except Exception:
        return pd.Series("unclear", index=ltf.index)

    if len(o) < 30:
        return pd.Series("unclear", index=ltf.index)

    fast, slow = cfg.regime.ema_fast, cfg.regime.ema_slow
    if len(o) < slow + 5:
        fast = max(10, min(fast, len(o) // 4))
        slow = max(fast + 5, min(slow, len(o) // 2))

    ef = ema(o["close"], fast)
    es = ema(o["close"], slow)
    bias = pd.Series(
        np.where(ef > es, "up", np.where(ef < es, "down", "unclear")),
        index=o.index).shift(1)
    return bias.reindex(ltf.index, method="ffill").fillna("unclear")


def regime_ok(row, direction, cfg) -> tuple[bool, str]:
    """Returns (allowed, reason). Cheap vetoes that kill most bad trades."""
    if not (cfg.regime.atr_pct_min <= row["atr_pct"] <= cfg.regime.atr_pct_max):
        return False, f"volatility out of band ({row['atr_pct']:.2f}%)"
    if row["adx"] < cfg.regime.adx_trend_min:
        return False, f"chop — ADX {row['adx']:.1f} < {cfg.regime.adx_trend_min}"
    if direction == "long" and row["rsi"] > cfg.regime.rsi_long_max:
        return False, f"long into exhaustion — RSI {row['rsi']:.0f}"
    if direction == "short" and row["rsi"] < cfg.regime.rsi_short_min:
        return False, f"short into exhaustion — RSI {row['rsi']:.0f}"
    return True, "ok"
