import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI — matches TradingView. Do NOT use a plain rolling mean."""
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    a = 1.0 / period                      # Wilder smoothing = alpha 1/n
    ru = up.ewm(alpha=a, adjust=False, min_periods=period).mean()
    rd = dn.ewm(alpha=a, adjust=False, min_periods=period).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1/period, adjust=False,
                             min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus  = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df)
    a = 1.0 / period
    tr_s = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(
        alpha=a, adjust=False, min_periods=period).mean() / tr_s
    mdi = 100 * pd.Series(minus, index=df.index).ewm(
        alpha=a, adjust=False, min_periods=period).mean() / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean().fillna(0.0)


def compute_all(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["atr"]      = atr(df, cfg.regime.atr_period)
    out["atr_pct"]  = out["atr"] / df["close"] * 100.0
    out["ema_fast"] = ema(df["close"], cfg.regime.ema_fast)
    out["ema_slow"] = ema(df["close"], cfg.regime.ema_slow)
    out["rsi"]      = rsi(df["close"], cfg.regime.rsi_period)
    out["adx"]      = adx(df, cfg.regime.adx_period)
    out["vol_ma"]   = df["volume"].rolling(20).mean()
    out["vol_ratio"] = df["volume"] / out["vol_ma"].replace(0, np.nan)
    return out
