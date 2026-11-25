"""Seven explainable, closed-candle strategy candidates.

Rules are transparent hypotheses, NOT proven edges. All rolling extrema use
shift(1); the caller must pass only completed OHLCV candles.
"""
import numpy as np
import pandas as pd
from strategy.setups import Setup, scan_bos_retest

STRATEGIES = (
    "BOS Retest", "Trend Pullback", "Momentum Breakout",
    "Range Reversal", "Liquidity Sweep", "EMA Reclaim",
    "Volatility Squeeze",
)


def _candidate(sym, df, ind, cfg, i, direction, pattern, reason, *, stop=None, quality=65):
    entry = float(df['close'].iloc[i]); a = float(ind['atr'].iloc[i])
    if not np.isfinite(a) or a <= 0 or entry <= 0:
        return None
    if stop is None:
        stop = float(df['low'].iloc[i]) - cfg.risk.stop_buffer_atr*a if direction == 'long' else float(df['high'].iloc[i]) + cfg.risk.stop_buffer_atr*a
    risk = (entry-stop) if direction == 'long' else (stop-entry)
    if not (np.isfinite(risk) and cfg.risk.min_stop_atr*a <= risk <= cfg.risk.max_stop_atr*a):
        return None
    tp1_r = max(1.5, min(float(cfg.risk.min_rr), 2.0))
    tp2_r = min(float(cfg.risk.max_tp_r), max(float(cfg.risk.min_rr), 3.0))
    if tp2_r < tp1_r:
        return None
    sign = 1 if direction == 'long' else -1
    return Setup(sym, direction, pattern, df.index[i], entry, float(stop),
                 entry + sign*tp1_r*risk, entry + sign*tp2_r*risk, a,
                 tp2_r, 0.0, [reason, 'closed-candle confirmation', 'ATR-based invalidation'],
                 {'candle_q': quality, 'strategy': pattern, 'confirmation': reason})


def scan_seven(sym, df, hb, ind, st, zu, zd, cfg, recent=3, *, zones_at=None):
    """Return deduplicated candidates for the latest `recent` closed bars."""
    if df is None or len(df) < 220:
        return []
    df = df.sort_index()
    close = df['close'].astype(float); high = df['high'].astype(float)
    low = df['low'].astype(float); vol = df['volume'].astype(float)
    ef = ind['ema_fast']; es = ind['ema_slow']; adx = ind['adx']; rsi = ind['rsi']
    atr = ind['atr']; vr = ind['vol_ratio']
    upper = high.rolling(20).max().shift(1)
    lower = low.rolling(20).min().shift(1)
    range_hi = high.rolling(35).max().shift(1)
    range_lo = low.rolling(35).min().shift(1)
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    width = 4*std / mid.replace(0, np.nan)
    squeeze = width.rolling(80).quantile(0.20).shift(1)
    hb = pd.Series(list(hb), index=df.index).fillna('unclear')
    out = []
    for s in scan_bos_retest(sym, df, hb.to_numpy(), ind, st, zu, zd, cfg, zones_at=zones_at):
        if s.ts >= df.index[max(0, len(df)-recent)]:
            s.meta['strategy'] = 'BOS Retest'
            out.append(s)
    for i in range(max(210, len(df)-max(1, int(recent))), len(df)):
        if not all(np.isfinite(float(v.iloc[i])) for v in (close, ef, es, adx, rsi, atr, vr, upper, lower, range_hi, range_lo, width)):
            continue
        a=float(atr.iloc[i]); p=float(close.iloc[i]); v=float(vr.iloc[i]); bias=str(hb.iloc[i]); trend=float(adx.iloc[i]); rs=float(rsi.iloc[i]);
        if a<=0: continue
        up = ef.iloc[i]>es.iloc[i] and bias!='down'
        down = ef.iloc[i]<es.iloc[i] and bias!='up'
        for direction in ('long','short'):
            sign=1 if direction=='long' else -1
            good = up if sign==1 else down
            # Trend pullback: previous bar touched fast EMA; this one closes back with trend.
            if good and trend>=20 and ((low.iloc[i-1]<=ef.iloc[i-1] and p>ef.iloc[i] and p>close.iloc[i-1]) if sign==1 else (high.iloc[i-1]>=ef.iloc[i-1] and p<ef.iloc[i] and p<close.iloc[i-1])):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'Trend Pullback','EMA50 pullback and trend continuation',stop=(min(low.iloc[i-1],low.iloc[i])-0.25*a if sign==1 else max(high.iloc[i-1],high.iloc[i])+0.25*a)))
            # Momentum breakout: prior 20-bar boundary, volume and ADX.
            if good and trend>=22 and v>=1.3 and (p>upper.iloc[i] if sign==1 else p<lower.iloc[i]):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'Momentum Breakout','20-bar breakout with above-average volume',stop=(upper.iloc[i]-a if sign==1 else lower.iloc[i]+a)))
            # Range reversal: weak trend, wick tests prior range boundary, closes back inside.
            if trend<22 and (low.iloc[i]<range_lo.iloc[i] and p>range_lo.iloc[i] and rs<48 if sign==1 else high.iloc[i]>range_hi.iloc[i] and p<range_hi.iloc[i] and rs>52):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'Range Reversal','Range boundary rejection in low ADX'))
            # Liquidity sweep: prior 20-bar extreme swept, reclaimed on close.
            if bias != ('down' if sign==1 else 'up') and v>=0.8 and (low.iloc[i]<lower.iloc[i]-0.05*a and p>lower.iloc[i] if sign==1 else high.iloc[i]>upper.iloc[i]+0.05*a and p<upper.iloc[i]):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'Liquidity Sweep','Prior swing extreme swept and reclaimed'))
            # EMA reclaim: candle crosses EMA50 in the direction of EMA200 trend.
            if good and (close.iloc[i-1]<=ef.iloc[i-1] and p>ef.iloc[i] and rs>=50 if sign==1 else close.iloc[i-1]>=ef.iloc[i-1] and p<ef.iloc[i] and rs<=50):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'EMA Reclaim','EMA50 close reclaim with EMA200 alignment'))
            # Volatility squeeze: prior narrow Bollinger width, current expansion.
            if good and v>=1.2 and np.isfinite(squeeze.iloc[i]) and width.iloc[i-1]<=squeeze.iloc[i] and (p>upper.iloc[i] if sign==1 else p<lower.iloc[i]):
                out.append(_candidate(sym,df,ind,cfg,i,direction,'Volatility Squeeze','Low-volatility compression followed by breakout'))
    return [s for s in out if s is not None and np.isfinite(s.entry) and np.isfinite(s.stop)]
