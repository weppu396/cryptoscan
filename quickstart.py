"""
quickstart.py - standalone, no folders, no package imports.
    pip install ccxt pandas numpy pyarrow
    python quickstart.py
"""
import ccxt
import pandas as pd
import numpy as np

SYMBOL, TF, LIMIT = "BTC/USDT:USDT", "4h", 800
EQUITY, RISK_PCT, MIN_RR = 10_000.0, 0.5, 2.0
EXCHANGE = "binanceusdm"


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n=14):
    d = c.diff()
    a = 1.0 / n
    ru = d.clip(lower=0).ewm(alpha=a, adjust=False, min_periods=n).mean()
    rd = (-d).clip(lower=0).ewm(alpha=a, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + ru / rd.replace(0, np.nan))).fillna(50)


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def swings(df, L=3, R=3):
    h, l, out = df["high"].values, df["low"].values, []
    for i in range(L, len(df) - R):
        if (h[i] > h[i - L:i]).all() and (h[i] > h[i + 1:i + R + 1]).all():
            out.append({"i": i, "p": float(h[i]), "t": "H", "c": i + R})
        if (l[i] < l[i - L:i]).all() and (l[i] < l[i + 1:i + R + 1]).all():
            out.append({"i": i, "p": float(l[i]), "t": "L", "c": i + R})
    return out


def main():
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True,
                                  "options": {"defaultType": "future"}})
    print(f"Downloading {SYMBOL} {TF} ...")
    raw = ex.fetch_ohlcv(SYMBOL, TF, limit=LIMIT)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").astype(float)

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)

    n = len(df)
    a = float(df["atr"].iat[-1])
    close = float(df["close"].iat[-1])
    trend = "up" if df["ema50"].iat[-1] > df["ema200"].iat[-1] else "down"

    print(f"\n{SYMBOL}  last close {close:,.2f}")
    print(f"bias {trend} | RSI {df['rsi'].iat[-1]:.1f} | ATR {a:,.2f}")

    sw = [s for s in swings(df) if s["c"] <= n - 1]
    hi = [s for s in sw if s["t"] == "H"]
    lo = [s for s in sw if s["t"] == "L"]
    if not hi or not lo:
        print("Not enough confirmed structure."); return

    if trend == "up":
        level = hi[-1]["p"]
        if close <= level:
            print(f"\nNo BOS yet - price {close:,.2f} is below the confirmed "
                  f"swing high {level:,.2f}. Wait for a close above it.")
            return
        stop = lo[-1]["p"] - 0.25 * a
        if stop >= close:
            stop = close - 1.5 * a
        risk = close - stop
        tp = close + MIN_RR * risk
        direction = "long"
    else:
        level = lo[-1]["p"]
        if close >= level:
            print(f"\nNo BOS yet - price {close:,.2f} is above the confirmed "
                  f"swing low {level:,.2f}. Wait for a close below it.")
            return
        stop = hi[-1]["p"] + 0.25 * a
        if stop <= close:
            stop = close + 1.5 * a
        risk = stop - close
        tp = close - MIN_RR * risk
        direction = "short"

    stop_atr = risk / a
    qty = EQUITY * RISK_PCT / 100 / risk

    print(f"\n--- candidate plan ({direction}) ---")
    print(f" broken level  {level:,.2f}  (BOS confirmed)")
    print(f" entry         {close:,.2f}")
    print(f" stop-loss     {stop:,.2f}   ({stop_atr:.2f} ATR)")
    print(f" take-profit   {tp:,.2f}   (R:R {MIN_RR:.1f})")
    print(f" size          {qty:,.4f}   risk ${EQUITY * RISK_PCT / 100:,.2f}")
    if stop_atr > 3.5:
        print(" NOTE: stop wider than 3.5 ATR - a tighter setup may be better.")
    print(f"\nInvalidation: a close beyond {stop:,.2f}.")


if __name__ == "__main__":
    main()