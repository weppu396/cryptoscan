import numpy as np
import pandas as pd


def report(trades, curve, cfg, bars_per_year=2190):
    if trades is None or trades.empty:
        return {"trades": 0, "note": "no trades passed the filters"}

    r = trades["r"].values.astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    eq = curve["equity"].values.astype(float)
    rets = pd.Series(eq).pct_change().dropna()
    years = max(len(curve) / bars_per_year, 1e-9)
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    dd = float(curve["dd"].min())
    gp = float(wins.sum())
    gl = float(abs(losses.sum()))
    sd = float(rets.std())

    return {
        "trades": int(len(r)),
        "win_rate_pct": round(len(wins) / len(r) * 100, 1),
        "expectancy_R": round(float(r.mean()), 3),
        "median_R": round(float(np.median(r)), 3),
        "total_R": round(float(r.sum()), 2),
        "profit_factor": round(gp / gl, 2) if gl else float("inf"),
        "avg_win_R": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_R": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "avg_mae_R": round(float(trades["mae_r"].mean()), 2),
        "max_dd_pct": round(dd * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "mar_ratio": round(cagr / abs(dd), 2) if dd else float("inf"),
        "sharpe": round(float(rets.mean()) / sd * np.sqrt(bars_per_year), 2) if sd else 0.0,
        "sortino": round(float(rets.mean()) / float(rets[rets < 0].std())
                         * np.sqrt(bars_per_year), 2) if (rets < 0).any() else 0.0,
        "avg_bars_held": round(float(trades["bars"].mean()), 1),
        "final_equity": round(float(eq[-1]), 2),
        "config_hash": cfg.fingerprint(),
    }


def bucket_stats(trades, bucket_size=10):
    """
    Real historical hit rate per score bucket. This is what turns the
    rating from a guess into an empirical estimate.
    """
    if trades is None or trades.empty or "score" not in trades.columns:
        return {"buckets": [], "n": 0}
    t = trades.dropna(subset=["score"]).copy()
    if t.empty:
        return {"buckets": [], "n": 0}
    t["bucket"] = (t["score"] // bucket_size * bucket_size).astype(int)
    out = []
    for b, g in t.groupby("bucket"):
        if len(g) < 5:
            continue
        out.append([int(b), int(b) + bucket_size,
                    round(float((g["r"] > 0).mean()), 3),
                    int(len(g)),
                    round(float(g["r"].mean()), 3)])
    out.sort(key=lambda x: x[0])
    return {"buckets": out, "n": int(len(t)),
            "note": "buckets need >=15 trades to be used by the rating engine"}