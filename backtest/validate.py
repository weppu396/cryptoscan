import numpy as np
import pandas as pd
from dataclasses import replace


def permutation_test(r_multiples, n=2000, seed=7):
    """Randomly flip each trade's sign. If the real mean R sits inside the
    shuffled distribution, the edge is luck."""
    r = np.asarray(r_multiples, dtype=float)
    if len(r) < 5:
        return {"note": "too few trades for a permutation test"}
    rng = np.random.default_rng(seed)
    null = np.array([(r * rng.choice([-1, 1], size=len(r))).mean() for _ in range(n)])
    p = float((null >= r.mean()).mean())
    return {"mean_R": round(float(r.mean()), 3),
            "null_mean_R": round(float(null.mean()), 4),
            "p_value": round(p, 4),
            "significant_at_5pct": bool(p < 0.05)}


def monte_carlo_drawdown(r_multiples, risk_pct=0.5, n=2000, seed=11):
    """Reshuffle trade order and measure the worst drawdown you could have
    suffered from sequencing alone."""
    r = np.asarray(r_multiples, dtype=float)
    if len(r) < 5:
        return {}
    rng = np.random.default_rng(seed)
    dds = []
    for _ in range(n):
        seq = rng.permutation(r) * risk_pct
        eq = 100.0 * np.cumprod(1 + seq / 100.0)
        peak = np.maximum.accumulate(eq)
        dds.append(float(((eq - peak) / peak).min()))
    dds = np.array(dds)
    return {"median_max_dd_pct": round(float(np.median(dds)) * 100, 2),
            "p95_worst_dd_pct": round(float(np.percentile(dds, 5)) * 100, 2)}


def walk_forward(pipe, df, cfg, folds=5, train_frac=0.7, grid=None):
    """
    Anchored walk-forward. For each fold, pick the best parameters on the
    train slice, then report results on the untouched test slice.
    `pipe(df_slice, cfg) -> (curve, trades)`.
    """
    if grid is None:
        grid = [{"min_rr": r, "max_stop_atr": s}
                for r in (1.5, 2.0, 2.5) for s in (2.5, 3.5)]

    n = len(df)
    fold_size = n // folds
    results = []

    for f in range(folds):
        start = f * fold_size
        end = start + fold_size if f < folds - 1 else n
        seg = df.iloc[start:end]
        cut = int(len(seg) * train_frac)
        train, test = seg.iloc[:cut], seg.iloc[cut:]
        if len(train) < 400 or len(test) < 100:
            continue

        best_cfg, best_score = None, -1e9
        for params in grid:
            trial = replace(cfg, risk=replace(cfg.risk, **params))
            try:
                _, tr = pipe(train, trial)
            except Exception:
                continue
            if tr.empty:
                continue
            score = float(tr["r"].sum())
            if score > best_score:
                best_score, best_cfg = score, trial

        if best_cfg is None:
            continue

        _, tr_test = pipe(test, best_cfg)
        results.append({
            "fold": f + 1,
            "train_bars": len(train),
            "test_bars": len(test),
            "chosen_min_rr": best_cfg.risk.min_rr,
            "chosen_max_stop_atr": best_cfg.risk.max_stop_atr,
            "oos_trades": int(len(tr_test)),
            "oos_total_R": round(float(tr_test["r"].sum()), 2) if not tr_test.empty else 0.0,
            "oos_expectancy_R": round(float(tr_test["r"].mean()), 3) if not tr_test.empty else 0.0,
        })

    if results:
        tot = sum(x["oos_total_R"] for x in results)
        print(f"\nWalk-forward: {len(results)} folds, total out-of-sample R = {tot:+.2f}")
    return pd.DataFrame(results)