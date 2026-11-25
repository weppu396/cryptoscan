import numpy as np
import pandas as pd


def run(df, signals, cfg, symbol="X"):
    fee = cfg.costs.fee_bps / 10_000.0
    slip = cfg.costs.slip_bps / 10_000.0
    equity = cfg.risk.account_equity
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    n = len(df)

    pos = None
    pending = None
    trades = []
    curve = []
    peak = equity
    last = n - 1

    for t in range(n):
        if pending is not None and pos is None:
            d = pending
            fill = o[t] * (1 + slip) if d["dir"] == "long" else o[t] * (1 - slip)
            # Reject wrong-side stops rather than turning invalid trades into
            # apparently positive risk/reward via abs().
            risk = (fill - d["stop"]) if d["dir"] == "long" else (d["stop"] - fill)
            if risk > 0 and np.isfinite(risk) and np.isfinite(fill) and fill > 0:
                qty = (equity * cfg.risk.risk_per_trade_pct / 100.0) / risk
                if qty > 0:
                    pos = {"dir": d["dir"], "stop": d["stop"], "tp1": d["tp1"],
                           "tp2": d["tp2"], "entry": fill, "qty": qty, "initial_qty": qty,
                           "realized_pnl": -qty * fill * fee,
                           "risk": risk, "bars": 0, "tp1_done": False,
                           "be": False, "entry_bar": t, "mae": 0.0,
                           "score": d.get("score"), "grade": d.get("grade"),
                           "strategy": d.get("strategy", "BOS Retest")}
                    equity -= qty * fill * fee
            pending = None

        if pos is not None:
            pos["bars"] += 1
            d = pos["dir"]
            entry = pos["entry"]
            risk = pos["risk"]
            exit_price = None
            tag = None

            adverse = (entry - l[t]) if d == "long" else (h[t] - entry)
            pos["mae"] = max(pos["mae"], adverse / risk)

            if d == "long":
                if o[t] < pos["stop"]:
                    exit_price, tag = o[t] * (1 - slip), "gap_stop"
                elif l[t] <= pos["stop"]:
                    exit_price, tag = pos["stop"] * (1 - slip), "stop"
                elif not pos["tp1_done"] and h[t] >= pos["tp1"]:
                    half = pos["qty"] * max(0.0, min(1.0, cfg.risk.partial_size))
                    tp1_pnl = (pos["tp1"] - entry) * half - half * pos["tp1"] * fee
                    equity += tp1_pnl
                    pos["realized_pnl"] += tp1_pnl
                    pos["qty"] -= half
                    pos["tp1_done"] = True
                    if not pos["be"] and (pos["tp1"] - entry) >= cfg.risk.move_stop_to_be_at_r * risk:
                        pos["stop"] = entry + fee * entry
                        pos["be"] = True
                if exit_price is None and h[t] >= pos["tp2"]:
                    exit_price, tag = pos["tp2"] * (1 - slip), "tp2"
                if exit_price is None and pos["bars"] >= cfg.risk.time_stop_bars:
                    exit_price, tag = c[t] * (1 - slip), "time_stop"
            else:
                if o[t] > pos["stop"]:
                    exit_price, tag = o[t] * (1 + slip), "gap_stop"
                elif h[t] >= pos["stop"]:
                    exit_price, tag = pos["stop"] * (1 + slip), "stop"
                elif not pos["tp1_done"] and l[t] <= pos["tp1"]:
                    half = pos["qty"] * max(0.0, min(1.0, cfg.risk.partial_size))
                    tp1_pnl = (entry - pos["tp1"]) * half - half * pos["tp1"] * fee
                    equity += tp1_pnl
                    pos["realized_pnl"] += tp1_pnl
                    pos["qty"] -= half
                    pos["tp1_done"] = True
                    if not pos["be"] and (entry - pos["tp1"]) >= cfg.risk.move_stop_to_be_at_r * risk:
                        pos["stop"] = entry - fee * entry
                        pos["be"] = True
                if exit_price is None and l[t] <= pos["tp2"]:
                    exit_price, tag = pos["tp2"] * (1 + slip), "tp2"
                if exit_price is None and pos["bars"] >= cfg.risk.time_stop_bars:
                    exit_price, tag = c[t] * (1 + slip), "time_stop"

            if exit_price is None and t == last:
                exit_price = c[t] * (1 - slip if d == "long" else 1 + slip)
                tag = "eod"

            if exit_price is not None:
                sign = 1 if d == "long" else -1
                exit_pnl = (exit_price - entry) * sign * pos["qty"] - pos["qty"] * exit_price * fee
                equity += exit_pnl
                pos["realized_pnl"] += exit_pnl
                trades.append({
                    "symbol": symbol, "dir": d, "strategy": pos.get("strategy", "BOS Retest"),
                    "entry": entry, "exit": exit_price,
                    "r": pos["realized_pnl"] / (pos["initial_qty"] * risk),
                    "net_pnl": pos["realized_pnl"], "tag": tag,
                    "bars": pos["bars"], "mae_r": pos["mae"],
                    "score": pos["score"], "grade": pos["grade"],
                    "entry_ts": df.index[pos["entry_bar"]], "exit_ts": df.index[t],
                })
                pos = None

        if t in signals and pos is None and pending is None and t < last:
            pending = dict(signals[t])

        peak = max(peak, equity)
        curve.append({"ts": df.index[t], "equity": equity,
                      "dd": (equity - peak) / peak if peak else 0.0})

    return pd.DataFrame(curve).set_index("ts"), pd.DataFrame(trades)
