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
            risk = abs(fill - d["stop"])
            if risk > 0:
                qty = (equity * cfg.risk.risk_per_trade_pct / 100.0) / risk
                if qty > 0:
                    pos = {"dir": d["dir"], "stop": d["stop"], "tp1": d["tp1"],
                           "tp2": d["tp2"], "entry": fill, "qty": qty,
                           "risk": risk, "bars": 0, "tp1_done": False,
                           "be": False, "entry_bar": t, "mae": 0.0,
                           "score": d.get("score")}
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
                    half = pos["qty"] * cfg.risk.partial_size
                    equity += (pos["tp1"] - entry) * half - half * pos["tp1"] * fee
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
                    half = pos["qty"] * cfg.risk.partial_size
                    equity += (entry - pos["tp1"]) * half - half * pos["tp1"] * fee
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
                equity += (exit_price - entry) * sign * pos["qty"] - pos["qty"] * exit_price * fee
                trades.append({
                    "symbol": symbol, "dir": d, "entry": entry, "exit": exit_price,
                    "r": (exit_price - entry) * sign / risk, "tag": tag,
                    "bars": pos["bars"], "mae_r": pos["mae"],
                    "score": pos["score"],
                    "entry_ts": df.index[pos["entry_bar"]], "exit_ts": df.index[t],
                })
                pos = None

        if t in signals and pos is None and pending is None and t < last:
            pending = dict(signals[t])

        peak = max(peak, equity)
        curve.append({"ts": df.index[t], "equity": equity,
                      "dd": (equity - peak) / peak if peak else 0.0})

    return pd.DataFrame(curve).set_index("ts"), pd.DataFrame(trades)