"""sizer.py - position size calculator. Leverage changes margin, never risk."""
import streamlit as st

MMR = 0.005  # maintenance margin rate, conservative


def size(balance, risk_pct, entry, stop, direction, lev, tp=None,
         fee_pct=0.05, slip_pct=0.05):
    dist = (entry - stop) if direction == "long" else (stop - entry)
    if balance <= 0 or dist <= 0 or entry <= 0:
        return {}
    risk_cash = balance * risk_pct / 100
    qty = risk_cash / dist
    notional = qty * entry
    margin = notional / max(lev, 1)
    fees = notional * (fee_pct + slip_pct) / 100
    liq = (entry * (1 - 1 / lev + MMR) if direction == "long"
           else entry * (1 + 1 / lev - MMR))
    be = entry + (fees / qty) if direction == "long" else entry - (fees / qty)
    rr = abs(tp - entry) / dist if tp else None
    return {"qty": qty, "notional": notional, "margin": margin,
            "risk_cash": risk_cash, "max_loss": risk_cash + fees,
            "fees": fees, "liq": liq, "be": be, "rr": rr,
            "liq_safe": (liq < stop) if direction == "long" else (liq > stop),
            "max_lev": entry / dist if dist else 0}


def render(balance_default: float = 1000.0, risk_default: float = 0.5):
    st.subheader("Position size calculator")
    st.caption("Leverage changes the margin posted, never the dollars at risk.")

    a, b = st.columns(2)
    balance = a.number_input("Account balance (USDT)", 1.0, 1e9,
                             float(balance_default), step=100.0)
    risk_pct = b.slider("Risk per trade (%)", 0.1, 5.0, float(risk_default), step=0.1)

    c, d = st.columns(2)
    entry = c.number_input("Entry price", 0.0, 1e9, 65000.0, format="%.6f")
    stop = d.number_input("Stop loss price", 0.0, 1e9, 63000.0, format="%.6f")

    e, f = st.columns(2)
    direction = e.radio("Direction", ["long", "short"], horizontal=True)
    lev = f.select_slider("Leverage", [1, 2, 3, 5, 10, 20, 25, 50, 100], value=5,
                          format_func=lambda x: f"{x}x" + (" (spot)" if x == 1 else ""))

    tp = st.number_input("Take profit price (0 = skip)", 0.0, 1e9, 0.0, format="%.6f")

    r = size(balance, risk_pct, entry, stop, direction, lev, tp or None)
    if not r:
        st.info("Balance, entry and stop must be positive, and stop must differ from entry.")
        return

    m = st.columns(4)
    m[0].metric("Position size", f"{r['qty']:.6g}")
    m[1].metric("Notional", f"${r['notional']:,.2f}")
    m[2].metric("Margin required", f"${r['margin']:,.2f}")
    m[3].metric("Max loss", f"${r['max_loss']:,.2f}",
                delta=f"{risk_pct:.1f}% of equity", delta_color="off")

    if r["rr"] is not None:
        st.metric("R:R", f"{r['rr']:.2f}",
                  delta="ok" if r["rr"] >= 2 else "thin",
                  delta_color="normal" if r["rr"] >= 2 else "inverse")

    st.caption(f"Break-even after fees {r['be']:,.6g} · est. liquidation {r['liq']:,.6g} "
               f"· max leverage before liq crosses stop {r['max_lev']:.1f}x")

    if not r["liq_safe"]:
        st.error("Liquidation is closer than your stop. Lower leverage or add margin.")
    if risk_pct > 2:
        st.warning("Above 2% risk per trade.")

    st.divider()
    st.caption("Reduce-only on every exit leg. Round size DOWN to the exchange step size.")
