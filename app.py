"""
app.py - CryptoScan web interface (Streamlit).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
from dataclasses import replace
import json
import pathlib

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import CFG
from data import resolve_exchange, build_universe, load
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.setups import scan_bos_retest
from strategy.risk import filter_setups
from strategy.rating import rate
from backtest.engine import run as engine_run
from backtest.metrics import report, bucket_stats
from backtest.validate import permutation_test, monte_carlo_drawdown


st.set_page_config(page_title="CryptoScan", page_icon="chart",
                   layout="wide", initial_sidebar_state="expanded")

OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}
GRADE_ORDER = ["A+", "A", "B", "C", "D"]
GRADE_COLOR = {"A+": "#16a34a", "A": "#22c55e", "B": "#eab308",
               "C": "#f97316", "D": "#ef4444"}


def tv_link(symbol: str) -> str:
    base = symbol.split(":")[0].replace("/", "")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{base}.P"


def load_stats():
    p = pathlib.Path("edge_stats.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def setup_status(s, ltf) -> str:
    cur = float(ltf["close"].iat[-1])
    tol = 0.5 * s.atr
    if s.direction == "long":
        if cur <= s.stop: return "INVALID"
        if cur >= s.tp2: return "TARGET_HIT"
        if cur >= s.tp1: return "RUNNING"
        if abs(cur - s.entry) <= tol: return "ACTIONABLE"
        if cur < s.entry: return "BETTER_PRICE"
        return "EXTENDED"
    else:
        if cur >= s.stop: return "INVALID"
        if cur <= s.tp2: return "TARGET_HIT"
        if cur <= s.tp1: return "RUNNING"
        if abs(cur - s.entry) <= tol: return "ACTIONABLE"
        if cur > s.entry: return "BETTER_PRICE"
        return "EXTENDED"


def build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity):
    return replace(
        CFG,
        scan=replace(CFG.scan, exchange=exchange, top_n_symbols=top,
                     min_quote_vol_24h=min_vol, ltf=ltf),
        risk=replace(CFG.risk, min_rr=min_rr, max_stop_atr=max_stop_atr,
                     risk_per_trade_pct=risk_pct, account_equity=equity),
    )


def scan_pipeline(cfg, recent_bars, min_grade):
    ex, ex_name = resolve_exchange(cfg, preferred=cfg.scan.exchange, verbose=False)
    universe = build_universe(ex, cfg)
    stats = load_stats()
    found = []

    for sym in universe:
        try:
            ltf = load(ex, sym, cfg.scan.ltf)
            if len(ltf) < 500:
                continue

            ind = compute_all(ltf, cfg)
            stt = build_structure(ltf, cfg.structure)
            hb = htf_bias(ltf, cfg)
            sw = find_swings(ltf, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(ltf, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(ltf, sw, ind["atr"], cfg.structure, "sup")

            raw = scan_bos_retest(sym, ltf, hb, ind, stt, zu, zd, cfg)
            kept = filter_setups(raw, ind, zu, zd, cfg)

            if recent_bars is not None and kept:
                cut = ltf.index[max(0, len(ltf) - 1 - recent_bars)]
                kept = [s for s in kept if s.ts >= cut]

            for s in kept:
                s.meta["status"] = setup_status(s, ltf)
                row = ind.loc[s.ts]
                zone = (zu[0] if zu else None) if s.direction == "long" \
                    else (zd[0] if zd else None)
                rt = rate(s, row, zone, cfg, stats)
                s.meta["rating"] = rt
                if min_grade and GRADE_ORDER.index(rt.grade) > GRADE_ORDER.index(min_grade):
                    continue
                found.append(s)
        except Exception:
            continue

    found.sort(key=lambda s: (-s.meta["rating"].confidence, -s.score))
    return found, ex_name


def backtest_pipeline(cfg, symbol, limit):
    ex, ex_name = resolve_exchange(cfg, preferred=cfg.scan.exchange, verbose=False)
    df = load(ex, symbol, cfg.scan.ltf, limit=limit)
    if len(df) < 300:
        return None, None, 0, 0, ex_name

    ind = compute_all(df, cfg)
    stt = build_structure(df, cfg.structure)
    hb = htf_bias(df, cfg)
    sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
    zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
    zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")

    raw = scan_bos_retest(symbol, df, hb, ind, stt, zu, zd, cfg)
    setups = filter_setups(raw, ind, zu, zd, cfg)

    signals = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        if loc not in signals:
            signals[loc] = {"dir": s.direction, "stop": s.stop,
                            "tp1": s.tp1, "tp2": s.tp2, "score": s.score}

    curve, trades = engine_run(df, signals, cfg, symbol)
    return curve, trades, len(raw), len(setups), ex_name


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr,
                risk_pct, equity, recent_bars, min_grade):
    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return scan_pipeline(cfg, recent_bars, min_grade)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr,
                    max_stop_atr, risk_pct, equity):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return backtest_pipeline(cfg, symbol, limit)


# ---------------- sidebar ----------------
st.sidebar.title("CryptoScan")
st.sidebar.caption("BOS + retest scanner with transparent grading")

st.sidebar.subheader("Market")
exch_pref = st.sidebar.selectbox(
    "Preferred exchange",
    ["auto", "binanceusdm", "bybit", "okx", "bitget"],
    index=0,
    help="auto tries each in turn and uses the first that responds")
ltf = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
top = st.sidebar.slider("Symbols to scan", 10, 150, 40, step=10)
min_vol = st.sidebar.select_slider(
    "Min 24h volume",
    options=[10e6, 30e6, 50e6, 100e6, 250e6],
    value=30e6, format_func=lambda v: f"${v/1e6:.0f}M")

st.sidebar.subheader("Filters")
recent = st.sidebar.slider("Signal freshness (bars)", 1, 12, 2)
min_grade = st.sidebar.select_slider("Minimum grade", GRADE_ORDER, value="B")
only_ok = st.sidebar.checkbox("Show only actionable", value=True)
limit = st.sidebar.slider("Max results", 3, 40, 10)

st.sidebar.subheader("Risk")
equity = st.sidebar.number_input("Account equity ($)", 100.0, 10_000_000.0,
                                 10_000.0, step=500.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1)
min_rr = st.sidebar.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25)
max_stop_atr = st.sidebar.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25)


tab_scan, tab_bt, tab_journal, tab_help = st.tabs(
    ["Live Scan", "Backtest", "Journal", "How to use"])


# ---------------- live scan ----------------
with tab_scan:
    st.subheader("Live scan")
    st.caption(f"Scanning top {top} symbols on {ltf}. "
               f"Grade {min_grade} and above. Signals from the last {recent} bars.")

    if st.button("Run scan", type="primary"):
        with st.spinner("Scanning. First run downloads candles - allow 1 to 2 minutes..."):
            setups, used = cached_scan(
                exch_pref, top, min_vol, ltf, min_rr, max_stop_atr,
                risk_pct, equity, recent, min_grade)
        st.session_state["setups"] = setups
        st.session_state["used_ex"] = used

    if st.session_state.get("used_ex"):
        used = st.session_state["used_ex"]
        if used == "binanceusdm":
            st.success(f"Connected to {used}")
        else:
            st.info(f"Using {used} (binance unavailable from this region - "
                    f"same data, no problem)")

    setups = st.session_state.get("setups", [])
    stats_src = load_stats()

    if stats_src:
        st.caption(f"Ratings calibrated on {stats_src.get('n', 0)} backtested trades.")
    else:
        st.caption("Ratings use heuristic base rates. Run a backtest to calibrate.")

    if not setups:
        st.info("No results yet. Press Run scan. Zero setups is a normal outcome.")
    else:
        rows = [s for s in setups
                if (not only_ok) or s.meta.get("status") in OK_STATUS][:limit]
        st.caption(f"{len(setups)} graded signals, {len(rows)} shown")

        if not rows:
            st.warning("Signals exist but none are actionable right now. "
                       "Uncheck 'Show only actionable'.")
        else:
            table = pd.DataFrame([{
                "grade": s.meta["rating"].grade,
                "conf": s.meta["rating"].confidence,
                "verdict": s.meta["rating"].verdict,
                "est_win%": s.meta["rating"].est_win_rate,
                "symbol": s.symbol, "dir": s.direction.upper(),
                "status": s.meta.get("status"),
                "entry": round(s.entry, 6), "stop": round(s.stop, 6),
                "tp1": round(s.tp1, 6), "tp2": round(s.tp2, 6),
                "R:R": round(s.rr, 2), "stop_ATR": round(s.stop_atr, 2),
                "qty": round(s.meta.get("qty", 0), 6),
                "risk_$": round(s.meta.get("risk_cash", 0), 2),
                "signal_bar": s.ts.strftime("%Y-%m-%d %H:%M"),
            } for s in rows])

            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download as CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name=f"setups_{pd.Timestamp.utcnow():%Y%m%d_%H%M}.csv",
                mime="text/csv")

            st.divider()
            st.subheader("Setup detail")

            for s in rows:
                rt = s.meta["rating"]
                stop_dist = abs(s.entry - s.stop)
                header = (f"{rt.grade}   {s.symbol}   {s.direction.upper()}   "
                          f"conf {rt.confidence}   [{s.meta.get('status')}]")
                with st.expander(header, expanded=(rt.grade in ("A+", "A"))):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(
                        f"<h2 style='color:{GRADE_COLOR[rt.grade]};margin:0'>"
                        f"{rt.grade}</h2>", unsafe_allow_html=True)
                    c2.metric("Confidence", f"{rt.confidence}/100")
                    c3.metric("Est. TP1 hit", f"{rt.est_win_rate}%")
                    c4.metric("Verdict", rt.verdict)
                    st.caption(f"probability basis: {rt.basis}")

                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entry", f"{s.entry:,.4f}")
                    c2.metric("Stop loss", f"{s.stop:,.4f}",
                              f"-{s.stop_atr:.2f} ATR", delta_color="inverse")
                    c3.metric("R:R", f"{s.rr:.2f}")
                    c1.metric("Take profit 1", f"{s.tp1:,.4f}",
                              f"+{abs(s.tp1-s.entry)/stop_dist:.1f}R")
                    c2.metric("Take profit 2", f"{s.tp2:,.4f}", f"+{s.rr:.1f}R")
                    c3.metric("Size", f"{s.meta.get('qty', 0):,.4f}")

                    st.caption(f"Risk ${s.meta.get('risk_cash', 0):,.2f} "
                               f"({risk_pct}% of ${equity:,.0f})   |   "
                               f"signal bar {s.ts}")

                    st.divider()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Works in favour**")
                        for x in rt.positives:
                            st.markdown(f":green[+ ] {x}")
                    with col_b:
                        st.markdown("**Works against**")
                        if rt.negatives:
                            for x in rt.negatives:
                                st.markdown(f":red[- ] {x}")
                        else:
                            st.caption("nothing material")

                    if rt.improve:
                        st.markdown("**To upgrade this setup**")
                        for x in rt.improve:
                            st.markdown(f"> {x}")

                    st.link_button("Open chart on TradingView", tv_link(s.symbol))


# ---------------- backtest ----------------
with tab_bt:
    st.subheader("Backtest")
    st.caption("Same logic, bar by bar, with fees and slippage. "
               "Writes edge_stats.json which calibrates the ratings.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    default_sym = "BTC/USDT:USDT" if exch_pref == "binanceusdm" else "BTC/USDT"
    bt_symbol = c1.text_input("Symbol", default_sym)
    bt_ltf = c2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1, key="bt_tf")
    bt_limit = c3.number_input("Bars", 1000, 20000, 6000, step=500)
    bt_go = c4.button("Run backtest", type="primary")

    if bt_go:
        with st.spinner("Backtesting. Downloading history if needed..."):
            curve, trades, n_raw, n_kept, used = cached_backtest(
                exch_pref, bt_symbol, bt_ltf, int(bt_limit),
                min_rr, max_stop_atr, risk_pct, equity)
        st.session_state["bt"] = (curve, trades, n_raw, n_kept, bt_symbol, bt_ltf, used)

    if "bt" in st.session_state:
        curve, trades, n_raw, n_kept, sym_used, tf_used, used = st.session_state["bt"]
        st.caption(f"{sym_used} {tf_used} via {used}: "
                   f"{n_raw} raw signals -> {n_kept} after filters")

        if trades is None or trades.empty:
            st.warning("No trades. Lower Minimum R:R to 1.5, raise Max stop ATR to 4.5.")
        else:
            if len(trades) < 30:
                st.warning(f"Only {len(trades)} trades - too few to conclude anything. "
                           f"Raise the bar count.")

            stats = report(trades, curve, CFG)
            c = st.columns(5)
            c[0].metric("Trades", stats["trades"])
            c[1].metric("Win rate", f"{stats['win_rate_pct']}%")
            c[2].metric("Expectancy", f"{stats['expectancy_R']} R")
            c[3].metric("Profit factor", stats["profit_factor"])
            c[4].metric("Max drawdown", f"{stats['max_dd_pct']}%")
            c = st.columns(5)
            c[0].metric("Total R", stats["total_R"])
            c[1].metric("Sharpe", stats["sharpe"])
            c[2].metric("Sortino", stats["sortino"])
            c[3].metric("Avg MAE", f"{stats['avg_mae_R']} R")
            c[4].metric("Avg bars held", stats["avg_bars_held"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"],
                                     name="equity", line=dict(width=2, color="#22c55e")))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                              title="Equity curve", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(x=curve.index, y=curve["dd"] * 100,
                                        name="drawdown", fill="tozeroy",
                                        line=dict(color="#ef4444")))
            dd_fig.update_layout(height=180, margin=dict(l=0, r=0, t=30, b=0),
                                 title="Drawdown %", template="plotly_dark")
            st.plotly_chart(dd_fig, use_container_width=True)

            st.subheader("Hit rate by score bucket")
            bs = bucket_stats(trades)
            if bs["buckets"]:
                st.dataframe(pd.DataFrame(bs["buckets"], columns=[
                    "score_from", "score_to", "win_rate", "trades", "expectancy_R"]),
                    use_container_width=True, hide_index=True)
                pathlib.Path("edge_stats.json").write_text(json.dumps(bs, indent=2))
                st.caption("This is what makes the ratings empirical instead of guessed.")
            else:
                st.caption("Not enough trades per bucket yet.")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Permutation test**")
                st.json(permutation_test(trades["r"].values))
            with c2:
                st.markdown("**Monte carlo drawdown**")
                st.json(monte_carlo_drawdown(trades["r"].values, risk_pct))

            st.subheader("Last 50 trades")
            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
            st.download_button("Download all trades",
                               trades.to_csv(index=False).encode("utf-8"),
                               "trades.csv", "text/csv")


# ---------------- journal ----------------
with tab_journal:
    st.subheader("Trade journal")
    st.caption("Log every trade. The 'followed rules' column is the one that matters.")

    with st.form("journal_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        j_symbol = c1.text_input("Symbol")
        j_dir = c2.selectbox("Direction", ["long", "short"])
        j_grade = c3.selectbox("Grade", GRADE_ORDER)
        j_result = c4.number_input("Result (R)", -5.0, 30.0, 0.0, step=0.1)
        c5, c6 = st.columns(2)
        j_followed = c5.selectbox("Followed the rules?", ["yes", "no"])
        j_notes = c6.text_input("Notes")
        if st.form_submit_button("Add trade") and j_symbol:
            row = pd.DataFrame([{
                "date": pd.Timestamp.utcnow().date().isoformat(),
                "symbol": j_symbol, "dir": j_dir, "grade": j_grade,
                "result_R": j_result, "followed": j_followed, "notes": j_notes}])
            try:
                old = pd.read_csv("journal.csv")
                row = pd.concat([old, row], ignore_index=True)
            except FileNotFoundError:
                pass
            row.to_csv("journal.csv", index=False)
            st.success(f"Logged {j_symbol}")

    try:
        j = pd.read_csv("journal.csv")
        if not j.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades logged", len(j))
            c2.metric("Expectancy", f"{j['result_R'].mean():.2f} R")
            c3.metric("Win rate", f"{(j['result_R'] > 0).mean() * 100:.0f}%")
            c4.metric("Rule adherence", f"{(j['followed'] == 'yes').mean() * 100:.0f}%")
            st.dataframe(j, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.caption("No journal yet.")


# ---------------- help ----------------
with tab_help:
    st.subheader("How to use this")
    st.markdown("""
### Daily routine

Run the scan just after each 4-hour close:
**00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.**

1. Press **Run scan**.
2. Keep only rows marked **ACTIONABLE** or **BETTER_PRICE**.
3. Take the top 3 by confidence, maximum.
4. Total open risk must stay under 3% of equity.
5. Limit order at entry. Stop at the stop loss. TP1 for 50%, TP2 for 50%.
6. When TP1 fills, move the stop to breakeven. Never widen it.
7. Close anything still open after 24 bars.

### Grade meaning

| Grade | Confidence | Action |
|---|---|---|
| A+ | 88-100 | Take it |
| A | 78-87 | Take it |
| B | 66-77 | Take if you have room |
| C | 52-65 | Watch |
| D | below 52 | Skip |

**est. TP1 hit rate** = chance price reaches TP1 before the stop.
Heuristic until you run a backtest, then it becomes the real measured rate.

### Status meaning

- **ACTIONABLE** - price is at entry. Place the order.
- **BETTER_PRICE** - price pulled back past entry. Better fill.
- **RUNNING** - TP1 already hit. Manage the rest.
- **EXTENDED** - price ran away. Wait.
- **INVALID** - stop already hit. Skip.

### Exchange note

Binance blocks US IPs. The app auto-switches to Bybit or OKX when that
happens - identical data, same symbols, no action needed from you.

### Before real money

Paper trade 4-8 weeks, minimum 30 signals. Then 25% size for 4 weeks.
""")


st.sidebar.divider()
st.sidebar.caption("Not financial advice. Backtest before risking money.")
