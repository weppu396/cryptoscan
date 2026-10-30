"""
app.py - CryptoScan web interface (Streamlit).
    pip install -r requirements.txt
    streamlit run app.py
"""
from dataclasses import replace
import json
import pathlib

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import data as datamod
import macro
import market_health as mh
import baserates
import portfolio
import risk_gate
import ticket
import alerts
import brief
from config import CFG
from data import (resolve_exchange, probe_exchanges, probe_proxy,
                  diagnose_binance, build_universe, load, list_symbols,
                  min_bars_needed, CANDIDATES)
from core.indicators import compute_all
from core.structure import build_structure, find_swings
from core.levels import build_zones
from core.regime import htf_bias
from strategy.setups import scan_bos_retest
from strategy.risk import filter_setups
from strategy.rating import rate
from backtest.engine import run as engine_run
from backtest.metrics import report, bucket_stats, grade_stats
from backtest.validate import permutation_test, monte_carlo_drawdown


st.set_page_config(page_title="CryptoScan", page_icon="chart",
                   layout="wide", initial_sidebar_state="expanded")

OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}
GRADE_ORDER = ["A+", "A", "B", "C", "D"]
GRADE_COLOR = {"A+": "#16a34a", "A": "#22c55e", "B": "#eab308",
               "C": "#f97316", "D": "#ef4444"}


def tv_link(symbol):
    base = symbol.split(":")[0].replace("/", "")
    suffix = ".P" if ":" in symbol else ""
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{base}{suffix}"


def load_stats():
    p = pathlib.Path("edge_stats.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def setup_status(s, ltf):
    cur = float(ltf["close"].iat[-1])
    tol = 0.5 * s.atr
    if s.direction == "long":
        if cur <= s.stop: return "INVALID"
        if cur >= s.tp2: return "TARGET_HIT"
        if cur >= s.tp1: return "RUNNING"
        if abs(cur - s.entry) <= tol: return "ACTIONABLE"
        if cur < s.entry: return "BETTER_PRICE"
        return "EXTENDED"
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


def scan_pipeline(cfg, recent_bars, min_grade, preferred, proxy):
    ex, ex_name = resolve_exchange(cfg, preferred=preferred, proxy=proxy)
    universe, note = build_universe(ex, cfg)
    diag = {"exchange": ex_name, "universe_note": note,
            "universe": len(universe), "loaded": 0, "too_short": 0,
            "raw_signals": 0, "after_filters": 0, "after_recency": 0,
            "after_grade": 0, "errors": 0, "sample": universe[:8]}

    bars = max(1500, min_bars_needed(cfg))
    stats, brt = load_stats(), baserates.load()
    found = []

    for sym in universe:
        try:
            ltf = load(ex, sym, cfg.scan.ltf, limit=bars)
            if len(ltf) < 500:
                diag["too_short"] += 1
                continue
            diag["loaded"] += 1

            ind = compute_all(ltf, cfg)
            stt = build_structure(ltf, cfg.structure)
            hb = htf_bias(ltf, cfg)
            sw = find_swings(ltf, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(ltf, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(ltf, sw, ind["atr"], cfg.structure, "sup")

            raw = scan_bos_retest(sym, ltf, hb, ind, stt, zu, zd, cfg)
            diag["raw_signals"] += len(raw)
            kept = filter_setups(raw, ind, zu, zd, cfg)
            diag["after_filters"] += len(kept)

            if recent_bars is not None and kept:
                cut = ltf.index[max(0, len(ltf) - 1 - recent_bars)]
                kept = [s for s in kept if s.ts >= cut]
            diag["after_recency"] += len(kept)

            for s in kept:
                s.meta["status"] = setup_status(s, ltf)
                row = ind.loc[s.ts]
                zone = (zu[0] if zu else None) if s.direction == "long" \
                    else (zd[0] if zd else None)
                rt = rate(s, row, zone, cfg, stats)
                s.meta["rating"] = rt
                if min_grade and GRADE_ORDER.index(rt.grade) > GRADE_ORDER.index(min_grade):
                    continue
                s.meta["base_rate"] = baserates.lookup(brt, s.direction, rt.grade, s.stop_atr)
                diag["after_grade"] += 1
                found.append(s)
        except Exception:
            diag["errors"] += 1

    found.sort(key=lambda s: (-s.meta["rating"].confidence, -s.score))
    return found, ex_name, diag


def backtest_pipeline(cfg, symbol, limit, preferred, proxy, min_grade=None):
    ex, ex_name = resolve_exchange(cfg, preferred=preferred, proxy=proxy)
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

    stats_src, graded = load_stats(), []
    for s in setups:
        try:
            row = ind.loc[s.ts]
        except KeyError:
            continue
        zone = (zu[0] if zu else None) if s.direction == "long" \
            else (zd[0] if zd else None)
        rt = rate(s, row, zone, cfg, stats_src)
        s.meta["rating"] = rt
        if min_grade and GRADE_ORDER.index(rt.grade) > GRADE_ORDER.index(min_grade):
            continue
        graded.append(s)
    setups = graded

    signals = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        signals.setdefault(loc, {"dir": s.direction, "stop": s.stop,
                                 "tp1": s.tp1, "tp2": s.tp2, "score": s.score,
                                 "grade": s.meta["rating"].grade})
    curve, trades = engine_run(df, signals, cfg, symbol)
    return curve, trades, len(raw), len(setups), ex_name


@st.cache_data(ttl=600, show_spinner=False)
def cached_connection(exchange, proxy):
    cfg = build_cfg(exchange, 40, 30e6, "4h", 2.0, 3.5, 0.5, 10000.0)
    ex, name = resolve_exchange(cfg, preferred=exchange, proxy=proxy)
    return name, list_symbols(ex)


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr,
                risk_pct, equity, recent_bars, min_grade, preferred, proxy):
    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return scan_pipeline(cfg, recent_bars, min_grade, preferred, proxy)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr,
                    max_stop_atr, risk_pct, equity, preferred, proxy,
                    min_grade="D"):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return backtest_pipeline(cfg, symbol, limit, preferred, proxy, min_grade)


@st.cache_data(ttl=300, show_spinner=False)
def cached_probe(proxy):
    return probe_exchanges(proxy or None)


@st.cache_data(ttl=120, show_spinner=False)
def cached_proxy_health(proxy):
    return probe_proxy(proxy)


@st.cache_data(ttl=120, show_spinner=False)
def cached_diagnose(proxy):
    return diagnose_binance(proxy or None)


# ---------------- sidebar ----------------
st.sidebar.title("CryptoScan")
st.sidebar.subheader("Exchange")
proxy_url = st.sidebar.text_input("Binance proxy URL (optional)", value="")
proxy = proxy_url.strip() or None

_opts = ["auto"] + CANDIDATES
_default = _opts.index("okx") if "okx" in _opts else 0
exch = st.sidebar.selectbox("Use this exchange", _opts, index=_default)

if st.sidebar.button("Check connection"):
    try:
        with st.spinner("resolving..."):
            name, syms = cached_connection(exch, proxy)
        st.session_state["resolved"] = name
        st.session_state["symbols"] = syms
        st.sidebar.success(f"resolved -> {name}")
    except Exception as e:
        st.sidebar.error(f"failed: {e}")

ltf = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
_top = st.sidebar.slider("Symbols to scan", 10, 150, 40, step=10)
_min_vol = st.sidebar.select_slider(
    "Min 24h volume", options=[1e6, 5e6, 10e6, 30e6, 50e6, 100e6],
    value=5e6, format_func=lambda v: f"${v/1e6:.0f}M")

st.sidebar.subheader("Filters")
recent = st.sidebar.slider("Signal freshness (bars)", 1, 24, 6)
min_grade = st.sidebar.select_slider("Minimum grade", GRADE_ORDER, value="B")
only_ok = st.sidebar.checkbox("Show only actionable", value=False)
_limit = st.sidebar.slider("Max results", 3, 40, 10)

st.sidebar.subheader("Risk")
equity = st.sidebar.number_input("Account equity ($)", 100.0, 10_000_000.0,
                                 10_000.0, step=500.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1)
min_rr = st.sidebar.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25)
max_stop_atr = st.sidebar.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25)

st.sidebar.subheader("Context")
if st.sidebar.button("Refresh health + macro"):
    with st.spinner("loading..."):
        st.session_state["health"] = mh.health()
        st.session_state["macro"] = macro.macro_report("BTC/USDT:USDT")

hstate = st.session_state.get("health")
if hstate:
    st.sidebar.metric("Market health", f"{hstate['score']}/100", hstate["verdict"])
    for r in hstate["reasons"]:
        st.sidebar.caption(f"- {r}")

mstate = (st.session_state.get("macro") or {}).get("state")
if mstate and mstate.get("blackout"):
    st.sidebar.warning(f"BLACKOUT: {mstate['blackout_reason']}")


tab_scan, tab_bt, tab_pf, tab_ex, tab_macro, tab_alert, tab_journal, tab_help = st.tabs(
    ["Scan", "Backtest", "Portfolio", "Exchanges", "Macro", "Alerts", "Journal", "Help"])


# ---------------- scan ----------------
with tab_scan:
    st.subheader("Scan")
    if st.button("Run scan", type="primary"):
        with st.spinner("Scanning..."):
            setups, used, diag = cached_scan(
                exch, _top, _min_vol, ltf, min_rr, max_stop_atr,
                risk_pct, equity, recent, min_grade, exch, proxy)
        st.session_state["setups"] = setups
        st.session_state["diag"] = diag

    if st.session_state.get("diag"):
        with st.expander("diagnostics", expanded=not st.session_state.get("setups")):
            st.json(st.session_state["diag"])

    setups = st.session_state.get("setups", [])
    if not setups:
        st.info("No results. Check diagnostics above.")
    else:
        rows = [s for s in setups
                if (not only_ok) or s.meta.get("status") in OK_STATUS][:_limit]
        st.caption(f"{len(setups)} graded, {len(rows)} shown")

        ctx_base = {"equity": float(equity),
                    "open_risk_pct": portfolio.open_risk_pct(float(equity)),
                    "open_majors": portfolio.majors_held(),
                    "macro": mstate or {},
                    "health": hstate or {}}

        for s in rows:
            rt = s.meta["rating"]
            br = s.meta.get("base_rate")
            gate_ctx = dict(ctx_base, base_rate=br)
            verdict, blocks, warns, mult = risk_gate.evaluate(s, gate_ctx)

            with st.expander(f"{rt.grade}  {s.symbol}  {s.direction.upper()}  "
                             f"[{s.meta.get('status')}]  {verdict}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<h2 style='color:{GRADE_COLOR[rt.grade]};margin:0'>"
                            f"{rt.grade}</h2>", unsafe_allow_html=True)
                c2.metric("Confidence", f"{rt.confidence}/100")
                c3.metric("Est. hit",
                          f"{br['rate']*100:.0f}%" if br else f"{rt.est_win_rate}%")
                c4.metric("Verdict", verdict)

                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"{s.entry:,.6f}")
                c2.metric("Stop", f"{s.stop:,.6f}", f"-{s.stop_atr:.2f} ATR",
                          delta_color="inverse")
                c3.metric("R:R", f"{s.rr:.2f}")
                c1.metric("TP1", f"{s.tp1:,.6f}")
                c2.metric("TP2", f"{s.tp2:,.6f}")
                c3.metric("Size x", f"{mult}")

                for b in blocks:
                    st.error(f"BLOCK: {b}")
                for w in warns:
                    st.warning(w)
                for r in rt.positives:
                    st.markdown(f":green[+ ] {r}")
                for r in rt.negatives:
                    st.markdown(f":red[- ] {r}")

                if st.button("Build ticket", key=f"t{s.symbol}{s.ts}"):
                    size = s.meta or {}
                    t = ticket.build(s, size, mult, (verdict, blocks, warns, mult),
                                     s.reasons)
                    ticket.save(t)
                    st.code(ticket.as_text(t))
                    if verdict == "PASS":
                        alerts.send("entry", t["id"],
                                    f"{t['grade']} {t['symbol']} {t['side']}",
                                    ticket.as_text(t))
                st.link_button("TradingView", tv_link(s.symbol))

    if st.button("Send desk brief"):
        with st.spinner("composing..."):
            hh = hstate or mh.health()
            mm = mstate or {}
            pp = st.session_state.get("pf_rows") or []
            body = brief.compose(hh, mm, setups, pp,
                                 portfolio.summary(pp, float(equity)))
        st.text(body)
        alerts.send("macro", f"brief{pd.Timestamp.now():%H%M}", "DESK BRIEF", body)


# ---------------- backtest ----------------
with tab_bt:
    st.subheader("Backtest")
    need = min_bars_needed(build_cfg(exch, 40, 30e6, ltf, min_rr,
                                     max_stop_atr, risk_pct, equity))
    st.info(f"Bars needed for a {CFG.scan.htf} bias on {ltf}: {need}. Use 4h.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    syms = st.session_state.get("symbols") or ["BTC/USDT", "ETH/USDT"]
    bt_symbol = c1.selectbox("Symbol", syms, index=0)
    bt_ltf = c2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1, key="bt_tf")
    bt_limit = c3.number_input("Bars", 1000, 30000, max(6000, need), step=500)
    bt_go = c4.button("Run backtest", type="primary")
    bt_min_grade = st.selectbox("Minimum grade", ["D", "C", "B", "A", "A+"], index=0)

    if bt_go:
        with st.spinner("Backtesting..."):
            curve, trades, n_raw, n_kept, used = cached_backtest(
                exch, bt_symbol, bt_ltf, int(bt_limit), min_rr, max_stop_atr,
                risk_pct, equity, exch, proxy, bt_min_grade)
        st.session_state["bt"] = (curve, trades, n_raw, n_kept, bt_symbol,
                                  bt_ltf, used, bt_min_grade)

    if "bt" in st.session_state:
        (curve, trades, n_raw, n_kept, sym_used, tf_used,
         used, g_used) = st.session_state["bt"]
        st.caption(f"{sym_used} {tf_used} via {used} | grade {g_used}: "
                   f"{n_raw} raw -> {n_kept} after filters+grade")

        if trades is None or trades.empty:
            st.warning("No trades. Lower min R:R to 1.5 or set grade back to D.")
        else:
            stats = report(trades, curve, CFG)
            c = st.columns(5)
            c[0].metric("Trades", stats["trades"])
            c[1].metric("Win rate", f"{stats['win_rate_pct']}%")
            c[2].metric("Expectancy", f"{stats['expectancy_R']} R")
            c[3].metric("PF", stats["profit_factor"])
            c[4].metric("Max DD", f"{stats['max_dd_pct']}%")

            gs = grade_stats(trades)
            if not gs.empty:
                st.subheader("Expectancy by grade")
                st.dataframe(gs, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"],
                                     line=dict(width=2, color="#22c55e")))
            fig.update_layout(height=300, template="plotly_dark",
                              title="Equity curve")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.json(permutation_test(trades["r"].values))
            with c2:
                st.json(monte_carlo_drawdown(trades["r"].values, risk_pct))

            if st.button("Build base rates from this run"):
                try:
                    ex_br, _ = resolve_exchange(
                        build_cfg(exch, 40, 5e6, bt_ltf, min_rr, max_stop_atr,
                                  risk_pct, equity), preferred=exch, proxy=proxy)
                    df_br = load(ex_br, sym_used, bt_ltf, limit=int(bt_limit))
                    tbl = baserates.build(df_br, CFG, sym_used,
                                          lambda s: (s.meta.get("rating").grade
                                                     if s.meta.get("rating") else "C"))
                    baserates.save(tbl)
                    st.dataframe(pd.DataFrame(
                        [{"signature": k, **v} for k, v in tbl.items()]),
                        use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"failed: {e}")

            st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)


# ---------------- portfolio ----------------
with tab_pf:
    st.subheader("Portfolio")
    pf_eq = float(equity)

    with st.form("add_pos"):
        a, b, c, d = st.columns(4)
        p_sym = a.text_input("Symbol", "BTC/USDT")
        p_qty = b.number_input("Qty", 0.0, 1e9, 0.0, format="%.8f")
        p_px = c.number_input("Price", 0.0, 1e9, 0.0, format="%.6f")
        p_stop = d.number_input("Stop (0 = none)", 0.0, 1e9, 0.0, format="%.6f")
        if st.form_submit_button("Add / increase") and p_sym and p_qty > 0 and p_px > 0:
            portfolio.add(p_sym, p_qty, p_px, p_stop or None)
            st.success(f"added {p_qty} {p_sym}")

    if st.button("Refresh health check"):
        with st.spinner("checking holdings..."):
            try:
                cfg_pf = build_cfg(exch, 40, 5e6, ltf, min_rr, max_stop_atr,
                                   risk_pct, pf_eq)
                ex_pf, _ = resolve_exchange(cfg_pf, preferred=exch, proxy=proxy)
                st.session_state["pf_rows"] = portfolio.health_check(ex_pf, cfg_pf, pf_eq)
            except Exception as e:
                st.error(f"failed: {e}")

    rows_pf = st.session_state.get("pf_rows", [])
    if rows_pf:
        summ = portfolio.summary(rows_pf, pf_eq)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions", summ["positions"])
        c2.metric("Value", f"${summ['value']:,.0f}")
        c3.metric("P&L", f"{summ['pnl_pct']:+.2f}%", f"${summ['pnl_usd']:,.0f}")
        c4.metric("Weight", f"{summ['weight_pct']:.1f}%")
        st.caption(f"open risk {portfolio.open_risk_pct(pf_eq)}% (cap 3.0) | "
                   f"majors {portfolio.majors_held()}/3")

        for r in rows_pf:
            tag = "RED" if r["severity"] >= 3 else "AMBER" if r["severity"] == 2 else "GREY"
            with st.expander(f"[{tag}] {r['symbol']}  qty {r['qty']}  P&L {r['pnl_pct']}%"):
                for sig in r["signals"]:
                    st.write(f"- {sig}")
                if st.button("Remove", key=f"rm{r['symbol']}"):
                    portfolio.remove(r["symbol"]); st.rerun()

        if summ["exits"] and st.button("Alert all exits"):
            for r in rows_pf:
                if r["severity"] >= 2:
                    alerts.send("exit", f"{r['symbol']}{r['severity']}",
                                f"EXIT {r['symbol']}", "\n".join(r["signals"]))
            st.success("sent")
    else:
        st.caption("no data yet - press Refresh health check")


# ---------------- exchanges ----------------
with tab_ex:
    st.subheader("Exchange health")
    c1, c2, c3 = st.columns(3)
    if c1.button("Test exchanges"):
        with st.spinner("probing..."):
            st.session_state["health_x"] = cached_probe(proxy)
    if c2.button("Test proxy"):
        with st.spinner("..."):
            st.session_state["proxy_h"] = cached_proxy_health(proxy)
    if c3.button("Diagnose binance"):
        with st.spinner("..."):
            st.session_state["diag_bi"] = cached_diagnose(proxy)

    if st.session_state.get("diag_bi"):
        st.json(st.session_state["diag_bi"])
    if st.session_state.get("health_x"):
        st.dataframe(pd.DataFrame([{
            "exchange": n, "ok": "YES" if r["ok"] else "no",
            "markets": r.get("markets", 0), "err": r.get("error", ""),
        } for n, r in st.session_state["health_x"].items()]),
            use_container_width=True, hide_index=True)


# ---------------- macro ----------------
with tab_macro:
    st.subheader("Macro & market health")
    if st.button("Refresh", type="primary"):
        with st.spinner("loading..."):
            st.session_state["health"] = mh.health()
            st.session_state["macro"] = macro.macro_report("BTC/USDT:USDT")

    hh = st.session_state.get("health")
    if hh:
        c1, c2, c3 = st.columns(3)
        c1.metric("Health score", f"{hh['score']}/100")
        c2.metric("Verdict", hh["verdict"])
        c3.metric("Size mult", f"x{mh.size_multiplier(hh)}")
        for r in hh["reasons"]:
            st.write(f"- {r}")

    rep = st.session_state.get("macro")
    if rep:
        s = rep["state"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Blackout", "YES" if s["blackout"] else "no")
        c2.metric("BTC funding", f"{s['funding_annualised_pct']}" if s["funding_annualised_pct"] is not None else "n/a")
        c3.metric("News tone", f"{s['sentiment']:+.2f}")
        for n in s["notes"]:
            st.write(f"- {n}")
        if rep["calendar"]:
            dfc = pd.DataFrame(rep["calendar"])
            dfc["when"] = pd.to_datetime(dfc["when"]).dt.strftime("%a %d %b %H:%M")
            st.dataframe(dfc[["when", "title", "impact"]],
                         use_container_width=True, hide_index=True)


# ---------------- alerts ----------------
with tab_alert:
    st.subheader("Alerts")
    c = alerts.config()
    with st.form("alertcfg"):
        st.markdown("**Telegram** - message @BotFather, /newbot, paste token")
        t1 = st.text_input("Bot token", c.get("telegram_bot_token", ""), type="password")
        t2 = st.text_input("Chat id", c.get("telegram_chat_id", ""))
        st.markdown("**Email** - Gmail needs an App Password")
        e1 = st.text_input("SMTP host", c.get("smtp_host", "smtp.gmail.com"))
        e2 = st.number_input("SMTP port", 1, 65535, int(c.get("smtp_port") or 587))
        e3 = st.text_input("SMTP user", c.get("smtp_user", ""))
        e4 = st.text_input("SMTP password", c.get("smtp_pass", ""), type="password")
        e5 = st.text_input("Send to", c.get("email_to", ""))
        w1 = st.text_input("Webhook URL", c.get("webhook", ""))
        if st.form_submit_button("Save"):
            alerts.save_config({"telegram_bot_token": t1, "telegram_chat_id": t2,
                                "smtp_host": e1, "smtp_port": e2, "smtp_user": e3,
                                "smtp_pass": e4, "email_to": e5, "webhook": w1})
            st.success("saved")

    if st.button("Send test alert"):
        used = alerts.send("risk", f"t{pd.Timestamp.now().timestamp()}",
                           "TEST", "Alerts working.")
        st.success(f"sent via {used}" if used else "no channel configured")


# ---------------- journal ----------------
with tab_journal:
    st.subheader("Journal")
    with st.form("jform", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        j_s = c1.text_input("Symbol")
        j_d = c2.selectbox("Dir", ["long", "short"])
        j_g = c3.selectbox("Grade", GRADE_ORDER)
        j_r = c4.number_input("Result R", -5.0, 30.0, 0.0, step=0.1)
        j_f = st.selectbox("Followed rules?", ["yes", "no"])
        if st.form_submit_button("Add") and j_s:
            row = pd.DataFrame([{"date": pd.Timestamp.now(tz="UTC").date().isoformat(),
                                 "symbol": j_s, "dir": j_d, "grade": j_g,
                                 "result_R": j_r, "followed": j_f}])
            try:
                row = pd.concat([pd.read_csv("journal.csv"), row], ignore_index=True)
            except FileNotFoundError:
                pass
            row.to_csv("journal.csv", index=False)
            st.success("logged")
    try:
        j = pd.read_csv("journal.csv")
        st.dataframe(j, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.caption("no journal yet")


# ---------------- help ----------------
with tab_help:
    st.markdown("""
### Order of operations
1. Refresh health + macro (sidebar)
2. Backtest grade D, then A+ - note both expectancies
3. Build base rates from the run
4. Scan, read the PASS/BLOCK verdicts
5. Build ticket on a PASS, place it on the exchange
6. Add to Portfolio with the stop
7. Refresh health check daily, act on RED/AMBER

### Gates (all must pass)
ACTIONABLE/BETTER_PRICE | grade A+/A | heat under 3% | under 3 majors |
chart matches | stop is a real invalidation

### Sizing
0.5% risk, scaled by the macro and health multipliers. Never round up.

### Never
Widen a stop. Average down. Trade a BLOCK. Skip the journal.
""")


st.sidebar.divider()
st.sidebar.caption("Not financial advice. Paper trade first.")
