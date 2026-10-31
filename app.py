"""
app.py - CryptoScan dashboard.

    pip install -r requirements.txt
    streamlit run app.py
"""
import inspect
import json
import pathlib
from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import auth
import market
import ui
from config import CFG

st.set_page_config(page_title="CryptoScan", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject()

USER = auth.gate()

from data import CANDIDATES, build_universe, load, probe_exchanges, resolve_exchange
from core.indicators import compute_all
from core.levels import build_zones
from core.regime import htf_bias
from core.structure import build_structure, find_swings
from strategy.rating import rate
from strategy.risk import filter_setups
from strategy.setups import scan_bos_retest
from backtest.engine import run as engine_run
from backtest.metrics import report
from backtest.validate import monte_carlo_drawdown, permutation_test

GRADE_ORDER = ["D", "C", "B", "A", "A+"]
GRADE_TONE = {"A+": "up", "A": "up", "B": "warn", "C": "mut", "D": "dn"}
OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}


# ---------------------------------------------------------------- helpers
def _resolve(cfg, preferred, proxy):
    """Call resolve_exchange across data.py versions."""
    sig = inspect.signature(resolve_exchange).parameters
    kw = {}
    if "preferred" in sig:
        kw["preferred"] = preferred
    if "proxy" in sig:
        kw["proxy"] = proxy
    try:
        return resolve_exchange(cfg, **kw)
    except TypeError:
        kw.pop("proxy", None)
        return resolve_exchange(cfg, **kw)


def build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity):
    return replace(
        CFG,
        scan=replace(CFG.scan, exchange=exchange, top_n_symbols=top,
                     min_quote_vol_24h=min_vol, ltf=ltf),
        risk=replace(CFG.risk, min_rr=min_rr, max_stop_atr=max_stop_atr,
                     risk_per_trade_pct=risk_pct, account_equity=equity),
    )


def setup_status(s, df) -> str:
    cur = float(df["close"].iloc[-1])
    tol = float(df["close"].iloc[-1]) * 0.0025
    if s.direction == "long":
        if cur <= s.stop:
            return "INVALID"
        if cur >= s.tp2:
            return "TARGET_HIT"
        if cur >= s.tp1:
            return "RUNNING"
        if abs(cur - s.entry) <= tol:
            return "ACTIONABLE"
        if cur < s.entry:
            return "BETTER_PRICE"
        return "EXTENDED"
    if cur >= s.stop:
        return "INVALID"
    if cur <= s.tp2:
        return "TARGET_HIT"
    if cur <= s.tp1:
        return "RUNNING"
    if abs(cur - s.entry) <= tol:
        return "ACTIONABLE"
    if cur > s.entry:
        return "BETTER_PRICE"
    return "EXTENDED"


def load_stats() -> dict:
    p = pathlib.Path("./state/edge_stats.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                equity, recent_bars, min_grade, preferred, proxy):
    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    universe = build_universe(ex, cfg)
    stats = load_stats()
    found = []
    diag = {"universe": len(universe), "loaded": 0, "raw": 0, "graded": 0}
    for sym in universe:
        try:
            df = load(ex, sym, cfg.scan.ltf)
            if df is None or len(df) < 500:
                continue
            diag["loaded"] += 1
            ind = compute_all(df, cfg)
            stt = build_structure(df, cfg.structure)
            hb = htf_bias(df, cfg)
            sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")
            raw = scan_bos_retest(sym, df, hb, ind, stt, zu, zd, cfg)
            diag["raw"] += len(raw)
            kept = filter_setups(raw, ind, zu, zd, cfg)
            if recent_bars and kept:
                cut = df.index[max(0, len(df) - 1 - recent_bars)]
                kept = [s for s in kept if s.ts >= cut]
            for s in kept:
                s.meta["status"] = setup_status(s, df)
                zone = (zu[0] if zu else None) if s.direction == "long" else (zd[0] if zd else None)
                rt = rate(s, ind.loc[s.ts], zone, cfg, stats)
                s.meta["rating"] = rt
                diag["graded"] += 1
                if min_grade and GRADE_ORDER.index(rt.grade) > GRADE_ORDER.index(min_grade):
                    continue
                found.append(s)
        except Exception:
            continue
    found.sort(key=lambda s: (-s.meta["rating"].confidence, -s.score))
    return found, ex_name, diag


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr, max_stop_atr,
                    risk_pct, equity, preferred, proxy):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    df = load(ex, symbol, cfg.scan.ltf, limit=limit)
    if df is None or len(df) < 300:
        return None, None, 0, 0
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
        signals.setdefault(loc, {"dir": s.direction, "stop": s.stop,
                                 "tp1": s.tp1, "tp2": s.tp2, "score": s.score})
    curve, trades = engine_run(df, signals, cfg, symbol)
    return curve, trades, len(raw), len(setups)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="auth-logo">◈</div>', unsafe_allow_html=True)
    st.markdown(f"**CryptoScan**")
    st.caption(f"signed in as `{USER}`")
    if st.button("Sign out", use_container_width=True):
        auth.logout()
    st.divider()

    exch = st.selectbox("Exchange", ["auto"] + CANDIDATES, index=0,
                        help="auto uses the first venue that answers")
    proxy = st.text_input("Binance proxy (optional)", value="",
                          placeholder="https://your-worker.workers.dev",
                          help="Cloudflare Worker that unblocks Binance from US servers")
    if st.button("Test venues", use_container_width=True):
        with st.spinner("probing..."):
            st.session_state["health"] = probe_exchanges(proxy.strip() or None)

    st.divider()
    ltf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1)
    top = st.slider("Symbols to scan", 10, 150, 40, step=10)
    min_vol = st.select_slider("Min 24h volume", [10e6, 30e6, 50e6, 100e6, 250e6],
                               value=30e6, format_func=lambda v: f"${v/1e6:.0f}M")
    recent = st.slider("Signal freshness (bars)", 1, 12, 2)
    min_grade = st.select_slider("Minimum grade", GRADE_ORDER, value="B")
    only_ok = st.checkbox("Actionable only", value=True)
    limit = st.slider("Max results", 3, 40, 10)

    st.divider()
    equity = st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0, step=500.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1)
    min_rr = st.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25)
    max_stop_atr = st.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25)

    st.divider()
    st.caption("Not financial advice. Backtest before risking money.")

# ---------------------------------------------------------------- top bar
st.markdown(
    f"""<div class="hero">
    <h1>Market desk</h1>
    <p>Break-of-structure scanner with transparent grading, portfolio heat and
    Monte Carlo validation. Signed in as <b>{USER}</b>.</p>
    </div>""",
    unsafe_allow_html=True,
)

ov = market.overview()
fng = ov.get("fng")
reg, tone = market.regime_label(fng)

c1, c2, c3, c4, c5 = st.columns(5)
ui.stat(c1, "Total market cap", market.money(ov["total_mcap"]),
        "up" if (ov["mcap_change_24h"] or 0) >= 0 else "dn")
ui.stat(c2, "24h change", market.pct(ov["mcap_change_24h"]),
        "up" if (ov["mcap_change_24h"] or 0) >= 0 else "dn")
ui.stat(c3, "BTC dominance", f"{ov['btc_dom']:.1f}%" if ov["btc_dom"] else "—")
ui.stat(c4, "Fear & Greed", f"{fng}" if fng is not None else "—", tone)
ui.stat(c5, "Regime", reg, tone)
st.caption("Source: CoinGecko and alternative.me. Cached 2 minutes.")

st.write("")

tabs = st.tabs(["Overview", "Scan", "Radar", "Portfolio", "Backtest",
                "Macro", "Alerts", "Journal", "Help"])
tab_ov, tab_scan, tab_radar, tab_pf, tab_bt, tab_macro, tab_alert, tab_journal, tab_help = tabs


# ---------------------------------------------------------------- overview
with tab_ov:
    ui.section("Coin detail", "Candles, trend state and the nearest structural zone.")
    d1, d2 = st.columns([2, 1])
    detail_sym = d1.text_input("Symbol", "BTC/USDT", key="detail_sym")
    detail_tf = d2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1, key="detail_tf")

    if st.button("Load detail", type="primary"):
        try:
            cfg = build_cfg(exch, top, min_vol, detail_tf, min_rr, max_stop_atr, risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df = load(ex, detail_sym, detail_tf, limit=400)
            if df is None or len(df) < 120:
                st.warning("Not enough candles for that pair on this venue.")
            else:
                ind = compute_all(df, cfg)
                hb = htf_bias(df, cfg)
                last = float(df["close"].iloc[-1])
                chg = (last / float(df["close"].iloc[-25]) - 1) * 100
                atr_pct = float(ind["atr"].iloc[-1]) / last * 100
                e50 = float(ind["ema50"].iloc[-1])
                e200 = float(ind["ema200"].iloc[-1])
                rsi = float(ind["rsi"].iloc[-1])

                m1, m2, m3, m4, m5 = st.columns(5)
                ui.stat(m1, "Last", f"{last:,.6g}")
                ui.stat(m2, "24 bar %", f"{chg:+.2f}%", "up" if chg >= 0 else "dn")
                ui.stat(m3, "ATR %", f"{atr_pct:.2f}%")
                ui.stat(m4, "RSI", f"{rsi:.1f}", "dn" if rsi > 70 else "up" if rsi < 30 else "")
                ui.stat(m5, "Bias", str(getattr(hb, "bias", hb)),
                        "up" if str(getattr(hb, "bias", "")).startswith("bull") else "dn")
                st.caption(f"EMA50 {e50:,.6g} · EMA200 {e200:,.6g} · venue {ex_name}")

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="price",
                    increasing_line_color="#16c784", decreasing_line_color="#ea3943"))
                fig.add_trace(go.Scatter(x=df.index, y=ind["ema50"], name="EMA50",
                                         line=dict(color="#14b8a6", width=1.2)))
                fig.add_trace(go.Scatter(x=df.index, y=ind["ema200"], name="EMA200",
                                         line=dict(color="#7f8798", width=1.2)))
                fig.update_layout(height=430, template="plotly_dark",
                                  paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                                  margin=dict(l=6, r=6, t=6, b=6),
                                  xaxis_rangeslider_visible=False,
                                  legend=dict(orientation="h", y=1.02))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    st.divider()
    ui.section("Movers", "Largest 24h moves in the top 100 by market cap.")
    rows = market.movers(8)
    if rows:
        st.dataframe(
            pd.DataFrame([{
                "coin": f"{r['symbol']} · {r['name']}",
                "price": f"${r['price']:,.6g}",
                "24h": f"{r['chg']:+.2f}%",
                "volume": market.money(r["vol"]),
            } for r in rows]),
            hide_index=True, use_container_width=True)
    else:
        st.info("Mover data unavailable right now.")


# ---------------------------------------------------------------- scan
with tab_scan:
    ui.section("Live scan", f"Grade {min_grade}+ · last {recent} bars · {ltf}")
    if st.button("Run scan", type="primary"):
        try:
            with st.spinner("Scanning..."):
                found, used, diag = cached_scan(
                    exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                    equity, recent, min_grade, exch, proxy.strip() or None)
            st.session_state["setups"] = found
            st.session_state["used_ex"] = used
            st.session_state["diag"] = diag
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    used = st.session_state.get("used_ex")
    if used:
        st.caption(f"venue: `{used}`")
    diag = st.session_state.get("diag")
    if diag:
        st.caption(f"universe {diag['universe']} · loaded {diag['loaded']} · "
                   f"raw {diag['raw']} · graded {diag['graded']}")

    setups = st.session_state.get("setups", [])
    if not setups:
        st.info("No results yet. Zero setups is a normal outcome.")
    else:
        rows = [s for s in setups if (not only_ok) or s.meta.get("status") in OK_STATUS][:limit]
        if not rows:
            st.warning("Signals exist but none are actionable. Uncheck Actionable only.")
        else:
            st.dataframe(pd.DataFrame([{
                "grade": s.meta["rating"].grade,
                "coin": s.symbol,
                "dir": s.direction.upper(),
                "status": s.meta.get("status"),
                "score": s.score,
                "entry": f"{s.entry:,.6g}",
                "stop": f"{s.stop:,.6g}",
                "tp1": f"{s.tp1:,.6g}",
                "tp2": f"{s.tp2:,.6g}",
            } for s in rows]), hide_index=True, use_container_width=True)

            for s in rows:
                rt = s.meta["rating"]
                with st.expander(f"{rt.grade} · {s.symbol} · {s.direction.upper()} · "
                                 f"{s.meta.get('status')}"):
                    st.markdown(f'<span class="pill {GRADE_TONE.get(rt.grade, "mut")}">'
                                f'{rt.grade} · {rt.score}/100</span>', unsafe_allow_html=True)
                    st.write(f"**Entry** {s.entry:,.6g} · **Stop** {s.stop:,.6g} · "
                             f"**TP1** {s.tp1:,.6g} · **TP2** {s.tp2:,.6g}")
                    st.write(f"Risk **{s.risk_pct:.2f}%** · R:R **{s.rr:.2f}**")
                    for p in rt.positives:
                        st.write(f"- {p}")
                    for n in rt.negatives:
                        st.write(f"- {n}")


# ---------------------------------------------------------------- radar
with tab_radar:
    ui.section("Value radar", "200-day zone proximity for accumulation.")
    try:
        import signals as sig_mod
        if st.button("Refresh radar"):
            st.session_state["radar"] = sig_mod.value_radar()
        radar = st.session_state.get("radar", [])
        if radar:
            st.dataframe(pd.DataFrame([{
                "coin": r["symbol"], "severity": r["sev"],
                "note": " · ".join(r["reasons"]),
            } for r in radar]), hide_index=True, use_container_width=True)
        else:
            st.info("Press Refresh radar.")
    except Exception as e:
        st.warning(f"Radar unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- portfolio
with tab_pf:
    ui.section("Portfolio", "Spot positions, heat and bucket split.")
    try:
        import portfolio as pf_mod
        with st.form("add_pos"):
            a, b, c, d = st.columns(4)
            p_sym = a.text_input("Symbol", "BTC/USDT")
            p_qty = b.number_input("Qty", 0.0, 1e9, 0.0, format="%.8f")
            p_px = c.number_input("Price", 0.0, 1e9, 0.0, format="%.6f")
            p_stop = d.number_input("Stop (0 = none)", 0.0, 1e9, 0.0, format="%.6f")
            if st.form_submit_button("Add / increase") and p_sym and p_qty > 0 and p_px > 0:
                pf_mod.add(p_sym, p_qty, p_px, p_stop or None)
                st.rerun()

        book = pf_mod.load_pf()
        if not book:
            st.info("No open positions.")
        else:
            df_book = pd.DataFrame(book)
            st.dataframe(df_book, hide_index=True, use_container_width=True)
            if {"bucket", "value"}.issubset(df_book.columns):
                fig = go.Figure(go.Pie(labels=df_book["bucket"],
                                       values=df_book["value"], hole=.55))
                fig.update_layout(height=320, template="plotly_dark",
                                  paper_bgcolor="#0f1116", margin=dict(l=6, r=6, t=6, b=6))
                st.plotly_chart(fig, use_container_width=True)
            if hasattr(pf_mod, "export_json"):
                st.download_button("Export backup", pf_mod.export_json(),
                                   file_name="portfolio.json")
    except Exception as e:
        st.warning(f"Portfolio unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- backtest
with tab_bt:
    ui.section("Backtest", "Same filters as the scanner, next-open fills.")
    b1, b2, b3 = st.columns(3)
    bt_sym = b1.text_input("Symbol", "BTC/USDT", key="bt_sym")
    bt_bars = b2.number_input("Bars", 500, 20000, 6000, step=500)
    bt_grade = b3.select_slider("Min grade", GRADE_ORDER, value="D", key="bt_grade")

    if st.button("Run backtest", type="primary"):
        try:
            with st.spinner("Running..."):
                curve, trades, raw_n, kept_n = cached_backtest(
                    exch, bt_sym, ltf, int(bt_bars), min_rr, max_stop_atr,
                    risk_pct, equity, exch, proxy.strip() or None)
            if curve is None:
                st.warning("Not enough history on this venue.")
            else:
                st.caption(f"raw {raw_n} · after filters {kept_n} · trades {len(trades)}")
                fig = go.Figure(go.Scatter(x=curve.index, y=curve.values,
                                           line=dict(color="#14b8a6", width=1.6)))
                fig.update_layout(height=340, template="plotly_dark",
                                  paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                                  margin=dict(l=6, r=6, t=6, b=6))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(trades.tail(50), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- macro
with tab_macro:
    ui.section("Macro", "Calendar blackouts and risk gate.")
    try:
        import macro as macro_mod
        state = macro_mod.state()
        st.json(state)
        if state.get("blackout"):
            st.warning(state.get("blackout_reason", "blackout active"))
    except Exception as e:
        st.warning(f"Macro unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- alerts
with tab_alert:
    ui.section("Alerts", "Telegram and email delivery.")
    try:
        import alerts as al_mod
        if st.button("Send test alert"):
            ok = al_mod.send("test", "test", "CryptoScan test",
                             "If you can read this, alerts work.")
            st.success("sent") if ok else st.error("not configured")
        st.caption(f"configured: {getattr(al_mod, 'is_configured', lambda: '?')()}")
    except Exception as e:
        st.warning(f"Alerts unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- journal
with tab_journal:
    ui.section("Journal", "Log every trade. The rules column is the one that matters.")
    with st.form("journal", clear_on_submit=True):
        j1, j2, j3, j4 = st.columns(4)
        j_sym = j1.text_input("Symbol")
        j_dir = j2.selectbox("Side", ["long", "short"])
        j_r = j3.number_input("Result (R)", -10.0, 10.0, 0.0, step=0.1)
        j_ok = j4.checkbox("Followed rules", value=True)
        j_note = st.text_input("Note")
        if st.form_submit_button("Log"):
            p = pathlib.Path("./state/journal.csv")
            p.parent.mkdir(exist_ok=True)
            row = pd.DataFrame([{"ts": pd.Timestamp.utcnow(), "symbol": j_sym,
                                 "dir": j_dir, "R": j_r, "followed": j_ok, "note": j_note}])
            row.to_csv(p, mode="a", header=not p.exists(), index=False)
            st.success("logged")
    if pathlib.Path("./state/journal.csv").exists():
        st.dataframe(pd.read_csv("./state/journal.csv").tail(100),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- help
with tab_help:
    ui.section("How to use this")
    st.markdown("""
**Flow.** Overview for context → Scan for setups → Portfolio for size and heat
→ Backtest to check a venue → Journal to record what you actually did.

**Grades.** A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52. B and up is the working
default. Grades are heuristic until you calibrate them with a backtest.

**Status.** ACTIONABLE price at entry · BETTER_PRICE pulled back past entry ·
RUNNING TP1 hit · EXTENDED ran away · INVALID stopped out.

**Venues.** Binance geo-blocks US datacentre IPs. Streamlit Cloud runs in the US,
so `auto` lands on OKX or binanceus. A Cloudflare Worker proxy unblocks Binance.

**Login.** Accounts live in `state/users.json`. Set `ADMIN_USER` and `ADMIN_PASS`
in Streamlit secrets so a redeploy reseeds your admin instead of resetting to the
default.
""")


st.divider()
st.caption("CryptoScan · not financial advice")
