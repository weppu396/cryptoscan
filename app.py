"""
app.py - CryptoScan.
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
import signals
import targets
import timemachine
import agent
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

# compatibility shim
import json as _json
if not hasattr(portfolio, "export_json"):
    portfolio.export_json = lambda: _json.dumps(portfolio.load_pf(), indent=2)
if not hasattr(portfolio, "import_json"):
    def _imp(t):
        try:
            d = _json.loads(t)
        except Exception:
            return -1
        if not isinstance(d, dict):
            return -1
        portfolio.save_pf(d)
        return len(d)
    portfolio.import_json = _imp
if not hasattr(portfolio, "gist_push"):
    portfolio.gist_push = lambda token, gid=None: None
if not hasattr(portfolio, "gist_pull"):
    portfolio.gist_pull = lambda token, gid: -1
if not hasattr(portfolio, "classify"):
    portfolio.classify = lambda h, a, w: ("medium", 50.0, None)
if not hasattr(portfolio, "bucket_totals"):
    portfolio.bucket_totals = lambda rows: {"low": 0, "medium": 0, "high": 0}


st.set_page_config(page_title="CryptoScan", page_icon="chart", layout="wide")

OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}
GRADE_ORDER = ["A+", "A", "B", "C", "D"]
GRADE_COLOR = {"A+": "#16a34a", "A": "#22c55e", "B": "#eab308",
               "C": "#f97316", "D": "#ef4444"}
BUCKET_COLOR = {"low": "#22c55e", "medium": "#eab308", "high": "#ef4444"}
WATCH = pathlib.Path("watchlist.json")


def grade_color(g):
    return GRADE_COLOR.get(g, "#94a3b8")


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


def load_watch():
    return json.loads(WATCH.read_text()) if WATCH.exists() else []


def save_watch(w):
    WATCH.write_text(json.dumps(sorted(set(w)), indent=2))


def setup_status(s, ltf):
    cur = float(ltf["close"].iat[-1])
    tol = 0.5 * s.atr
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


def build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity):
    return replace(
        CFG,
        scan=replace(CFG.scan, exchange=exchange, top_n_symbols=top,
                     min_quote_vol_24h=min_vol, ltf=ltf),
        risk=replace(CFG.risk, min_rr=min_rr, max_stop_atr=max_stop_atr,
                     risk_per_trade_pct=risk_pct, account_equity=equity))


def scan_pipeline(cfg, recent_bars, min_grade, preferred, proxy):
    ex, ex_name = resolve_exchange(cfg, preferred=preferred, proxy=proxy)
    universe, note = build_universe(ex, cfg)
    diag = {"exchange": ex_name, "universe_note": note,
            "universe": len(universe), "loaded": 0, "too_short": 0,
            "raw_signals": 0, "after_filters": 0, "after_recency": 0,
            "after_grade": 0, "errors": 0, "sample": universe[:8]}
    bars = max(1500, min_bars_needed(cfg))
    stats, brt = load_stats(), baserates.load()
    fng = signals.fear_greed()
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
                s.meta["base_rate"] = baserates.lookup(brt, s.direction,
                                                       rt.grade, s.stop_atr)
                s.meta["_df"] = ltf
                s.meta["_idx"] = int(ltf.index.get_loc(s.ts))
                try:
                    s.meta["sim"] = timemachine.simulate(
                        ltf, s.meta["_idx"], s.entry, s.stop, s.tp1, s.tp2,
                        horizon=cfg.risk.time_stop_bars, n=4000)
                    s.meta["tm_v"], s.meta["tm_r"] = timemachine.verdict(
                        s.meta["sim"], s.meta["base_rate"])
                except Exception:
                    s.meta["sim"] = {"error": "simulation failed"}
                diag["after_grade"] += 1
                found.append(s)
        except Exception:
            diag["errors"] += 1

    found.sort(key=lambda s: (-s.meta["rating"].confidence, -s.score))
    return found, ex_name, diag, fng


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
        zone = (zu[0] if zu else None) if s.direction == "long" else (zd[0] if zd else None)
        rt = rate(s, row, zone, cfg, stats_src)
        s.meta["rating"] = rt
        if min_grade and GRADE_ORDER.index(rt.grade) > GRADE_ORDER.index(min_grade):
            continue
        graded.append(s)
    setups = graded
    sig = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        sig.setdefault(loc, {"dir": s.direction, "stop": s.stop, "tp1": s.tp1,
                             "tp2": s.tp2, "score": s.score,
                             "grade": s.meta["rating"].grade})
    curve, trades = engine_run(df, sig, cfg, symbol)
    return curve, trades, len(raw), len(setups), ex_name


@st.cache_data(ttl=600, show_spinner=False)
def cached_connection(exchange, proxy):
    cfg = build_cfg(exchange, 40, 30e6, "4h", 2.0, 3.5, 0.5, 10000.0)
    ex, name = resolve_exchange(cfg, preferred=exchange, proxy=proxy)
    return name, list_symbols(ex)


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr,
                risk_pct, equity, recent, min_grade, preferred, proxy):
    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return scan_pipeline(cfg, recent, min_grade, preferred, proxy)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr, max_stop_atr,
                    risk_pct, equity, preferred, proxy, min_grade="D"):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr,
                    max_stop_atr, risk_pct, equity)
    return backtest_pipeline(cfg, symbol, limit, preferred, proxy, min_grade)


# ============================================================
# sidebar
# ============================================================
st.sidebar.title("CryptoScan")

proxy_url = st.sidebar.text_input(
    "Binance proxy URL (optional)", value="",
    help="Only needed for Binance. Leave blank for OKX, MEXC, KuCoin.")

_opts = ["auto"] + CANDIDATES
_default = _opts.index("okx") if "okx" in _opts else 0
exch = st.sidebar.selectbox(
    "Exchange", _opts, index=_default,
    help="Where prices come from. OKX works from this server. Binance needs a proxy.")

if st.sidebar.button("Check connection", help="Test the exchange before scanning."):
    try:
        with st.spinner("resolving..."):
            name, syms = cached_connection(exch, proxy)
        st.session_state["resolved"] = name
        st.session_state["symbols"] = syms
        st.sidebar.success(f"resolved -> {name}")
    except Exception as e:
        st.sidebar.error(str(e))

ltf = st.sidebar.selectbox(
    "Timeframe", ["1h", "4h", "1d"], index=1,
    help="Length of one candle. 4h is the sweet spot: enough noise filtered, enough trades.")
_top = st.sidebar.slider(
    "Symbols to scan", 10, 150, 40, step=10,
    help="How many top-volume coins to check. More = slower but wider.")
_min_vol = st.sidebar.select_slider(
    "Min 24h volume", [1e6, 5e6, 10e6, 30e6, 50e6], value=5e6,
    format_func=lambda v: f"${v/1e6:.0f}M",
    help="Skip illiquid coins. Thin venues like Binance US need $5M or lower.")

recent = st.sidebar.slider(
    "Freshness (bars)", 1, 24, 6,
    help="Only show signals from the last N candles. 6 bars on 4h = 24 hours.")
min_grade = st.sidebar.select_slider(
    "Min grade", GRADE_ORDER, value="B",
    help="A+ best, D worst. B keeps the noise down.")
only_ok = st.sidebar.checkbox(
    "Actionable only", value=False,
    help="Hide signals where price already ran away or already stopped.")
_limit = st.sidebar.slider("Max results", 3, 40, 10,
                           help="How many cards to show.")

equity = st.sidebar.number_input(
    "Equity ($)", 100.0, 10_000_000.0, 10_000.0, step=500.0,
    help="Total account size in dollars. Used only to size trades, never to place them.")
risk_pct = st.sidebar.slider(
    "Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1,
    help="Percent of equity lost if the stop is hit. 0.5 is standard.")
min_rr = st.sidebar.slider(
    "Min R:R", 1.5, 5.0, 2.0, step=0.25,
    help="Reward divided by risk. Below 2 the maths gets thin.")
max_stop_atr = st.sidebar.slider(
    "Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25,
    help="Reject setups needing a stop wider than this many ATRs.")

st.sidebar.subheader("Context")
if st.sidebar.button("Refresh health + macro",
                     help="Load BTC dominance, stablecoin supply, econ calendar, funding, fear index."):
    with st.spinner("loading..."):
        st.session_state["health"] = mh.health()
        st.session_state["macro"] = macro.macro_report("BTC/USDT:USDT")
        st.session_state["fng"] = signals.fear_greed()

hh = st.session_state.get("health")
if hh:
    st.sidebar.metric("Market health", f"{hh['score']}/100", hh["verdict"],
                      help="0-100. Below 35 halves every position size automatically.")
fg_side = st.session_state.get("fng")
if fg_side and fg_side.get("value") is not None:
    st.sidebar.metric("Fear & Greed", fg_side["value"], fg_side["label"],
                      help="0 = extreme fear, 100 = extreme greed. Buys are better low, exits better high.")
mm = (st.session_state.get("macro") or {}).get("state")
if mm and mm.get("blackout"):
    st.sidebar.warning(mm["blackout_reason"])

# ---- proxy compatibility shim ----
_proxy_raw = ""
for _n in ("proxy_url", "proxy_input", "binance_proxy", "proxy"):
    if _n in globals() and globals()[_n] is not None:
        _proxy_raw = str(globals()[_n])
        break
proxy = _proxy_raw.strip() or None
_top = globals().get("_top", globals().get("top", 40))
_min_vol = globals().get("_min_vol", globals().get("min_vol", 30e6))
_limit = globals().get("_limit", globals().get("limit", 10))

(tab_scan, tab_radar, tab_pf, tab_bt, tab_macro,
 tab_alert, tab_journal, tab_help) = st.tabs(
    ["Scan", "Radar", "Portfolio", "Backtest", "Macro", "Alerts", "Journal", "Help"])


# ============================================================
# scan
# ============================================================
with tab_scan:
    st.subheader("Scan")
    if st.button("Run scan", type="primary",
                 help="Check every symbol for the BOS-retest setup."):
        with st.spinner("Scanning..."):
            found, used, diag, fng = cached_scan(
                exch, _top, _min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                equity, recent, min_grade, exch, proxy)
        st.session_state["setups"] = found
        st.session_state["diag"] = diag
        st.session_state["fng"] = fng

    if st.session_state.get("diag"):
        with st.expander("Diagnostics", expanded=not st.session_state.get("setups")):
            st.json(st.session_state["diag"])

    setups = st.session_state.get("setups", [])
    if not setups:
        st.info("No results yet. Open diagnostics for the reason.")
    else:
        rows = [s for s in setups
                if (not only_ok) or s.meta.get("status") in OK_STATUS][:_limit]
        st.caption(f"{len(setups)} graded, {len(rows)} shown")
        base_ctx = {"open_risk_pct": portfolio.open_risk_pct(float(equity)),
                    "open_majors": portfolio.majors_held(),
                    "macro": mm or {}, "health": hh or {}}

        for s in rows:
            rt = s.meta["rating"]
            br = s.meta.get("base_rate")
            sim = s.meta.get("sim") or {}
            verdict, blocks, warns, mult = risk_gate.evaluate(
                s, dict(base_ctx, base_rate=br, sim=sim))
            with st.expander(f"{rt.grade}  {s.symbol}  {s.direction.upper()}  "
                             f"[{s.meta.get('status')}]  {verdict}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(
                    f"<h2 style='color:{grade_color(rt.grade)};margin:0'>{rt.grade}</h2>",
                    unsafe_allow_html=True)
                c2.metric("Confidence", f"{rt.confidence}/100",
                          help="0-100 quality score from trend, volume, stop, R:R, level, RSI.")
                c3.metric("Est hit", f"{br['rate']*100:.0f}%" if br else f"{rt.est_win_rate}%",
                          help="Chance of reaching TP1 before the stop. Measured if backtested, guessed if not.")
                c4.metric("Verdict", verdict, help="PASS = every gate cleared. BLOCK = do not trade.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"{s.entry:,.6f}",
                          help="Limit price to buy at. Not market — this exact price.")
                c2.metric("Stop", f"{s.stop:,.6f}", f"-{s.stop_atr:.2f} ATR",
                          delta_color="inverse",
                          help="Price where you are wrong. Sell here, lose the planned amount.")
                c3.metric("R:R", f"{s.rr:.2f}",
                          help="Reward divided by risk. 3.00 means you win 3x what you risk.")

                for b in blocks:
                    st.error(f"BLOCK: {b}")
                for w in warns:
                    st.warning(w)

                if not sim.get("error"):
                    st.markdown(f"**Time machine: {s.meta.get('tm_v', 'UNKNOWN')}** "
                                f"- {s.meta.get('tm_r', '')}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("P(TP1)", f"{sim['p_tp1_first']*100:.1f}%",
                              help="Share of 4,000 simulated futures that reached TP1 first.")
                    c2.metric("P(stop)", f"{sim['p_stop_first']*100:.1f}%",
                              help="Share that stopped out first.")
                    c3.metric("Sim R", f"{sim['expectancy_R']:+.2f}",
                              help="Average result across all simulated paths. Positive is required.")
                    c4.metric("Break-even", f"{(sim.get('breakeven_p') or 0)*100:.0f}%",
                              help="Hit rate needed to break even. P(TP1) must be above this.")

                st.markdown("**Exit ladder**")
                st.dataframe(pd.DataFrame(targets.exit_ladder(
                    s.entry, abs(s.entry - s.stop))),
                    use_container_width=True, hide_index=True)

                for r in rt.positives:
                    st.markdown(f":green[+ ] {r}")
                for r in rt.negatives:
                    st.markdown(f":red[- ] {r}")

                if agent.available():
                    if st.button("AI review", key=f"ai{s.symbol}{s.ts}",
                                 help="Ask the model for a second opinion. Cannot override a BLOCK."):
                        with st.spinner("reviewing..."):
                            rev = agent.review(agent.build_context(s, s.meta, base_ctx))
                        if rev.get("error"):
                            st.caption(rev["error"])
                        else:
                            st.markdown(f"**AI: {rev.get('verdict')}** "
                                        f"({rev.get('confidence')}/100)")
                            for x in rev.get("reasons", []):
                                st.markdown(f":green[+ ] {x}")
                            for x in rev.get("risks", []):
                                st.markdown(f":red[- ] {x}")
                            st.caption(f"invalidation: {rev.get('invalidation')}")

                if st.button("Build ticket", key=f"t{s.symbol}{s.ts}",
                             help="Create the bracket order: entry, stop, TP1, TP2, size."):
                    t = ticket.build(s, s.meta or {}, mult,
                                     (verdict, blocks, warns, mult), s.reasons)
                    ticket.save(t)
                    st.code(ticket.as_text(t))
                    if verdict == "PASS":
                        alerts.send("entry", t["id"],
                                    f"{t['grade']} {t['symbol']} {t['side']}",
                                    ticket.as_text(t))
                st.link_button("TradingView", tv_link(s.symbol))


# ============================================================
# radar
# ============================================================
with tab_radar:
    st.subheader("Entry radar")
    st.caption("Long-term accumulation zones: price below the 200-day average, "
               "low RSI, drying volume, fear in the market.")
    watch = load_watch()
    c1, c2 = st.columns([3, 1])
    new_sym = c1.text_input("Add to watchlist", "BTC/USDT",
                            help="Coin you would actually want to own long term.")
    if c2.button("Add", help="Save it to the watchlist file.") and new_sym:
        watch.append(new_sym)
        save_watch(watch)
        st.rerun()
    if watch:
        st.caption("watching: " + ", ".join(watch))

    if st.button("Scan watchlist", type="primary",
                 help="Score every watched coin on 200-day distance, RSI, volume and fear.") and watch:
        fng = signals.fear_greed()
        st.session_state["fng"] = fng
        try:
            cfg_r = build_cfg(exch, 40, 5e6, ltf, min_rr, max_stop_atr, risk_pct, equity)
            ex_r, _ = resolve_exchange(cfg_r, preferred=exch, proxy=proxy)
        except Exception as e:
            st.error(str(e))
        else:
            out = []
            for sym in watch:
                try:
                    df = load(ex_r, sym, "1d", limit=1500)
                    if len(df) < 250:
                        continue
                    sig, sev, vz = signals.entry_signals(df, fng)
                    out.append({"symbol": sym, "zone": vz["zone"],
                                "dist_sma": vz["dist_sma"], "rsi": vz["rsi"],
                                "sev": sev, "reasons": sig, "sma200": vz["sma200"]})
                except Exception:
                    continue
            out.sort(key=lambda r: -r["sev"])
            shown = [r for r in out if r["sev"] > 0]
            if not shown:
                st.info("Nothing at value yet.")
            for r in shown:
                d = r["dist_sma"]
                with st.expander(f"{r['symbol']}  {r['zone']}  "
                                 f"{(f'{d:+.1f}% vs 200d' if d is not None else 'n/a')}  "
                                 f"sev {r['sev']}"):
                    for x in r["reasons"]:
                        st.write(f"- {x}")
                    if r["sma200"]:
                        st.write("**Accumulation ladder** — buy 25% at each rung")
                        st.dataframe(pd.DataFrame(targets.accumulation_ladder(r["sma200"])),
                                     use_container_width=True, hide_index=True)
                    if st.button("Alert me", key=f"ra{r['symbol']}",
                                 help="Send this to Telegram or email."):
                        alerts.send("entry", f"radar{r['symbol']}{r['sev']}",
                                    f"ENTRY WATCH {r['symbol']}", "\n".join(r["reasons"]))
                        st.success("sent")


# ============================================================
# portfolio
# ============================================================
with tab_pf:
    st.subheader("Portfolio")
    pf = portfolio.load_pf()

    with st.form("add_pos"):
        a, b, c, d = st.columns(4)
        a_s = a.text_input("Symbol", "BTC/USDT",
                           help="Pair you own, e.g. BTC/USDT. Spot only.")
        a_q = b.number_input("Qty", 0.0, 1e9, 0.0, format="%.8f",
                             help="How many coins. Not dollars — units of the coin.")
        a_p = c.number_input("Price", 0.0, 1e9, 0.0, format="%.6f",
                             help="Your average buy price per ONE coin.")
        a_st = d.number_input("Stop (0 = none)", 0.0, 1e9, 0.0, format="%.6f",
                              help="Price where you give up on this position. Leave 0 if unprotected.")
        if st.form_submit_button("Add") and a_s and a_q > 0 and a_p > 0:
            portfolio.add(a_s, a_q, a_p, a_st or None)
            st.rerun()

    if pf:
        dfp = pd.DataFrame([{"symbol": k, "qty": v["qty"], "avg": v["avg"],
                             "stop": v.get("stop") or 0.0, "tp": v.get("tp") or 0.0}
                            for k, v in pf.items()])
        edited = st.data_editor(dfp, num_rows="dynamic", use_container_width=True,
                                key="pfedit")
        c1, c2, c3 = st.columns(3)
        if c1.button("Save edits", help="Write the table changes back to the file."):
            new = {}
            for _, r in edited.iterrows():
                if not r["symbol"] or r["qty"] <= 0:
                    continue
                new[r["symbol"]] = {"qty": float(r["qty"]), "avg": float(r["avg"]),
                                    "stop": float(r["stop"]) or None,
                                    "tp": float(r["tp"]) or None,
                                    "opened": None, "notes": []}
            portfolio.save_pf(new)
            st.success("saved")
            st.rerun()
        if c2.button("Scan holdings", help="Check every holding for stop proximity, structure breaks, overbought."):
            with st.spinner("checking..."):
                try:
                    cfg_p = build_cfg(exch, 40, 5e6, ltf, min_rr, max_stop_atr,
                                      risk_pct, float(equity))
                    ex_p, _ = resolve_exchange(cfg_p, preferred=exch, proxy=proxy)
                    st.session_state["pf_rows"] = portfolio.health_check(
                        ex_p, cfg_p, float(equity))
                except Exception as e:
                    st.error(str(e))
        if c3.button("Alert all exits", help="Push every RED and AMBER holding to your phone."):
            sent = 0
            for r in st.session_state.get("pf_rows", []):
                if r["severity"] >= 2:
                    alerts.send("exit", f"{r['symbol']}{r['severity']}",
                                f"EXIT {r['symbol']}", "\n".join(r["signals"]))
                    sent += 1
            st.success(f"sent {sent}")

        rows_pf = st.session_state.get("pf_rows", [])
        if rows_pf:
            summ = portfolio.summary(rows_pf, float(equity))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positions", summ["positions"])
            c2.metric("Value", f"${summ['value']:,.0f}",
                      help="Current market value of everything you hold.")
            c3.metric("P&L", f"{summ['pnl_pct']:+.2f}%", f"${summ['pnl_usd']:,.0f}")
            c4.metric("Weight", f"{summ['weight_pct']:.1f}%",
                      help="Portfolio value as a percent of equity.")
            st.caption(f"open risk {portfolio.open_risk_pct(float(equity))}% (cap 3.0) | "
                       f"majors {portfolio.majors_held()}/3")

            tot = portfolio.bucket_totals(rows_pf)
            p1, p2 = st.columns(2)
            with p1:
                bfig = go.Figure(go.Pie(
                    labels=[k for k in tot if tot[k] > 0],
                    values=[v for k, v in tot.items() if v > 0],
                    marker=dict(colors=[BUCKET_COLOR[k] for k in tot if tot[k] > 0]),
                    hole=0.45))
                bfig.update_layout(height=300, template="plotly_dark",
                                   title="By risk bucket", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(bfig, use_container_width=True)
            with p2:
                cfig = go.Figure(go.Pie(
                    labels=[r["symbol"] for r in rows_pf],
                    values=[r.get("value") or 0 for r in rows_pf],
                    hole=0.45))
                cfig.update_layout(height=300, template="plotly_dark",
                                   title="By coin", margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(cfig, use_container_width=True)

            st.caption("Low = small weight, wide stop buffer, calm coin. "
                       "High = large weight, wide or missing stop, volatile coin.")

            for r in rows_pf:
                tag = ("RED" if r["severity"] >= 3 else "AMBER" if r["severity"] == 2 else "GREY")
                mark = " <<< ACT" if r["severity"] >= 2 else ""
                with st.expander(f"[{tag}] {r['symbol']}  {r['bucket'].upper()} risk  "
                                 f"qty {r['qty']}  P&L {r['pnl_pct']}%{mark}"):
                    for x in r["signals"]:
                        st.write(f"- {x}")
                    if r["stop"] and r["avg"]:
                        st.write("**Exit ladder**")
                        st.dataframe(pd.DataFrame(targets.exit_ladder(
                            r["avg"], abs(r["avg"] - r["stop"]))),
                            use_container_width=True, hide_index=True)
                    if st.button("Remove", key=f"rm{r['symbol']}",
                                 help="Delete this holding from the portfolio."):
                        portfolio.remove(r["symbol"])
                        st.rerun()

        st.divider()
        st.download_button("Export backup", portfolio.export_json(),
                           "portfolio.json", "application/json",
                           help="Download a copy. Do this weekly — cloud wipes files.")
        up = st.file_uploader("Restore backup", type=["json"],
                              help="Load a previously exported file.")
        if up and st.button("Import"):
            n = portfolio.import_json(up.read().decode())
            st.success(f"imported {n}" if n >= 0 else "invalid file")
            st.rerun()

        with st.expander("GitHub Gist sync (survives restarts)"):
            st.caption("Token needs the 'gist' scope.")
            gt = st.text_input("Token", type="password")
            gi = st.text_input("Gist id (blank to create)")
            c1, c2 = st.columns(2)
            if c1.button("Push to gist") and gt:
                gid = portfolio.gist_push(gt, gi or None)
                st.success(f"pushed: {gid}" if gid else "failed")
            if c2.button("Pull from gist") and gt and gi:
                n = portfolio.gist_pull(gt, gi)
                st.success(f"pulled {n}" if n >= 0 else "failed")
                st.rerun()
    else:
        st.info("No holdings yet. Add one above.")


# ============================================================
# backtest
# ============================================================
with tab_bt:
    st.subheader("Backtest")
    need = min_bars_needed(build_cfg(exch, 40, 30e6, ltf, min_rr, max_stop_atr,
                                     risk_pct, equity))
    st.info(f"Bars needed for a {CFG.scan.htf} bias on {ltf}: {need}. Use 4h.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    syms = st.session_state.get("symbols") or ["BTC/USDT", "ETH/USDT"]
    b_s = c1.selectbox("Symbol", syms, index=0, help="Coin to replay.")
    b_tf = c2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1, help="Keep 4h.")
    b_n = c3.number_input("Bars", 1000, 30000, max(6000, need), step=500,
                          help="How far back to replay. More bars = more trades = more reliable.")
    b_go = c4.button("Run", type="primary", help="Replay history with fees and slippage.")
    b_g = st.selectbox("Min grade", ["D", "C", "B", "A", "A+"], index=0,
                       help="D tests everything. A+ tests only the best. Compare both.")

    if b_go:
        with st.spinner("Backtesting..."):
            curve, trades, n_raw, n_kept, used = cached_backtest(
                exch, b_s, b_tf, int(b_n), min_rr, max_stop_atr, risk_pct,
                equity, exch, proxy, b_g)
        st.session_state["bt"] = (curve, trades, n_raw, n_kept, b_s, b_tf, used, b_g)

    if "bt" in st.session_state:
        (curve, trades, n_raw, n_kept, su, tu, used, gu) = st.session_state["bt"]
        st.caption(f"{su} {tu} via {used} grade {gu}: {n_raw} raw -> {n_kept} kept")
        if trades is None or trades.empty:
            st.warning("No trades. Lower min R:R to 1.5 or set grade to D.")
        else:
            stats = report(trades, curve, CFG)
            c = st.columns(5)
            c[0].metric("Trades", stats["trades"])
            c[1].metric("Win %", f"{stats['win_rate_pct']}%",
                        help="Percent of trades that made money.")
            c[2].metric("Expectancy", f"{stats['expectancy_R']}R",
                        help="Average result per trade. Must be positive to trade.")
            c[3].metric("PF", stats["profit_factor"], help="Won divided by lost. Above 1 is profitable.")
            c[4].metric("Max DD", f"{stats['max_dd_pct']}%",
                        help="Worst peak-to-trough dip. Your pain tolerance test.")
            gs = grade_stats(trades)
            if not gs.empty:
                st.dataframe(gs, use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"],
                                     line=dict(color="#22c55e")))
            fig.update_layout(height=280, template="plotly_dark", title="Equity curve")
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.json(permutation_test(trades["r"].values))
            with c2:
                st.json(monte_carlo_drawdown(trades["r"].values, risk_pct))
            if st.button("Build base rates", help="Turn these trades into measured hit rates per setup type."):
                try:
                    cfg_b = build_cfg(exch, 40, 5e6, tu, min_rr, max_stop_atr, risk_pct, equity)
                    ex_b, _ = resolve_exchange(cfg_b, preferred=exch, proxy=proxy)
                    dfb = load(ex_b, su, tu, limit=int(b_n))
                    tbl = baserates.build(dfb, CFG, su, lambda s: (
                        s.meta["rating"].grade if s.meta.get("rating") else "C"))
                    baserates.save(tbl)
                    st.dataframe(pd.DataFrame([{"sig": k, **v} for k, v in tbl.items()]),
                                 use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(str(e))


# ============================================================
# macro
# ============================================================
with tab_macro:
    st.subheader("Macro & market health")
    if st.button("Refresh", type="primary", help="Reload health, calendar, funding, fear index."):
        with st.spinner("loading..."):
            st.session_state["health"] = mh.health()
            st.session_state["macro"] = macro.macro_report("BTC/USDT:USDT")
            st.session_state["fng"] = signals.fear_greed()
    if hh:
        c1, c2, c3 = st.columns(3)
        c1.metric("Health", f"{hh['score']}/100")
        c2.metric("Verdict", hh["verdict"])
        c3.metric("Size x", mh.size_multiplier(hh))
        for r in hh["reasons"]:
            st.write(f"- {r}")
    rep = st.session_state.get("macro")
    if rep:
        for n in rep["state"]["notes"]:
            st.write(f"- {n}")
        if rep["calendar"]:
            dfc = pd.DataFrame(rep["calendar"])
            dfc["when"] = pd.to_datetime(dfc["when"]).dt.strftime("%a %d %b %H:%M")
            st.dataframe(dfc[["when", "title", "impact"]],
                         use_container_width=True, hide_index=True)
    fgs = st.session_state.get("fng")
    if fgs and fgs.get("series"):
        st.line_chart(pd.DataFrame({"fear_greed": fgs["series"][::-1]}))
    if st.button("Send desk brief", help="Compose and send the full summary to your phone."):
        body = brief.compose(hh or mh.health(), (rep or {}).get("state") or {},
                             st.session_state.get("setups") or [],
                             st.session_state.get("pf_rows") or [],
                             portfolio.summary(st.session_state.get("pf_rows") or [],
                                               float(equity)))
        st.text(body)
        alerts.send("macro", f"brief{pd.Timestamp.now():%H%M}", "DESK BRIEF", body)


# ============================================================
# alerts
# ============================================================
with tab_alert:
    st.subheader("Alerts")
    c = alerts.config()
    with st.form("alertcfg"):
        t1 = st.text_input("Telegram bot token", c.get("telegram_bot_token", ""),
                           type="password", help="From @BotFather. Keep secret.")
        t2 = st.text_input("Telegram chat id", c.get("telegram_chat_id", ""),
                           help="Your numeric chat id from getUpdates.")
        e1 = st.text_input("SMTP host", c.get("smtp_host", "smtp.gmail.com"),
                           help="Mail server. Gmail = smtp.gmail.com.")
        e2 = st.number_input("SMTP port", 1, 65535, int(c.get("smtp_port") or 587),
                             help="587 for STARTTLS. Do not change unless your host says so.")
        e3 = st.text_input("SMTP user", c.get("smtp_user", ""), help="Your email address.")
        e4 = st.text_input("SMTP password", c.get("smtp_pass", ""), type="password",
                           help="Gmail needs an App Password, not your login password.")
        e5 = st.text_input("Email to", c.get("email_to", ""), help="Where alerts land.")
        w1 = st.text_input("Webhook", c.get("webhook", ""), help="Optional Discord or custom URL.")
        if st.form_submit_button("Save"):
            alerts.save_config({"telegram_bot_token": t1, "telegram_chat_id": t2,
                                "smtp_host": e1, "smtp_port": e2, "smtp_user": e3,
                                "smtp_pass": e4, "email_to": e5, "webhook": w1})
            st.success("saved")

    if st.button("Send test alert", help="Confirm at least one channel works."):
        u = alerts.send("risk", f"t{pd.Timestamp.now().timestamp()}", "TEST",
                        "If you see this, alerts work.")
        st.success(f"sent via {u}" if u else "no channel configured")

    st.divider()
    st.subheader("AI agent")
    ac = agent.config()
    with st.form("agentcfg"):
        u = st.text_input("Base URL", ac.get("base_url", "https://api.venice.ai/api/v1"),
                          help="OpenAI-compatible endpoint. Venice default is pre-filled.")
        k = st.text_input("API key", ac.get("api_key", ""), type="password",
                          help="Stored locally. Never sent anywhere except the endpoint.")
        m = st.text_input("Model", ac.get("model", ""), help="Model id, e.g. a Venice model name.")
        if st.form_submit_button("Save agent"):
            agent.save_config({"base_url": u, "api_key": k, "model": m})
            st.success("saved")


# ============================================================
# journal
# ============================================================
with tab_journal:
    st.subheader("Journal")
    with st.form("jform", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        j_s = c1.text_input("Symbol", help="What you traded.")
        j_d = c2.selectbox("Dir", ["long", "short"], help="Direction.")
        j_g = c3.selectbox("Grade", GRADE_ORDER, help="Grade the tool gave it.")
        j_r = c4.number_input("Result R", -5.0, 30.0, 0.0, step=0.1,
                              help="Result in R multiples. -1 = full stop. +3 = full target.")
        j_f = st.selectbox("Followed rules?", ["yes", "no"],
                           help="Be honest. This column decides what you fix.")
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
        st.dataframe(pd.read_csv("journal.csv"), use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.caption("empty")


# ============================================================
# help
# ============================================================
with tab_help:
    st.markdown("""
## What this app does

It scans hundreds of charts for one pattern, scores each one, simulates the
outcome 4,000 times, and tells you the entry, the stop, the targets and the
size. It also watches what you already hold and shouts when a position is in
trouble or has hit a target.

It does not place orders. You click.

## The three jobs

| Job | Tab | What you get |
|---|---|---|
| Find new trades | Scan | Rated setups with exact numbers and a PASS/BLOCK verdict |
| Find long-term buys | Radar | Coins below their 200-day average with fear, low RSI, dry volume |
| Watch what you hold | Portfolio | Risk buckets, exit ladders, RED/AMBER flags |

## How to read the numbers

- **Grade** — quality score. A+ is best, D is worst.
- **Verdict** — PASS means every rule cleared. BLOCK means do not trade.
- **R:R** — reward divided by risk. 3 means you win three times what you risk.
- **P(TP1)** — how often the target was reached first in simulation.
- **Break-even** — the hit rate you need. P(TP1) must sit above it.
- **Risk bucket** — low/medium/high, from position size, stop width and coin volatility.

## Daily order

1. Sidebar: Refresh health + macro.
2. Radar: scan the watchlist. Anything at deep value goes on the accumulation ladder.
3. Scan: take PASS cards only, time machine SUPPORTED or MARGINAL.
4. Place the trade, then add it to Portfolio with its stop.
5. Portfolio: Scan holdings. Act on RED, sell into the ladder on AMBER.

## Rules

- 0.5% risk per trade, never rounded up
- never widen a stop
- the ladder decides the exit, not your mood
- greed above 80 with a pump score above 55 means reduce
- a BLOCK from the gate beats any opinion, including the AI's
""")


st.sidebar.divider()
st.sidebar.caption("Not financial advice. Paper trade first.")
