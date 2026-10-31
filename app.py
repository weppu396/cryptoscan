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
MIN_BARS = 300

# venues ranked by perpetual depth. shallow spot-only venues go last.
VENUE_ORDER = ["binance", "binanceusdm", "okx", "bybit", "bitget",
               "kucoinfutures", "mexc", "gateio", "binanceus"]
MIN_PERPS = 20


# ---------------------------------------------------------------- helpers
def _call(mod, names, *a, **k):
    """Call the first attribute that exists. Survives module drift."""
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f(*a, **k)
    raise AttributeError(f"{mod.__name__} has none of {names}")


def _has(mod, names) -> bool:
    return any(callable(getattr(mod, n, None)) for n in names)


def _perps(ex) -> int:
    """Count USDT perpetual markets. A proxy for usable depth."""
    try:
        ex.load_markets()
    except Exception:
        return 0
    return sum(1 for s, m in ex.markets.items()
               if ":" in s and m.get("quote") == "USDT" and m.get("active"))


def _resolve(cfg, preferred, proxy):
    """Pick a venue with real perp depth. Never silently lands on binanceus."""
    sig = inspect.signature(resolve_exchange).parameters
    base = {"preferred": preferred} if "preferred" in sig else {}
    if "proxy" in sig:
        base["proxy"] = proxy

    if preferred and preferred != "auto":
        order = [preferred] + [v for v in VENUE_ORDER if v != preferred]
    else:
        order = list(VENUE_ORDER)

    best, best_n = None, -1
    for name in order:
        kw = dict(base)
        if "preferred" in sig:
            kw["preferred"] = name
        try:
            ex, used = resolve_exchange(cfg, **kw)
        except TypeError:
            kw.pop("proxy", None)
            try:
                ex, used = resolve_exchange(cfg, **kw)
            except Exception:
                continue
        except Exception:
            continue
        n = _perps(ex)
        if n >= MIN_PERPS:
            return ex, used
        if n > best_n:
            best, best_n = (ex, used), n
    if best:
        return best
    raise RuntimeError("no reachable venue")


def _universe(ex, cfg, min_vol):
    """build_universe with a volume floor that backs off until it fills."""
    last = []
    for div in (1, 5, 25, 0):
        floor = min_vol / div if div else 0
        try:
            u = build_universe(ex, replace(cfg, scan=replace(
                cfg.scan, min_quote_vol_24h=floor)))
        except Exception:
            u = []
        if len(u) >= 15:
            return u, floor
        last = u or last
    return last, 0


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
    tol = cur * 0.0025
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


# ---------------------------------------------------------------- pipelines
@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                equity, recent_bars, min_grade, preferred, proxy):
    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    universe, floor = _universe(ex, cfg, min_vol)
    stats = load_stats()
    found = []
    diag = {"venue": ex_name, "universe": len(universe), "floor": floor,
            "loaded": 0, "short": 0, "raw": 0, "graded": 0}

    for sym in universe:
        try:
            df = load(ex, sym, cfg.scan.ltf)
            if df is None or len(df) < MIN_BARS:
                diag["short"] += 1
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
    return found, diag


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr, max_stop_atr,
                    risk_pct, equity, preferred, proxy):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, _ = _resolve(cfg, preferred, proxy)
    df = load(ex, symbol, cfg.scan.ltf, limit=limit)
    if df is None or len(df) < MIN_BARS:
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
    st.markdown('<div class="auth-logo">◈</div>', unsafe_allow_html=True)
    st.markdown("**CryptoScan**")
    st.caption(f"signed in as `{USER}`")
    if st.button("Sign out", use_container_width=True,
                 help="End this session and return to the login screen."):
        auth.logout()
    st.divider()

    exch = st.selectbox(
        "Exchange", ["auto"] + CANDIDATES, index=0,
        help="auto scores venues by perpetual-market depth and picks the "
             "deepest one that answers. binanceus is spot-only and thin, so "
             "auto avoids it unless nothing else works.")
    proxy = st.text_input(
        "Binance proxy (optional)", value="",
        placeholder="https://your-worker.workers.dev",
        help="Cloudflare Worker URL. Only needed for Binance, which returns "
             "HTTP 451 to US datacentre IPs.")
    if st.button("Test venues", use_container_width=True,
                 help="Probe every venue and show which ones answer, with latency."):
        with st.spinner("probing..."):
            st.session_state["health"] = probe_exchanges(proxy.strip() or None)

    h = st.session_state.get("health")
    if h:
        st.dataframe(pd.DataFrame([{
            "venue": n, "ok": "YES" if r["ok"] else "no",
            "markets": r.get("markets", 0), "ms": r["ms"],
        } for n, r in h.items()]), hide_index=True, use_container_width=True)

    st.divider()
    ltf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1,
                       help="Bar size. 4h is the default swing horizon; 1d needs "
                            "far more history to warm the EMA200 up.")
    top = st.slider("Symbols to scan", 10, 150, 40, step=10,
                    help="How many of the highest-volume pairs to check. "
                         "More symbols means a slower scan.")
    min_vol = st.select_slider(
        "Min 24h volume", [10e6, 30e6, 50e6, 100e6, 250e6], value=30e6,
        format_func=lambda v: f"${v/1e6:.0f}M",
        help="Liquidity floor. The scanner lowers this automatically if it "
             "would otherwise leave you with almost nothing to check.")
    recent = st.slider("Signal freshness (bars)", 1, 12, 2,
                       help="How many bars back a break may be and still count. "
                            "Small numbers mean fresher, fewer signals.")
    min_grade = st.select_slider("Minimum grade", GRADE_ORDER, value="B",
                                 help="A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52. "
                                      "B and up is the working default.")
    only_ok = st.checkbox("Actionable only", value=True,
                          help="Hide setups that are already extended, invalid "
                               "or past target.")
    limit = st.slider("Max results", 3, 40, 10,
                      help="Cap on cards rendered.")

    st.divider()
    equity = st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0,
                             step=500.0,
                             help="Total account size. Position size is derived "
                                  "from this and risk per trade.")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1,
                         help="Percent of equity lost if the stop is hit. "
                              "0.5% means $50 on a $10,000 account.")
    min_rr = st.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25,
                       help="Reward-to-risk floor. Setups below it are dropped.")
    max_stop_atr = st.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25,
                             help="Stop distance cap in ATR units. Wider stops "
                                  "get skipped rather than sized down.")

    st.divider()
    st.caption("Not financial advice. Backtest before risking money.")

# ---------------------------------------------------------------- header
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
st.caption("Source: CoinGecko and alternative.me. Cached two minutes.")

st.write("")

tabs = st.tabs(["Overview", "Scan", "Radar", "Portfolio", "Backtest",
                "Macro", "Alerts", "Journal", "Guide"])
(tab_ov, tab_scan, tab_radar, tab_pf, tab_bt,
 tab_macro, tab_alert, tab_journal, tab_guide) = tabs


# ---------------------------------------------------------------- overview
with tab_ov:
    ui.section("Coin detail", "Candles, trend state and venue depth.")
    d1, d2 = st.columns([2, 1])
    detail_sym = d1.text_input("Symbol", "BTC/USDT", key="detail_sym",
                               help="Unified ccxt symbol, e.g. BTC/USDT for spot "
                                    "or BTC/USDT:USDT for a perpetual.")
    detail_tf = d2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1,
                             key="detail_tf", help="Chart resolution.")

    if st.button("Load detail", type="primary",
                 help="Fetch candles and render the chart with EMAs."):
        try:
            cfg = build_cfg(exch, top, min_vol, detail_tf, min_rr,
                            max_stop_atr, risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df = load(ex, detail_sym, detail_tf, limit=400)
            if df is None or len(df) < 120:
                st.warning(f"Not enough candles for {detail_sym} on {ex_name}. "
                           f"Try the other symbol format or another venue.")
            else:
                ind = compute_all(df, cfg)
                hb = htf_bias(df, cfg)
                last = float(df["close"].iloc[-1])
                chg = (last / float(df["close"].iloc[-25]) - 1) * 100
                atr_pct = float(ind["atr"].iloc[-1]) / last * 100
                rsi = float(ind["rsi"].iloc[-1])
                bias = str(getattr(hb, "bias", hb))

                m1, m2, m3, m4, m5 = st.columns(5)
                ui.stat(m1, "Last", f"{last:,.6g}")
                ui.stat(m2, "24 bar %", f"{chg:+.2f}%", "up" if chg >= 0 else "dn")
                ui.stat(m3, "ATR %", f"{atr_pct:.2f}%")
                ui.stat(m4, "RSI", f"{rsi:.1f}",
                        "dn" if rsi > 70 else "up" if rsi < 30 else "")
                ui.stat(m5, "Bias", bias, "up" if bias.startswith("bull") else "dn")
                st.caption(f"EMA50 {float(ind['ema50'].iloc[-1]):,.6g} · "
                           f"EMA200 {float(ind['ema200'].iloc[-1]):,.6g} · "
                           f"venue {ex_name}")

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="price",
                    increasing_line_color="#16c784",
                    decreasing_line_color="#ea3943"))
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
        st.dataframe(pd.DataFrame([{
            "coin": f"{r['symbol']} · {r['name']}",
            "price": f"${r['price']:,.6g}",
            "24h": f"{r['chg']:+.2f}%",
            "volume": market.money(r["vol"]),
        } for r in rows]), hide_index=True, use_container_width=True)
    else:
        st.info("Mover data unavailable right now.")


# ---------------------------------------------------------------- scan
with tab_scan:
    ui.section("Live scan", f"Grade {min_grade}+ · last {recent} bars · {ltf}")
    if st.button("Run scan", type="primary",
                 help="Check every symbol in the universe for the setup."):
        try:
            with st.spinner("Scanning..."):
                found, diag = cached_scan(
                    exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                    equity, recent, min_grade, exch, proxy.strip() or None)
            st.session_state["setups"] = found
            st.session_state["diag"] = diag
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    diag = st.session_state.get("diag")
    if diag:
        st.caption(f"venue `{diag['venue']}` · universe {diag['universe']} "
                   f"(floor ${diag['floor']/1e6:.0f}M) · loaded {diag['loaded']} · "
                   f"too short {diag['short']} · raw {diag['raw']} · "
                   f"graded {diag['graded']}")
        if diag["venue"] == "binanceus":
            st.warning("Landed on binanceus, which is spot-only and thin. "
                       "Set Exchange to `okx` or `bybit` for a real universe.")

    setups = st.session_state.get("setups", [])
    if not setups:
        st.info("No results yet. Zero setups is a normal outcome.")
    else:
        rows = [s for s in setups
                if (not only_ok) or s.meta.get("status") in OK_STATUS][:limit]
        if not rows:
            st.warning("Signals exist but none are actionable. "
                       "Uncheck Actionable only.")
        else:
            st.dataframe(pd.DataFrame([{
                "grade": s.meta["rating"].grade, "coin": s.symbol,
                "dir": s.direction.upper(), "status": s.meta.get("status"),
                "score": s.score, "entry": f"{s.entry:,.6g}",
                "stop": f"{s.stop:,.6g}", "tp1": f"{s.tp1:,.6g}",
                "tp2": f"{s.tp2:,.6g}",
            } for s in rows]), hide_index=True, use_container_width=True)

            for s in rows:
                rt = s.meta["rating"]
                with st.expander(f"{rt.grade} · {s.symbol} · "
                                 f"{s.direction.upper()} · {s.meta.get('status')}"):
                    st.markdown(
                        f'<span class="pill {GRADE_TONE.get(rt.grade, "mut")}">'
                        f'{rt.grade} · {rt.score}/100</span>',
                        unsafe_allow_html=True)
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
        if st.button("Refresh radar", help="Scan majors for deep discount zones."):
            st.session_state["radar"] = _call(
                sig_mod, ["value_radar", "radar", "scan_radar"])
        radar = st.session_state.get("radar", [])
        if radar:
            st.dataframe(pd.DataFrame([{
                "coin": r.get("symbol"), "severity": r.get("sev"),
                "note": " · ".join(r.get("reasons", [])),
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
            p_sym = a.text_input("Symbol", "BTC/USDT",
                                 help="Pair you own.")
            p_qty = b.number_input("Qty", 0.0, 1e9, 0.0, format="%.8f",
                                   help="Units of the coin, not dollars.")
            p_px = c.number_input("Price", 0.0, 1e9, 0.0, format="%.6f",
                                  help="Your average buy price.")
            p_stop = d.number_input("Stop (0 = none)", 0.0, 1e9, 0.0,
                                    format="%.6f",
                                    help="Price where the position is invalidated.")
            if st.form_submit_button("Add / increase") and p_sym and p_qty > 0 and p_px > 0:
                _call(pf_mod, ["add", "add_position"], p_sym, p_qty, p_px,
                      p_stop or None)
                st.rerun()

        book = _call(pf_mod, ["load_pf", "load", "positions", "all"])
        if isinstance(book, dict):
            book = [{"symbol": k, **v} for k, v in book.items()]
        if not book:
            st.info("No open positions.")
        else:
            df_book = pd.DataFrame(book)
            if "value" not in df_book.columns and {"qty", "price"}.issubset(df_book.columns):
                df_book["value"] = df_book["qty"] * df_book["price"]
            if "bucket" not in df_book.columns:
                df_book["bucket"] = "unset"
            st.dataframe(df_book, hide_index=True, use_container_width=True)
            if "value" in df_book.columns:
                fig = go.Figure(go.Pie(labels=df_book["bucket"],
                                       values=df_book["value"], hole=.55))
                fig.update_layout(height=320, template="plotly_dark",
                                  paper_bgcolor="#0f1116",
                                  margin=dict(l=6, r=6, t=6, b=6))
                st.plotly_chart(fig, use_container_width=True)
            if _has(pf_mod, ["export_json"]):
                st.download_button("Export backup", pf_mod.export_json(),
                                   file_name="portfolio.json")
    except Exception as e:
        st.warning(f"Portfolio unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- backtest
with tab_bt:
    ui.section("Backtest", "Same filters as the scanner, next-open fills.")
    b1, b2, b3 = st.columns(3)
    bt_sym = b1.text_input("Symbol", "BTC/USDT", key="bt_sym",
                           help="Use BTC/USDT:USDT on perp venues.")
    bt_bars = b2.number_input("Bars", 500, 20000, 6000, step=500,
                              help="History depth. More bars means a slower run "
                                   "and a longer warm-up for EMA200.")
    bt_grade = b3.select_slider("Min grade", GRADE_ORDER, value="D",
                                key="bt_grade",
                                help="Grade floor applied inside the run.")

    if st.button("Run backtest", type="primary",
                 help="Replay the strategy and report expectancy in R."):
        try:
            with st.spinner("Running..."):
                curve, trades, raw_n, kept_n = cached_backtest(
                    exch, bt_sym, ltf, int(bt_bars), min_rr, max_stop_atr,
                    risk_pct, equity, exch, proxy.strip() or None)
            if curve is None:
                st.warning("Not enough history on this venue for that symbol.")
            else:
                st.caption(f"raw {raw_n} · after filters {kept_n} · "
                           f"trades {len(trades)}")
                fig = go.Figure(go.Scatter(x=curve.index, y=curve.values,
                                           line=dict(color="#14b8a6", width=1.6)))
                fig.update_layout(height=340, template="plotly_dark",
                                  paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                                  margin=dict(l=6, r=6, t=6, b=6))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(trades.tail(50), hide_index=True,
                             use_container_width=True)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- macro
with tab_macro:
    ui.section("Macro", "Calendar blackouts and the risk gate.")
    try:
        import macro as macro_mod
        state = _call(macro_mod, ["state", "macro_state", "evaluate",
                                  "snapshot", "get_state"])
        if isinstance(state, dict):
            if state.get("blackout"):
                st.warning(state.get("blackout_reason", "blackout active"))
            st.json(state)
        else:
            st.write(state)
    except Exception as e:
        st.warning(f"Macro unavailable: {type(e).__name__}: {e}")
        st.caption("Expected one of: state(), macro_state(), evaluate(), "
                   "snapshot(), get_state(). Add one of those names to macro.py.")


# ---------------------------------------------------------------- alerts
with tab_alert:
    ui.section("Alerts", "Telegram and email delivery.")
    try:
        import alerts as al_mod
        configured = None
        if _has(al_mod, ["is_configured", "configured"]):
            try:
                configured = _call(al_mod, ["is_configured", "configured"])
            except Exception:
                configured = None
        if configured is not None:
            st.caption(f"configured: {configured}")
        if st.button("Send test alert",
                     help="Fire one test message to every configured channel."):
            ok = _call(al_mod, ["send"], "test", "test", "CryptoScan test",
                       "If you can read this, alerts work.")
            if ok:
                st.success("Test sent.")
            else:
                st.error("Not configured. Add Telegram or SMTP values first.")
    except Exception as e:
        st.warning(f"Alerts unavailable: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- journal
with tab_journal:
    ui.section("Journal", "Log every trade. The rules column is the one that matters.")
    with st.form("journal", clear_on_submit=True):
        j1, j2, j3, j4 = st.columns(4)
        j_sym = j1.text_input("Symbol", help="Pair traded.")
        j_dir = j2.selectbox("Side", ["long", "short"], help="Direction.")
        j_r = j3.number_input("Result (R)", -10.0, 10.0, 0.0, step=0.1,
                              help="Outcome in R. -1 means the stop was hit.")
        j_ok = j4.checkbox("Followed rules", value=True,
                           help="Did you take it exactly as the card said?")
        j_note = st.text_input("Note", help="What you saw, what you did.")
        if st.form_submit_button("Log"):
            p = pathlib.Path("./state/journal.csv")
            p.parent.mkdir(exist_ok=True)
            pd.DataFrame([{"ts": pd.Timestamp.utcnow(), "symbol": j_sym,
                           "dir": j_dir, "R": j_r, "followed": j_ok,
                           "note": j_note}]).to_csv(
                p, mode="a", header=not p.exists(), index=False)
            st.success("Logged.")
    if pathlib.Path("./state/journal.csv").exists():
        st.dataframe(pd.read_csv("./state/journal.csv").tail(100),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- guide
with tab_guide:
    ui.section("User guide")
    st.markdown("""
### First run
1. Sidebar, set **Exchange** to `okx` or `bybit`. Leave `auto` only if you want
   the app to choose — it now refuses thin venues.
2. Leave **Min 24h volume** at $30M. The scanner lowers it on its own if the
   universe would be tiny.
3. Press **Test venues** to confirm what answers from your host.
4. Press **Run scan** in the Scan tab.

### Reading the diagnostic line
`venue · universe · loaded · too short · raw · graded`

- **universe 2** — venue too shallow. Switch venue.
- **loaded 0** — nothing had enough history. Usually means a spot-only venue.
- **raw 0** — no break-of-structure at all in the window. Normal in chop.
- **raw > 0, graded 0** — filters rejected everything. Loosen grade or R:R.

### Grades
A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52. The card lists positives and
negatives so you can see exactly why. Grades are heuristic until a backtest
calibrates them.

### Status
ACTIONABLE price at entry · BETTER_PRICE pulled back past entry · RUNNING TP1
hit · EXTENDED ran away · INVALID stopped out.

### Position sizing
`qty = (equity × risk%) ÷ (entry − stop)`. At $10,000 with 0.5% risk and a $200
stop distance that is $50 ÷ $200 = 0.25 units. The app never sizes for you.

### Daily routine
1. Overview for context. Fear & Greed under 25 favours accumulation.
2. Scan. Work top grade down.
3. Check portfolio heat before adding a correlated position.
4. Place entry, stop and TP1 together. Never a market order without a stop.
5. Log it in the Journal the same day.

### Venues
Binance geo-blocks US datacentre IPs with HTTP 451. Streamlit Cloud runs in the
US. A Cloudflare Worker proxy in the sidebar unblocks it. Without one, use OKX,
bybit or bitget.

### Rules that matter more than the tool
- Risk 0.5% per trade, 3% total heat.
- Max two correlated positions in the same direction.
- No new entries within 24h of a major macro print.
- Paper trade 30 signals before real money.
- If expectancy in R is negative over 50 trades, stop and re-tune.

### Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| universe under 5 | thin venue | pick okx or bybit |
| loaded 0 | spot-only venue, no perps | same |
| macro tab warning | `macro.py` has no `state()` | add one, or ignore |
| Nothing reachable | all venues blocked | add the proxy URL |
| Scan very slow | too many symbols | lower to 20 |
""")


st.divider()
st.caption("CryptoScan · not financial advice")
