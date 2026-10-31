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
SCAN_BARS = 1200
HAS_FRAGMENT = hasattr(st, "fragment")

VENUE_ORDER = ["binance", "binanceusdm", "okx", "bybit", "bitget",
               "kucoinfutures", "mexc", "gateio", "binanceus"]
MIN_PERPS = 20

MAJORS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK",
          "TON", "SUI", "DOT", "NEAR", "APT", "ARB", "OP", "INJ", "SEI", "TIA"]


# ---------------------------------------------------------------- helpers
def _call(mod, names, *a, **k):
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f(*a, **k)
    raise AttributeError(f"{mod.__name__} has none of {names}")


def _has(mod, names) -> bool:
    return any(callable(getattr(mod, n, None)) for n in names)


def _resolve(cfg, preferred, proxy):
    sig = inspect.signature(resolve_exchange).parameters
    base = {"preferred": preferred} if "preferred" in sig else {}
    if "proxy" in sig:
        base["proxy"] = proxy
    order = ([preferred] + [v for v in VENUE_ORDER if v != preferred]
             if preferred and preferred != "auto" else list(VENUE_ORDER))
    best, best_n = None, -1
    for name in order:
        kw = dict(base)
        if "preferred" in sig:
            kw["preferred"] = name
        try:
            try:
                ex, used = resolve_exchange(cfg, **kw)
            except TypeError:
                kw.pop("proxy", None)
                ex, used = resolve_exchange(cfg, **kw)
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


def _perps(ex) -> int:
    try:
        ex.load_markets()
    except Exception:
        return 0
    return sum(1 for s, m in ex.markets.items()
               if ":" in s and m.get("quote") == "USDT" and m.get("active"))


def _alts(sym: str) -> list[str]:
    out = [sym]
    out.append(sym.split(":")[0] if ":" in sym else f"{sym}:USDT")
    return out


def _universe(ex, min_vol: int, top: int) -> tuple[list[str], int, int]:
    ex.load_markets()
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        tickers = {}
    rows = []
    for sym, m in ex.markets.items():
        if not m.get("active", True) or m.get("quote") != "USDT":
            continue
        qv = float((tickers.get(sym) or {}).get("quoteVolume") or 0)
        rows.append((sym, qv, bool(m.get("swap") or m.get("contract"))))
    perps = [r for r in rows if r[2]]
    pool = perps if len(perps) >= 10 else rows
    floor = min_vol
    filtered = [r for r in pool if r[1] >= floor]
    if len(filtered) < 15:
        floor = 0
        filtered = pool
    filtered.sort(key=lambda r: -r[1])
    syms = [r[0] for r in filtered[:top]]
    if not syms:
        base = "BTC/USDT:USDT" if len(perps) >= 10 else "BTC/USDT"
        syms = [base.replace("BTC", c) for c in MAJORS if base.replace("BTC", c) in ex.markets]
    return syms, floor, len(perps)


def _load(ex, sym: str, tf: str, limit: int):
    for cand in _alts(sym):
        try:
            df = load(ex, cand, tf, limit=limit)
        except TypeError:
            try:
                df = load(ex, cand, tf)
            except Exception:
                df = None
        except Exception:
            df = None
        if df is not None and len(df) >= MIN_BARS:
            return df, cand
    return None, sym


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
                equity, recent_bars, min_grade, preferred, proxy, relax):
    if relax:
        min_rr = max(1.2, min_rr * 0.7)
        max_stop_atr = max_stop_atr * 1.4
        idx = GRADE_ORDER.index(min_grade)
        min_grade = GRADE_ORDER[max(0, idx - 1)]

    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    universe, floor, perps = _universe(ex, int(min_vol), top)
    stats = load_stats()
    found = []
    diag = {"venue": ex_name, "perps": perps, "universe": len(universe),
            "floor": floor, "loaded": 0, "err": 0, "raw": 0, "kept": 0,
            "graded": 0, "below_grade": 0, "relax": relax,
            "rr": min_rr, "stop": max_stop_atr}

    for sym in universe:
        df, used_sym = _load(ex, sym, cfg.scan.ltf, SCAN_BARS)
        if df is None:
            diag["err"] += 1
            continue
        diag["loaded"] += 1
        try:
            ind = compute_all(df, cfg)
            stt = build_structure(df, cfg.structure)
            hb = htf_bias(df, cfg)
            sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")
            raw = scan_bos_retest(used_sym, df, hb, ind, stt, zu, zd, cfg)
            diag["raw"] += len(raw)
            kept = filter_setups(raw, ind, zu, zd, cfg)
            diag["kept"] += len(kept)
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
                    diag["below_grade"] += 1
                    continue
                found.append(s)
        except Exception:
            diag["err"] += 1
            continue

    found.sort(key=lambda s: (-s.meta["rating"].confidence, -s.score))
    return found, diag


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr, max_stop_atr,
                    risk_pct, equity, preferred, proxy):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, _ = _resolve(cfg, preferred, proxy)
    df, used = _load(ex, symbol, cfg.scan.ltf, int(limit))
    if df is None:
        return None, None, 0, 0
    ind = compute_all(df, cfg)
    stt = build_structure(df, cfg.structure)
    hb = htf_bias(df, cfg)
    sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
    zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
    zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")
    raw = scan_bos_retest(used, df, hb, ind, stt, zu, zd, cfg)
    setups = filter_setups(raw, ind, zu, zd, cfg)
    signals = {}
    for s in setups:
        loc = df.index.get_loc(s.ts)
        signals.setdefault(loc, {"dir": s.direction, "stop": s.stop,
                                 "tp1": s.tp1, "tp2": s.tp2, "score": s.score})
    curve, trades = engine_run(df, signals, cfg, used)
    return curve, trades, len(raw), len(setups)


@st.cache_data(ttl=900, show_spinner=False)
def cached_radar(exchange, preferred, proxy):
    """200-day SMA distance across majors. No signals.py dependency."""
    cfg = build_cfg(exchange, 40, 30e6, "1d", 2.0, 3.5, 0.5, 10000)
    ex, name = _resolve(cfg, preferred, proxy)
    out = []
    for c in MAJORS[:12]:
        df, used = _load(ex, f"{c}/USDT:USDT", "1d", 400)
        if df is None:
            df, used = _load(ex, f"{c}/USDT", "1d", 400)
        if df is None:
            continue
        sma = df["close"].rolling(200).mean().iloc[-1]
        if pd.isna(sma):
            continue
        px = float(df["close"].iloc[-1])
        dist = (px / float(sma) - 1) * 100
        try:
            rsi = float(compute_all(df, cfg)["rsi"].iloc[-1])
        except Exception:
            rsi = None
        sev = 2 if dist <= 0 else 1 if dist <= 6 else 0
        reasons = [f"{dist:+.1f}% vs 200d SMA"]
        if rsi is not None:
            reasons.append(f"RSI {rsi:.0f}")
        out.append({"symbol": used, "sev": sev, "dist": dist,
                    "reasons": reasons, "venue": name})
    out.sort(key=lambda r: r["dist"])
    return out


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
        help="auto scores venues by perpetual depth and picks the deepest that "
             "answers. Spot-only venues are avoided.")
    proxy = st.text_input(
        "Binance proxy (optional)", value="",
        placeholder="https://your-worker.workers.dev",
        help="Cloudflare Worker URL. Only needed for Binance, which returns "
             "HTTP 451 to US datacentre IPs.")
    if st.button("Test venues", use_container_width=True,
                 help="Probe every venue and list which ones answer."):
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
                       help="Bar size. 4h is the swing default.")
    top = st.slider("Symbols to scan", 10, 150, 40, step=10,
                    help="How many top-volume pairs to check. More means slower.")
    min_vol = st.select_slider(
        "Min 24h volume", [10e6, 30e6, 50e6, 100e6, 250e6], value=30e6,
        format_func=lambda v: f"${v/1e6:.0f}M",
        help="Liquidity floor. Drops to zero automatically if it would leave "
             "you with almost nothing.")
    recent = st.slider("Signal freshness (bars)", 1, 12, 2,
                       help="How many bars back a break may be and still count.")
    min_grade = st.select_slider("Minimum grade", GRADE_ORDER, value="B",
                                 help="A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52.")
    only_ok = st.checkbox("Actionable only", value=True,
                          help="Hide extended, invalid and past-target setups.")
    relax = st.checkbox(
        "Relax filters", value=False,
        help="Widens R:R floor to 70% and stop cap to 140%, and drops the grade "
             "floor one step. Use when a scan returns raw breaks but nothing "
             "survives filtering.")
    limit = st.slider("Max results", 3, 40, 10, help="Cap on cards rendered.")

    st.divider()
    equity = st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0,
                             step=500.0, help="Used for position sizing.")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1,
                         help="Equity lost if the stop is hit.")
    min_rr = st.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25,
                       help="Reward-to-risk floor. Lower means more signals.")
    max_stop_atr = st.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25,
                             help="Stop distance cap in ATR units.")

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


def _strip() -> None:
    ui.live_label()
    ui.marquee(market.ticker_rows(24))


if HAS_FRAGMENT:
    _strip = st.fragment(run_every="30s")(_strip)
_strip()

ov = market.overview()
fng = ov.get("fng")
stab = market.stable_ratio(ov)
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
    detail_sym = d1.text_input("Symbol", "BTC/USDT:USDT", key="detail_sym",
                               help="Unified ccxt symbol. BTC/USDT spot, "
                                    "BTC/USDT:USDT perpetual.")
    detail_tf = d2.selectbox("Timeframe", ["1h", "4h", "1d"], index=1,
                             key="detail_tf", help="Chart resolution.")

    if st.button("Load detail", type="primary",
                 help="Fetch candles and render the chart with EMAs."):
        try:
            cfg = build_cfg(exch, top, min_vol, detail_tf, min_rr,
                            max_stop_atr, risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df, used = _load(ex, detail_sym, detail_tf, 400)
            if df is None:
                st.warning(f"Not enough candles for {detail_sym} on {ex_name}.")
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
                st.caption(f"venue {ex_name} · symbol {used}")

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
                    equity, recent, min_grade, exch, proxy.strip() or None, relax)
            st.session_state["setups"] = found
            st.session_state["diag"] = diag
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    diag = st.session_state.get("diag")
    if diag:
        st.caption(f"venue `{diag['venue']}` · perps {diag['perps']} · "
                   f"universe {diag['universe']} · loaded {diag['loaded']} · "
                   f"failed {diag['err']} · raw {diag['raw']} · "
                   f"passed filters {diag['kept']} · graded {diag['graded']} · "
                   f"below grade {diag['below_grade']}"
                   + (f" · RELAXED R:R {diag['rr']:.2f} stop {diag['stop']:.1f}" if diag["relax"] else ""))

        if diag["raw"] and not diag["kept"]:
            st.error(f"{diag['raw']} breaks found, all rejected by filters. "
                     f"Tick **Relax filters** in the sidebar and rerun, or lower "
                     f"Minimum R:R.")
        elif diag["kept"] and not diag["graded"]:
            st.warning("Setups passed filters but all were below your grade floor. "
                       "Lower Minimum grade or tick Relax filters.")
        elif diag["below_grade"] and not st.session_state.get("setups"):
            st.warning(f"{diag['below_grade']} setup(s) graded below {min_grade}. "
                       f"Lower the grade floor to see them.")
        elif not diag["raw"] and diag["loaded"] > 10:
            st.info("Pipeline healthy, no break-of-structure in this window. "
                    "Normal in chop. Try 1h.")

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
    ui.section("Value radar", "Distance to the 200-day SMA across majors.")
    st.caption("Negative means price is under the 200-day average — a "
               "multi-cycle accumulation zone, not a buy signal on its own.")
    if st.button("Refresh radar", help="Recompute distances from the venue."):
        try:
            with st.spinner("loading daily candles..."):
                st.session_state["radar"] = cached_radar(
                    exch, exch, proxy.strip() or None)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    radar = st.session_state.get("radar", [])
    if radar:
        st.dataframe(pd.DataFrame([{
            "coin": r["symbol"],
            "zone": "deep" if r["sev"] == 2 else "near" if r["sev"] == 1 else "extended",
            "vs 200d": f"{r['dist']:+.1f}%",
            "detail": " · ".join(r["reasons"]),
        } for r in radar]), hide_index=True, use_container_width=True)

        bars = go.Figure(go.Bar(
            x=[r["dist"] for r in radar], y=[r["symbol"] for r in radar],
            orientation="h",
            marker_color=["#16c784" if r["dist"] <= 0 else
                          "#f0b90b" if r["dist"] <= 6 else "#ea3943"
                          for r in radar]))
        bars.update_layout(height=360, template="plotly_dark",
                           paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                           margin=dict(l=6, r=6, t=6, b=6),
                           xaxis_title="% vs 200d SMA")
        st.plotly_chart(bars, use_container_width=True)
    else:
        st.info("Press Refresh radar.")


# ---------------------------------------------------------------- portfolio
with tab_pf:
    ui.section("Portfolio", "Spot positions, heat and bucket split.")
    try:
        import portfolio as pf_mod
        with st.form("add_pos"):
            a, b, c, d = st.columns(4)
            p_sym = a.text_input("Symbol", "BTC/USDT", help="Pair you own.")
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
    bt_sym = b1.text_input("Symbol", "BTC/USDT:USDT", key="bt_sym",
                           help="Use BTC/USDT:USDT on perpetual venues.")
    bt_bars = b2.number_input("Bars", 500, 20000, 6000, step=500,
                              help="History depth. More bars is slower but warms "
                                   "EMA200 properly.")
    bt_grade = b3.select_slider("Min grade", GRADE_ORDER, value="D",
                                key="bt_grade", help="Grade floor inside the run.")

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
    ui.section("Macro", "Sentiment, dominance and dry powder — read as pictures.")

    try:
        import macro as macro_mod
        mstate = _call(macro_mod, ["state", "macro_state", "evaluate",
                                   "snapshot", "get_state"])
        if isinstance(mstate, dict) and mstate.get("blackout"):
            st.warning(mstate.get("blackout_reason", "Macro blackout active."))
    except Exception:
        pass

    label, ltone, notes = market.stance(fng, ov["btc_dom"], stab)
    s1, s2, s3 = st.columns(3)
    ui.stat(s1, "Stance", label, ltone)
    ui.stat(s2, "Stablecoin share", f"{stab:.1f}%" if stab is not None else "—")
    ui.stat(s3, "Fear & Greed", f"{fng}" if fng is not None else "—", tone)

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Fear & Greed — 0 panic, 100 euphoria")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=fng if fng is not None else 0,
            gauge={"axis": {"range": [0, 100], "tickcolor": "#7f8798"},
                   "bar": {"color": "#14b8a6", "thickness": .28},
                   "steps": [{"range": [0, 25], "color": "#2a1214"},
                             {"range": [25, 45], "color": "#2b2110"},
                             {"range": [45, 55], "color": "#161a21"},
                             {"range": [55, 75], "color": "#122a1c"},
                             {"range": [75, 100], "color": "#0f3b24"}]}))
        gauge.update_layout(height=260, template="plotly_dark",
                            paper_bgcolor="#0f1116",
                            margin=dict(l=18, r=18, t=10, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    with g2:
        st.caption("Market cap split — BTC vs ETH vs everything else")
        labels, values = market.dominance_split(ov)
        if labels:
            pie = go.Figure(go.Pie(labels=labels, values=values, hole=.55,
                                   marker=dict(colors=["#f0b90b", "#627eea", "#3a4150"])))
            pie.update_layout(height=260, template="plotly_dark",
                              paper_bgcolor="#0f1116",
                              margin=dict(l=18, r=18, t=10, b=10))
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("Dominance data unavailable.")

    caps, _ = market.btc_history(90)
    if caps:
        st.caption("BTC market cap, 90 days — the direction money is actually moving")
        xs = [pd.Timestamp(c[0], unit="ms") for c in caps]
        ys = [c[1] / 1e9 for c in caps]
        line = go.Figure(go.Scatter(x=xs, y=ys, fill="tozeroy",
                                    line=dict(color="#14b8a6", width=1.8),
                                    fillcolor="rgba(20,184,166,.12)"))
        line.update_layout(height=260, template="plotly_dark",
                           paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                           margin=dict(l=6, r=6, t=6, b=6),
                           yaxis_title="market cap, $B")
        st.plotly_chart(line, use_container_width=True)

    st.divider()
    st.markdown("#### What these mean")
    for n in notes:
        st.write(f"- {n}")
    st.markdown("""
**Fear & Greed** aggregates volatility, momentum, chatter and dominance into one
number. It is a crowd gauge, not a timing tool. Deep fear gives better prices but
worse follow-through.

**Dominance** shows where market cap sits. Rising BTC dominance means capital is
retreating into the largest asset and alts underperform. Falling dominance is
rotation into alts, where alt breakouts actually run.

**Stablecoin share** is unspent buying power. Rising is fuel, not a signal.

**Market cap trend** shows expansion or contraction. Longs work better in
expansion, shorts in contraction.

**Blackout** comes from the economic calendar. Rate decisions and CPI prints
distort intraday structure.
""")


# ---------------------------------------------------------------- alerts
with tab_alert:
    ui.section("Alerts", "Telegram and email delivery.")
    try:
        import alerts as al_mod
        if _has(al_mod, ["is_configured", "configured"]):
            try:
                st.caption(f"configured: {_call(al_mod, ['is_configured', 'configured'])}")
            except Exception:
                pass
        if st.button("Send test alert",
                     help="Fire one test message to every configured channel."):
            ok = _call(al_mod, ["send"], "test", "test", "CryptoScan test",
                       "If you can read this, alerts work.")
            if ok:
                st.success("Test sent.")
            else:
                st.error("Not configured yet. Follow the steps below.")
    except Exception as e:
        st.warning(f"Alerts unavailable: {type(e).__name__}: {e}")

    st.divider()
    st.markdown("#### Telegram setup")
    st.markdown("""
1. Open Telegram, message **@BotFather**, send `/newbot`, follow the prompts.
2. Copy the token it gives you.
3. Message **@userinfobot** to get your numeric chat id.
4. Streamlit Cloud → Settings → **Secrets**, add:

```toml
TELEGRAM_TOKEN = "9144109318:AAEl1Mj2_XBqOJIImkRFH_2HAumleexDcHY"
TELEGRAM_CHAT_ID = "6193241326"
Save, reboot the app, press Send test alert.
""")
st.markdown("#### Email setup")
st.markdown("""
Gmail needs an app password, not your normal one.
toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "you@gmail.com"
SMTP_PASS = "the-16-char-app-password"
ALERT_TO = "you@gmail.com"
""")

---------------------------------------------------------------- journal
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

---------------------------------------------------------------- guide
with tab_guide:
ui.section("User guide")
st.markdown("""

Getting signals to appear
Set Exchange to okx or bybit.
Set Minimum grade to C and tick Relax filters.
Press Run scan.
A 4h scan on 40 majors typically produces a handful of raw breaks and few
survivors. That is the strategy being picky, not a bug.

Reading the diagnostic line
venue · perps · universe · loaded · failed · raw · passed filters · graded · below grade

raw > 0, passed filters 0 — filters too tight. Tick Relax filters.
passed filters > 0, graded 0 — all below your grade floor. Lower it.
below grade > 0, no results — signals exist, just not at your standard.
loaded 0 — venue refuses candles. Try BTC/USDT:USDT.
raw 0, loaded high — genuine chop. Try 1h.
Grades
A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52.

Status
ACTIONABLE at entry · BETTER_PRICE pulled back · RUNNING TP1 hit · EXTENDED
ran away · INVALID stopped out.

Sizing
qty = (equity × risk%) ÷ (entry − stop).

Radar
Distance to the 200-day SMA. Under 0 means multi-cycle discount territory — the
zone to ladder into slowly, not a single-entry signal.

Daily routine
Macro tab for stance.
Scan, work top grade down.
Check heat before a correlated add.
Entry, stop and TP1 together.
Log it the same day.
""")
st.divider()
st.caption("CryptoScan · not financial advice")
