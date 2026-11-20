"""
app.py - CryptoScan. Spot mode and Bot mode.

    pip install -r requirements.txt
    streamlit run app.py
"""
import inspect
import json
import os
import pathlib
import smtplib
from dataclasses import replace
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import auth
import chart
import learn
import market
import paper
import sizer
import smc
import ui
from config import AUTH_ENABLED, CFG

st.set_page_config(page_title="CryptoScan", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject()


def _auth_on() -> bool:
    """Secrets win over config, so login can be toggled from the Cloud UI."""
    try:
        if "AUTH_ENABLED" in st.secrets:
            return str(st.secrets["AUTH_ENABLED"]).lower() in ("1", "true", "yes")
    except Exception:
        pass
    return bool(AUTH_ENABLED)


AUTH_ON = _auth_on()
USER = auth.gate() if AUTH_ON else "local"

from data import CANDIDATES, build_universe, load, probe_exchanges, resolve_exchange
from core.indicators import compute_all
from core.levels import build_zones
from core.mtf import aligned, htf_for, structure_bias
from core.regime import htf_bias
from core.structure import build_structure, find_swings
from core.volume import avwap_context, volume_profile
from strategy.rating import rate
from strategy.risk import filter_setups
from strategy.setups import scan_bos_retest
from backtest.engine import run as engine_run
from backtest.metrics import report
from backtest.validate import monte_carlo_drawdown, permutation_test

GRADE_ORDER = ["D", "C", "B", "A", "A+"]
GRADE_RANK = {g: i for i, g in enumerate(GRADE_ORDER)}
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

DEFAULT_TG_TOKEN = "9144109318:AAEl1Mj2_XBqOJIImkRFH_2HAumleexDcHY"
DEFAULT_TG_CHAT = "6193241326"
CREDS_FILE = pathlib.Path("./state/alerts.json")


# ---------------------------------------------------------------- helpers
def _call(mod, names, *a, **k):
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f(*a, **k)
    raise AttributeError(f"{mod.__name__} has none of {names}")


def _has(mod, names) -> bool:
    return any(callable(getattr(mod, n, None)) for n in names)


def _ind(ind, *names):
    """First indicator column that exists, else None. Survives naming drift."""
    for n in names:
        try:
            s = ind[n]
        except Exception:
            s = None
        if s is not None:
            return s
    return None


def _last(ind, *names, default=0.0) -> float:
    s = _ind(ind, *names)
    try:
        v = float(s.iloc[-1])
        return v if v == v else default
    except Exception:
        return default


def _rank(g) -> int:
    return GRADE_RANK.get(str(g).strip(), 0)


def _norm(rt) -> dict:
    """Coerce whatever rate() returned into a plain dict."""
    out = {"grade": "D", "score": 0, "confidence": 0.0,
           "positives": [], "negatives": [], "improve": []}
    if rt is None:
        return out
    if isinstance(rt, dict):
        src = rt
    elif isinstance(rt, (tuple, list)):
        src = {}
        if len(rt) > 0:
            src["grade"] = rt[0]
        if len(rt) > 1:
            src["score"] = rt[1]
    else:
        src = {k: getattr(rt, k, None) for k in out}
    for k in out:
        v = src.get(k)
        if v is not None:
            out[k] = v
    try:
        out["score"] = int(out["score"] or 0)
    except Exception:
        out["score"] = 0
    try:
        out["confidence"] = float(out["confidence"] or 0)
    except Exception:
        out["confidence"] = 0.0
    for k in ("positives", "negatives", "improve"):
        try:
            out[k] = list(out[k] or [])
        except Exception:
            out[k] = []
    if str(out["grade"]) not in GRADE_RANK:
        out["grade"] = "D"
    return out


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
        syms = [base.replace("BTC", c) for c in MAJORS
                if base.replace("BTC", c) in ex.markets]
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


# ---------------------------------------------------------------- creds
def _saved() -> dict:
    if CREDS_FILE.exists():
        try:
            return json.loads(CREDS_FILE.read_text())
        except Exception:
            return {}
    return {}


def _cred(key: str) -> str:
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    if os.environ.get(key):
        return os.environ[key]
    saved = _saved().get(key)
    if saved:
        return str(saved)
    if key == "TELEGRAM_TOKEN":
        return DEFAULT_TG_TOKEN
    if key == "TELEGRAM_CHAT_ID":
        return DEFAULT_TG_CHAT
    return "587" if key == "SMTP_PORT" else ""


def _save_creds(d: dict) -> None:
    CREDS_FILE.parent.mkdir(exist_ok=True)
    CREDS_FILE.write_text(json.dumps(d, indent=2))


def _send_telegram(text: str) -> tuple[bool, str]:
    token, chat = _cred("TELEGRAM_TOKEN"), _cred("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False, "missing token or chat id"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text, "disable_web_page_preview": True},
            timeout=15)
        if r.ok:
            return True, "sent"
        return False, f"telegram {r.status_code}: {r.text[:160]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _send_email(text: str) -> tuple[bool, str]:
    host, user = _cred("SMTP_HOST"), _cred("SMTP_USER")
    pw, to = _cred("SMTP_PASS"), _cred("ALERT_TO")
    if not all([host, user, pw, to]):
        return False, "email not configured"
    try:
        msg = MIMEText(text)
        msg["Subject"] = "CryptoScan alert"
        msg["From"], msg["To"] = user, to
        with smtplib.SMTP(host, int(_cred("SMTP_PORT") or 587), timeout=20) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def notify(text: str) -> tuple[bool, str]:
    ok, msg = _send_telegram(text)
    if ok:
        return True, "telegram sent"
    ok2, msg2 = _send_email(text)
    if ok2:
        return True, "email sent"
    return False, f"{msg} | {msg2}"


# ---------------------------------------------------------------- pipelines
@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                equity, recent_bars, min_grade, preferred, proxy, relax,
                need_smc, long_only):
    if relax:
        min_rr = max(1.2, min_rr * 0.7)
        max_stop_atr = max_stop_atr * 1.4
        min_grade = GRADE_ORDER[max(0, _rank(min_grade) - 1)]

    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    universe, floor, perps = _universe(ex, int(min_vol), top)
    stats = load_stats()
    found = []
    diag = {"venue": ex_name, "perps": perps, "universe": len(universe),
            "floor": floor, "loaded": 0, "err": 0, "raw": 0, "kept": 0,
            "stale": 0, "graded": 0, "below_grade": 0, "smc_skip": 0,
            "mtf_skip": 0, "vol_skip": 0,
            "relax": relax, "rr": min_rr, "stop": max_stop_atr,
            "recent": recent_bars}

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
            if long_only:
                kept = [s for s in kept if s.direction == "long"]
            diag["kept"] += len(kept)
            if recent_bars and kept:
                cut = df.index[max(0, len(df) - 1 - recent_bars)]
                fresh = [s for s in kept if s.ts >= cut]
                diag["stale"] += len(kept) - len(fresh)
                kept = fresh
            for s in kept:
                s.meta["status"] = setup_status(s, df)
                zone = (zu[0] if zu else None) if s.direction == "long" \
                    else (zd[0] if zd else None)
                s.meta["r"] = _norm(rate(s, ind.loc[s.ts], zone, cfg, stats))
                diag["graded"] += 1
                if _rank(s.meta["r"]["grade"]) < _rank(min_grade):
                    diag["below_grade"] += 1
                    continue

                try:
                    s.meta["smc"] = smc.confluence(
                        df, s.direction, s.entry,
                        _last(ind, "atr", "atr14", "atr_14", default=0.0),
                        idx=df.index.get_loc(s.ts))
                except Exception:
                    s.meta["smc"] = {"score": 0, "flags": [], "warnings": []}
                if need_smc and s.meta["smc"]["score"] < 2:
                    diag["smc_skip"] += 1
                    continue

                htf_df, _ = _load(ex, used_sym, htf_for(cfg.scan.ltf), 400)
                hb_s = structure_bias(htf_df, cfg) if htf_df is not None else "range"
                lb_s = str(stt["trend"].iloc[-1])
                ok_mtf, mtf_why = aligned(hb_s, lb_s, s.direction)
                s.meta["mtf"] = {"ok": ok_mtf, "why": mtf_why, "htf": hb_s}

                anchor = int(stt["prev_swing_low" if s.direction == "long"
                                 else "prev_swing_high"].iloc[-1])
                av = (avwap_context(df, s.direction,
                                    _last(ind, "atr", default=0.0), anchor)
                      if np.isfinite(anchor) else
                      {"score": 0, "flags": [], "warnings": ["no anchor"], "dist": 0.0})
                s.meta["avwap"] = av
                vp = volume_profile(df)
                s.meta["vp"] = vp
                s.meta["in_value"] = bool(vp and vp["val"] <= s.entry <= vp["vah"])

                if not ok_mtf:
                    diag["mtf_skip"] += 1
                    continue
                if av["score"] < 40:
                    diag["vol_skip"] += 1
                    continue

                found.append(s)
        except Exception:
            diag["err"] += 1
            continue

    found.sort(key=lambda s: -(s.meta["r"]["confidence"]
                               + s.meta.get("smc", {}).get("score", 0) * 3
                               + s.meta.get("avwap", {}).get("score", 0) * 0.4
                               + (15 if s.meta.get("in_value") else 0)
                               + s.meta.get("candle_q", 0) * 0.2))
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
            rsi = _last(compute_all(df, cfg), "rsi", "rsi14", "rsi_14",
                        default=float("nan"))
            rsi = None if rsi != rsi else rsi
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
    st.caption(f"signed in as {USER}" if AUTH_ON else "local mode · login disabled")
    if AUTH_ON and st.button("Sign out", use_container_width=True,
                             help="End this session and return to the login screen."):
        auth.logout()

    MODE = st.radio("Desk", ["Spot", "Bot"], horizontal=True, index=0,
                    help="Spot is long-only accumulation on daily charts. "
                         "Bot is the long and short scanner. They are separate "
                         "because they are different jobs.")
    st.divider()

    exch = st.selectbox("Exchange", ["auto"] + CANDIDATES, index=0,
                        help="auto picks the deepest venue that answers.")
    proxy = st.text_input("Binance proxy (optional)", value="",
                          placeholder="https://your-worker.workers.dev",
                          help="Only needed for Binance from US servers.")
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
    if MODE == "Spot":
        ltf = st.selectbox("Timeframe", ["4h", "1d", "1w"], index=1,
                           help="Spot accumulation is a daily and weekly job.")
    else:
        ltf = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=1,
                           help="Bar size for the scanner.")

    top = st.slider("Symbols to scan", 10, 150, 40, step=10,
                    help="How many top-volume pairs to check.")
    min_vol = st.select_slider(
        "Min 24h volume", [10e6, 30e6, 50e6, 100e6, 250e6], value=30e6,
        format_func=lambda v: f"${v/1e6:.0f}M", help="Liquidity floor.")
    recent = st.slider("Signal freshness (bars)", 1, 48, 12,
                       help="How many bars back a break may be. Weekends are "
                            "thin, so 12 to 24 is practical.")
    min_grade = st.select_slider("Minimum grade", GRADE_ORDER, value="C",
                                 help="A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52.")
    only_ok = st.checkbox("Actionable only", value=False,
                          help="Hide extended and invalid setups.")
    relax = st.checkbox("Relax filters", value=True,
                        help="Widens R:R to 70% and stop cap to 140%, drops the "
                             "grade floor one step.")
    need_smc = st.checkbox("Require SMC confluence", value=False,
                           help="Only show setups with at least two smart money "
                                "factors.")

    st.divider()
    equity = st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0,
                             step=500.0, help="Used for sizing.")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1,
                         help="Equity lost if the stop is hit.")
    min_rr = st.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25,
                       help="Reward-to-risk floor.")
    max_stop_atr = st.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25,
                             help="Stop distance cap in ATR.")
    st.divider()
    st.caption("Not financial advice. Backtest before risking money.")

# ---------------------------------------------------------------- header
st.markdown(
    f"""<div class="hero">
    <h1>{'Spot accumulation desk' if MODE == 'Spot' else 'Bot swing desk'}</h1>
    <p>{'Long-only, daily structure, laddered entries into demand.' if MODE == 'Spot'
        else 'Long and short break-of-structure scanner with full playbook.'}
    {f'Signed in as <b>{USER}</b>.' if AUTH_ON else 'Local mode.'}</p>
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

if MODE == "Spot":
    NAMES = ["Overview", "Accumulate", "SMC", "Sizer", "Radar", "Portfolio",
             "Learn", "Guide"]
else:
    NAMES = ["Overview", "Scan", "SMC", "Sizer", "Radar", "Portfolio", "Paper",
             "Backtest", "Macro", "Alerts", "Journal", "Learn", "Guide"]
T = dict(zip(NAMES, st.tabs(NAMES)))


# ---------------------------------------------------------------- plan card
def render_plan(plan: dict, symbol: str, ladder: bool = False) -> None:
    conf = plan["confluence"]
    for label, ok in plan["checks"]:
        st.write(("PASS  " if ok else "FAIL  ") + label)
    st.write(f"**Verdict** {plan['verdict']} · {plan['passed']}/{len(plan['checks'])}")
    if conf["flags"]:
        st.write("**Supporting** " + " · ".join(conf["flags"]))
    if conf["warnings"]:
        st.write("**Against** " + " · ".join(conf["warnings"]))
    st.write(f"**Entry** {plan['entry']:,.6g} · **Stop** {plan['sl']:,.6g} · "
             f"**TP1** {plan['tp1']:,.6g} · **TP2** {plan['tp2']:,.6g}")
    st.write(f"Risk ${plan['size_usd']:.2f} ({risk_pct}% of equity) · "
             f"qty {plan['qty']:.6g} · R:R {plan['rr1']:.2f} / {plan['rr2']:.2f}")
    if ladder:
        st.write("**Ladder**")
        st.dataframe(pd.DataFrame([{
            "tranche": t["tranche"], "price": f"{t['price']:,.6g}",
            "qty": f"{t['qty']:.6g}", "weight": f"{t['weight']*100:.0f}%",
        } for t in smc.ladder(plan)]), hide_index=True, use_container_width=True)
    st.caption(f"session {conf['session']['name']} · volatility "
               f"{conf['session']['vol']}")


# ---------------------------------------------------------------- overview
with T["Overview"]:
    ui.section("Coin detail", "Candles, structure state and venue depth.")
    d1, d2 = st.columns([2, 1])
    detail_sym = d1.text_input("Symbol", "BTC/USDT:USDT", key="detail_sym",
                               help="Unified ccxt symbol.")
    detail_tf = d2.selectbox("Timeframe", ["1h", "4h", "1d", "1w"], index=2,
                             key="detail_tf", help="Chart resolution.")
    if st.button("Load detail", type="primary", help="Fetch and chart."):
        try:
            cfg = build_cfg(exch, top, min_vol, detail_tf, min_rr,
                            max_stop_atr, risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df, used = _load(ex, detail_sym, detail_tf, 400)
            if df is None:
                st.warning(f"Not enough candles for {detail_sym} on {ex_name}.")
            else:
                ind = compute_all(df, cfg)
                trend, notes = smc.trend_state(df)
                last = float(df["close"].iloc[-1])
                chg = (last / float(df["close"].iloc[-25]) - 1) * 100
                rsi = _last(ind, "rsi", "rsi14", "rsi_14", default=50.0)
                m1, m2, m3, m4 = st.columns(4)
                ui.stat(m1, "Last", f"{last:,.6g}")
                ui.stat(m2, "24 bar %", f"{chg:+.2f}%", "up" if chg >= 0 else "dn")
                ui.stat(m3, "RSI", f"{rsi:.1f}",
                        "dn" if rsi > 70 else "up" if rsi < 30 else "")
                ui.stat(m4, "Structure", trend,
                        "up" if trend == "up" else "dn" if trend == "down" else "warn")
                st.caption(" · ".join(notes) or "no clear swing sequence")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="price",
                    increasing_line_color="#16c784",
                    decreasing_line_color="#ea3943"))
                fast = _ind(ind, "ema50", "ema_50", "ema_fast", "ema20")
                slow = _ind(ind, "ema200", "ema_200", "ema_slow")
                if fast is not None:
                    fig.add_trace(go.Scatter(x=df.index, y=fast, name="fast EMA",
                                             line=dict(color="#14b8a6", width=1.2)))
                if slow is not None:
                    fig.add_trace(go.Scatter(x=df.index, y=slow, name="slow EMA",
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
    ui.section("Movers", "Largest 24h moves in the top 100.")
    rows = market.movers(8)
    if rows:
        st.dataframe(pd.DataFrame([{
            "coin": f"{r['symbol']} · {r['name']}",
            "price": f"${r['price']:,.6g}", "24h": f"{r['chg']:+.2f}%",
            "volume": market.money(r["vol"]),
        } for r in rows]), hide_index=True, use_container_width=True)
    else:
        st.info("Mover data unavailable.")


# ---------------------------------------------------------------- spot
if MODE == "Spot":
    with T["Accumulate"]:
        ui.section("Accumulation plans", "Long-only, staged into demand on daily.")
        st.caption("Deep discount majors first. Each plan is a ladder, not a "
                   "single buy. Stop sits under the zone.")
        if st.button("Find candidates", type="primary",
                     help="Rank majors by distance below the 200-day average."):
            try:
                with st.spinner("loading..."):
                    st.session_state["radar"] = cached_radar(
                        exch, exch, proxy.strip() or None)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

        cands = [r for r in st.session_state.get("radar", []) if r["sev"] >= 1]
        if not cands:
            st.info("No deep-discount majors right now, or press Find candidates.")
        for r in cands[:6]:
            try:
                cfg = build_cfg(exch, top, min_vol, "1d", min_rr,
                                max_stop_atr, risk_pct, equity)
                ex, _ = _resolve(cfg, exch, proxy.strip() or None)
                df, used = _load(ex, r["symbol"], "1d", 400)
                if df is None:
                    continue
                plan = smc.playbook(df, "long", equity, risk_pct)
                with st.expander(f"{used} · {r['dist']:+.1f}% vs 200d SMA · "
                                 f"{plan['verdict']}"):
                    render_plan(plan, used, ladder=True)
            except Exception as e:
                st.caption(f"{r['symbol']}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- scan
if MODE == "Bot":
    with T["Scan"]:
        ui.section("Live scan", f"Grade {min_grade}+ · last {recent} bars · {ltf}")
        if st.button("Run scan", type="primary", help="Check the universe."):
            try:
                with st.spinner("Scanning..."):
                    found, diag = cached_scan(
                        exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                        equity, recent, min_grade, exch, proxy.strip() or None,
                        relax, need_smc, False)
                st.session_state["setups"] = found
                st.session_state["diag"] = diag
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.exception(e)

        diag = st.session_state.get("diag")
        if diag:
            st.caption(f"venue {diag['venue']} · perps {diag['perps']} · "
                       f"universe {diag['universe']} · loaded {diag['loaded']} · "
                       f"failed {diag['err']} · raw {diag['raw']} · "
                       f"kept {diag['kept']} · stale {diag['stale']} · "
                       f"graded {diag['graded']} · below grade "
                       f"{diag['below_grade']} · smc {diag['smc_skip']} · "
                       f"mtf {diag['mtf_skip']} · volume {diag['vol_skip']}")

        setups = st.session_state.get("setups", [])
        if not setups:
            st.info("No results yet.")
        else:
            rows = [s for s in setups
                    if (not only_ok) or s.meta.get("status") in OK_STATUS][:15]
            st.dataframe(pd.DataFrame([{
                "grade": s.meta["r"]["grade"], "score": s.meta["r"]["score"],
                "symbol": s.symbol, "side": s.direction,
                "candle": s.meta.get("candle") or "—",
                "smc": s.meta.get("smc", {}).get("score", 0),
                "avwap": round(s.meta.get("avwap", {}).get("dist", 0.0), 2),
                "in value": "yes" if s.meta.get("in_value") else "no",
                "status": s.meta.get("status"),
                "entry": f"{s.entry:,.6g}", "stop": f"{s.stop:,.6g}",
                "rr": round(s.rr, 2),
            } for s in rows]), hide_index=True, use_container_width=True)

            for s in rows[:10]:
                r = _norm(s.meta.get("r"))
                sm = s.meta.get("smc", {}) or {}
                with st.expander(f"{r['grade']} · {s.symbol} · "
                                 f"{s.direction.upper()} · "
                                 f"{s.meta.get('status')} · smc "
                                 f"{int(sm.get('score', 0)):+d}"):
                    st.markdown(
                        f'<span class="pill {GRADE_TONE.get(r["grade"], "mut")}">'
                        f'{r["grade"]} · {r["score"]}/100</span>',
                        unsafe_allow_html=True)
                    st.write(f"**Entry** {s.entry:,.6g} · **Stop** {s.stop:,.6g} · "
                             f"**TP1** {s.tp1:,.6g} · **TP2** {s.tp2:,.6g}")
                    if sm.get("flags"):
                        st.write("**Supporting** " + " · ".join(sm["flags"]))
                    if sm.get("warnings"):
                        st.write("**Against** " + " · ".join(sm["warnings"]))
                    for p in r["positives"]:
                        st.write(f"- {p}")
                    for n in r["negatives"]:
                        st.write(f"- {n}")
                    c1, c2 = st.columns(2)
                    if c1.button("Alert me", key=f"al{s.symbol}"):
                        ok, msg = notify(
                            f"{r['grade']} {s.symbol} {s.direction} "
                            f"entry {s.entry:,.6g} stop {s.stop:,.6g} "
                            f"tp1 {s.tp1:,.6g}")
                        st.success(msg) if ok else st.error(msg)
                    if c2.button("Log to paper", key=f"pp{s.symbol}"):
                        paper.open_trade(s.symbol, s.direction, s.entry, s.stop,
                                         s.tp1, s.tp2,
                                         s.meta.get("qty", 0), equity,
                                         " · ".join(sm.get("flags", [])[:3]))
                        st.success("Paper trade opened.")


# ---------------------------------------------------------------- smc
with T["SMC"]:
    ui.section("Smart money map",
               "Structure, zones, volume, the trade and its invalidation — on one chart.")

    s1, s2, s3 = st.columns([2, 1, 1])
    smc_sym = s1.text_input("Symbol", "BTC/USDT:USDT", key="smc_sym")
    smc_tf = s2.selectbox("Entry timeframe", ["3m", "5m", "15m", "1h", "4h", "1d"],
                          index=2, key="smc_tf",
                          help="Lower = earlier entries, more noise.")
    side = s3.radio("Side", ["long", "short"], horizontal=True)

    if st.button("Build map", type="primary", help="Fetch, mark and plan."):
        try:
            cfg = build_cfg(exch, top, min_vol, smc_tf, min_rr, max_stop_atr,
                            risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df, used = _load(ex, smc_sym, smc_tf, 500)
            if df is None:
                st.warning(f"No data for {smc_sym} on {ex_name}.")
            else:
                htf_name = htf_for(smc_tf)
                hdf, _ = _load(ex, smc_sym, htf_name, 400)
                hb = structure_bias(hdf, cfg)
                stt_tab = build_structure(df, cfg.structure)
                lb = str(stt_tab["trend"].iloc[-1])
                ok_mtf, mtf_why = aligned(hb, lb, side)

                plan = smc.playbook(df, side, equity, risk_pct)
                atr = plan["atr"]

                anchor = int(stt_tab["prev_swing_low" if side == "long"
                                     else "prev_swing_high"].iloc[-1])
                av = (avwap_context(df, side, atr, anchor) if np.isfinite(anchor)
                      else {"score": 0, "flags": [], "warnings": ["no anchor"],
                            "dist": 0.0})
                vp = volume_profile(df)
                in_va = bool(vp and vp["val"] <= plan["entry"] <= vp["vah"])

                k = st.columns(5)
                ui.stat(k[0], "HTF structure", hb,
                        "up" if hb == "up" else "dn" if hb == "down" else "warn")
                ui.stat(k[1], "LTF structure", lb,
                        "up" if lb == "up" else "dn" if lb == "down" else "warn")
                ui.stat(k[2], "MTF", "aligned" if ok_mtf else "against",
                        "up" if ok_mtf else "dn")
                ui.stat(k[3], "AVWAP", f"{av['dist']:+.2f} ATR",
                        "up" if av["score"] >= 40 else "dn")
                ui.stat(k[4], "In value area", "yes" if in_va else "no",
                        "up" if in_va else "warn")
                st.caption(f"pair {smc_tf}/{htf_name} · {mtf_why}")

                st.plotly_chart(chart.build(df, cfg, side, plan),
                                use_container_width=True,
                                config={"scrollZoom": True, "displaylogo": False})

                with st.expander("How to read this chart", expanded=False):
                    st.markdown(chart.legend_md())

                st.divider()
                ui.section(f"Playbook · {side.upper()}")
                render_plan(plan, used, ladder=(side == "long"))

                st.divider()
                ui.section("Order card", "Copy to the exchange. Reduce-only on exits.")
                st.code(
                    f"symbol      {used}\n"
                    f"side        {'BUY / LONG' if side == 'long' else 'SELL / SHORT'}\n"
                    f"type        LIMIT  (post-only)\n"
                    f"qty         {plan['qty']:.8g}\n"
                    f"entry       {plan['entry']:,.8g}\n"
                    f"stop        {plan['sl']:,.8g}   reduce-only\n"
                    f"tp1 50%     {plan['tp1']:,.8g}   reduce-only\n"
                    f"tp2 50%     {plan['tp2']:,.8g}   reduce-only\n"
                    f"risk        ${plan['size_usd']:.2f} ({risk_pct}% of {equity:,.0f})\n"
                    f"R:R         {plan['rr1']:.2f} / {plan['rr2']:.2f}\n"
                    f"invalidate  {'close below' if side == 'long' else 'close above'} "
                    f"{plan['sl']:,.8g}\n"
                    f"time stop   {CFG.risk.time_stop_bars} bars",
                    language="text")

                if st.button("Log to paper book", key="paper_log"):
                    paper.open_trade(used, side, plan["entry"], plan["sl"],
                                     plan["tp1"], plan["tp2"], plan["qty"], equity,
                                     " | ".join(plan["confluence"]["flags"][:3]))
                    st.success("Paper trade opened. Close it from the Paper tab.")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    st.divider()
    st.markdown(smc.READING_ORDER)


# ---------------------------------------------------------------- sizer
with T["Sizer"]:
    ui.section("Position size", "Leverage changes margin, never risk.")
    sizer.render(equity, risk_pct)


# ---------------------------------------------------------------- radar
with T["Radar"]:
    ui.section("Value radar", "Distance to the 200-day SMA across majors.")
    if st.button("Refresh radar", help="Recompute from the venue."):
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
            "vs 200d": f"{r['dist']:+.1f}%", "detail": " · ".join(r["reasons"]),
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
with T["Portfolio"]:
    ui.section("Portfolio", "Spot positions, heat and bucket split.")
    try:
        import portfolio as pf_mod
        with st.form("add_pos"):
            a, b, c, d = st.columns(4)
            p_sym = a.text_input("Symbol", "BTC/USDT", help="Pair you own.")
            p_qty = b.number_input("Qty", 0.0, 1e9, 0.0, format="%.8f",
                                   help="Units of the coin.")
            p_px = c.number_input("Price", 0.0, 1e9, 0.0, format="%.6f",
                                  help="Average buy price.")
            p_stop = d.number_input("Stop (0 = none)", 0.0, 1e9, 0.0,
                                    format="%.6f", help="Invalidation price.")
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


# ---------------------------------------------------------------- paper
if MODE == "Bot":
    with T["Paper"]:
        ui.section("Paper book", "Forward test. Promote to live only on the gate.")
        cleared, why = paper.gate()
        (st.success if cleared else st.info)(f"Promotion gate: {why}")

        s = paper.stats()
        if s.get("n"):
            m = st.columns(5)
            ui.stat(m[0], "Closed", str(s["n"]))
            ui.stat(m[1], "Win rate", f"{s['win_rate']}%")
            ui.stat(m[2], "Profit factor", str(s.get("pf") or "—"),
                    "up" if (s.get("pf") or 0) >= 1.3 else "dn")
            ui.stat(m[3], "Expectancy", f"{s['expectancy']:+.2f}R",
                    "up" if s["expectancy"] > 0 else "dn")
            ui.stat(m[4], "Avg R", f"{s['avg_r']:+.2f}")

        opens = paper.open_rows()
        if not opens:
            st.info("No open paper trades.")
        else:
            st.dataframe(pd.DataFrame([{
                "id": r["id"], "symbol": r["symbol"], "side": r["dir"],
                "entry": f"{r['entry']:,.6g}", "stop": f"{r['stop']:,.6g}",
                "tp1": f"{r['tp1']:,.6g}", "tp2": f"{r['tp2']:,.6g}",
                "qty": f"{r['qty']:.6g}", "opened": r["ts"][:16],
            } for r in opens]), hide_index=True, use_container_width=True)

            with st.form("close_paper"):
                c1, c2, c3 = st.columns(3)
                tid = c1.selectbox("Ticket", [r["id"] for r in opens])
                px = c2.number_input("Exit price", 0.0, 1e9, 0.0, format="%.6f")
                tag = c3.selectbox("Reason", ["tp1", "tp2", "stop", "time_stop",
                                              "manual"])
                if st.form_submit_button("Close") and px > 0:
                    paper.close_trade(tid, px, tag)
                    st.rerun()

        if paper.book():
            st.download_button("Export paper book",
                               json.dumps(paper.book(), indent=2),
                               file_name="paper.json")


# ---------------------------------------------------------------- backtest
if MODE == "Bot":
    with T["Backtest"]:
        ui.section("Backtest", "Same filters, next-open fills.")
        b1, b2 = st.columns(2)
        bt_sym = b1.text_input("Symbol", "BTC/USDT:USDT", key="bt_sym",
                               help="Use the perp format.")
        bt_bars = b2.number_input("Bars", 500, 20000, 6000, step=500,
                                  help="History depth.")
        if st.button("Run backtest", type="primary", help="Replay the strategy."):
            try:
                with st.spinner("Running..."):
                    curve, trades, raw_n, kept_n = cached_backtest(
                        exch, bt_sym, ltf, int(bt_bars), min_rr, max_stop_atr,
                        risk_pct, equity, exch, proxy.strip() or None)
                if curve is None:
                    st.warning("Not enough history.")
                else:
                    st.caption(f"raw {raw_n} · after filters {kept_n} · "
                               f"trades {len(trades)}")
                    fig = go.Figure(go.Scatter(x=curve.index, y=curve.values,
                                               line=dict(color="#14b8a6", width=1.6)))
                    fig.update_layout(height=340, template="plotly_dark",
                                      paper_bgcolor="#0f1116",
                                      plot_bgcolor="#0f1116",
                                      margin=dict(l=6, r=6, t=6, b=6))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(trades.tail(50), hide_index=True,
                                 use_container_width=True)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- macro
if MODE == "Bot":
    with T["Macro"]:
        ui.section("Macro", "Sentiment, dominance and dry powder.")
        try:
            import macro as macro_mod
            mstate = _call(macro_mod, ["state", "macro_state", "evaluate",
                                       "snapshot", "get_state"])
            if isinstance(mstate, dict) and mstate.get("blackout"):
                st.warning(mstate.get("blackout_reason", "Blackout active."))
        except Exception:
            pass
        label, ltone, notes = market.stance(fng, ov["btc_dom"], stab)
        s1, s2 = st.columns(2)
        ui.stat(s1, "Stance", label, ltone)
        ui.stat(s2, "Stablecoin share", f"{stab:.1f}%" if stab is not None else "—")
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=fng if fng is not None else 0,
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#14b8a6", "thickness": .28},
                   "steps": [{"range": [0, 25], "color": "#2a1214"},
                             {"range": [25, 45], "color": "#2b2110"},
                             {"range": [45, 55], "color": "#161a21"},
                             {"range": [55, 75], "color": "#122a1c"},
                             {"range": [75, 100], "color": "#0f3b24"}]}))
        g.update_layout(height=250, template="plotly_dark",
                        paper_bgcolor="#0f1116",
                        margin=dict(l=18, r=18, t=10, b=10))
        st.plotly_chart(g, use_container_width=True)
        for n in notes:
            st.write(f"- {n}")


# ---------------------------------------------------------------- alerts
if MODE == "Bot":
    with T["Alerts"]:
        ui.section("Alerts", "Telegram and email.")
        tg1, tg2 = st.columns([2, 1])
        tg_token = tg1.text_input("Telegram bot token", value=_cred("TELEGRAM_TOKEN"),
                                  type="password", help="From @BotFather.")
        tg_chat = tg2.text_input("Chat id", value=_cred("TELEGRAM_CHAT_ID"),
                                 help="Numeric id from @userinfobot.")
        with st.expander("Email (optional)"):
            e1, e2 = st.columns(2)
            smtp_host = e1.text_input("SMTP host", value=_cred("SMTP_HOST"))
            smtp_port = e2.text_input("SMTP port", value=_cred("SMTP_PORT"))
            smtp_user = st.text_input("SMTP user", value=_cred("SMTP_USER"))
            smtp_pass = st.text_input("SMTP password", value=_cred("SMTP_PASS"),
                                      type="password")
            alert_to = st.text_input("Send to", value=_cred("ALERT_TO"))
        a1, a2 = st.columns(2)
        payload = {"TELEGRAM_TOKEN": tg_token, "TELEGRAM_CHAT_ID": tg_chat,
                   "SMTP_HOST": smtp_host, "SMTP_PORT": smtp_port,
                   "SMTP_USER": smtp_user, "SMTP_PASS": smtp_pass,
                   "ALERT_TO": alert_to}
        if a1.button("Save settings", use_container_width=True):
            _save_creds(payload)
            st.success("Saved.")
        if a2.button("Send test", type="primary", use_container_width=True):
            _save_creds(payload)
            with st.spinner("sending..."):
                ok, msg = notify("CryptoScan test — alerts working.")
            st.success(msg) if ok else st.error(msg)
        st.caption("400 chat not found means message your bot /start once. "
                   "401 means the token is wrong.")


# ---------------------------------------------------------------- journal
if MODE == "Bot":
    with T["Journal"]:
        ui.section("Journal", "Log every trade, including the ones you skipped.")
        with st.form("journal", clear_on_submit=True):
            j1, j2, j3, j4 = st.columns(4)
            j_sym = j1.text_input("Symbol", help="Pair traded.")
            j_dir = j2.selectbox("Side", ["long", "short"], help="Direction.")
            j_r = j3.number_input("Result (R)", -10.0, 10.0, 0.0, step=0.1,
                                  help="-1 means stopped.")
            j_ok = j4.checkbox("Followed rules", value=True,
                               help="Taken exactly as planned?")
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


# ---------------------------------------------------------------- learn
with T["Learn"]:
    ui.section("Learn", "Work the stages in order. Do not skip to stage 5.")
    for stage, title, status, items in learn.STAGES:
        tone = "up" if status == "started" else "warn" if status == "next" else "mut"
        with st.expander(f"{stage} · {title}"):
            st.markdown(f'<span class="pill {tone}">{status}</span>',
                        unsafe_allow_html=True)
            for it in items:
                st.write(f"- {it}")
    st.divider()
    st.markdown("#### Priority order")
    st.dataframe(pd.DataFrame([{"#": a, "concept": b, "why": c}
                               for a, b, c in learn.PRIORITY]),
                 hide_index=True, use_container_width=True)
    st.divider()
    st.markdown("#### How the concepts fit together")
    st.markdown(learn.FRAMEWORK)
    st.divider()
    st.markdown("#### A daily sequence")
    st.markdown(learn.DAILY)


# ---------------------------------------------------------------- guide
with T["Guide"]:
    ui.section("Guide")
    st.markdown("""
### Spot vs Bot
Spot is long-only accumulation on daily charts, laddered into demand. Bot is
the long and short scanner. They are separate because the jobs differ: spot
wants time and patience, bot wants structure and a tight stop.

### Getting bot signals
Grade C, freshness 12, Relax on, then Run scan. Read the diagnostic line. If
`mtf` or `volume` rejections dominate, the market is out of phase — that is
information, not a fault.

### Playbook verdict
Every SMC plan scores seven checks: trend, structure break, zone, range
position, liquidity taken, R:R of 2 or more, stop inside 3 ATR. Six or more is
an A, five a B at half size, four is watch only, below four skip.

### Grades
A+ 88+ · A 78+ · B 66+ · C 52+ · D below 52.

### Sizing
qty = (equity × risk percent) ÷ (entry − stop)

### Order placement
Entry is a post-only limit. Stop is a stop-market, reduce-only, triggered on
mark price. TP1 and TP2 are limits, reduce-only, 50% each. Round size DOWN to
the step size. Set isolated margin and leverage low enough that liquidation
sits beyond the stop — the Sizer tells you the max safe leverage.

### Rules
- 0.5% risk per trade, 3% total heat.
- Two correlated positions maximum per direction.
- No entries within 24h of a major macro print.
- Paper trade 30 signals and clear the gate before real money.
""")


st.divider()
st.caption("CryptoScan · not financial advice")
