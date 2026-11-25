"""
app.py - CryptoScan. Spot mode and Trade mode.

    pip install -r requirements.txt
    streamlit run app.py
"""
import inspect
import json
import math
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
import board
import chart
import learn
import market
import paper
import sizer
import smc
import sop
import ui
from config import AUTH_ENABLED, CFG
from core import reversal_alerts
from core.models import Layer
from core.tfs import CHART_TFS, cap_htf, cap_ltf, chart_minutes, data_tf
from strategy import reversal

try:
    import live
    LIVE_OK = True
except Exception:
    LIVE_OK = False

try:
    import analysis_page
    ANALYSIS_OK, ANALYSIS_ERR = True, ""
except Exception as _e:
    ANALYSIS_OK, ANALYSIS_ERR = False, f"{type(_e).__name__}: {_e}"

st.set_page_config(page_title="CryptoScan by Chris", page_icon="◈", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject()


def _auth_on() -> bool:
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
from strategy.risk import filter_setups, plan_rr
from core.opportunities import assess as assess_opportunity, current_only
from strategy.setups import scan_bos_retest
from strategy.multistrategy import scan_seven, STRATEGIES
from backtest.engine import run as engine_run
from backtest.metrics import report
from backtest.validate import monte_carlo_drawdown, permutation_test

GRADE_ORDER = ["D", "C", "B", "A", "A+"]
GRADE_RANK = {g: i for i, g in enumerate(GRADE_ORDER)}
GRADE_TONE = {"A+": "up", "A": "up", "B": "warn", "C": "mut", "D": "dn"}
OK_STATUS = {"ACTIONABLE", "BETTER_PRICE"}
MIN_BARS = 300
HAS_FRAGMENT = hasattr(st, "fragment")
CHART_BARS = 3000
DEFAULT_TF = "30m"

RATIO = {"3m": 5, "5m": 6, "15m": 4, "1h": 4, "4h": 6, "1d": 1, "1w": 1}
HTF_BARS = 260
TF_MIN = {"1s": 1 / 60, "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
          "1h": 60, "4h": 240, "1d": 1440}
OHLC = ("open", "high", "low", "close")

LOGIC_MD = """
### The pair is chosen, and both sides are capped

The higher timeframe is picked from **30m or 1h** — never slower than 1h — by
how cleanly the market is structured: swing separation measured in ATR, swing
density, and whether volatility sits in a workable band.

The lower timeframe is picked from **3m or 5m** — never slower than 5m — by
running the full assessment on each and keeping the better setup. The HTF
decides location; the LTF decides whether the trigger actually fired. Both are
printed on every card, alert and chart title, and both pass through a clamp so
a stale label cannot leak through.

30m and 1h are resampled from the loaded 15m frame, so a lower timeframe is
never invented.

| role | higher timeframe | lower timeframe |
|---|---|---|
| what it decides | trend, major pivots, zones, the candidate | trigger, minor pivots, sweep, candle |
| why | context: are we at a reversal location at all | timing: is the reversal actually starting |

### Sequence

1. **HTF trend.** `build_structure` returns up, down or range. A long needs the
   HTF not-down, a short not-up. Continuation is skipped.
2. **HTF location.** `core.levels.build_zones` clusters swing prices. The
   nearest unmitigated zone within 6 ATR becomes the candidate.
3. **Proximity.** Measured in HTF ATR. Past 6 ATR the coin drops off. Inside
   1.5 ATR it is `APPROACHING`.
4. **LTF trigger.** In the zone, the LTF must print a CHOCH or BOS in the
   reversal direction within 15 bars **and** a candle scoring 60+ from
   `core.candles.quality` — the same ATR grader the scanner uses.
5. **Bracket.** Entry is the zone edge facing price. Stop is the far edge plus
   0.35 × LTF ATR. TP1 is the next opposing zone or 1.5R, TP2 the next one or
   3.0R, capped by `max_tp_r`.
6. **Gates.** `ACTIVE_ENTRY` needs R:R ≥ `min_rr`, stop ≤ `max_stop_atr`,
   travel over 12 bars ≤ 4 ATR, and no open paper position.

### Only future trades are listed

A row is `executable` only when it is `APPROACHING`, `DEVELOPING` or
`ACTIVE_ENTRY`, **and** R:R still clears `min_rr`, **and** the move is not
extended, **and** you are not already in it. Anything else is marked
`past / played` and pushed below the live rows, or dropped from the board.
`MISSED` and `INVALIDATED` never appear as live.

### Statuses

| status | meaning |
|---|---|
| `APPROACHING` | inside 1.5 ATR, no trigger. ETA is a projection, not a promise |
| `DEVELOPING` | in the zone, trigger not satisfied |
| `CONFIRMED` | triggered, but entry gone, R:R fails, or already in |
| `ACTIVE_ENTRY` | new, confirmed, in range, gates pass |
| `MISSED` | ran past the zone |
| `INVALIDATED` | stop breached |

Closed candles only. The forming bar never produces a claim.
"""


def _scan_bars(ltf: str) -> int:
    return max(1200, HTF_BARS * RATIO.get(ltf, 6))


def fresh_bars(hours: float, tf: str) -> int:
    """Hours of freshness expressed in bars of the timeframe actually scanned.

    12 bars of 5m is one hour, not twelve. This is what the slider means now.
    """
    mins = max(chart_minutes(tf), 1e-6)
    return max(1, min(4000, int(round(float(hours) * 60.0 / mins))))


VENUE_ORDER = ["okx", "bybit", "kucoinfutures", "mexc", "gateio", "bitget",
               "binanceusdm", "binance", "binanceus"]
MIN_PERPS = 20

MAJORS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK",
          "TON", "SUI", "DOT", "NEAR", "APT", "ARB", "OP", "INJ", "SEI", "TIA"]

DEFAULT_TG_TOKEN = ""
DEFAULT_TG_CHAT = ""
CREDS_FILE = pathlib.Path("./state/alerts.json")


def _call(mod, names, *a, **k):
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f(*a, **k)
    raise AttributeError(f"{mod.__name__} has none of {names}")


def _has(mod, names) -> bool:
    return any(callable(getattr(mod, n, None)) for n in names)


def _ind(ind, *names):
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


def _anchor(stt, direction: str):
    col = "prev_swing_low" if direction == "long" else "prev_swing_high"
    try:
        s = stt[col].tail(60).dropna()
        return int(float(s.iloc[-1])) if len(s) else None
    except Exception:
        return None


def _closed(df):
    """Exclude only a candle that is still forming, not a historical closed bar."""
    if df is None or len(df) <= 2 or not isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        # This helper is also used on historical/HTF frames; infer duration only
        # when consecutive timestamps are available and consistently spaced.
        timestamps = df.index
        duration = timestamps[-1] - timestamps[-2]
        if duration <= pd.Timedelta(0):
            return df
        last = pd.Timestamp(timestamps[-1])
        last = last.tz_localize("UTC") if last.tzinfo is None else last
        if pd.Timestamp.now(tz="UTC") < last + duration:
            return df.iloc[:-1]
    except (ValueError, TypeError, OverflowError):
        pass
    return df


def _tidy(df):
    if df is None or not len(df):
        return df
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    elif d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    for c in OHLC:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    keep = [c for c in OHLC if c in d.columns]
    d = d.dropna(subset=keep)
    if keep:
        d = d[np.isfinite(d[keep].to_numpy(float)).all(axis=1)]
    return d


def _detail_figure(df, tf: str, bars: int = 120, symbol: str = "") -> tuple:
    """Plain candle chart collapsing only real holes. No candle is invented."""
    d = _tidy(df)
    hist = d  # do not discard older candles when zooming or panning
    view = hist.tail(bars)
    diffs = pd.Series(hist.index).diff().dropna()
    good = diffs[diffs > pd.Timedelta(0)]
    step = good.median() if len(good) else pd.Timedelta(minutes=1)

    breaks = []
    for ts, gap in zip(hist.index[1:], diffs):
        if gap > step * 3 and len(breaks) < 30:
            breaks.append(dict(bounds=[ts - gap, ts]))

    # Keep historical price levels within the Y range while panning left.
    lo, hi = float(hist["low"].min()), float(hist["high"].max())
    pad = (hi - lo) * 0.03 or max(abs(hi), 1.0) * 0.001
    m = TF_MIN.get(tf, 60)
    tick = "%H:%M:%S" if m < 1 else "%H:%M" if m < 60 else (
        "%d %b %H:%M" if m < 1440 else "%d %b")

    fig = go.Figure(go.Candlestick(
        x=hist.index, open=hist["open"], high=hist["high"], low=hist["low"],
        close=hist["close"], increasing_line_color="#16c784",
        decreasing_line_color="#ea3943", increasing_fillcolor="#16c784",
        decreasing_fillcolor="#ea3943", whiskerwidth=0.6))
    fig.update_layout(height=460, template="plotly_dark",
                      paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                      margin=dict(l=58, r=10, t=85, b=22), showlegend=False,
                      dragmode="pan", uirevision=f"detail|{symbol}|{tf}",
                      title=dict(text=f"{tf} · {len(view)} shown of "
                                      f"{len(hist)} loaded · "
                                      f"{len(breaks)} real gap(s) collapsed",
                                 x=0, xanchor="left",
                                 font=dict(size=10, color="#7f8798")),
                      xaxis=dict(type="date", showgrid=False, nticks=8,
                                 tickformat=tick, rangeslider=dict(visible=False),
                                 range=[view.index[0], view.index[-1]],
                                 autorange=False, rangebreaks=breaks,
                                 fixedrange=False, constrain="range"),
                      yaxis=dict(range=[lo - pad, hi + pad], autorange=False,
                                 side="left", gridcolor="#1b1f28",
                                 tickformat=",.6g", fixedrange=False))
    from chart_axes import add_independent_axis_controls
    add_independent_axis_controls(fig, [view.index[0], view.index[-1]],
                                  [lo - pad, hi + pad])
    return fig, len(breaks)


def _norm(rt) -> dict:
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

    def attempt(name):
        kw = {"preferred": name} if "preferred" in sig else {}
        if "proxy" in sig:
            kw["proxy"] = proxy
        try:
            try:
                return resolve_exchange(cfg, **kw)
            except TypeError:
                kw.pop("proxy", None)
                return resolve_exchange(cfg, **kw)
        except Exception:
            return None

    order = []
    for v in (preferred, st.session_state.get("_venue"), *VENUE_ORDER):
        if v and v != "auto" and v not in order:
            order.append(v)

    for i, name in enumerate(order):
        got = attempt(name)
        if not got:
            continue
        ex, used = got
        if i == 0 or _perps(ex) >= MIN_PERPS:
            st.session_state["_venue"] = name
            st.session_state["_venue_warn"] = (
                None if i == 0 else f"{preferred} did not answer, using {name}")
            return ex, used
    raise RuntimeError("no reachable venue — press Test venues")


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


def _universe(ex, min_vol: int, top: int):
    ex.load_markets()
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        tickers = {}
    rows = []
    for sym, m in ex.markets.items():
        if not m.get("active", True) or m.get("quote") != "USDT":
            continue
        if not (m.get("swap") or m.get("spot")):
            continue
        t = tickers.get(sym) or {}
        try:
            qv = float(t.get("quoteVolume") or 0)
            if not math.isfinite(qv) or qv <= 0:
                qv = float(t.get("baseVolume") or 0) * float(t.get("last") or 0)
            if not math.isfinite(qv) or qv <= 0:
                qv = 0.0
        except (TypeError, ValueError, OverflowError):
            qv = 0.0
        rows.append((sym, qv, bool(m.get("swap") or m.get("contract"))))
    perps = [r for r in rows if r[2]]
    pool = perps if len(perps) >= 10 else rows
    floor = min_vol
    filtered = [r for r in pool if r[1] >= floor]
    # Do not silently dilute the liquidity floor to fill the quota.
    filtered.sort(key=lambda r: -r[1])
    syms = [r[0] for r in filtered[:min(100, max(1, int(top)))]]
    if not syms:
        return [], floor, len(perps)
    return syms, floor, len(perps)


def _load(ex, sym: str, tf: str, limit: int, min_bars: int = MIN_BARS):
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
        if df is not None and len(df) >= min_bars:
            return df, cand
    return None, sym


def build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity):
    return replace(
        CFG,
        scan=replace(CFG.scan, exchange=exchange, top_n_symbols=top,
                     min_quote_vol_24h=min_vol, ltf=cap_ltf(ltf),
                     htf=cap_htf(htf_for(ltf))),
        risk=replace(CFG.risk, min_rr=min_rr, max_stop_atr=max_stop_atr,
                     risk_per_trade_pct=risk_pct, account_equity=equity),
    )


def setup_status(s, df, atr=0.0) -> str:
    """A flat 0.25% band is noise on a 5m bar and marks everything EXTENDED.
    The band scales with ATR, floored at 0.12% of price."""
    cur = float(df["close"].iloc[-1])
    tol = max(cur * 0.0012, 0.35 * float(atr or 0.0))
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


def _send_telegram(text: str):
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


def _send_email(text: str):
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


def notify(text: str):
    ok, msg = _send_telegram(text)
    if ok:
        return True, "telegram sent"
    ok2, msg2 = _send_email(text)
    if ok2:
        return True, "email sent"
    return False, f"{msg} | {msg2}"


@st.cache_data(ttl=120, show_spinner=False)
def cached_scan(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                equity, recent_h, min_grade, preferred, proxy, relax,
                need_smc, long_only, need_mtf=False, need_vol=False):
    if relax:
        min_rr = max(1.2, min_rr * 0.7)
        max_stop_atr = max_stop_atr * 1.4
        min_grade = GRADE_ORDER[max(0, _rank(min_grade) - 1)]

    cfg = build_cfg(exchange, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, ex_name = _resolve(cfg, preferred, proxy)
    universe, floor, perps = _universe(ex, int(min_vol), top)
    bars = _scan_bars(cfg.scan.ltf)
    fresh = fresh_bars(recent_h, cfg.scan.ltf)
    stats = load_stats()
    found = []
    diag = {"venue": ex_name, "perps": perps, "universe": len(universe),
            "floor": floor, "loaded": 0, "err": 0, "raw": 0, "kept": 0,
            "stale": 0, "graded": 0, "below_grade": 0, "smc_skip": 0,
            "mtf_skip": 0, "vol_skip": 0, "no_anchor": 0, "bias_blank": 0,
            "past_skip": 0, "actionable": 0, "better": 0, "extended": 0,
            "invalid": 0, "htf": cfg.scan.htf, "ltf": cfg.scan.ltf,
            "bars": bars, "recent": recent_h, "fresh": fresh}

    for sym in universe:
        df, used_sym = _load(ex, sym, cfg.scan.ltf, bars)
        if df is None:
            diag["err"] += 1
            continue
        # Never score a setup from an in-progress candle: its high, low,
        # volume and close are not final and can repaint on the next refresh.
        df = _closed(df)
        if df is None or len(df) < MIN_BARS:
            diag["err"] += 1
            continue
        diag["loaded"] += 1
        try:
            ind = compute_all(df, cfg)
            stt = build_structure(df, cfg.structure)
            hb = htf_bias(df, cfg)
            if not (hb == "up").any() and not (hb == "down").any():
                diag["bias_blank"] += 1
            sw = find_swings(df, cfg.structure.swing_left, cfg.structure.swing_right)
            zu = build_zones(df, sw, ind["atr"], cfg.structure, "res")
            zd = build_zones(df, sw, ind["atr"], cfg.structure, "sup")
            raw = scan_seven(used_sym, df, hb, ind, stt, zu, zd, cfg, recent=max(3, fresh + 1))
            diag["raw"] += len(raw)
            kept = filter_setups(raw, ind, zu, zd, cfg)
            if long_only:
                kept = [s for s in kept if s.direction == "long"]
            diag["kept"] += len(kept)
            if kept:
                cut = df.index[max(0, len(df) - 1 - fresh)]
                fresh_kept = [s for s in kept if s.ts >= cut]
                diag["stale"] += len(kept) - len(fresh_kept)
                kept = fresh_kept
            atr_now = _last(ind, "atr", default=0.0)
            for s in kept:
                s.meta["status"], s.meta["status_reason"] = assess_opportunity(
                    s, df, max_age_bars=fresh,
                    timeframe_seconds=chart_minutes(cfg.scan.ltf) * 60)
                s.meta["htf"] = cfg.scan.htf
                s.meta["ltf"] = cfg.scan.ltf
                stt_s = s.meta["status"]
                if stt_s == "ACTIONABLE":
                    diag["actionable"] += 1
                elif stt_s == "BETTER_PRICE":
                    diag["better"] += 1
                elif stt_s == "EXTENDED":
                    diag["extended"] += 1
                elif stt_s == "INVALID":
                    diag["invalid"] += 1
                else:
                    diag["past_skip"] += 1
                if stt_s not in OK_STATUS:
                    continue
                zone = (zu[0] if zu else None) if s.direction == "long" \
                    else (zd[0] if zd else None)
                s.meta["r"] = _norm(rate(s, ind.loc[s.ts], zone, cfg, stats))
                diag["graded"] += 1
                if _rank(s.meta["r"]["grade"]) < _rank(min_grade):
                    diag["below_grade"] += 1
                    continue
                try:
                    s.meta["smc"] = smc.confluence(
                        _closed(df), s.direction, s.entry,
                        _last(ind, "atr", default=0.0),
                        idx=df.index.get_loc(s.ts))
                except Exception:
                    s.meta["smc"] = {"score": 0, "flags": [], "warnings": []}
                if need_smc and s.meta["smc"]["score"] < 2:
                    diag["smc_skip"] += 1
                    continue

                anchor = _anchor(stt, s.direction)
                av = (avwap_context(_closed(df), s.direction,
                                    _last(ind, "atr", default=0.0), anchor)
                      if anchor is not None
                      else {"score": 0, "flags": [], "warnings": ["no anchor"],
                            "dist": 0.0, "usable": False})
                if anchor is None:
                    diag["no_anchor"] += 1
                s.meta["avwap"] = av
                vp = volume_profile(_closed(df))
                s.meta["in_value"] = bool(vp and vp["val"] <= s.entry <= vp["vah"])
                if need_vol and av["score"] < 40:
                    diag["vol_skip"] += 1
                    continue

                if need_mtf:
                    htf_df, _ = _load(ex, used_sym, cfg.scan.htf,
                                      cfg.scan.htf_bars, 60)
                    hb_s = structure_bias(_closed(htf_df), cfg) \
                        if htf_df is not None else "range"
                    ok_mtf, why = aligned(hb_s, str(stt["trend"].iloc[-1]), s.direction)
                    s.meta["mtf"] = {"ok": ok_mtf, "why": why, "htf": hb_s}
                    if not ok_mtf:
                        diag["mtf_skip"] += 1
                        continue
                else:
                    s.meta["mtf"] = {"ok": True, "why": "off", "htf": "—"}

                found.append(s)
        except Exception:
            diag["err"] += 1
            continue

    found = current_only(found)
    return found, diag


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest(exchange, symbol, ltf, limit, min_rr, max_stop_atr,
                    risk_pct, equity, preferred, proxy, strategy="BOS Retest"):
    cfg = build_cfg(exchange, 40, 30e6, ltf, min_rr, max_stop_atr, risk_pct, equity)
    ex, _ = _resolve(cfg, preferred, proxy)
    df, used = _load(ex, symbol, cfg.scan.ltf, int(limit), 200)
    if df is None:
        return None, None, 0, 0, 0
    from run_backtest import build_trades
    curve, trades, raw_n, kept_n = build_trades(_closed(df), cfg, used, strategy)
    return curve, trades, raw_n, kept_n, len(df)


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
            rsi = _last(compute_all(df, cfg), "rsi", default=float("nan"))
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


@st.cache_data(ttl=120, show_spinner=False)
def cached_reversal(exchange, preferred, proxy, symbols):
    """3m, 5m and 15m per symbol. HTF 30m/1h, LTF 3m/5m, both by setup."""
    cfg = build_cfg(exchange, 40, 30e6, reversal.SCALP, 2.0, 3.5, 0.5, 10000)
    ex, name = _resolve(cfg, preferred, proxy)
    try:
        held = {r["symbol"] for r in paper.open_rows()}
    except Exception:
        held = set()
    rows = []
    for c in symbols:
        try:
            ldf, used = _load(ex, f"{c}/USDT:USDT", reversal.SCALP,
                              reversal.LTF_BARS, reversal.MIN_LTF_BARS)
            if ldf is None:
                continue
            fdf, _ = _load(ex, used, reversal.FAST, reversal.LTF_BARS,
                           reversal.MIN_LTF_BARS)
            if fdf is None:
                continue
            mdf, _ = _load(ex, used, reversal.BASE, reversal.LTF_BARS,
                           reversal.MIN_LTF_BARS)
            if mdf is None:
                continue
            src = {reversal.SCALP: _closed(ldf), reversal.FAST: _closed(fdf),
                   reversal.BASE: _closed(mdf)}
            r = reversal.evaluate(used, src, cfg, already=used in held)
            if r:
                rows.append(r)
        except Exception:
            continue
    return rows, name


with st.sidebar:
    st.markdown('<div class="auth-logo">◈</div>', unsafe_allow_html=True)
    st.markdown("**CryptoScan by Chris**")
    st.caption(f"signed in as {USER}" if AUTH_ON else "local mode · login disabled")
    if AUTH_ON and st.button("Sign out", use_container_width=True):
        auth.logout()

    MODE = st.radio("Desk", ["Spot", "Trade"], horizontal=True, index=0)

    exch = st.selectbox("Exchange", ["auto"] + CANDIDATES, index=0,
                        help="auto walks a cloud-friendly list. Pick one "
                             "directly to stop the fallback.")
    if st.session_state.get("_venue"):
        st.caption(f"in use: {st.session_state['_venue']}")
    if st.session_state.get("_venue_warn"):
        st.warning(st.session_state["_venue_warn"])
    if st.button("Reset venue", use_container_width=True):
        st.session_state.pop("_venue", None)
        st.session_state.pop("_venue_warn", None)
        st.rerun()

    proxy = st.text_input("Binance proxy (optional)", value="",
                          placeholder="https://your-worker.workers.dev")
    if st.button("Test venues", use_container_width=True):
        with st.spinner("probing..."):
            st.session_state["health"] = probe_exchanges(proxy.strip() or None)
    h = st.session_state.get("health")
    if h:
        st.dataframe(pd.DataFrame([{"venue": n, "ok": "YES" if r["ok"] else "no",
                                    "markets": r.get("markets", 0), "ms": r["ms"]}
                                   for n, r in h.items()]),
                     hide_index=True, use_container_width=True)

    st.divider()
    ltf = st.selectbox("Timeframe", ["4h", "1d"] if MODE == "Spot" else ["1h", "4h", "1d"],
                       index=1)
    st.caption(f"scan {cap_ltf(ltf)} · bias {cap_htf(htf_for(ltf))} · "
               f"{_scan_bars(cap_ltf(ltf))} bars · trigger never slower than 5m")
    st.caption("Liquidity-ranked universe: maximum 100 coins; no unlimited scan.")
    top = st.slider("High-liquidity coins to scan", 10, 100, 100, step=10)
    min_vol = st.select_slider("Min 24h volume",
                               [10e6, 30e6, 50e6, 100e6, 250e6], value=30e6,
                               format_func=lambda v: f"${v/1e6:.0f}M")
    recent = st.slider("Signal freshness (hours)", 1, 72, 12,
                       help="Wall-clock age a setup may have and still be "
                            "listed. Converted to bars of the scanned timeframe, "
                            "so 12h is 144 bars of 5m.")
    min_grade = st.select_slider("Minimum grade", GRADE_ORDER, value="C")
    only_ok = st.checkbox("Actionable only", value=True)
    relax = st.checkbox("Relax filters", value=True)
    need_smc = st.checkbox("Require SMC confluence", value=False)
    need_mtf = st.checkbox("Require MTF alignment", value=False,
                           help="Both structures must agree. Strict.")
    need_vol = st.checkbox("Require AVWAP confluence", value=False,
                           help="Price must be on the correct side of the anchor "
                                "VWAP. Strict.")

    st.divider()
    equity = st.number_input("Account equity ($)", 100.0, 10_000_000.0, 10_000.0,
                             step=500.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 0.5, step=0.1)
    min_rr = st.slider("Minimum R:R", 1.5, 5.0, 2.0, step=0.25)
    max_stop_atr = st.slider("Max stop (ATR)", 1.5, 6.0, 3.5, step=0.25)
    st.divider()
    st.caption("Not financial advice. Backtest before risking money.")

st.markdown(
    f"""<div class="hero">
    <h1>CryptoScan by Chris</h1>
    <p><b>{'Spot accumulation desk' if MODE == 'Spot' else 'Trade desk'}</b><br>
    {'Long-only, daily structure, laddered entries into demand.' if MODE == 'Spot'
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
    NAMES = ["Overview", "Accumulate", "Setups", "SMC", "Sizer", "Radar",
             "Portfolio", "SOP", "Analysis", "Learn", "Guide"]
else:
    NAMES = ["Overview", "Setups", "Scan", "SMC", "Sizer", "Radar", "Portfolio",
             "Paper", "Backtest", "Macro", "Alerts", "Journal", "SOP", "Analysis",
             "Learn", "Guide"]
T = dict(zip(NAMES, st.tabs(NAMES)))


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
             f"**TP1** {plan['tp1']:,.6g} ({plan['rr1']:.2f}R) · "
             f"**TP2** {plan['tp2']:,.6g} ({plan['rr2']:.2f}R) · "
             f"weighted {plan['weighted_rr']:.2f}R")
    st.write(f"Risk ${plan['size_usd']:.2f} ({risk_pct}% of equity) · "
             f"qty {plan['qty']:.6g}")
    if ladder:
        st.dataframe(pd.DataFrame([{
            "tranche": t["tranche"], "price": f"{t['price']:,.6g}",
            "qty": f"{t['qty']:.6g}", "weight": f"{t['weight']*100:.0f}%",
        } for t in smc.ladder(plan)]), hide_index=True, use_container_width=True)
    st.caption(f"session {conf['session']['name']} · volatility "
               f"{conf['session']['vol']}")


def order_card(plan: dict, symbol: str, side: str, tf: str = "") -> None:
    rr1 = plan.get("rr1", 0.0)
    rr2 = plan.get("rr2", 0.0)
    w_rr = plan.get("weighted_rr", (rr1 + rr2) / 2 if (rr1 or rr2) else 0.0)
    bars = CFG.risk.time_stop_bars
    mins = TF_MIN.get(tf or "", 0)
    stop_txt = (f"{bars} bars @ {tf} ≈ {bars * mins:.0f} min"
                if mins else f"{bars} bars @ {tf or 'setup tf'}")
    st.code(
        f"symbol      {symbol}\n"
        f"side        {'BUY / LONG' if side == 'long' else 'SELL / SHORT'}\n"
        f"type        LIMIT  (post-only)\n"
        f"qty         {plan['qty']:.8g}\n"
        f"entry       {plan['entry']:,.8g}\n"
        f"stop        {plan['sl']:,.8g}   reduce-only   (1.00R)\n"
        f"tp1 50%     {plan['tp1']:,.8g}   reduce-only   ({rr1:.2f}R)\n"
        f"tp2 50%     {plan['tp2']:,.8g}   reduce-only   ({rr2:.2f}R)\n"
        f"risk        ${plan['size_usd']:.2f} ({risk_pct}% of {equity:,.0f})\n"
        f"R:R         tp1 {rr1:.2f} · tp2 {rr2:.2f} · weighted {w_rr:.2f}\n"
        f"invalidate  {'close below' if side == 'long' else 'close above'} "
        f"{plan['sl']:,.8g}\n"
        f"time stop   {stop_txt}",
        language="text")


with T["Setups"]:
    board.render(cached_scan, build_cfg, _resolve, _load, _norm, order_card,
                 notify, exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                 equity, recent, proxy.strip() or None, relax, need_smc,
                 need_mtf, need_vol)


with T["Overview"]:
    ui.section("Coin detail", "Candles, structure state and venue depth.")
    d1, d2 = st.columns([2, 1])
    detail_sym = d1.text_input("Symbol", "BTC/USDT:USDT", key="detail_sym")
    detail_tf = d2.selectbox("Timeframe",
                             ["1s", "1m", "5m", "15m", "1h", "4h", "1d", "1w"],
                             index=5)
    if st.button("Load detail", type="primary"):
        try:
            cfg = build_cfg(exch, top, min_vol, detail_tf, min_rr, max_stop_atr,
                            risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df, used = _load(ex, detail_sym, detail_tf, CHART_BARS, 100)
            if df is None:
                st.warning(f"Not enough candles for {detail_sym} on {ex_name}.")
            else:
                ind = compute_all(df, cfg)
                trend, notes = smc.trend_state(df)
                last = float(df["close"].iloc[-1])
                chg = (last / float(df["close"].iloc[-25]) - 1) * 100
                rsi = _last(ind, "rsi", default=50.0)
                m1, m2, m3, m4 = st.columns(4)
                ui.stat(m1, "Last", f"{last:,.6g}")
                ui.stat(m2, "24 bar %", f"{chg:+.2f}%", "up" if chg >= 0 else "dn")
                ui.stat(m3, "RSI", f"{rsi:.1f}",
                        "dn" if rsi > 70 else "up" if rsi < 30 else "")
                ui.stat(m4, "Structure", trend,
                        "up" if trend == "up" else "dn" if trend == "down" else "warn")
                st.caption(" · ".join(notes) or "no clear swing sequence")
                fig, gaps = _detail_figure(df, detail_tf, symbol=used)
                st.plotly_chart(fig, use_container_width=True,
                                config={"scrollZoom": True,
                                        "displaylogo": False})
                st.caption(
                    f"{used} · {detail_tf} · real venue holes collapsed: {gaps}. "
                    "No candle is invented. The whole loaded history is in the "
                    "chart, so zooming out reveals earlier candles. Drag to pan, "
                    "scroll to zoom, double-click to reset. Drag the time axis "
                    "to zoom X independently, or the price axis to zoom Y; "
                    "Time/Price buttons provide separate axis controls.")
                st.session_state["detail_ctx"] = {"ex": ex, "sym": used}
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    dctx = st.session_state.get("detail_ctx")
    if dctx and LIVE_OK and st.toggle("Live ticks", key="detail_live"):
        live.render(dctx["ex"], dctx["sym"], "long", "detail_live_seed")

    st.divider()
    ui.section("Movers", "Largest 24h moves in the top 100.")
    rows = market.movers(8)
    if rows:
        st.dataframe(pd.DataFrame([{"coin": f"{r['symbol']} · {r['name']}",
                                    "price": f"${r['price']:,.6g}",
                                    "24h": f"{r['chg']:+.2f}%",
                                    "volume": market.money(r["vol"])}
                                   for r in rows]),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Mover data unavailable.")


if MODE == "Spot":
    with T["Accumulate"]:
        ui.section("Accumulation plans", "Long-only, staged into demand on daily.")
        if st.button("Find candidates", type="primary"):
            try:
                with st.spinner("loading..."):
                    st.session_state["radar"] = cached_radar(
                        exch, exch, proxy.strip() or None)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
        cands = [r for r in st.session_state.get("radar", []) if r["sev"] >= 1]
        if not cands:
            st.info("No deep-discount majors right now.")
        for r in cands[:6]:
            try:
                cfg = build_cfg(exch, top, min_vol, "1d", min_rr, max_stop_atr,
                                risk_pct, equity)
                ex, _ = _resolve(cfg, exch, proxy.strip() or None)
                df, used = _load(ex, r["symbol"], "1d", 400)
                if df is None:
                    continue
                plan = smc.playbook(_closed(df), "long", equity, risk_pct)
                with st.expander(f"{used} · {r['dist']:+.1f}% vs 200d SMA · "
                                 f"{plan['verdict']}"):
                    render_plan(plan, used, ladder=True)
            except Exception as e:
                st.caption(f"{r['symbol']}: {type(e).__name__}: {e}")


if MODE == "Trade":
    with T["Scan"]:
        ui.section("Live scan", f"Grade {min_grade}+ · last {recent}h · "
                                f"scanning {cap_ltf(ltf)}")
        if st.button("Run scan", type="primary"):
            try:
                with st.spinner("Scanning..."):
                    found, diag = cached_scan(
                        exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct,
                        equity, recent, min_grade, exch, proxy.strip() or None,
                        relax, need_smc, False, need_mtf, need_vol)
                st.session_state["setups"] = found
                st.session_state["diag"] = diag
                st.session_state["last_scan_at"] = pd.Timestamp.now(tz="UTC")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.exception(e)
        diag = st.session_state.get("diag")
        if diag:
            st.caption(f"venue {diag['venue']} · universe {diag['universe']} · "
                       f"loaded {diag['loaded']} · failed {diag['err']} · raw "
                       f"{diag['raw']} · kept {diag['kept']} · live "
                       f"{diag.get('actionable', 0) + diag.get('better', 0)} · "
                       f"played {diag.get('extended', 0)} · invalid "
                       f"{diag.get('invalid', 0)} · below grade "
                       f"{diag['below_grade']} · blank bias {diag['bias_blank']} "
                       f"· scan {diag.get('ltf', '5m')} · HTF {diag['htf']} · "
                       f"freshness {diag.get('recent', 12)}h "
                       f"({diag.get('fresh', 0)} bars)")
        chosen_strategies = st.multiselect("Strategies", list(STRATEGIES),
                                           default=list(STRATEGIES), key="bot_strategy_filter")
        live_rows = current_only([s for s in st.session_state.get("setups", [])
                     if s.meta.get("strategy", "BOS Retest") in chosen_strategies])
        scan_at = st.session_state.get("last_scan_at")
        if scan_at is not None:
            scan_age = (pd.Timestamp.now(tz="UTC") - scan_at).total_seconds()
            st.caption(f"Last scan: {scan_at.strftime('%Y-%m-%d %H:%M:%S UTC')} · "
                       f"{scan_age/60:.1f} minutes ago")
            if scan_age > 2 * chart_minutes(cap_ltf(ltf)) * 60:
                live_rows = []
                st.warning("Scan is too old for a current-entry decision. Run scan again.")
        best_per_coin = st.checkbox("Show only the best strategy per coin", value=True)
        if best_per_coin:
            unique = {}
            for candidate in live_rows:
                unique.setdefault(candidate.symbol, candidate)
            live_rows = list(unique.values())
        if diag and not live_rows:
            st.info("Nothing executable in this window. Raise freshness hours, "
                    "keep Relax on, or lower the minimum grade.")
        if live_rows:
            st.caption("Strategy candidates are rule-based, not guaranteed returns. 'Confidence' is a heuristic unless calibrated on sufficient out-of-sample trades.")
            st.dataframe(pd.DataFrame([{
                "Coin": s.symbol, "Strategy": s.meta.get("strategy", s.pattern),
                "Score": _norm(s.meta.get("r")).get("confidence", 0),
                "Side": s.direction.upper(), "Status": s.meta.get("status"),
                "Grade": _norm(s.meta.get("r")).get("grade", "—"),
                "Entry": s.entry, "Stop": s.stop, "TP1": s.tp1, "TP2": s.tp2,
                "Weighted R:R": round(s.weighted_rr, 2),
                "Potential gross % to TP2 (not guaranteed)": round(abs(s.tp2/s.entry-1)*100, 2),
            } for s in live_rows]), hide_index=True, use_container_width=True)
        for s in live_rows[:10]:
            r = _norm(s.meta.get("r"))
            with st.expander(f"{r['grade']} · {s.symbol} · {s.direction.upper()} · "
                             f"{s.meta.get('strategy', s.pattern)} · {s.meta.get('status')}"):
                st.write(f"**Pair** {s.meta.get('htf')}/{s.meta.get('ltf')} · "
                         f"**Entry** {s.entry:,.6g} · **Stop** {s.stop:,.6g} · "
                         f"**TP1** {s.tp1:,.6g} ({s.rr1:.2f}R) · "
                         f"**TP2** {s.tp2:,.6g} ({s.rr2:.2f}R) · "
                         f"weighted {s.weighted_rr:.2f}R")


with T["SMC"]:
    ui.section("Smart money map",
               "Structure, zones, volume, the trade and its invalidation.")
    st.caption("Up to two confirmed BOS levels per timeframe. Yellow is HTF, "
               "cyan is LTF. Each line starts at the pivot wick. HTF is "
               "30m or 1h, LTF is 3m or 5m. Internal pivots 3+3, external 5+5, "
               "confirmed only. Drag to pan, scroll to zoom.")
    s1, s2, s3 = st.columns([2, 1, 1])
    smc_sym = s1.text_input("Symbol", "BTC/USDT:USDT", key="smc_sym")
    smc_tf = s2.selectbox("Entry timeframe",
                          ["3m", "5m", "15m", "30m", "1h", "4h", "1d"],
                          index=3, key="smc_tf")
    side = s3.radio("Side", ["long", "short"], horizontal=True)
    chart_tf = chart.st_selectbox(
        "smc_chart_tf", CHART_TFS.index(DEFAULT_TF), default=DEFAULT_TF)
    if st.button("Build map", type="primary"):
        try:
            cfg = build_cfg(exch, top, min_vol, smc_tf, min_rr, max_stop_atr,
                            risk_pct, equity)
            ex, ex_name = _resolve(cfg, exch, proxy.strip() or None)
            df, used = _load(ex, smc_sym, smc_tf, _scan_bars(smc_tf))
            if df is None:
                st.warning(f"No data for {smc_sym} on {ex_name}.")
            else:
                cdf = _closed(df)
                frames = {smc_tf: cdf}
                for tf in (reversal.SCALP, reversal.FAST, reversal.BASE):
                    if tf == smc_tf:
                        continue
                    d, _u = _load(ex, used, tf, reversal.LTF_BARS,
                                  reversal.MIN_LTF_BARS)
                    if d is not None:
                        frames[tf] = _closed(d)
                htf_tf, ltf_tf, _q = reversal.choose_pair(frames, cfg)
                hdf_c = reversal._frame(frames, htf_tf)
                ldf_c = frames.get(ltf_tf)
                structure = chart.merge_frames(hdf_c, ldf_c, cfg, htf_tf, ltf_tf)
                hb = (structure.state_at(Layer.EXTERNAL, len(hdf_c) - 1)
                      if hdf_c is not None else "unclear")
                lb = (structure.state_at(Layer.INTERNAL, len(ldf_c) - 1)
                      if ldf_c is not None else "unclear")
                stt_tab = build_structure(cdf, cfg.structure)
                ok_mtf, why = aligned(hb, lb, side)
                plan = smc.playbook(cdf, side, equity, risk_pct)
                anchor = _anchor(stt_tab, side)
                av = (avwap_context(cdf, side, plan["atr"], anchor)
                      if anchor is not None else
                      {"score": 0, "flags": [], "warnings": ["no anchor"],
                       "dist": 0.0, "usable": False})
                vp = volume_profile(cdf)
                in_va = bool(vp and vp["val"] <= plan["entry"] <= vp["vah"])
                av_txt = (f"{av['dist']:+.2f} ATR" if av.get("usable", True)
                          else "n/a")
                k = st.columns(5)
                ui.stat(k[0], "HTF structure", hb,
                        "up" if hb == "up" else "dn" if hb == "down" else "warn")
                ui.stat(k[1], "LTF structure", lb,
                        "up" if lb == "up" else "dn" if lb == "down" else "warn")
                ui.stat(k[2], "MTF", "aligned" if ok_mtf else "against",
                        "up" if ok_mtf else "dn")
                ui.stat(k[3], "AVWAP", av_txt,
                        "up" if av["score"] >= 40 else "dn")
                ui.stat(k[4], "In value area", "yes" if in_va else "no",
                        "up" if in_va else "warn")
                st.caption(f"HTF {htf_tf} · LTF {ltf_tf} · {why} · closed "
                           f"candles only · both auto-picked and capped")

                disp = df
                display_tf = data_tf(chart_tf)
                if display_tf != smc_tf or len(df) < CHART_BARS:
                    d2, _v = _load(ex, used, display_tf, CHART_BARS)
                    if d2 is not None and (display_tf != smc_tf or len(d2) > len(df)):
                        disp = d2
                    elif d2 is None and display_tf != smc_tf:
                        st.caption(f"{chart_tf} candles unavailable. Showing {smc_tf}.")
                        chart_tf = smc_tf
                st.plotly_chart(chart.build(disp, cfg, side, plan,
                                            chart_tf=chart_tf, analysis_df=cdf,
                                            htf_df=hdf_c, ltf_df=ldf_c,
                                            pair=(htf_tf, ltf_tf), symbol=used),
                                use_container_width=True,
                                config={"scrollZoom": True, "displaylogo": False})
                st.caption("Volume profile: wider bars are HTF, narrower bars are LTF. "
                           "Green and red estimate up- and down-candle volume. Eight "
                           "windows of history sit behind the visible bars, so "
                           "zooming out keeps revealing earlier candles.")
                st.session_state["live_ctx"] = {"ex": ex, "sym": used, "side": side}
                with st.expander("How to read this chart"):
                    st.markdown(chart.legend_md())
                st.divider()
                render_plan(plan, used, ladder=(side == "long"))
                st.divider()
                order_card(plan, used, side, ltf_tf)
                if st.button("Log to paper book", key="paper_log"):
                    paper.open_trade(used, side, plan["entry"], plan["sl"],
                                     plan["tp1"], plan["tp2"], plan["qty"], equity,
                                     " | ".join(plan["confluence"]["flags"][:3]))
                    st.success("Paper trade opened.")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    lctx = st.session_state.get("live_ctx")
    if lctx and LIVE_OK and st.toggle("Live ticks", key="smc_live"):
        live.render(lctx["ex"], lctx["sym"], lctx["side"], "smc_live_seed")

    st.divider()
    with st.expander("Full trading logic — how the pair is chosen and used"):
        st.markdown(LOGIC_MD)
    st.markdown(smc.READING_ORDER)


with T["Sizer"]:
    ui.section("Position size", "Leverage changes margin, never risk.")
    sizer.render(equity, risk_pct)


with T["Radar"]:
    ui.section("Value radar", "Distance to the 200-day SMA across majors.")
    if st.button("Refresh radar"):
        try:
            with st.spinner("loading daily candles..."):
                st.session_state["radar"] = cached_radar(
                    exch, exch, proxy.strip() or None)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    radar = st.session_state.get("radar", [])
    if radar:
        st.dataframe(pd.DataFrame([{"coin": r["symbol"],
                                    "zone": "deep" if r["sev"] == 2 else
                                            "near" if r["sev"] == 1 else "extended",
                                    "vs 200d": f"{r['dist']:+.1f}%",
                                    "detail": " · ".join(r["reasons"])}
                                   for r in radar]),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Press Refresh radar.")

    st.divider()
    ui.section("Reversal radar",
               "HTF 30m/1h, LTF 3m/5m, both auto-picked. Approaching first. "
               "Only setups still executable are listed.")
    st.caption("Zones from the HTF engine. Confirmation from the LTF CHoCH/BOS "
               "plus the ATR candle grader. Played, extended and already-held "
               "coins are kept out of the live table. Closed candles only.")
    with st.expander("How the pair and the statuses are produced"):
        st.markdown(LOGIC_MD)
    if st.button("Scan reversals", key="rev_scan", type="primary"):
        try:
            with st.spinner("scoring HTF, then triggers..."):
                rows, venue = cached_reversal(exch, exch, proxy.strip() or None,
                                              MAJORS[:12])
            st.session_state["rev_rows"] = rows
            st.session_state["rev_venue"] = venue
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    rev_rows = st.session_state.get("rev_rows", [])
    if rev_rows:
        live, dead = reversal.rank(rev_rows)
        future = reversal.future_only(rev_rows)
        st.caption(f"venue {st.session_state.get('rev_venue', '—')} · "
                   f"{len(future)} executable · {len(live) - len(future)} "
                   f"played · {len(dead)} stale")
        if live:
            st.dataframe(pd.DataFrame([{
                "status": r["status"], "pair": f"{r['htf']}/{r['ltf']}",
                "symbol": r["symbol"], "side": r["direction"],
                "zone": f"{r['zone'][0]:,.6g}–{r['zone'][1]:,.6g}",
                "dist ATR": r["dist_atr"], "entry": f"{r['entry']:,.6g}",
                "stop": f"{r['stop']:,.6g}", "tp1": f"{r['tp1']:,.6g}",
                "rr": round(r["rr"], 2),
                "eta": f"~{r['eta_min']:.0f}m" if r["eta_min"] else "—",
                "exec": "yes" if r.get("future") else "past",
                "why": " · ".join(r["reasons"]) or "—"} for r in live]),
                hide_index=True, use_container_width=True)

        entries = reversal.active_entries(rev_rows)
        st.markdown("##### Active entries")
        if not entries:
            st.info("No new confirmed reversal is still in its entry range.")
        else:
            for r in entries:
                st.write(f"**{r['symbol']} {r['direction'].upper()}** "
                         f"({r['htf']}/{r['ltf']}) · entry {r['entry']:,.6g} · "
                         f"stop {r['stop']:,.6g} · tp1 {r['tp1']:,.6g} · "
                         f"tp2 {r['tp2']:,.6g} · R:R {r['rr']:.2f}")

        with st.expander(f"Missed / invalidated ({len(dead)})"):
            st.dataframe(pd.DataFrame([{
                "status": r["status"], "pair": f"{r['htf']}/{r['ltf']}",
                "symbol": r["symbol"], "side": r["direction"],
                "zone": f"{r['zone'][0]:,.6g}–{r['zone'][1]:,.6g}",
                "rr": round(r["rr"], 2)} for r in dead]),
                hide_index=True, use_container_width=True)
    else:
        st.info("Press Scan reversals.")


with T["Portfolio"]:
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
            if _has(pf_mod, ["export_json"]):
                st.download_button("Export backup", pf_mod.export_json(),
                                   file_name="portfolio.json")
    except Exception as e:
        st.warning(f"Portfolio unavailable: {type(e).__name__}: {e}")


if MODE == "Trade":
    with T["Paper"]:
        ui.section("Paper book", "Forward test. Promote to live only on the gate.")
        cleared, why = paper.gate()
        (st.success if cleared else st.info)(f"Promotion gate: {why}")
        s = paper.stats()
        if s.get("n"):
            m = st.columns(5)
            ui.stat(m[0], "Closed", str(s["n"]))
            ui.stat(m[1], "Win rate", f"{s['win_rate']}%")
            ui.stat(m[2], "Profit factor", str(s.get("pf") or "—"))
            ui.stat(m[3], "Expectancy", f"{s['expectancy']:+.2f}R")
            ui.stat(m[4], "Avg R", f"{s['avg_r']:+.2f}")
        opens = paper.open_rows()
        if not opens:
            st.info("No open paper trades.")
        else:
            st.dataframe(pd.DataFrame([{"id": r["id"], "symbol": r["symbol"],
                                        "side": r["dir"],
                                        "entry": f"{r['entry']:,.6g}",
                                        "stop": f"{r['stop']:,.6g}",
                                        "qty": f"{r['qty']:.6g}"}
                                       for r in opens]),
                         hide_index=True, use_container_width=True)
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


if MODE == "Trade":
    with T["Backtest"]:
        ui.section("Backtest", "Independent strategy backtesting · historical simulation, next-open fills.")
        st.caption("Each strategy is tested separately. These are historical hypotheses, not proof of future profitability.")
        b1, b2 = st.columns(2)
        bt_sym = b1.text_input("Symbol", "BTC/USDT:USDT", key="bt_sym")
        bt_bars = b2.number_input("Bars", 500, 20000, 6000, step=500)
        bt_strategy = st.selectbox("Strategy", list(STRATEGIES) + ["Compare all seven"],
                                   key="bt_strategy")
        st.caption(f"runs on {cap_ltf(ltf)} — the same timeframe the scan uses.")
        if st.button("Run backtest", type="primary"):
            try:
                with st.spinner("Running..."):
                    if bt_strategy == "Compare all seven":
                        comparison = []
                        for strategy_name in STRATEGIES:
                            cr, tr, nr, nk, nb = cached_backtest(
                                exch, bt_sym, ltf, int(bt_bars), min_rr, max_stop_atr,
                                risk_pct, equity, exch, proxy.strip() or None, strategy_name)
                            if cr is not None:
                                stats = report(tr, cr, build_cfg(exch, 40, 30e6, ltf,
                                                    min_rr, max_stop_atr, risk_pct, equity))
                                comparison.append({"Strategy": strategy_name, "Raw": nr,
                                    "Filtered": nk, "Trades": len(tr),
                                    "Expectancy (R)": stats.get("expectancy_R"),
                                    "Max drawdown (%)": stats.get("max_dd_pct")})
                        st.dataframe(pd.DataFrame(comparison), hide_index=True,
                                     use_container_width=True)
                        st.caption("In-sample comparison only; do not rank future profits from this table.")
                        st.stop()
                    curve, trades, raw_n, kept_n, bars_n = cached_backtest(
                        exch, bt_sym, ltf, int(bt_bars), min_rr, max_stop_atr,
                        risk_pct, equity, exch, proxy.strip() or None, bt_strategy)
                if curve is None:
                    st.warning(f"Not enough history for {bt_sym}: the venue "
                               f"returned fewer than 200 bars of {cap_ltf(ltf)}. "
                               f"Pick a venue with deeper short-timeframe "
                               f"history, or lower the bar count.")
                else:
                    st.caption(f"bars {bars_n} · raw {raw_n} · after filters "
                               f"{kept_n} · trades {len(trades)}")
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


if MODE == "Trade":
    with T["Macro"]:
        ui.section("Macro", "Sentiment, dominance and dry powder.")
        label, ltone, notes = market.stance(fng, ov["btc_dom"], stab)
        s1, s2 = st.columns(2)
        ui.stat(s1, "Stance", label, ltone)
        ui.stat(s2, "Stablecoin share", f"{stab:.1f}%" if stab is not None else "—")
        for n in notes:
            st.write(f"- {n}")


if MODE == "Trade":
    with T["Alerts"]:
        ui.section("Alerts", "Telegram and email.")
        tg1, tg2 = st.columns([2, 1])
        tg_token = tg1.text_input("Telegram bot token", value=_cred("TELEGRAM_TOKEN"),
                                  type="password")
        tg_chat = tg2.text_input("Chat id", value=_cred("TELEGRAM_CHAT_ID"))
        if st.button("Send test", type="primary"):
            _save_creds({"TELEGRAM_TOKEN": tg_token, "TELEGRAM_CHAT_ID": tg_chat})
            with st.spinner("sending..."):
                ok, msg = notify("CryptoScan test — alerts working.")
            st.success(msg) if ok else st.error(msg)

        st.divider()
        ui.section("Reversal alerts",
                   "Early warning 5-10 minutes out, then a separate confirmed "
                   "entry. Reuses the same Telegram credentials.")
        st.caption("ETA comes from distance divided by the HTF ATR per bar. It "
                   "is an estimate. Duplicate and stale alerts are suppressed, "
                   "and only executable rows are sent. Nothing is auto-traded. "
                   "Polling runs while this tab is open or on button press — "
                   "there is no background worker.")
        a1, a2 = st.columns(2)
        auto = a1.checkbox("Poll automatically", value=False,
                           help="Re-scans every 90s while this tab is open.")
        a2.caption("Manual poll uses the same dedupe as the automatic one.")
        state = st.session_state.setdefault("rev_alerts", reversal_alerts.State())

        def _poll() -> None:
            try:
                rows, _v = cached_reversal(exch, exch, proxy.strip() or None,
                                           MAJORS[:12])
                cfg = build_cfg(exch, top, min_vol, reversal.SCALP, min_rr,
                                max_stop_atr, risk_pct, equity)
                batch = state.due(reversal.future_only(rows), cfg.risk.min_rr)
                state.prune()
                for r, kind, text in batch:
                    notify(text)
                st.session_state["rev_sent"] = (
                    f"{len(batch)} alert(s) at {pd.Timestamp.utcnow():%H:%M:%S} UTC")
            except Exception as e:
                st.session_state["rev_sent"] = f"{type(e).__name__}: {e}"

        if auto and HAS_FRAGMENT:
            st.fragment(run_every="90s")(_poll)()
        elif st.button("Poll now", key="rev_poll"):
            with st.spinner("checking zones..."):
                _poll()
        if st.session_state.get("rev_sent"):
            st.caption(st.session_state["rev_sent"])


if MODE == "Trade":
    with T["Journal"]:
        ui.section("Journal", "Log every trade, including the ones you skipped.")
        with st.form("journal", clear_on_submit=True):
            j1, j2, j3, j4 = st.columns(4)
            j_sym = j1.text_input("Symbol")
            j_dir = j2.selectbox("Side", ["long", "short"])
            j_r = j3.number_input("Result (R)", -10.0, 10.0, 0.0, step=0.1)
            j_ok = j4.checkbox("Followed rules", value=True)
            j_note = j4.text_input("Note")
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


with T["SOP"]:
    sop.render()


with T["Analysis"]:
    if not ANALYSIS_OK:
        st.error("Analysis module failed to load. Spot and Trade are unaffected.")
        st.code(ANALYSIS_ERR, language="text")
        st.caption("Run `python check_install.py` locally to find the broken module.")
    else:
        analysis_page.render(exch, top, min_vol, min_rr, max_stop_atr, risk_pct,
                             equity, proxy.strip() or None, build_cfg, _resolve,
                             _load, CFG, notify)


with T["Learn"]:
    ui.section("Learn", "Work the stages in order.")
    for stage, title, status, items in learn.STAGES:
        tone = "up" if status == "started" else "warn" if status == "next" else "mut"
        with st.expander(f"{stage} · {title}"):
            st.markdown(f'<span class="pill {tone}">{status}</span>',
                        unsafe_allow_html=True)
            for it in items:
                st.write(f"- {it}")
    st.divider()
    st.dataframe(pd.DataFrame([{"#": a, "concept": b, "why": c}
                               for a, b, c in learn.PRIORITY]),
                 hide_index=True, use_container_width=True)
    st.markdown(learn.FRAMEWORK)
    st.markdown(learn.DAILY)


with T["Guide"]:
    ui.section("Guide")
    st.markdown(LOGIC_MD)
    st.markdown("""
### Venue
`auto` walks a list and stops at the first venue that answers with enough
perps. It pins that venue for the session. To stop the switching, pick one
directly — `okx` or `bybit` work from Cloud, `binanceus` has almost no perps.
`Reset venue` clears the pin.

### Chart timeframes
The chart selector is display-only and defaults to 30m. It re-reads candles,
never the strategy. The structure pair is capped on both sides: HTF 30m or 1h,
LTF 3m or 5m.

### Pivots
Internal 3 left + 3 right, external 5 left + 5 right, from `config.StructCfg`.
A pivot is only knowable right bars later; that lag is the anti-repainting
guarantee. Two structure lines per timeframe, anchored at the wick, run to the
right edge with the exact price and the BOS/CHoCH tag.

### Freshness
The freshness slider is hours, not bars. It is converted to bars of the
timeframe actually scanned, so the same setting means the same wall-clock age
whether the trigger side resolves to 3m or 5m.

### Live ticks
Available on Coin detail, SMC and Analysis. A ticker is polled once a second
and each tick is folded into a one-second OHLC bucket, so you get real candles,
not a line. Display only — the strategy still reads closed 3m/5m bars.

### Zoom and history
Pan is the default drag. Scroll or drag a box to zoom; X and Y are independent.
Roughly eight windows of history sit behind the visible bars, so zooming out
keeps revealing earlier candles. Double-click resets.

### Coin detail gaps
Sparse history on 1H, 4H and 1D is a venue fact. The detail chart collapses
only real holes and prints how many it hid.

### Setups vs Analysis
Setups is the scanner board, filtered to what can still be executed. Analysis
is the point-in-time workflow.

### R:R
TP1, TP2 and the weighted plan are quoted separately. A 50/50 exit between TP1
and TP2 does not earn the TP2 R — the weighted figure is the one to trust.

### Sizing
qty = (equity × risk percent) ÷ (entry − stop)

### Placement
Post-only limit entry, stop-market reduce-only on mark price, TP1 and TP2 limits
reduce-only at 50% each. Round down to step size. Isolated margin, leverage low
enough that liquidation sits past the stop.

### Volume profile
The right-hand gutter is an OHLCV estimate. Blue is HTF context, orange is LTF
trigger, both proportional and clear of the candles.
""")

st.divider()
st.caption("CryptoScan · not financial advice")
