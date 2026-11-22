"""board.py - the setups board. Ranked cards, charted, with the reasoning shown."""
import streamlit as st
import pandas as pd

import chart
import paper
import ui
from config import CFG
from core.tfs import CHART_TFS, cap_htf, cap_ltf, chart_minutes, data_tf
from strategy import reversal
from strategy.risk import plan_rr

FUTURE = {"ACTIONABLE", "BETTER_PRICE"}
DEFAULT_CTF = "30m"


def _auto_pair(load_fn, ex, sym):
    """(htf, ltf, frames). HTF 30m/1h, LTF 3m/5m, chosen on structure."""
    cached = st.session_state.get(f"pair{sym}")
    if cached:
        return cached[0], cached[1], cached[2]
    frames = {}
    for tf in ("3m", "5m", "15m"):
        try:
            d, _u = load_fn(ex, sym, tf, 1200, 200)
        except Exception:
            d = None
        if d is not None and len(d) > 2:
            frames[tf] = d.iloc[:-1]
    htf, ltf, _q = reversal.choose_pair(frames, CFG)
    st.session_state[f"pair{sym}"] = (htf, ltf, frames)
    return htf, ltf, frames


def _card(s, r, sm, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity,
          exch, proxy, cfg_fn, resolve_fn, load_fn, order_card, notify):
    conf = r["confidence"]
    pair = (cap_htf(s.meta.get("htf") or "30m"), cap_ltf(s.meta.get("ltf") or "5m"))
    with st.expander(f"{r['grade']} · {s.symbol} · {s.direction.upper()} · "
                     f"{s.meta.get('status')} · conf {conf:.0f}/100"):
        st.caption(f"structure pair **{pair[0]} HTF / {pair[1]} LTF** · HTF from "
                   f"30m/1h, LTF from 3m/5m, both by setup quality")
        try:
            cfg = cfg_fn(exch, top, min_vol, ltf, min_rr, max_stop_atr,
                         risk_pct, equity)
            ex, _ = resolve_fn(cfg, exch, proxy)
            htf, trig, frames = _auto_pair(load_fn, ex, s.symbol)
            pair = (htf, trig)
            st.caption(f"live pair **{htf} HTF / {trig} LTF**")
            df, _u = load_fn(ex, s.symbol, ltf, 1500)
            if df is None:
                st.caption("chart: not enough candles")
            else:
                ctf = st.selectbox("Chart timeframe", CHART_TFS,
                                   index=CHART_TFS.index(DEFAULT_CTF),
                                   key=f"ctf{s.symbol}",
                                   help="Display only. Structure uses the live "
                                        "pair above.")
                disp = df
                if chart_minutes(ctf) < chart_minutes(ltf):
                    d2, _v = load_fn(ex, s.symbol, data_tf(ctf), 800)
                    if d2 is not None:
                        disp = d2
                cdf = df.iloc[:-1] if len(df) > 2 else df
                htf_df = reversal._frame(frames, htf)
                ltf_df = frames.get(trig)
                plan = {"entry": s.entry, "sl": s.stop, "tp1": s.tp1,
                        "tp2": s.tp2}
                st.plotly_chart(chart.build(disp, cfg, s.direction, plan,
                                            chart_tf=ctf, analysis_df=cdf,
                                            htf_df=htf_df, ltf_df=ltf_df,
                                            pair=pair),
                                use_container_width=True,
                                config={"scrollZoom": True, "displaylogo": False})
                with st.expander("How to read this chart"):
                    st.markdown(chart.legend_md())
        except Exception as e:
            st.caption(f"chart error: {type(e).__name__}: {e}")

        rr = plan_rr(s.entry, s.stop, s.tp1, s.tp2, 0.5)

        st.write(f"**Logic** {' · '.join(s.reasons) or '—'}")
        if sm.get("flags"):
            st.write("**Supporting** " + " · ".join(sm["flags"]))
        if sm.get("warnings"):
            st.write("**Against** " + " · ".join(sm["warnings"]))
        st.write(f"**Entry** {s.entry:,.6g} · **Stop** {s.stop:,.6g} · "
                 f"**TP1** {s.tp1:,.6g} ({rr['rr1']:.2f}R) · "
                 f"**TP2** {s.tp2:,.6g} ({rr['rr2']:.2f}R) · "
                 f"weighted {rr['weighted']:.2f}R")
        for p in r["positives"]:
            st.write(f"- {p}")
        for n in r["negatives"]:
            st.write(f"- {n}")
        av = s.meta.get("avwap", {})
        av_txt = (f"{av.get('dist', 0):+.2f} ATR" if av.get("usable", True)
                  else "n/a")
        st.write(f"**AVWAP** {av_txt} · **in value area** "
                 f"{'yes' if s.meta.get('in_value') else 'no'} · **candle** "
                 f"{s.meta.get('candle') or '—'} ({s.meta.get('candle_q', 0)})")

        order_card({"qty": s.meta.get("qty", 0), "entry": s.entry, "sl": s.stop,
                    "tp1": s.tp1, "tp2": s.tp2,
                    "size_usd": s.meta.get("risk_cash", 0),
                    "rr1": rr["rr1"], "rr2": rr["rr2"],
                    "weighted_rr": rr["weighted"]},
                   s.symbol, s.direction, pair[1])

        c1, c2 = st.columns(2)
        if c1.button("Alert me", key=f"ba{s.symbol}"):
            ok, msg = notify(f"{r['grade']} {s.symbol} {s.direction} "
                             f"entry {s.entry:,.6g} stop {s.stop:,.6g}")
            st.success(msg) if ok else st.error(msg)
        if c2.button("Log to paper", key=f"bp{s.symbol}"):
            paper.open_trade(s.symbol, s.direction, s.entry, s.stop, s.tp1, s.tp2,
                             s.meta.get("qty", 0), equity,
                             " · ".join(sm.get("flags", [])[:3]))
            st.success("Paper trade opened.")


def render(scan_fn, cfg_fn, resolve_fn, load_fn, norm_fn, order_card, notify,
           exch, top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity,
           recent, proxy, relax, need_smc, need_mtf, need_vol):
    ui.section("Setups board", "Ranked, charted, with the reasoning shown.")
    st.caption("Press Build board. Only setups still executable from here are "
               "listed — played and invalidated ones are dropped. Charts are "
               "capped at 6 to stay inside rate limits.")

    g1, g2 = st.columns(2)
    b_top = g1.slider("Universe size", 10, 60, 25, step=5, key="b_top")
    b_grade = g2.select_slider("Minimum grade", ["D", "C", "B", "A", "A+"],
                               value="C", key="b_grade")

    if st.button("Build board", type="primary"):
        try:
            with st.spinner("Scanning and grading..."):
                found, diag = scan_fn(exch, b_top, min_vol, ltf, min_rr,
                                      max_stop_atr, risk_pct, equity, recent,
                                      b_grade, exch, proxy, relax, need_smc,
                                      False, need_mtf, need_vol)
            st.session_state["board"] = found
            st.session_state["board_diag"] = diag
            for k in [k for k in st.session_state if k.startswith("pair")]:
                st.session_state.pop(k, None)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.exception(e)

    bd = st.session_state.get("board_diag")
    if bd:
        st.caption(f"venue {bd['venue']} · universe {bd['universe']} · loaded "
                   f"{bd['loaded']} · failed {bd['err']} · raw {bd['raw']} · "
                   f"kept {bd['kept']} · stale {bd['stale']} · live "
                   f"{bd.get('actionable', 0) + bd.get('better', 0)} "
                   f"(actionable {bd.get('actionable', 0)} / better "
                   f"{bd.get('better', 0)}) · played {bd.get('extended', 0)} · "
                   f"invalid {bd.get('invalid', 0)} · below grade "
                   f"{bd['below_grade']} · mtf {bd['mtf_skip']} · volume "
                   f"{bd['vol_skip']} · no anchor {bd['no_anchor']} · scan "
                   f"{bd.get('ltf', '5m')} · HTF {bd.get('htf', '1h')} · "
                   f"freshness {bd.get('recent', 12)}h")
        if bd["universe"] < 8:
            st.warning(f"Only {bd['universe']} symbols on this venue. If it is "
                       f"binanceus, pick okx or bybit — it has almost no perps.")
        if bd["loaded"] and not bd["raw"]:
            st.warning("Candles loaded but no break-of-structure in the window. "
                       "Raise freshness to 48h, keep Relax on, or widen the "
                       "universe.")
        if bd["loaded"] and bd["err"] >= bd["loaded"]:
            st.warning("Most symbols failed — rate limiting. Lower universe size "
                       "or choose a venue manually.")
        if bd["kept"] and not (bd.get("actionable", 0) + bd.get("better", 0)):
            st.warning("Setups were found but every one has already run away "
                       "from its entry. Raise freshness hours, or wait for a "
                       "pullback into the zone.")
        if bd["stale"] > bd["kept"]:
            st.caption(f"{bd['stale']} setups fell outside the {bd['recent']}h "
                       f"freshness window. Raise it to see more.")

    board = [s for s in st.session_state.get("board", [])
             if s.meta.get("status") in FUTURE]
    if not board:
        st.info("No board yet, or every setup has already played out.")
        return

    st.dataframe(pd.DataFrame([{
        "grade": s.meta["r"]["grade"],
        "conf": round(s.meta["r"]["confidence"]),
        "symbol": s.symbol, "side": s.direction,
        "candle": s.meta.get("candle") or "—",
        "smc": s.meta.get("smc", {}).get("score", 0),
        "status": s.meta.get("status"),
        "entry": f"{s.entry:,.6g}", "stop": f"{s.stop:,.6g}",
        "tp1": f"{s.tp1:,.6g} ({s.rr1:.2f}R)",
        "tp2": f"{s.tp2:,.6g} ({s.rr2:.2f}R)",
        "weighted": round(s.weighted_rr, 2),
    } for s in board]), hide_index=True, use_container_width=True)

    st.divider()
    for s in board[:6]:
        _card(s, norm_fn(s.meta.get("r")), s.meta.get("smc", {}) or {},
              b_top, min_vol, ltf, min_rr, max_stop_atr, risk_pct, equity,
              exch, proxy, cfg_fn, resolve_fn, load_fn, order_card, notify)
