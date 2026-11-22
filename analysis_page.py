"""analysis_page.py - the analysis workflow. Spot and Bot tabs untouched."""
import pandas as pd
import streamlit as st

import chart2
import core.lifecycle as lc
import core.structure2 as s2
import smc
import ui
from core import alerts2, nextaction, pit, readiness
from core.candles import latest as latest_candle
from core.indicators import compute_all
from core.models import Decision, Layer, ReadinessInput, Stage, new_id
from core.volume import avwap_context, volume_profile
from strategy.risk import plan_rr

try:
    import live
    LIVE_OK = True
except Exception:
    LIVE_OK = False

MODE_LABEL = {"structure": "Structure", "analysis": "Analysis", "execution": "Execution"}
TF_BIAS = {"4h": "1d", "1h": "4h", "15m": "1h"}
STAGE_OF = {Decision.WAIT: Stage.NEW_SETUP, Decision.PREPARE: Stage.APPROACHING,
            Decision.READY: Stage.READY, Decision.TRIGGERED: Stage.TRIGGERED}


def _g(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _bias(htf, cfg) -> str:
    if htf is None or len(htf) < 60:
        return "unclear"
    f = s2.run(htf, "", "4h", cfg)
    return f.state_at(Layer.EXTERNAL, len(htf) - 1)


def _panel(n: dict):
    st.markdown(f"**DECISION** `{n['decision']}` — {n['action']}")
    rows = {
        "Price": f"{n['price']:,.6g}",
        "Level": f"{n['level']:,.6g}" if n["level"] else "—",
        "Next zone": f"{n['zone'][0]:,.6g} - {n['zone'][1]:,.6g}" if n["zone"] else "—",
        "Missing": ", ".join(n["missing"]) or "none",
        "Timeframe": n["timeframe"],
        "Invalidation": f"{n['invalidate']:,.6g}" if n["invalidate"] else "—",
    }
    st.dataframe(pd.DataFrame([rows]).T.rename(columns={0: "value"}),
                 use_container_width=True)
    if n["actionable"]:
        st.success("Actionable — every enabled mandatory check passed.")
    else:
        st.info("Not actionable.")


def render(exch, top, min_vol, min_rr, max_stop_atr, risk_pct, equity, proxy,
           build_cfg, resolve, load, cfg_base, notify_fn):
    ui.section("Analysis", "Point-in-time structure, readiness and next action.")
    c1, c2, c3 = st.columns([2, 1, 1])
    sym = c1.text_input("Symbol", "BTC/USDT:USDT", key="an_sym")
    setup_tf = c2.selectbox("Setup timeframe", ["1h", "4h", "15m"], index=0,
                            key="an_tf")
    side = c3.radio("Direction", ["short", "long"], horizontal=True, key="an_side")

    o1, o2, o3 = st.columns(3)
    vol_opt = o1.checkbox("Require volume", key="an_vol")
    av_opt = o2.checkbox("Require AVWAP", key="an_av")
    va_opt = o3.checkbox("Require value area", key="an_va")

    mode = st.radio("Chart mode", list(MODE_LABEL), horizontal=True,
                    format_func=lambda m: MODE_LABEL[m], key="an_mode")

    if st.button("Analyse", type="primary"):
        st.session_state["an_run"] = True

    if not st.session_state.get("an_run"):
        st.caption("Press Analyse.")
        return

    def _poster(text):
        try:
            ok, _msg = notify_fn(text)
        except Exception:
            return False, None
        return bool(ok), None

    try:
        cfg = build_cfg(exch, top, min_vol, setup_tf, min_rr, max_stop_atr,
                        risk_pct, equity)
        ex, venue = resolve(cfg, exch, proxy)
        df, used = load(ex, sym, setup_tf, 600)
        if df is None:
            st.warning(f"No data for {sym} on {venue}.")
            return

        bias_tf = TF_BIAS.get(setup_tf, "4h")
        htf, _ = load(ex, used, bias_tf, cfg.scan.htf_bars, 60)
        bias = (pit.bias_at(htf, df.index[-1], lambda f: _bias(f, cfg))
                if htf is not None else "unclear")

        frame = s2.run(df, used, setup_tf, cfg)
        t = len(df) - 1
        ext = frame.state_at(Layer.EXTERNAL, t)
        inn = frame.state_at(Layer.INTERNAL, t)
        ranging = bool(frame.ranging[t]) if frame.ranging else False
        price = float(df["close"].iloc[-1])
        ind = compute_all(df, cfg)
        atr = float(ind["atr"].iloc[-1] or 0) or price * 0.01

        want = "up" if side == "long" else "down"
        bias_ok = bias == want
        bias_reason = f"HTF {bias} vs {side}"
        struct = frame.last(Layer.INTERNAL, "bull" if side == "long" else "bear", t)
        struct_ok = struct is not None
        struct_level = float(_g(struct, "level")) if struct_ok else None

        _, cname, cq = latest_candle(df, atr, side)
        candle_ok = cq >= 60

        zones = [z for z in smc.supply_demand(df)
                 if z["dir"] == ("demand" if side == "long" else "supply")]
        zones.sort(key=lambda z: -z["idx"])
        zone = (zones[0]["bot"], zones[0]["top"]) if zones else None
        if zone:
            dist = min(abs(price - zone[0]), abs(price - zone[1])) / atr
            in_zone = zone[0] <= price <= zone[1]
        else:
            dist, in_zone = 99.0, False

        anchor = None
        pool = [s for s in (getattr(frame, "swings", None) or [])
                if _g(s, "layer") == Layer.EXTERNAL
                and _g(s, "kind") == ("L" if side == "long" else "H")
                and int(_g(s, "confirmed_at", 0)) <= t]
        if pool:
            anchor = int(_g(pool[-1], "idx"))
        av = (avwap_context(df, side, atr, anchor) if anchor is not None
              else {"score": 0, "dist": 0.0})
        vp = volume_profile(df)
        in_va = bool(vp and vp["val"] <= price <= vp["vah"])

        if zone:
            entry = zone[1] if side == "long" else zone[0]
            inval = (zone[0] - 0.3 * atr) if side == "long" else (zone[1] + 0.3 * atr)
        else:
            entry = price
            inval = price - 1.5 * atr if side == "long" else price + 1.5 * atr
        sign = 1.0 if side == "long" else -1.0
        tp1 = entry + sign * 3.0 * atr
        tp2 = entry + sign * 5.0 * atr
        rr_map = plan_rr(entry, inval, tp1, tp2)
        rr = rr_map["rr1"]
        risk_ok = rr >= cfg.risk.min_rr

        optional = {}
        if vol_opt:
            optional["volume confirmation missing"] = av["score"] >= 40
        if av_opt:
            optional["AVWAP not on the correct side"] = av["score"] >= 40
        if va_opt:
            optional["entry outside the value area"] = in_va

        ri = ReadinessInput(True, bias_ok, bias_reason, dist, 1.0, in_zone,
                            struct_ok, candle_ok, risk_ok, False, False, False,
                            optional)
        decision, missing = readiness.evaluate(ri)

        nav = nextaction.build(decision, used, side, price,
                               struct_level, zone, missing, setup_tf, inval)

        k = st.columns(5)
        ui.stat(k[0], "HTF bias", bias,
                "up" if bias == "up" else "dn" if bias == "down" else "warn")
        ui.stat(k[1], "External", ext + (" · ranging" if ranging else ""),
                "up" if ext == "up" else "dn" if ext == "down" else "warn")
        ui.stat(k[2], "Internal", inn,
                "up" if inn == "up" else "dn" if inn == "down" else "warn")
        ui.stat(k[3], "Candle", f"{cname or '—'} {cq}", "up" if candle_ok else "warn")
        ui.stat(k[4], "R:R", f"{rr:.2f}", "up" if risk_ok else "dn")

        _panel(nav)

        try:
            frames = {}
            if htf is not None and len(htf):
                frames[Layer.EXTERNAL] = htf
            frames[Layer.INTERNAL] = df
            st.plotly_chart(
                chart2.build(df, frame, side,
                             {"entry": entry, "stop": inval, "tp1": tp1, "tp2": tp2},
                             mode=mode, cfg=cfg, frames=frames),
                use_container_width=True, config={"scrollZoom": True})
        except Exception as e:
            st.warning(f"chart unavailable: {type(e).__name__}: {e}")
            st.caption("The metrics above do not depend on the chart. It is "
                       "display only.")

        if LIVE_OK and st.toggle("Live ticks", key="an_live"):
            live.render(ex, used, side, "an_live_seed")

        st.divider()
        sid = st.session_state.setdefault("an_sid", new_id())
        st.code(f"setup {sid}\nentry {entry:,.6g}\nstop {inval:,.6g}\n"
                f"tp1 {tp1:,.6g} ({rr_map['rr1']:.2f}R)\n"
                f"tp2 {tp2:,.6g} ({rr_map['rr2']:.2f}R)\n"
                f"weighted {rr_map['weighted']:.2f}R\n"
                f"invalidate {'close below' if side == 'long' else 'close above'} "
                f"{inval:,.6g}", language="text")

        a1, a2, a3 = st.columns(3)
        if a1.button("Send stage alert"):
            stg = STAGE_OF.get(decision, Stage.NEW_SETUP)
            txt = alerts2.message(stg, used, side, setup_tf, sid, "—", zone, price,
                                  bias, missing, inval, nav["action"])
            st.caption(alerts2.send(sid, stg, txt, 0, _poster))
        if a2.button("Log lifecycle"):
            lc.transition(sid, "setup", "FORMING", str(df.index[-1]), price, "created")
            if decision in (Decision.READY, Decision.TRIGGERED):
                lc.transition(sid, "setup", "READY", str(df.index[-1]), price)
            st.caption("logged")
        if a3.button("Kill alerts"):
            alerts2.kill(sid)
            st.caption("killed")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.exception(e)
