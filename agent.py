"""agent.py - optional LLM reviewer. OpenAI-compatible endpoints."""
import hashlib
import json
import pathlib

import requests

CONF = pathlib.Path("agent_config.json")
CACHE = pathlib.Path("agent_cache.json")

SYSTEM = (
    "You are a crypto risk reviewer. You receive a fully computed trade "
    "context. You do not compute prices. Return ONLY compact JSON with keys: "
    "verdict (TAKE|CONSIDER|SKIP), confidence (0-100 int), "
    "reasons (max 4 short strings), risks (max 4 short strings), "
    "invalidation (one string). Be blunt. If evidence is thin, say SKIP."
)


def config():
    if CONF.exists():
        try:
            return json.loads(CONF.read_text())
        except Exception:
            pass
    return {}


def save_config(c):
    CONF.write_text(json.dumps(c, indent=2))


def available():
    c = config()
    return bool(c.get("base_url") and c.get("api_key") and c.get("model"))


def _cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def build_context(setup, meta, ctx):
    rt = meta.get("rating")
    sim = meta.get("sim") or {}
    br = meta.get("base_rate") or {}
    return {
        "symbol": setup.symbol, "direction": setup.direction,
        "status": meta.get("status"),
        "grade": rt.grade if rt else None,
        "confidence": rt.confidence if rt else None,
        "rr": round(setup.rr, 2), "stop_atr": round(setup.stop_atr, 2),
        "positives": rt.positives if rt else [],
        "negatives": rt.negatives if rt else [],
        "measured_hit_rate": br.get("rate"), "measured_n": br.get("n"),
        "sim_p_tp1": sim.get("p_tp1_first"), "sim_p_stop": sim.get("p_stop_first"),
        "sim_expectancy_R": sim.get("expectancy_R"),
        "sim_breakeven_p": sim.get("breakeven_p"),
        "market_health": (ctx.get("health") or {}).get("verdict"),
        "blackout": (ctx.get("macro") or {}).get("blackout"),
        "funding_pct": (ctx.get("macro") or {}).get("funding_annualised_pct"),
        "open_risk_pct": ctx.get("open_risk_pct"),
        "majors_held": ctx.get("open_majors"),
    }


def review(context):
    if not available():
        return {"error": "no agent configured"}
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
    cache = _cache()
    if key in cache:
        return {**cache[key], "cached": True}
    c = config()
    try:
        r = requests.post(
            f"{c['base_url'].rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {c['api_key']}",
                     "Content-Type": "application/json"},
            json={"model": c["model"], "temperature": 0.2, "max_tokens": 500,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": json.dumps(context)}]},
            timeout=60)
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```")
        out = json.loads(text)
        out["cached"] = False
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    cache[key] = out
    CACHE.write_text(json.dumps(cache))
    return out
