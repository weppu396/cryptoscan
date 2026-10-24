import json, pathlib, requests
STATE = pathlib.Path("./state/alerted.json")
STATE.parent.mkdir(exist_ok=True)


def notify(setups, cfg, webhook=None):
    seen = json.loads(STATE.read_text()) if STATE.exists() else {}
    fresh = []
    for s in setups:
        key = f"{s.symbol}|{s.pattern}|{s.ts.date()}"
        if seen.get(key):
            continue
        seen[key] = 1
        fresh.append(s)
    STATE.write_text(json.dumps(seen))

    for s in fresh:
        msg = (f"{s.symbol} {s.direction.upper()} | {s.pattern} | score {s.score}\n"
               f"entry {s.entry:,.4f} | SL {s.stop:,.4f} | TP2 {s.tp2:,.4f} | R:R {s.rr:.2f}")
        if webhook:
            requests.post(webhook, json={"content": msg}, timeout=15)
    return len(fresh)