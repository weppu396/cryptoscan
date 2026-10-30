"""risk_gate.py - hard PASS/BLOCK with written reasons."""


def evaluate(setup, ctx):
    blocks, warns, mult = [], [], 1.0

    if setup.rr < 2.0:
        blocks.append(f"R:R {setup.rr:.2f} below 2.0")
    if setup.stop_atr > 3.5:
        blocks.append(f"stop {setup.stop_atr:.2f} ATR wider than 3.5")
    if setup.stop_atr < 0.4:
        blocks.append(f"stop {setup.stop_atr:.2f} ATR too tight")

    heat = ctx.get("open_risk_pct", 0) + 0.5
    if heat > 3.0:
        blocks.append(f"portfolio heat would reach {heat:.1f}% (cap 3.0%)")
    if ctx.get("open_majors", 0) >= 3:
        blocks.append("already holding 3 correlated majors")

    macro = ctx.get("macro") or {}
    if macro.get("blackout"):
        mult *= 0.5
        warns.append(f"blackout: {macro.get('blackout_reason')} - size halved")
    fund = macro.get("funding_annualised_pct")
    if fund is not None and abs(fund) > 40:
        mult *= 0.75
        warns.append(f"funding {fund:+.0f}% annualised - crowded")

    health = ctx.get("health") or {}
    if health.get("verdict") == "hostile":
        mult *= 0.5
        warns.append("market health hostile - size halved")
    elif health.get("verdict") == "cautious":
        mult *= 0.75
        warns.append("market health cautious - size reduced")

    br = ctx.get("base_rate")
    if br:
        if br["ci_low"] < 0.25:
            warns.append(f"measured hit {br['rate']*100:.0f}% "
                         f"(CI {br['ci_low']*100:.0f}-{br['ci_high']*100:.0f}%, n={br['n']}) - marginal")
        elif br["rate"] >= 0.45:
            warns.append(f"measured hit {br['rate']*100:.0f}% on n={br['n']}")

    return ("BLOCK" if blocks else "PASS"), blocks, warns, round(mult, 2)
