"""Selection policy for user-facing scan results."""

LIVE_STATUSES = {"ACTIONABLE", "BETTER_PRICE"}


def best_live_per_market(ranked_setups):
    """Keep the first eligible candidate for each exact exchange symbol."""
    selected = {}
    for setup in ranked_setups:
        if setup.meta.get("status") in LIVE_STATUSES:
            selected.setdefault(setup.symbol, setup)
    return list(selected.values())


def signal_key(venue, setup):
    """Identify a setup across reruns and changes to sizing or confidence."""
    return (venue, setup.symbol, setup.direction, setup.pattern,
            setup.meta.get("ltf"), setup.ts.isoformat(),
            float(setup.entry), float(setup.stop), float(setup.tp1), float(setup.tp2))
