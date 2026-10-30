"""
targets.py - realistic spot ladders.

For an entry: tranches down so averaging works.
For an exit: scale-out ladder with % gains, so you bank progressively
instead of trying to pick one perfect top.
"""


def accumulation_ladder(first_entry, tranches=4, step_pct=8.0):
    """Buy more as price falls. Each tranche is the same cash amount."""
    out = []
    for i in range(tranches):
        price = first_entry * (1 - step_pct / 100) ** i
        out.append({"tranche": i + 1, "price": round(price, 8),
                    "drop_pct": round((1 - price / first_entry) * 100, 2)})
    return out


def exit_ladder(entry, risk, structure_targets=None, extended=None):
    """
    Scale out. Percentages sum to 100.
    risk = entry - stop (positive number).
    """
    lvl = []

    def add(pct, price, tag):
        lvl.append({"pct": pct, "price": round(price, 8),
                    "gain_pct": round((price / entry - 1) * 100, 2), "tag": tag})

    add(30, entry + risk * 1.0, "de-risk, cover the stop")
    add(30, entry + risk * 2.0, "2R")
    if structure_targets:
        for i, t in enumerate(structure_targets[:1]):
            add(25, t, "next resistance")
    else:
        add(25, entry + risk * 3.5, "3.5R")
    add(15, entry * 1.5 if extended is None else extended, "runner to +50%")
    return lvl


def breakeven_pct(entry, stop):
    return round((entry / stop - 1) * 100, 2)
