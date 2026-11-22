HTF_PREF_MARGIN = 2.0


def choose_pair(src: dict, cfg) -> tuple:
    """(htf, ltf, quality). 30m is preferred unless 1h is clearly cleaner.

    The slower HTF needs twice the confirmation time, so an equal-quality 1h
    keeps 30m structure off the chart. A margin makes 30m win ties.
    """
    best = None
    for htf in HTF_CHOICES:
        hdf = _frame(src, cap_htf(htf))
        for ltf in LTF_CHOICES:
            q = quality(hdf, _frame(src, ltf), cfg)
            if q < 0:
                continue
            score = q - (0.0 if htf == HTF_CHOICES[0] else HTF_PREF_MARGIN)
            if best is None or score > best[3]:
                best = (cap_htf(htf), ltf, q, score)
    if best is None:
        return (HTF_CHOICES[0], LTF_CHOICES[0], -1.0)
    return best[0], best[1], best[2]
