"""core/mtf.py - explicit HTF/LTF pairing with a structure-based bias."""
from .structure import build_structure

PAIRS = {"3m": "15m", "5m": "30m", "15m": "1h", "1h": "4h", "4h": "1d"}


def htf_for(ltf: str) -> str:
    return PAIRS.get(ltf, "1d")


def structure_bias(df, cfg) -> str:
    """'up' / 'down' / 'range' from the last HTF structure state."""
    if df is None or len(df) < 50:
        return "range"
    try:
        return str(build_structure(df, cfg.structure)["trend"].iloc[-1])
    except Exception:
        return "range"


def aligned(htf_bias: str, ltf_bias: str, direction: str) -> tuple[bool, str]:
    want = "up" if direction == "long" else "down"
    if htf_bias != want:
        return False, f"HTF {htf_bias} vs {direction}"
    if ltf_bias != want:
        return False, f"LTF {ltf_bias} vs {direction}"
    return True, "aligned"
