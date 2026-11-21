"""check_install.py - import every module the Analysis tab depends on.

Imports in dependency order so the first FAIL is the real culprit.
`--all` walks the tree instead.
"""
import importlib
import pathlib
import sys

CHAIN = [
    "config", "core.models", "store",
    "core.indicators", "core.volume", "core.candles", "core.structure2",
    "core.readiness", "core.nextaction", "core.lifecycle", "core.pit",
    "app_alerts", "core.alerts2", "chart2", "analysis_page",
]

APP = ["data", "core.mtf", "core.regime", "core.structure", "core.levels",
       "strategy.setups", "strategy.risk", "strategy.rating",
       "backtest.engine", "backtest.metrics", "backtest.validate",
       "ui", "market", "smc", "sizer", "board", "paper", "portfolio",
       "sop", "learn", "auth"]


def _walk() -> list:
    out = []
    for p in pathlib.Path(".").rglob("*.py"):
        rel = p.with_suffix("")
        if any(part in ("tests", ".devcontainer", "build", "dist")
               for part in rel.parts):
            continue
        if rel.name in ("app", "check_install", "conftest"):
            continue
        out.append(".".join(rel.parts))
    return sorted(out)


def run(mods: list) -> int:
    bad = 0
    for m in mods:
        try:
            importlib.import_module(m)
            print(f"OK    {m}")
        except Exception as e:
            bad += 1
            print(f"FAIL  {m}  ->  {type(e).__name__}: {e}")
    print(f"\n{bad} failing of {len(mods)}")
    return 1 if bad else 0


if __name__ == "__main__":
    targets = _walk() if "--all" in sys.argv else CHAIN + APP
    sys.exit(run(targets))
