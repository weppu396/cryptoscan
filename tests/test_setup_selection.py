from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import paper
from strategy.selection import best_live_per_market, signal_key
from strategy.setups import Setup


def make_setup(symbol, direction="long", status="BETTER_PRICE", entry=10):
    return Setup(symbol, direction, "bos_retest", pd.Timestamp("2026-11-28", tz="UTC"),
                 entry, 9, 12, 13, 1, 2, 80,
                 meta={"status": status, "ltf": "5m", "qty": 1,
                       "r": {"grade": "A+", "confidence": 88,
                             "positives": [], "negatives": []}})


def test_selects_first_ranked_live_candidate_per_exact_instrument():
    invalid = make_setup("TAO/USDT:USDT", status="INVALID")
    best = make_setup(invalid.symbol)
    alternate = make_setup(invalid.symbol, direction="short", entry=11)
    spot = make_setup("TAO/USDT")
    icp = make_setup("ICP/USDT:USDT")
    assert best_live_per_market([invalid, best, alternate, spot, icp]) == [best, spot, icp]


def test_board_renders_duplicate_scan_results_once():
    app = AppTest.from_string("""
import streamlit as st
import board
from test_setup_selection import make_setup

def scan(*args):
    return ([make_setup("TAO/USDT:USDT"), make_setup("TAO/USDT:USDT", entry=11),
             make_setup("ICP/USDT:USDT")], {
        "venue": "okx", "universe": 10, "loaded": 10, "err": 0,
        "raw": 3, "kept": 3, "stale": 0, "actionable": 0, "better": 3,
        "below_grade": 0, "mtf_skip": 0, "vol_skip": 0, "no_anchor": 0,
    })

board.render(scan, lambda *args: None, lambda *args: (None, "okx"),
             lambda *args: (None, args[1]), lambda rating: rating,
             lambda *args: None, lambda *args: (True, "sent"),
             "auto", 25, 0, "5m", 2, 3, 0.5, 10000, 12, None,
             False, False, False, False)
""").run()
    app.button[0].click().run()
    assert not app.exception
    assert list(app.dataframe[0].value["symbol"]) == ["TAO/USDT:USDT", "ICP/USDT:USDT"]
    assert len([item for item in app.expander if "TAO/USDT:USDT" in item.label]) == 1
    app.button(key="baTAO/USDT:USDT").click().run()
    app.run()
    assert app.button(key="baTAO/USDT:USDT").disabled
    app.slider(key="b_top").set_value(30).run()
    assert not app.dataframe


def test_paper_retries_do_not_create_another_trade(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paper, "BOOK", tmp_path / "paper.json")
    setup = make_setup("TAO/USDT:USDT")
    key = signal_key("okx", setup)

    def log():
        return paper.open_trade(setup.symbol, setup.direction, setup.entry,
                                setup.stop, setup.tp1, setup.tp2, 1, 10000, signal=key)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: log(), range(8))) == 1
    trade, = paper.open_rows()
    paper.close_trade(trade["id"], 12, "target")
    assert not log()
    assert len(paper.book()) == 1
