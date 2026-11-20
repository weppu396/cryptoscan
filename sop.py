"""sop.py - the desk procedure, start to finish."""
import streamlit as st

import ui

TEXT = """
### Phase 0 — Setup, once

1. **Pick the venue manually.** Sidebar → Exchange. Leave `auto` only if you do
   not care which venue answers. `Test venues` shows what is alive. BinanceUS
   has almost no perps — use okx or bybit.
2. Set `Account equity` to your real balance. It is the only source of size.
3. `Risk per trade` 0.5%. Never above 1% until 50 logged trades.
4. `Minimum R:R` 2.0. `Max stop (ATR)` 3.5.
5. Keep `Relax filters` on. Leave `Require MTF` and `Require AVWAP` **off** while
   learning. In a downtrend, tick `Actionable only` and work the shorts.

### Phase 1 — Context, once per session

6. Overview → BTC/USDT:USDT, 4h, Load detail. If BTC is mid-impulse against your
   intended side, halve size or sit out.
7. Macro. A blackout means no new entries.
8. Session: London open to NY overlap, 07:00–21:00 UTC.

### Phase 2 — Find candidates

9. Setups → Build board. Grade C, universe 25, timeframe 4h.
10. Read the diagnostic line first. `raw 0` = no breaks, raise freshness. `failed`
    near `loaded` = rate limits, lower the universe or switch venue.
11. Work down the board. Skip anything not `ACTIONABLE` or `BETTER_PRICE`.

### Phase 3 — Qualify

12. Open a card. Read the chart top to bottom: structure, location, liquidity,
    volume, acceptance, break, candle, range position.
13. Rebuild the same symbol in SMC at the entry timeframe (HTF/4) and confirm the
    stat strip: both structures with you, AVWAP on the right side, entry in value.
14. Verdict A or B only. C is watch. D is skip.

### Phase 4 — Size

15. Sizer. Balance, 0.5%, entry and stop from the card, direction, leverage.
16. Liquidation inside the stop → lower leverage. Max safe leverage is printed.
17. Round size **down** to the step size. Recheck risk is under 0.5%.

### Phase 5 — Place

18. Exchange: isolated margin, leverage set, symbol is the **perp**.
19. Entry: **post-only limit** at the card's entry.
20. Stop: **stop-market, reduce-only**, triggered on **mark price**.
21. TP1 and TP2: **limits, reduce-only**, 50% each.
22. Verify one position and three working orders, two reduce-only.

### Phase 6 — Manage

23. Unfilled after 2 bars → cancel and re-evaluate.
24. TP1 → 50% off, stop to breakeven. Change nothing else.
25. 24 bars without TP1 → close at market.
26. Journal it, including whether you followed the rules.

### Phase 7 — Promote

27. Log to paper every setup you would have taken.
28. Gate: 30 closed paper trades, profit factor ≥ 1.0, positive expectancy. Then
    live at 0.25% risk.
29. Backtest before any parameter change: 300 trades, PF ≥ 1.3, drawdown ≤ 15%,
    permutation p < 0.05. Otherwise the change is noise.
"""


def render() -> None:
    ui.section("Desk SOP", "Start to finish. Do not skip a gate.")
    st.markdown(TEXT)
