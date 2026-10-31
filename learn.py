"""learn.py - staged curriculum content."""

STAGES = [
    ("Stage 1", "Market structure", "started",
     ["Uptrend, downtrend and sideways market",
      "Higher High (HH), Higher Low (HL)",
      "Lower High (LH), Lower Low (LL)",
      "BOS and CHoCH"]),
    ("Stage 2", "Supply, demand and order blocks", "next",
     ["Supply zones", "Demand zones", "Order blocks",
      "Valid versus weak zones", "How to mark zones on a chart"]),
    ("Stage 3", "Liquidity", "later",
     ["Liquidity and stop-loss clusters", "Equal highs and equal lows",
      "Liquidity sweep or stop run", "Inducement and false breakouts"]),
    ("Stage 4", "Imbalances and price delivery", "later",
     ["Fair value gap", "Imbalance", "Mitigation",
      "Premium and discount zones", "Fibonacci as a supporting tool"]),
    ("Stage 5", "Building a complete trade", "later",
     ["Entry confirmation", "Stop loss", "Take profit",
      "Risk-to-reward ratio", "Position sizing", "Trade management"]),
    ("Stage 6", "Advanced confirmation and practice", "later",
     ["Multi-timeframe analysis", "Candlestick patterns",
      "Sessions and volatility", "Backtesting and journal",
      "Psychology and discipline"]),
]

PRIORITY = [
    ("1", "Supply and demand", "Where price may react"),
    ("2", "Order blocks", "Areas of buying or selling interest"),
    ("3", "Liquidity", "Where stop orders cluster"),
    ("4", "Liquidity sweep", "Whether price reversed after taking a level"),
    ("5", "Fair value gap", "Areas of rapid movement and imbalance"),
    ("6", "Premium and discount", "Whether price is high or low in a range"),
    ("7", "Entry confirmation", "When a setup justifies entering"),
    ("8", "Stop loss and take profit", "How to manage risk and exits"),
    ("9", "Risk-to-reward ratio", "Whether reward justifies risk"),
    ("10", "Multi-timeframe analysis", "How structures fit together"),
]

FRAMEWORK = """
A worked buy example, in order:

1. Trend is bullish on the higher timeframe.
2. Price falls into a demand zone or order block.
3. A sweep takes the lows under the zone, then price reclaims it.
4. A bullish CHoCH or BOS confirms the reclaim.
5. Entry at the zone, stop under the sweep low, targets at the range high and
   the 1.272 extension.
6. Size so the stop costs 0.5% of the account.

If any of steps 1 to 4 is missing, it is a weaker trade. If step 1 or 4 is
missing, it is not this setup at all.
"""

DAILY = """
A 21-day sequence, one lesson and one exercise per day:

Days 1-4 · structure. Mark HH/HL and LH/LL on 20 charts. No entries.
Days 5-8 · supply and demand. Mark 30 zones, then review which held.
Days 9-11 · order blocks. Compare against the zones you already marked.
Days 12-14 · liquidity. Find equal highs and lows, then find the sweeps.
Days 15-17 · gaps and premium/discount. Note which gaps got filled.
Days 18-19 · entries, stops and targets on paper.
Days 20-21 · full framework, 10 paper trades, log every one.

Roughly 30 minutes a day. Do not skip to stage 5.
"""
