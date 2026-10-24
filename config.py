from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class Risk:
    account_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    max_open_risk_pct: float = 3.0
    max_positions: int = 5
    max_correlated_positions: int = 3
    min_rr: float = 2.0
    max_tp_r: float = 5.0
    min_stop_atr: float = 0.4
    max_stop_atr: float = 3.5
    stop_buffer_atr: float = 0.25
    partial_at_r: float = 1.5
    partial_size: float = 0.5
    move_stop_to_be_at_r: float = 1.0
    time_stop_bars: int = 24


@dataclass(frozen=True)
class Structure:
    swing_left: int = 3
    swing_right: int = 3
    zone_tol_atr: float = 0.30
    min_zone_touches: int = 2
    zone_lookback: int = 400
    max_wait_bars: int = 20
    require_rejection_candle: bool = True


@dataclass(frozen=True)
class Regime:
    ema_fast: int = 50
    ema_slow: int = 200
    adx_period: int = 14
    adx_trend_min: float = 20.0
    atr_period: int = 14
    atr_pct_min: float = 0.40
    atr_pct_max: float = 6.00
    rsi_period: int = 14
    rsi_long_max: float = 78.0
    rsi_short_min: float = 22.0


@dataclass(frozen=True)
class Costs:
    fee_bps: float = 5.0
    slip_bps: float = 3.0
    funding_bps_per_day: float = 1.0
    use_funding: bool = False


@dataclass(frozen=True)
class Scan:
    exchange: str = "binanceusdm"
    quote: str = "USDT"
    htf: str = "1d"
    ltf: str = "4h"
    min_quote_vol_24h: float = 30_000_000
    top_n_symbols: int = 150
    exclude_suffixes: tuple = ("USDC", "BUSD", "DAI", "TUSD", "FDUSD")


@dataclass(frozen=True)
class Config:
    risk: Risk = Risk()
    structure: Structure = Structure()
    regime: Regime = Regime()
    costs: Costs = Costs()
    scan: Scan = Scan()

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


CFG = Config()