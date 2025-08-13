from pydantic import BaseModel, Field, field_validator
from typing import Literal

Timeframe = Literal["1m","3m","5m","15m","30m","1h","4h","1d"]

class StrategyEMA(BaseModel):
    fast_windows: list[int]
    slow_windows: list[int]
    sl_stop_pct: float = Field(ge=0, le=0.5)
    tp_stop_pct: float = Field(ge=0, le=1.0)

class StrategyBB(BaseModel):
    window_list: list[int]
    k_list: list[float]
    sl_stop_pct: float = Field(ge=0, le=0.5)
    tp_stop_pct: float = Field(ge=0, le=1.0)

class Strategies(BaseModel):
    ema_cross: StrategyEMA
    bb_meanrev: StrategyBB

class Config(BaseModel):
    symbols: list[str]
    timeframe: Timeframe
    start_days: int = Field(gt=0, le=3650)
    rate_limit_ms: int = Field(gt=0, le=5000)
    fees_bps: float = Field(ge=0, le=1000)
    slippage_bps: float = Field(ge=0, le=100)
    init_cash: float = Field(gt=0)
    strategies: Strategies

    @field_validator('symbols')
    def symbols_not_empty(cls, v):
        if not v:
            raise ValueError('symbols list must not be empty')
        return v
