# src/risk/position_sizing.py
from __future__ import annotations
import numpy as np
import pandas as pd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high = high.astype(float); low = low.astype(float); close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def stop_by_atr(entry_price: float, atr_value: float, atr_mult: float, side: str = "long") -> float:
    if side == "long":
        return entry_price - atr_mult * atr_value
    else:
        return entry_price + atr_mult * atr_value

def size_for_risk(equity: float, risk_pct: float, entry: float, stop: float) -> float:
    risk_amount = equity * risk_pct
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    qty = risk_amount / per_unit_risk
    return float(max(qty, 0.0))
