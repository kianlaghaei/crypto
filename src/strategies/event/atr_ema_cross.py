# src/strategies/event/ema_cross.py
from __future__ import annotations
import pandas as pd

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def build_signals_ema_cross(close: pd.Series, fast: int, slow: int) -> tuple[pd.Series, pd.Series]:
    """Cross-up generates entry (at next open), cross-down generates exit (at next open)."""
    f = ema(close.astype(float), fast)
    s = ema(close.astype(float), slow)
    cross_up = (f.shift(1) <= s.shift(1)) & (f > s)
    cross_dn = (f.shift(1) >= s.shift(1)) & (f < s)
    entries = cross_up.fillna(False).astype(bool)
    exits = cross_dn.fillna(False).astype(bool)
    entries.name = "entry_signal"
    exits.name = "exit_signal"
    return entries, exits
