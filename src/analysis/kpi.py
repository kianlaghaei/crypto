# src/analysis/kpi.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _to_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x)

def sharpe(returns: pd.Series, rf: float = 0.0, freq: int = 365) -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    excess = r - rf / freq
    mu = excess.mean(); sigma = excess.std(ddof=1)
    return float(mu / sigma) if sigma > 0 else float("nan")

def sortino(returns: pd.Series, rf: float = 0.0, freq: int = 365) -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    excess = r - rf / freq
    downside = excess.copy()
    downside[downside > 0] = 0
    dd = np.sqrt((downside**2).mean())
    mu = excess.mean()
    return float(mu / dd) if dd > 0 else float("nan")

def calmar(returns: pd.Series, freq: int = 365) -> float:
    r = _to_series(returns).dropna()
    if r.empty:
        return float("nan")
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (freq / len(r)) - 1
    running_max = cum.cummax()
    dd = (cum / running_max - 1).min()
    mdd = abs(dd)
    return float(ann / mdd) if mdd > 0 else float("nan")

def omega(returns: pd.Series, tau: float = 0.0) -> float:
    r = _to_series(returns).dropna().sort_values()
    if r.empty:
        return float("nan")
    gains = (r[r > tau] - tau).sum()
    losses = (tau - r[r <= tau]).sum()
    return float(gains / losses) if losses > 0 else float("inf")
