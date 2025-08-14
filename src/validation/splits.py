# src/validation/splits.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Tuple, List

def rolling_time_splits(
    index: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling (train_start, train_end, test_start, test_end) tuples."""
    step_days = step_days or test_days
    start = index.min().normalize()
    end = index.max().normalize()

    splits = []
    t0 = start
    while True:
        tr_start = t0
        tr_end = tr_start + pd.Timedelta(days=train_days) - pd.Timedelta(seconds=1)
        te_start = tr_end + pd.Timedelta(seconds=1)
        te_end = te_start + pd.Timedelta(days=test_days) - pd.Timedelta(seconds=1)
        if te_end > end:
            break
        splits.append((tr_start, tr_end, te_start, te_end))
        t0 = t0 + pd.Timedelta(days=step_days)
    return splits
