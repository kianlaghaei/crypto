import numpy as np
import pandas as pd
from src.strategies.ema_cross import build_signals_ema_cross
from src.strategies.bb_meanrev import build_signals_bb_meanrev

def test_ema_cross():
    close = pd.Series(np.random.rand(100))
    entries, exits = build_signals_ema_cross(close, [10], [30])
    assert entries.shape == exits.shape
    assert entries.shape[0] == 100

def test_bb_meanrev():
    close = pd.Series(np.random.rand(100))
    entries, exits = build_signals_bb_meanrev(close, [20], [2.0])
    assert entries.shape == exits.shape
    assert entries.shape[0] == 100
