import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Tuple


def build_signals_bb_meanrev(close: pd.Series, window_list, k_list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bb = vbt.BBANDS.run(close, window=np.array(window_list), alpha=np.array(k_list))
    entries = close.vbt.crossed_below(bb.lower)
    exits   = close.vbt.crossed_above(bb.middle)
    return entries, exits
