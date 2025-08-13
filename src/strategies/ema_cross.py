import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Tuple


def build_signals_ema_cross(close: pd.Series, fast_ws, slow_ws) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fast = vbt.MA.run(close, window=np.array(fast_ws), short_name="fast").ma
    slow = vbt.MA.run(close, window=np.array(slow_ws), short_name="slow").ma
    entries = fast.vbt.crossed_above(slow)
    exits   = fast.vbt.crossed_below(slow)
    return entries, exits
