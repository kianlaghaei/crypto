import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Tuple


def build_signals_ema_cross(close: pd.Series, fast_ws, slow_ws) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build entry/exit signals for EMA cross strategy.

    Args:
        close (pd.Series): Close price series.
        fast_ws (list[int]): Fast EMA window sizes.
        slow_ws (list[int]): Slow EMA window sizes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Entry and exit signals (DataFrames).

    Example:
        >>> import pandas as pd, numpy as np, vectorbt as vbt
        >>> close = pd.Series(np.random.rand(100))
        >>> entries, exits = build_signals_ema_cross(close, [10], [30])
    """
    fast = vbt.MA.run(close, window=np.array(fast_ws), short_name="fast").ma
    slow = vbt.MA.run(close, window=np.array(slow_ws), short_name="slow").ma
    entries = fast.vbt.crossed_above(slow)
    exits   = fast.vbt.crossed_below(slow)
    return entries, exits
