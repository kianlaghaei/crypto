import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Tuple


def build_signals_bb_meanrev(close: pd.Series, window_list, k_list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build entry/exit signals for Bollinger Bands mean reversion strategy.

    Args:
        close (pd.Series): Close price series.
        window_list (list[int]): BB window sizes.
        k_list (list[float]): BB k (alpha) values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Entry and exit signals (DataFrames).

    Example:
        >>> import pandas as pd, numpy as np, vectorbt as vbt
        >>> close = pd.Series(np.random.rand(100))
        >>> entries, exits = build_signals_bb_meanrev(close, [20], [2.0])
    """
    bb = vbt.BBANDS.run(close, window=np.array(window_list), alpha=np.array(k_list))
    entries = close.vbt.crossed_below(bb.lower)
    exits   = close.vbt.crossed_above(bb.middle)
    return entries, exits
