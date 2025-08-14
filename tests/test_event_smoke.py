import numpy as np
import pandas as pd
from pathlib import Path

from src.engine.event_backtester import EventBacktester


def test_event_smoke(tmp_path: Path):
    ts = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    price = np.linspace(100, 110, 5)
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price + 1,
        },
        index=ts,
    )

    entries = pd.Series([True, False, False, True, False], index=ts)
    exits = pd.Series([False, False, True, False, False], index=ts)

    bt = EventBacktester(
        df=df,
        entries=entries,
        exits=exits,
        init_cash=1000,
        fees_bps=10,
        slippage_bps=1,
        risk_pct_per_trade=0.1,
        stop_mode="percent",
        sl_pct=0.01,
        tp_pct=0.02,
    )
    bt.run()
    bt.save_outputs(tmp_path)

    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "summary.json").exists()
