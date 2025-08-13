import pandas as pd
import numpy as np
from src.backtest.run_backtest import run

def test_smoke(tmp_path):
    # Minimal config
    cfg = {
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "start_days": 1,
        "rate_limit_ms": 200,
        "fees_bps": 10,
        "slippage_bps": 2,
        "init_cash": 8000,
        "strategies": {
            "ema_cross": {
                "fast_windows": [10],
                "slow_windows": [30],
                "sl_stop_pct": 0.02,
                "tp_stop_pct": 0.04
            },
            "bb_meanrev": {
                "window_list": [20],
                "k_list": [2.0],
                "sl_stop_pct": 0.015,
                "tp_stop_pct": 0.02
            }
        }
    }
    # Fake data
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=100, freq="H"),
        "close": np.random.rand(100) * 100 + 1000
    }).set_index("datetime")
    data_path = tmp_path / "BTC-USDT_1h.csv"
    df.to_csv(data_path)
    import shutil
    shutil.copy(str(data_path), "data/raw/BTC-USDT_1h.csv")
    run(str(tmp_path / "cfg.yaml"), "ema_cross", tmp_path)
