# tests/test_backtest_smoke.py
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.backtest.run_backtest import run


def test_smoke(tmp_path: Path):
    # --- Minimal config ---
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
                "tp_stop_pct": 0.04,
            },
            "bb_meanrev": {
                "window_list": [20],
                "k_list": [2.0],
                "sl_stop_pct": 0.015,
                "tp_stop_pct": 0.02,
            },
        },
    }

    # --- Fake data (ساخت دیتای تست) ---
    # FutureWarningِ فرکانس را با استفاده از 'h' حل می‌کنیم
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=100, freq="h"),
            "close": np.random.rand(100) * 100 + 1000,
        }
    ).set_index("datetime")

    # ذخیرهٔ داده در tmp و کپی به مسیر مورد انتظار کد
    data_path = tmp_path / "BTC-USDT_1h.csv"
    df.to_csv(data_path)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    (Path("data/raw") / "BTC-USDT_1h.csv").write_text(data_path.read_text(encoding="utf-8"), encoding="utf-8")

    # --- نوشتن فایل کانفیگ در tmp_path ---
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    # --- اجرای بک‌تست دود (Smoke) ---
    outdir = run(str(cfg_path), "ema_cross", tmp_path)

    # بررسی خروجی‌های کلیدی
    assert (outdir / "grid_results.csv").exists()
    assert (outdir / "report.html").exists()
