import pytest
from src.utils.config import Config


def base_cfg():
    return {
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


def test_valid_limits():
    cfg = base_cfg()
    cfg["daily_loss_limit_pct"] = 0.5
    cfg["max_trades_per_day"] = 50
    parsed = Config.model_validate(cfg)
    assert parsed.daily_loss_limit_pct == 0.5
    assert parsed.max_trades_per_day == 50


def test_invalid_daily_loss_limit():
    cfg = base_cfg()
    cfg["daily_loss_limit_pct"] = 1.5
    with pytest.raises(Exception):
        Config.model_validate(cfg)


def test_invalid_max_trades():
    cfg = base_cfg()
    cfg["max_trades_per_day"] = 0
    with pytest.raises(Exception):
        Config.model_validate(cfg)
