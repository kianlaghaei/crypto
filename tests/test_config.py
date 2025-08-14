import pytest
from pydantic import ValidationError
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


def test_required_fields():
    required = [
        "symbols",
        "timeframe",
        "start_days",
        "rate_limit_ms",
        "fees_bps",
        "slippage_bps",
        "init_cash",
        "strategies",
    ]
    for field in required:
        cfg = base_cfg()
        cfg.pop(field)
        with pytest.raises(ValidationError):
            Config.model_validate(cfg)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("start_days", 0),
        ("rate_limit_ms", -1),
        ("fees_bps", -5),
        ("slippage_bps", 200),
        ("init_cash", -100),
        ("symbols", []),
    ],
)
def test_invalid_values(field, bad):
    cfg = base_cfg()
    cfg[field] = bad
    with pytest.raises(Exception):
        Config.model_validate(cfg)
