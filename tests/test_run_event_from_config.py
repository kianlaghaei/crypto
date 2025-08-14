import subprocess
import sys
from pathlib import Path
import textwrap
import pytest


def test_invalid_config_raises(tmp_path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            symbols: []
            timeframe: 1h
            start_days: 1
            rate_limit_ms: 200
            fees_bps: 10
            slippage_bps: 2
            init_cash: 8000
            strategies:
              ema_cross:
                fast_windows: [10]
                slow_windows: [30]
                sl_stop_pct: 0.02
                tp_stop_pct: 0.04
              bb_meanrev:
                window_list: [20]
                k_list: [2.0]
                sl_stop_pct: 0.015
                tp_stop_pct: 0.02
            """
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "src.scripts.run_event_from_config",
        "--cfg",
        str(cfg_path),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
