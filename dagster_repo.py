# dagster_repo.py
from __future__ import annotations
from pathlib import Path

from dagster import job, op, In, Out, get_dagster_logger

from src.optimize.optuna_runner import run_optuna

@op(out=Out(bool))
def fetch_ohlcv_op():
    log = get_dagster_logger()
    # TODO: call your src/data/fetch_coinex.py with symbols/timeframes from config
    log.info("Fetching OHLCV ... (placeholder)")
    return True

@op(ins={"upstream": In(bool)}, out=Out(bool))
def validate_data_op(upstream: bool):
    log = get_dagster_logger()
    # TODO: run data quality checks (missing bars, outliers)
    log.info("Validating data ... (placeholder)")
    return True

@op(ins={"upstream": In(bool)}, out=Out(Path))
def run_optuna_op(upstream: bool) -> Path:
    log = get_dagster_logger()
    outdir = run_optuna(
        cfg_path="config/config.yaml",
        strategy="ema_cross",
        symbol=None,
        n_trials=100,
        study_name="ema_daily",
        storage="sqlite:///out/optuna/optuna.db",
    )
    log.info(f"Optuna done: {outdir}")
    return outdir

@job
def research_pipeline():
    data_ok = fetch_ohlcv_op()
    validated = validate_data_op(data_ok)
    run_optuna_op(validated)
