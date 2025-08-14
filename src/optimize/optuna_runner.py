# src/optimize/optuna_runner.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna
import vectorbt as vbt

from src.utils.io import load_yaml


def load_price(sym: str, timeframe: str) -> pd.Series:
    pq_path = Path("data/parquet") / f"{sym.replace('/','-')}_{timeframe}.parquet"
    csv_path = Path("data/raw") / f"{sym.replace('/','-')}_{timeframe}.csv"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        if "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["datetime"], utc=True))
        elif "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Parquet missing datetime/timestamp: {pq_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")
    else:
        raise FileNotFoundError(f"Missing data for {sym} {timeframe}. Run fetch_coinex.py first.")
    return df["close"].astype(float)


def objective_ema_cross(
    trial: optuna.trial.Trial,
    close: pd.Series,
    fees: float,
    slip: float,
    sl: float,
    tp: float,
    timeframe: str,
) -> float:
    fast = trial.suggest_int("fast", 5, 120)
    slow = trial.suggest_int("slow", fast + 5, 400)

    ma_fast = vbt.MA.run(close, window=fast).ma
    ma_slow = vbt.MA.run(close, window=slow).ma

    entries = ma_fast.vbt.crossed_above(ma_slow).to_frame()
    exits = ma_fast.vbt.crossed_below(ma_slow).to_frame()

    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=fees, slippage=slip,
        sl_stop=sl, tp_stop=tp,
        init_cash=10_000, freq=timeframe,
    )

    sharpe = float(np.asarray(pf.sharpe_ratio()).item())
    trial.report(sharpe, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return sharpe


def objective_bb_meanrev(
    trial: optuna.trial.Trial,
    close: pd.Series,
    fees: float,
    slip: float,
    sl: float,
    tp: float,
    timeframe: str,
) -> float:
    window = trial.suggest_int("window", 10, 200)
    k = trial.suggest_float("k", 1.0, 4.0)

    bb = vbt.BBANDS.run(close, window=window, alpha=k)
    entries = close.vbt.crossed_below(bb.lower).to_frame()
    exits = close.vbt.crossed_above(bb.middle).to_frame()

    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=fees, slippage=slip,
        sl_stop=sl, tp_stop=tp,
        init_cash=10_000, freq=timeframe,
    )
    sharpe = float(np.asarray(pf.sharpe_ratio()).item())
    trial.report(sharpe, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return sharpe


def run_optuna(
    cfg_path: str,
    strategy: str,
    symbol: Optional[str],
    n_trials: int,
    study_name: str,
    storage: Optional[str] = None,
) -> Path:
    raw_cfg: dict = load_yaml(cfg_path)
    timeframe: str = raw_cfg["timeframe"]
    fees = float(raw_cfg.get("fees_bps", 0.0)) / 1e4
    slip = float(raw_cfg.get("slippage_bps", 0.0)) / 1e4

    if strategy == "ema_cross":
        s = raw_cfg["strategies"]["ema_cross"]
    elif strategy == "bb_meanrev":
        s = raw_cfg["strategies"]["bb_meanrev"]
    else:
        raise ValueError("Unknown strategy")

    sl = float(s.get("sl_stop_pct", 0.0))
    tp = float(s.get("tp_stop_pct", 0.0))

    symbols = [symbol] if symbol else list(raw_cfg["symbols"])
    results_dir = Path("out/optuna") / f"{strategy}_{timeframe}_{study_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    for sym in symbols:
        close = load_price(sym, timeframe)

        def _obj(trial: optuna.trial.Trial) -> float:
            if strategy == "ema_cross":
                return objective_ema_cross(trial, close, fees, slip, sl, tp, timeframe)
            else:
                return objective_bb_meanrev(trial, close, fees, slip, sl, tp, timeframe)

        study.optimize(_obj, n_trials=n_trials, gc_after_trial=True, show_progress_bar=False)

        best = study.best_trial
        (results_dir / f"best_params_{sym.replace('/','-')}.json").write_text(
            json.dumps({"value": best.value, "params": best.params}, indent=2),
            encoding="utf-8",
        )

    (results_dir / "study_summary.json").write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "n_trials": len(study.trials),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--strategy", choices=["ema_cross", "bb_meanrev"], required=True)
    ap.add_argument("--symbol", help="Optional: optimize a single symbol")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--study", default="default_study")
    ap.add_argument("--storage", help='e.g. "sqlite:///out/optuna/optuna.db"', default=None)
    args = ap.parse_args()

    out = run_optuna(args.cfg, args.strategy, args.symbol, args.trials, args.study, args.storage)
    print(f"Optuna results → {out}")


if __name__ == "__main__":
    main()
