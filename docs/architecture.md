# Project Architecture

This document outlines the major components of this crypto trading research scaffold and how they fit together.

## Data fetching (`src/data`)

### `fetch_coinex.py`
Fetches historical OHLCV data from the CoinEx exchange via **ccxt**. It resumes from existing Parquet files, respects the exchange rate limit, performs basic integrity checks, and stores the result in `data/parquet/` (optionally also writes raw CSVs). Structured logs are written under `logs/`.

## Backtesting (`src/backtest`)

### `run_backtest.py`
Runs vectorised grid backtests for the built‑in strategies. Features include:
- EMA cross and Bollinger Bands mean reversion strategies
- fees, slippage, stop‑loss and take‑profit handling
- optional cap on trades per day
- CSV export of grid results, parameter JSON, simple HTML report, equity chart and a pickled `Portfolio` for the best Sharpe ratio

## Analysis (`src/analysis`)

### `advanced_report.py`
Generates additional KPIs and plots from a backtest output directory. It loads the saved portfolio, exports equity/returns/drawdown/trades series, and saves Plotly charts (orders, PnL distribution, monthly returns, rolling Sharpe, underwater, turnover) as PNG/HTML alongside a small HTML index.

## Signal generation (`src/signals`)

### `make_signals.py`
Builds entry/exit signal CSV files for a single symbol and strategy using stored price data. The signals are aligned with price, normalised to a standard schema and saved in `out/signals/` for use by the paper trading simulator or other tooling.

## Execution (`src/exec`)

### `paper.py`
A lightweight paper trading simulator that consumes a signals CSV and the project configuration. It applies fees, slippage and an optional daily loss limit, tracks cash, and outputs a trades CSV together with structured logs.

## Strategies (`src/strategies`)

- `ema_cross.py` – constructs moving‑average crossover entry/exit signals.
- `bb_meanrev.py` – constructs Bollinger Bands mean‑reversion signals.

## Utilities (`src/utils`)

- `config.py` – Pydantic models describing the configuration schema (symbols, timeframe, strategy parameters, etc.).
- `io.py` – helper for loading YAML configs with helpful error messages.

## Other directories

- `config/` – sample configuration files.
- `data/` – raw CSV and Parquet price data.
- `logs/` – runtime logs.
- `tests/` – unit tests for I/O, strategies and smoke backtests.
- `notebooks/` – exploratory research notebooks.

