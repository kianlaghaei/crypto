# Daytrading (CoinEx) — vectorbt + CCXT

A lightweight research scaffold for exploring crypto trading strategies on the CoinEx exchange. Data is pulled with [ccxt](https://github.com/ccxt/ccxt) and strategies are backtested using [vectorbt](https://vectorbt.pro/).

## Project structure

| Path | Description |
| ---- | ----------- |
| `src/data/fetch_coinex.py` | Fetch OHLCV data and store it as Parquet (resumes from previous runs). |
| `src/backtest/run_backtest.py` | Run grid backtests for EMA‑cross and Bollinger‑Band mean‑reversion strategies. |
| `src/analysis/advanced_report.py` | Generate KPIs and Plotly charts from a backtest run directory. |
| `src/signals/make_signals.py` | Build entry/exit signal CSVs for a symbol and strategy. |
| `src/exec/paper.py` | Simple paper‑trading simulator that replays signal CSVs. |
| `src/strategies/` | Strategy helpers used for building signals. |
| `src/utils/` | Configuration and I/O utilities. |
| `tests/` | Unit tests and smoke checks. |

More details can be found in [docs/architecture.md](docs/architecture.md).

## Quickstart

1. `python -m venv .venv && source .venv/bin/activate`  
   *(Windows: `.venv\Scripts\activate`)*
2. `pip install -r requirements.txt`
3. `cp .env.example .env` *(optional; only needed for private endpoints)*
4. Edit `config/config.yaml` (symbols, timeframe, strategy parameters)
5. Fetch data:  `python src/data/fetch_coinex.py --cfg config/config.yaml`
6. Run backtest: `python src/backtest/run_backtest.py --cfg config/config.yaml --strategy ema_cross`
7. See outputs in `out/backtests/<run_id>/` (CSV summary, equity chart, etc.)

## Development

### Pre-commit hooks

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Type checking

```sh
mypy .
```

### Tests

```sh
pytest
```

## Notes

- Built for quick experimentation; Docker and live trading are out of scope.
- Output directories:
  - Backtests → `out/backtests/<run_id>/`
  - Logs → `logs/`
  - Price data → `data/parquet/`
