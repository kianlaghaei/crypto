# Daytrading (CoinEx) — vectorbt + CCXT

## Quickstart

1) python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
2) pip install -r requirements.txt
3) cp .env.example .env  # (optional; only needed for private endpoints)
4) Edit config/config.yaml (symbols, timeframe, start_days)
5) Fetch data:
   python src/data/fetch_coinex.py --cfg config/config.yaml
6) Run backtest (EMA Cross example):
   python src/backtest/run_backtest.py --cfg config/config.yaml --strategy ema_cross
7) See outputs in out/backtests/<run_id>/ (summary.csv, equity.png)

## Additional Development Info

### Pre-commit Setup

Install pre-commit and set up hooks:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks: ruff (lint/format), pyupgrade (syntax upgrade)

### Type checking

Run mypy:

```sh
mypy .
```

### CI

GitHub Actions workflow: `.github/workflows/ci.yml` (lint + smoke backtest)

## Config schema

See `src/utils/config.py` for Pydantic model. Edit `config/config.yaml` for symbols, timeframe, strategy params, etc.

## Output paths

- Backtests: `out/backtests/<run_id>/`
- Logs: `logs/`
- Parquet data: `data/parquet/`

## Known limitations

- No Docker support (out of scope)
- Paper trading is stub only (no live orders)
- Only EMA/BB strategies implemented

## Development

### Pre-commit hooks

Install pre-commit and set up hooks:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks: ruff (lint/format), pyupgrade (syntax upgrade)

## Notes

- This scaffold uses vectorbt for fast research/backtests and ccxt for data.
- Execution (paper/live) is out-of-scope here; we can add a CCXT bridge later.
