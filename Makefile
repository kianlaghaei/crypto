.PHONY: setup lock lint test bt
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip pip-tools
lock:
	pip-compile -o requirements.txt requirements.in
	pip install -r requirements.txt
lint:
	ruff check .
	ruff format --check .
test:
	pytest -q || true
bt:
	python src/backtest/run_backtest.py --cfg config/config.yaml --strategy ema_cross
