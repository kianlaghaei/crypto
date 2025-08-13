import argparse
import time
from pathlib import Path
import ccxt
import pandas as pd
from loguru import logger
from src.utils.io import load_yaml
from src.utils.config import Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signals', required=True, help='Path to signals CSV')
    ap.add_argument('--cfg', required=True, help='Path to config.yaml')
    ap.add_argument('--max_daily_loss', type=float, default=0.02)
    args = ap.parse_args()

    raw_cfg = load_yaml(args.cfg)
    cfg = Config.model_validate(raw_cfg)

    df = pd.read_csv(args.signals)
    ex = ccxt.coinex({'enableRateLimit': True})
    ex.load_markets()

    cash = cfg.init_cash
    trades = []
    daily_loss = 0
    last_day = None
    block_trading = False
    for row in df.itertuples():
        ts = getattr(row, 'timestamp', None)
        dt = pd.to_datetime(ts, unit='ms', utc=True)
        day = dt.date()
        if last_day is None or day != last_day:
            daily_loss = 0
            block_trading = False
            last_day = day
        if block_trading:
            continue
        price = getattr(row, 'close', None)
        side = getattr(row, 'side', 1)
        exit_price = getattr(row, 'exit_price', price)
        pnl = (side * (exit_price - price)) - cfg.fees_bps/1e4 * price
        cash += pnl
        trades.append({'ts': ts, 'price': price, 'pnl': pnl, 'day': str(day)})
        daily_loss += min(0, pnl)
        if abs(daily_loss) > args.max_daily_loss * cfg.init_cash:
            logger.warning(f'Daily loss limit hit for {day}, blocking further trades until next day.')
            block_trading = True
    # Log summary
    logger.info(f'Trades: {len(trades)}, Final PnL: {cash-cfg.init_cash:.2f}')
    with open('logs/paper_run.jsonl', 'w') as f:
        for t in trades:
            f.write(str(t) + '\n')

if __name__ == '__main__':
    main()
