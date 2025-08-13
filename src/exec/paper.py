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
    daily_loss = 0
    trades = []
    for i, row in df.iterrows():
        # Simulate trade
        price = row['close']
        pnl = (row.get('side', 1) * (row.get('exit_price', price) - price)) - cfg.fees_bps/1e4 * price
        cash += pnl
        trades.append({'ts': row['timestamp'], 'price': price, 'pnl': pnl})
        daily_loss += min(0, pnl)
        if abs(daily_loss) > args.max_daily_loss * cfg.init_cash:
            logger.warning('Daily loss limit hit, stopping simulation.')
            break
    # Log summary
    logger.info(f'Trades: {len(trades)}, Final PnL: {cash-cfg.init_cash:.2f}')
    with open('logs/paper_run.jsonl', 'w') as f:
        for t in trades:
            f.write(str(t) + '\n')

if __name__ == '__main__':
    main()
