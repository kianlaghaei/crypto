import argparse, time
from pathlib import Path
import ccxt
import pandas as pd
from loguru import logger
from src.utils.io import load_yaml
from src.utils.config import Config


def fetch_ohlcv(ex, symbol: str, timeframe: str, start_ms: int, limit: int = 1000):
    import ccxt
    rows = []
    since = start_ms
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    max_retries = 5
    retry_wait = 2
    while True:
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Network/Exchange error: {e}, retry {attempt+1}/{max_retries}")
                time.sleep(retry_wait * (2 ** attempt))
        else:
            logger.error(f"Max retries exceeded for {symbol} {timeframe}")
            break
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < limit:
            break
        time.sleep(ex.rateLimit/1000)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    raw_cfg = load_yaml(args.cfg)
    try:
        cfg = Config.model_validate(raw_cfg)
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    log_path = Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    logger.add(log_path, format="{{'level': '{level}', 'ts': '{time:YYYY-MM-DD HH:mm:ss}', 'msg': '{message}', 'module': '{module}'}}", serialize=True)

    ex = ccxt.coinex({"enableRateLimit": True})
    ex.load_markets()

    timeframe = cfg["timeframe"]
    start_days = int(cfg["start_days"]) 
    start_ms = ex.milliseconds() - start_days * 24 * 60 * 60 * 1000

    for sym in cfg["symbols"]:
        if sym not in ex.symbols:
            logger.warning(f"Symbol not in CoinEx markets: {sym}")
            continue
        logger.info(f"Fetching {sym} {timeframe} for ~{start_days} days…")
        # Resume: check for existing parquet
        pq_dir = Path("data/parquet")
        pq_dir.mkdir(parents=True, exist_ok=True)
        pq_path = pq_dir / f"{sym.replace('/', '-')}_{timeframe}.parquet"
        last_ts = None
        if pq_path.exists():
            try:
                prev_df = pd.read_parquet(pq_path)
                if not prev_df.empty:
                    last_ts = int(prev_df["timestamp"].max())
                    logger.info(f"Resuming from timestamp {last_ts} for {sym}")
            except Exception as e:
                logger.warning(f"Failed to read existing parquet: {e}")
        fetch_start = last_ts + 1 if last_ts else start_ms
        rows = fetch_ohlcv(ex, sym, timeframe, fetch_start)
        if not rows:
            logger.warning(f"No new data for {sym}")
            continue
        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        # Dedup and append to parquet
        if pq_path.exists():
            try:
                prev_df = pd.read_parquet(pq_path)
                df = pd.concat([prev_df, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            except Exception as e:
                logger.warning(f"Failed to merge with existing parquet: {e}")
        df.to_parquet(pq_path)
        logger.info(f"Saved {len(df):,} rows → {pq_path}")

if __name__ == "__main__":
    main()
