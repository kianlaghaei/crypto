# src/data/fetch_coinex.py
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import ccxt
import pandas as pd
from loguru import logger

from src.utils.io import load_yaml
from src.utils.config import Config


def fetch_ohlcv(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
    retry_wait_sec: int = 2,
    max_iters: int = 500,
    rate_limit_ms: int | None = None,
) -> List[list]:
    """
    Fetch OHLCV rows (ts, o, h, l, c, v) from CoinEx via ccxt with retry/backoff and hard iteration cap.
    Respects exchange rateLimit or an explicit rate_limit_ms if provided.
    """
    rows: List[list] = []
    since = start_ms
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    iters = 0

    while iters < max_iters:
        # retry with exponential backoff
        batch = None
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                logger.warning(f"Network/Exchange error: {e} | retry {attempt + 1}/{max_retries}")
                time.sleep(retry_wait_sec * (2**attempt))
        if batch is None:
            logger.error(f"Max retries exceeded for {symbol} {timeframe}")
            break

        if not batch:
            break

        rows.extend(batch)
        since = batch[-1][0] + tf_ms

        if len(batch) < limit:
            # no more data right now
            break

        sleep_ms = rate_limit_ms if rate_limit_ms is not None else getattr(ex, "rateLimit", 200)
        time.sleep(sleep_ms / 1000)
        iters += 1

    return rows


def main():
    ap = argparse.ArgumentParser(description="Fetch OHLCV from CoinEx and store to Parquet with resume")
    ap.add_argument("--cfg", required=True, help="Path to YAML config file (config/config.yaml)")
    ap.add_argument("--outdir", default="data/raw", help="(Optional) write raw CSVs here (for debugging)")
    ap.add_argument("--parquet-dir", default="data/parquet", help="Directory to store Parquet files")
    args = ap.parse_args()

    # Load & validate config
    raw_cfg = load_yaml(args.cfg)
    cfg = Config.model_validate(raw_cfg)

    # Ensure output dirs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pq_dir = Path(args.parquet_dir)
    pq_dir.mkdir(parents=True, exist_ok=True)

    # Structured logs
    logdir = Path("logs")
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"fetch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.jsonl"
    logger.add(log_path, serialize=True, level="INFO", enqueue=True)

    # Exchange
    ex = ccxt.coinex({"enableRateLimit": True})
    ex.load_markets()

    timeframe = cfg.timeframe
    start_days = int(cfg.start_days)
    start_ms = ex.milliseconds() - start_days * 24 * 60 * 60 * 1000
    rate_limit_ms = getattr(cfg, "rate_limit_ms", None)

    for sym in cfg.symbols:
        if sym not in ex.symbols:
            logger.warning(f"Symbol not in CoinEx markets: {sym}")
            continue

        logger.info(f"Fetching {sym} {timeframe} for ~{start_days} days…")

        # Resume from existing parquet
        pq_path = pq_dir / f"{sym.replace('/', '-')}_{timeframe}.parquet"
        last_ts: int | None = None
        if pq_path.exists():
            try:
                prev_df = pd.read_parquet(pq_path)
                if not prev_df.empty and "timestamp" in prev_df:
                    last_ts = int(prev_df["timestamp"].max())
                    logger.info(f"Resuming from timestamp {last_ts} for {sym}")
            except Exception as e:
                logger.warning(f"Failed to read existing parquet ({pq_path}): {e}")

        fetch_start = last_ts + 1 if last_ts else start_ms

        rows = fetch_ohlcv(
            ex,
            sym,
            timeframe,
            fetch_start,
            limit=1000,
            max_retries=5,
            retry_wait_sec=2,
            max_iters=500,
            rate_limit_ms=rate_limit_ms,
        )
        if not rows:
            logger.warning(f"No new data for {sym}")
            continue

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)

        # Integrity checks
        bad_high_low = int((df["high"] < df["low"]).sum())
        bad_volume = int((df["volume"] < 0).sum())
        nan_count = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
        if bad_high_low > 0:
            logger.warning(f"{bad_high_low} rows with high < low")
        if bad_volume > 0:
            logger.warning(f"{bad_volume} rows with negative volume")
        if nan_count > 0:
            logger.warning(f"{nan_count} NaN values in OHLCV columns")

        # Time gap reporting (very rough heuristic)
        ts_sorted = df["timestamp"].sort_values()
        diffs = ts_sorted.diff().dropna()
        if not diffs.empty:
            median_gap = diffs.median()
            gaps = int((diffs > 2 * median_gap).sum())
            if gaps > 0:
                logger.warning(f"{gaps} time gaps detected (threshold: 2×median interval)")

        # Downcast numeric columns for memory efficiency
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Optional raw CSV (debug)
        raw_csv = outdir / f"{sym.replace('/', '-')}_{timeframe}.csv"
        try:
            df.reset_index().to_csv(raw_csv, index=False)
        except Exception as e:
            logger.warning(f"Failed to write raw CSV ({raw_csv}): {e}")

        # Merge with existing Parquet (dedup on timestamp)
        if pq_path.exists():
            try:
                prev_df = pd.read_parquet(pq_path)
                df = (
                    pd.concat([prev_df, df], axis=0)
                    .drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                )
            except Exception as e:
                logger.warning(f"Failed to merge with existing parquet: {e}")

        # Save Parquet
        df.to_parquet(pq_path, compression="zstd")
        logger.info(f"Saved {len(df):,} rows → {pq_path}")


if __name__ == "__main__":
    # Run as a module from project root:
    #   python -m src.data.fetch_coinex --cfg config/config.yaml
    main()
