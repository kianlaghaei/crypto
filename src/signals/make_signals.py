# src/signals/make_signals.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from src.utils.io import load_yaml
from src.utils.config import Config
from src.strategies.ema_cross import build_signals_ema_cross
from src.strategies.bb_meanrev import build_signals_bb_meanrev


def _load_price(symbol: str, timeframe: str) -> pd.Series:
    pq = Path("data/parquet") / f"{symbol.replace('/','-')}_{timeframe}.parquet"
    csv = Path("data/raw") / f"{symbol.replace('/','-')}_{timeframe}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
        if "datetime" in df.columns:
            idx = pd.to_datetime(df["datetime"], utc=True)
        elif "timestamp" in df.columns:
            idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
        else:
            raise ValueError("Parquet must contain datetime/timestamp or have DatetimeIndex")
        close = pd.Series(df["close"].astype(float).to_numpy(), index=idx, name="close")
        return close
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["datetime"]).set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df["close"].astype(float)
    raise FileNotFoundError(f"Price not found for {symbol} {timeframe}. Run fetcher first.")


def _as_1d_series(x: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize entries/exits to a 1D boolean Series."""
    if isinstance(x, pd.Series):
        return x.astype(bool)
    # DataFrame: اگر چندستونه است، ستون اول را برمی‌داریم
    if x.shape[1] == 1:
        return x.iloc[:, 0].astype(bool)
    # اگر چند ستون دارد (پارامترهای گرید)، منطقی‌ترین کار انتخاب ستون اول است
    return x.iloc[:, 0].astype(bool)


def make_signals(cfg: Config, strategy: str, symbol: str, outdir: Path, **params) -> Path:
    timeframe = cfg.timeframe
    close = _load_price(symbol, timeframe)

    if strategy == "ema_cross":
        fast = int(params.get("fast", 30))
        slow = int(params.get("slow", 40))
        entries, exits = build_signals_ema_cross(close, [fast], [slow])
        tag = f"{symbol.replace('/','-')}_{timeframe}_ema_f{fast}_s{slow}"
    elif strategy == "bb_meanrev":
        window = int(params.get("window", 20))
        k = float(params.get("k", 2.0))
        entries, exits = build_signals_bb_meanrev(close, [window], [k])
        tag = f"{symbol.replace('/','-')}_{timeframe}_bb_w{window}_k{k}"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # نرمال‌سازی به سری ۱بعدی
    entries_s = _as_1d_series(entries)
    exits_s = _as_1d_series(exits)

    # همترازی ایندکس‌ها و قیمت
    idx = entries_s.index.union(exits_s.index).union(close.index)
    entries_s = entries_s.reindex(idx, fill_value=False)
    exits_s = exits_s.reindex(idx, fill_value=False)
    close_aligned = close.reindex(idx).astype(float)

    # ساخت دیتافریم خروجی به فرمت entry/exit/close
    sig = pd.DataFrame(
        {
            "datetime": idx,
            "entry": entries_s.astype(int).values,
            "exit": exits_s.astype(int).values,
            "close": close_aligned.values,
        }
    ).dropna(subset=["close"]).sort_values("datetime")

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{tag}.csv"
    sig.to_csv(out_csv, index=False)
    logger.info(f"Saved signals → {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Generate signals CSV for paper/backtest")
    ap.add_argument("--cfg", required=True, help="Path to config.yaml")
    ap.add_argument("--strategy", choices=["ema_cross", "bb_meanrev"], required=True)
    ap.add_argument("--symbol", required=True, help="e.g. ETH/USDT")
    ap.add_argument("--outdir", default="out/signals")
    # strategy params
    ap.add_argument("--fast", type=int)
    ap.add_argument("--slow", type=int)
    ap.add_argument("--window", type=int)
    ap.add_argument("--k", type=float)
    args = ap.parse_args()

    raw = load_yaml(args.cfg)
    cfg = Config.model_validate(raw)

    out_csv = make_signals(
        cfg,
        args.strategy,
        args.symbol,
        Path(args.outdir),
        fast=args.fast,
        slow=args.slow,
        window=args.window,
        k=args.k,
    )
    print(out_csv)


if __name__ == "__main__":
    # اجرا از ریشه‌ی پروژه:
    #   python -m src.signals.make_signals --cfg config/config.yaml --strategy ema_cross --symbol ETH/USDT --fast 30 --slow 40
    main()
