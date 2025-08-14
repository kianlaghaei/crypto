# src/scripts/run_event_from_config.py
from __future__ import annotations
import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.engine.event_backtester import EventBacktester
from src.strategies.event.atr_ema_cross import build_signals_ema_cross
from src.utils.io import load_yaml
from src.utils.config import Config


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    pq = Path("data/parquet") / f"{symbol.replace('/','-')}_{timeframe}.parquet"
    csv = Path("data/raw") / f"{symbol.replace('/','-')}_{timeframe}.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Missing data for {symbol} {timeframe}. Fetch first.")
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index(dt)
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index(dt)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV must have datetime/timestamp")
    df = df.sort_index()
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    keep = needed + (["volume"] if "volume" in df.columns else [])
    return df[keep].astype(float)


def main():
    ap = argparse.ArgumentParser(
        description="Event-driven backtest (percent SL/TP) from YAML config"
    )
    ap.add_argument(
        "--cfg", required=True, help="Path to config.yaml (like the snippet you sent)"
    )
    ap.add_argument("--out", default="out/event_backtests", help="Output root folder")
    args = ap.parse_args()

    raw = load_yaml(args.cfg)
    cfg = Config.model_validate(raw)

    symbols = cfg.symbols
    timeframe = cfg.timeframe
    fees_bps = float(cfg.fees_bps)
    slippage_bps = float(cfg.slippage_bps)
    init_cash = float(cfg.init_cash)
    daily_loss_limit_pct = cfg.daily_loss_limit_pct
    max_trades_per_day = cfg.max_trades_per_day
    risk_pct = float(raw.get("risk_pct_per_trade", 0.01))  # optional in YAML

    strat_cfg = cfg.strategies.ema_cross
    fast_list = list(strat_cfg.fast_windows)
    slow_list = list(strat_cfg.slow_windows)
    sl_stop_pct = float(strat_cfg.sl_stop_pct)
    tp_stop_pct = float(strat_cfg.tp_stop_pct)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) / f"ema_event_grid_{timeframe}_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = {"sharpe": -1e9, "dir": None, "row": None}

    for sym in symbols:
        df = load_ohlcv(sym, timeframe)

        for fwin in fast_list:
            for swin in slow_list:
                # سیگنال‌ها
                entries, exits = build_signals_ema_cross(df["close"], fwin, swin)

                # بک‌تستر
                bt = EventBacktester(
                    df=df,
                    entries=entries,
                    exits=exits,
                    init_cash=init_cash,
                    fees_bps=fees_bps,
                    slippage_bps=slippage_bps,
                    risk_pct_per_trade=risk_pct,
                    stop_mode="percent",
                    sl_pct=sl_stop_pct,
                    tp_pct=tp_stop_pct,
                    max_trades_per_day=max_trades_per_day,
                    daily_loss_limit_pct=daily_loss_limit_pct,
                    pessimistic_same_bar=True,
                )
                bt.run()

                # ذخیره خروجی‌های هر گرید در پوشه اختصاصی
                leaf = f"{sym.replace('/','-')}_f{fwin}_s{swin}"
                leaf_dir = outdir / leaf
                bt.save_outputs(leaf_dir)

                k = bt.kpis()
                row = {
                    "symbol": sym,
                    "strategy": "ema_cross_event",
                    "params": f"fast={fwin},slow={swin},sl%={sl_stop_pct},tp%={tp_stop_pct}",
                    **k,
                }
                rows.append(row)

                if pd.notna(k["sharpe"]) and k["sharpe"] > best["sharpe"]:
                    best = {"sharpe": k["sharpe"], "dir": leaf_dir, "row": row}

    # خلاصه گرید
    dfres = pd.DataFrame(rows)
    dfres.to_csv(outdir / "grid_results.csv", index=False)

    # گزارش HTML ساده
    if not dfres.empty:
        best_row = dfres.sort_values("sharpe", ascending=False).iloc[0].to_dict()
        # نمودار بازده کل روی گرید (به ترتیب سطرها)
        fig, ax = plt.subplots(figsize=(9, 3))
        dfres["total_return"].reset_index(drop=True).plot(ax=ax)
        ax.set_title("Total Return across grid rows")
        ax.set_xlabel("Row #")
        ax.set_ylabel("Total Return")
        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight")
        buf.seek(0)
        svg = buf.getvalue().decode()
    else:
        best_row = {}
        svg = "<p>No results</p>"

    html = f"""
    <html><head><meta charset="utf-8"><title>Event Backtest Report</title></head><body>
    <h2>Grid Results</h2>
    {dfres.to_html(index=False)}
    <h2>Best Parameters (by Sharpe)</h2>
    <pre>{json.dumps(best_row, indent=2)}</pre>
    <h2>Total Return Chart (per grid row)</h2>
    {svg}
    <h3>Notes</h3>
    <ul>
      <li>Signals executed at next open; pessimistic same-bar policy (SL before TP).</li>
      <li>Fees (taker) = {fees_bps} bps per side, slippage = {slippage_bps} bps per trade.</li>
      <li>Risk per trade = {risk_pct*100:.2f}% of equity.</li>
    </ul>
    </body></html>
    """
    (outdir / "report.html").write_text(html, encoding="utf-8")

    # پارامترها برای ثبت
    params = {
        "run_id": run_id,
        "timeframe": timeframe,
        "symbols": symbols,
        "config_used": raw,
    }
    (outdir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print(f"Saved event-driven grid backtest → {outdir}")
    if best["dir"] is not None:
        print(f"Best by Sharpe: {best['row']}")
        print(f"Best outputs folder: {best['dir']}")


if __name__ == "__main__":
    main()
