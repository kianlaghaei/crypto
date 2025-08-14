# src/backtest/run_backtest.py
import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from loguru import logger

from src.utils.io import load_yaml
from src.utils.config import Config
from src.strategies.ema_cross import build_signals_ema_cross
from src.strategies.bb_meanrev import build_signals_bb_meanrev


# ---------- helpers ----------
def apply_max_trades_per_day(
    entries: pd.DataFrame,
    close: pd.Series,
    max_trades_per_day: int | None,
) -> pd.DataFrame:
    """Limit number of entries per UTC day in a vectorized way (broadcast to all columns)."""
    if entries.empty or not max_trades_per_day:
        return entries
    idx = close.index
    day = idx.tz_convert("UTC").normalize()
    first_col = entries.iloc[:, 0].fillna(False)
    daily_counts = first_col.groupby(day).cumsum().fillna(0)
    allowed_mask = (daily_counts <= max_trades_per_day) | daily_counts.isna()
    allowed_df = pd.DataFrame(
        np.tile(allowed_mask.values.reshape(-1, 1), entries.shape[1]),
        index=entries.index,
        columns=entries.columns,
    )
    return entries & allowed_df


def kpis_scalar(pf: vbt.portfolio.base.Portfolio) -> dict:
    """Return scalar KPIs for a single-column portfolio safely."""
    # این توابع در حالت تک‌ستونی اسکالر برمی‌گردانند؛ به float تبدیلشان می‌کنیم
    total_return = float(np.asarray(pf.total_return()).item())
    max_dd = float(np.asarray(pf.max_drawdown()).item())
    sharpe = float(np.asarray(pf.sharpe_ratio()).item())

    # KPI های مبتنی بر معاملات
    t = pf.trades
    wins = int(np.asarray(t.winning.count()).item())
    tot = int(np.asarray(t.count()).item())
    winrate = float((wins / tot) * 100.0) if tot > 0 else float("nan")

    profits = float(np.asarray(t.winning.pnl.sum()).item())
    losses = float(np.asarray(t.losing.pnl.sum()).item())  # منفی
    loss_abs = -losses
    profit_factor = float(profits / loss_abs) if loss_abs > 0 else float("nan")

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "winrate": winrate,
        "profit_factor": profit_factor,
        "trades": tot,
    }


def load_price(sym: str, timeframe: str) -> pd.Series:
    """Load close price, preferring Parquet, fallback to CSV."""
    pq_path = Path("data/parquet") / f"{sym.replace('/','-')}_{timeframe}.parquet"
    csv_path = Path("data/raw") / f"{sym.replace('/','-')}_{timeframe}.csv"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        if "datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["datetime"], utc=True))
        elif "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Parquet missing datetime/timestamp: {pq_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["datetime"]).set_index("datetime")
    else:
        raise FileNotFoundError(f"Missing data for {sym} {timeframe}. Run fetch_coinex.py first.")

    return df["close"].astype(float)


# ---------- main run ----------
def run(cfg_path: str, strategy: str, outdir: str | Path) -> Path:
    raw_cfg: dict = load_yaml(cfg_path)
    cfg: Config = Config.model_validate(raw_cfg)

    # structured logs
    logdir = Path("logs"); logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.jsonl"
    logger.add(log_path, serialize=True, level="INFO", enqueue=True)

    results_rows = []
    best_by_sharpe = {"sharpe": -1e9, "pf": None, "sym": None, "label": None}

    for sym in cfg.symbols:
        timeframe = cfg.timeframe
        try:
            close = load_price(sym, timeframe)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        fees = cfg.fees_bps / 1e4
        slip = cfg.slippage_bps / 1e4
        init_cash = float(cfg.init_cash)

        if strategy == "ema_cross":
            s = cfg.strategies.ema_cross
            fast_ws = list(s.fast_windows)
            slow_ws = list(s.slow_windows)
            sl = float(s.sl_stop_pct); tp = float(s.tp_stop_pct)
            max_tpd = getattr(s, "max_trades_per_day", None)

            # برای هر ترکیب جداگانه سیگنال و بک‌تست بساز؛ پایدار برای هر شکل گرید
            for fwin in fast_ws:
                for swin in slow_ws:
                    # سیگنال‌ها (1 ستونه)
                    fast = vbt.MA.run(close, window=fwin).ma
                    slow = vbt.MA.run(close, window=swin).ma
                    entries = fast.vbt.crossed_above(slow).to_frame()
                    exits = fast.vbt.crossed_below(slow).to_frame()

                    # قید معاملات روزانه
                    entries = apply_max_trades_per_day(entries, close, max_tpd)

                    # پورتفوی
                    pf = vbt.Portfolio.from_signals(
                        close, entries, exits,
                        fees=fees, slippage=slip,
                        sl_stop=sl, tp_stop=tp,
                        init_cash=init_cash, freq=timeframe,
                    )

                    k = kpis_scalar(pf)
                    results_rows.append({
                        "symbol": sym,
                        "strategy": "ema_cross",
                        "params": f"fast={fwin},slow={swin},sl={sl},tp={tp}",
                        **k,
                    })

                    if np.isfinite(k["sharpe"]) and k["sharpe"] > best_by_sharpe["sharpe"]:
                        best_by_sharpe = {"sharpe": k["sharpe"], "pf": pf, "sym": sym,
                                          "label": f"EMA fast={fwin} slow={swin}"}

        elif strategy == "bb_meanrev":
            s = cfg.strategies.bb_meanrev
            ws = list(s.window_list)
            ks = list(s.k_list)
            sl = float(s.sl_stop_pct); tp = float(s.tp_stop_pct)
            max_tpd = getattr(s, "max_trades_per_day", None)

            for win in ws:
                for kval in ks:
                    bb = vbt.BBANDS.run(close, window=win, alpha=kval)
                    entries = close.vbt.crossed_below(bb.lower).to_frame()
                    exits = close.vbt.crossed_above(bb.middle).to_frame()

                    entries = apply_max_trades_per_day(entries, close, max_tpd)

                    pf = vbt.Portfolio.from_signals(
                        close, entries, exits,
                        fees=fees, slippage=slip,
                        sl_stop=sl, tp_stop=tp,
                        init_cash=init_cash, freq=timeframe,
                    )

                    k = kpis_scalar(pf)
                    results_rows.append({
                        "symbol": sym,
                        "strategy": "bb_meanrev",
                        "params": f"window={win},k={kval},sl={sl},tp={tp}",
                        **k,
                    })

                    if np.isfinite(k["sharpe"]) and k["sharpe"] > best_by_sharpe["sharpe"]:
                        best_by_sharpe = {"sharpe": k["sharpe"], "pf": pf, "sym": sym,
                                          "label": f"BB win={win} k={kval}"}
        else:
            raise ValueError("Unknown strategy")

    # Outputs
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = Path(outdir) / f"{strategy}_grid_{cfg.timeframe}_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(outdir / "grid_results.csv", index=False)

    # params.json
    try:
        import importlib.metadata as imd
        versions = {p: imd.version(p) for p in ["vectorbt", "pandas", "numpy", "numba"]}
    except Exception:
        versions = {}
    params = {
        "run_id": run_id,
        "strategy": strategy,
        "symbols": cfg.symbols,
        "timeframe": cfg.timeframe,
        "config": raw_cfg,
        "package_versions": versions,
    }
    (outdir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    # Simple HTML report
    if not df_results.empty:
        best_row = df_results.sort_values("sharpe", ascending=False).iloc[0]
        # plot distribution of total_return across grid
        fig, ax = plt.subplots()
        df_results["total_return"].plot(ax=ax)
        ax.set_title("Total Return across Grid")
        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight"); buf.seek(0)
        svg_data = buf.getvalue().decode()
    else:
        best_row = None
        svg_data = "<p>No results</p>"

    html = f"""
    <html><head><meta charset="utf-8"><title>Backtest Report</title></head><body>
    <h2>Grid Results</h2>
    {df_results.to_html(index=False)}
    <h2>Best Parameters (by Sharpe)</h2>
    <pre>{(best_row.to_dict() if best_row is not None else 'No results')}</pre>
    <h2>Total Return Chart (per grid row)</h2>
    {svg_data}
    <h3>Note</h3>
    <p>If a candle hits both SL and TP, we assume SL is triggered first (pessimistic same-bar policy).</p>
    </body></html>
    """
    (outdir / "report.html").write_text(html, encoding="utf-8")

    # Save best equity chart if available
    if best_by_sharpe["pf"] is not None:
        eq = best_by_sharpe["pf"].value()
        fig = eq.plot(title=f"Best Equity — {best_by_sharpe['label']} ({best_by_sharpe['sym']} {cfg.timeframe})")
        fig.figure.savefig(outdir / "best_equity.png", dpi=150, bbox_inches="tight")
        best_by_sharpe["pf"].save(outdir / "best_portfolio.pkl")

    logger.info(f"Saved grid results & report → {outdir}")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--strategy", choices=["ema_cross", "bb_meanrev"], required=True)
    ap.add_argument("--out", default="out/backtests")
    args = ap.parse_args()

    # Run from project root:
    #   python -m src.backtest.run_backtest --cfg config/config.yaml --strategy ema_cross
    run(args.cfg, args.strategy, args.out)


if __name__ == "__main__":
    main()
