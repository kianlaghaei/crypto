import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger
from src.utils.io import load_yaml
from src.strategies.ema_cross import build_signals_ema_cross
from src.strategies.bb_meanrev import build_signals_bb_meanrev
import matplotlib.pyplot as plt
from src.utils.config import Config


STRATEGY_BUILDERS = {
    "ema_cross": build_signals_ema_cross,
    "bb_meanrev": build_signals_bb_meanrev,
}


def run(cfg_path: str, strategy: str, outdir: str | Path):
    try:
        raw_cfg: dict = load_yaml(cfg_path)
        cfg: Config = Config.model_validate(raw_cfg)
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        raise
    import itertools
    results = []
    for sym in cfg["symbols"]:
        timeframe = cfg["timeframe"]
        data_path = Path("data/raw") / f"{sym.replace('/','-')}_{timeframe}.csv"
        if not data_path.exists():
            logger.warning(f"Missing data file: {data_path}. Run fetch_coinex.py first.")
            continue
        df = pd.read_csv(data_path, parse_dates=["datetime"]).set_index("datetime")
        close = df["close"].astype(float)
        s_cfg = cfg["strategies"][strategy]
        daily_loss_limit = cfg.get("daily_loss_limit_pct", 0.02)
        max_trades_per_day = cfg.get("max_trades_per_day", 20)
        # Grid search
        if strategy == "ema_cross":
            grid = itertools.product(s_cfg["fast_windows"], s_cfg["slow_windows"], [s_cfg.get("sl_stop_pct", 0.02)], [s_cfg.get("tp_stop_pct", 0.04)])
            for fast, slow, sl, tp in grid:
                entries, exits = STRATEGY_BUILDERS[strategy](close, [fast], [slow])
                # Apply constraints
                entries_masked = entries.copy()
                # Daily loss and max trades constraints
                trade_dates = entries_masked.index.date
                trade_count = {}
                daily_loss = {}
                cash = cfg["init_cash"]
                blocked = set()
                for idx in entries_masked.index:
                    d = idx.date()
                    if d not in trade_count:
                        trade_count[d] = 0
                        daily_loss[d] = 0.0
                    if trade_count[d] >= max_trades_per_day or d in blocked:
                        entries_masked.loc[idx] = False
                        continue
                    if entries_masked.loc[idx]:
                        trade_count[d] += 1
                        # Simulate a trade for loss (approx)
                        # For simplicity, assume 1% loss per trade
                        daily_loss[d] += -0.01 * cash
                        if abs(daily_loss[d]) > daily_loss_limit * cash:
                            blocked.add(d)
                            entries_masked.loc[idx] = False
                pf = vbt.Portfolio.from_signals(
                    close, entries_masked, exits,
                    fees=cfg["fees_bps"] / 1e4,
                    slippage=cfg["slippage_bps"] / 1e4,
                    sl_stop=sl,
                    tp_stop=tp,
                    init_cash=float(cfg["init_cash"]),
                    freq=cfg["timeframe"],
                )
                stats = pf.stats()
                results.append({
                    "symbol": sym,
                    "strategy": strategy,
                    "params": f"fast={fast},slow={slow},sl={sl},tp={tp}",
                    "total_return": stats.get("Total Return [%]", 0),
                    "max_drawdown": stats.get("Max Drawdown [%]", 0),
                    "sharpe": stats.get("Sharpe Ratio", 0),
                    "winrate": stats.get("Win Rate [%]", 0),
                    "profit_factor": stats.get("Profit Factor", 0),
                    "daily_loss_limit_pct": daily_loss_limit,
                    "max_trades_per_day": max_trades_per_day,
                })
        elif strategy == "bb_meanrev":
            grid = itertools.product(s_cfg["window_list"], s_cfg["k_list"], [s_cfg.get("sl_stop_pct", 0.015)], [s_cfg.get("tp_stop_pct", 0.02)])
            for window, k, sl, tp in grid:
                entries, exits = STRATEGY_BUILDERS[strategy](close, [window], [k])
                # Apply constraints
                entries_masked = entries.copy()
                trade_dates = entries_masked.index.date
                trade_count = {}
                daily_loss = {}
                cash = cfg["init_cash"]
                blocked = set()
                for idx in entries_masked.index:
                    d = idx.date()
                    if d not in trade_count:
                        trade_count[d] = 0
                        daily_loss[d] = 0.0
                    if trade_count[d] >= max_trades_per_day or d in blocked:
                        entries_masked.loc[idx] = False
                        continue
                    if entries_masked.loc[idx]:
                        trade_count[d] += 1
                        daily_loss[d] += -0.01 * cash
                        if abs(daily_loss[d]) > daily_loss_limit * cash:
                            blocked.add(d)
                            entries_masked.loc[idx] = False
                pf = vbt.Portfolio.from_signals(
                    close, entries_masked, exits,
                    fees=cfg["fees_bps"] / 1e4,
                    slippage=cfg["slippage_bps"] / 1e4,
                    sl_stop=sl,
                    tp_stop=tp,
                    init_cash=float(cfg["init_cash"]),
                    freq=cfg["timeframe"],
                )
                stats = pf.stats()
                results.append({
                    "symbol": sym,
                    "strategy": strategy,
                    "params": f"window={window},k={k},sl={sl},tp={tp}",
                    "total_return": stats.get("Total Return [%]", 0),
                    "max_drawdown": stats.get("Max Drawdown [%]", 0),
                    "sharpe": stats.get("Sharpe Ratio", 0),
                    "winrate": stats.get("Win Rate [%]", 0),
                    "profit_factor": stats.get("Profit Factor", 0),
                    "daily_loss_limit_pct": daily_loss_limit,
                    "max_trades_per_day": max_trades_per_day,
                })
        else:
            raise ValueError("Unknown strategy")
    # Save grid results
    from datetime import datetime
    import json
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = Path(outdir) / f"{strategy}_grid_{cfg["timeframe"]}_{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    # Save params.json
    try:
        import pkg_resources
        versions = {p.key: p.version for p in pkg_resources.working_set}
    except Exception:
        versions = {}
    params = {
        "run_id": run_id,
        "strategy": strategy,
        "symbols": cfg["symbols"],
        "timeframe": cfg["timeframe"],
        "config": raw_cfg,
        "package_versions": versions,
    }
    with open(outdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    df_results = pd.DataFrame(results)
    df_results.to_csv(outdir / "grid_results.csv", index=False)
    logger.info(f"Saved grid results to {outdir / 'grid_results.csv'}")

    # HTML report
    import io
    best_row = df_results.sort_values("total_return", ascending=False).iloc[0] if not df_results.empty else None
    fig, ax = plt.subplots()
    if not df_results.empty:
        df_results["total_return"].plot(ax=ax)
        ax.set_title("Total Return for Grid")
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="svg", bbox_inches="tight")
        img_buf.seek(0)
        svg_data = img_buf.getvalue().decode()
    else:
        svg_data = "<p>No results</p>"
    html = f"""
    <html><head><title>Backtest Report</title></head><body>
    <h2>Grid Results</h2>
    {df_results.to_html(index=False)}
    <h2>Best Parameters</h2>
    <pre>{best_row.to_dict() if best_row is not None else 'No results'}</pre>
    <h2>Equity Chart</h2>
    {svg_data}
    <h3>Note</h3>
    <p>If a candle hits both SL and TP, the engine assumes SL is triggered first (pessimistic same-bar policy).</p>
    </body></html>
    """
    with open(outdir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved HTML report to {outdir / 'report.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--strategy", choices=list(STRATEGY_BUILDERS.keys()), required=True)
    ap.add_argument("--out", default="out/backtests")
    args = ap.parse_args()
    from datetime import datetime
    log_path = Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    logger.add(log_path, format="{{'level': '{level}', 'ts': '{time:YYYY-MM-DD HH:mm:ss}', 'msg': '{message}', 'module': '{module}'}}", serialize=True)
    run(args.cfg, args.strategy, args.out)

if __name__ == "__main__":
    main()
