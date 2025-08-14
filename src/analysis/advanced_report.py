# src/analysis/advanced_report.py
import argparse
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import pandas as pd
import vectorbt as vbt

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _ensure_single_column_pf(pf: vbt.portfolio.base.Portfolio) -> vbt.portfolio.base.Portfolio:
    """Make sure Portfolio is single-column (needed by some vectorbt plots)."""
    try:
        cols = list(pf.wrapper.columns)
        if len(cols) == 0:
            return pf
        # pick the first column deterministically
        return pf.select_one(column=cols[0], group_by=False)
    except Exception:
        # Portfolio may already be single-column / lacks columns attr
        try:
            # Some wrappers still require select_one even if single
            return pf.select_one(column=0, group_by=False)
        except Exception:
            return pf


def _try_write_plotly(fig, out_png: Path, out_html: Path) -> Path:
    """Try saving a Plotly figure as PNG (needs kaleido). Fallback to HTML."""
    # Prefer PNG for easy preview, but gracefully fall back to HTML.
    try:
        import plotly.io as pio  # noqa: F401
        fig.write_image(str(out_png), scale=2)  # requires kaleido
        return out_png
    except Exception:
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        return out_html


def _safe_make_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_best_portfolio(run_dir: Path) -> vbt.portfolio.base.Portfolio:
    """Load best_portfolio.pkl (or first *.pkl as fallback)."""
    pf_path = run_dir / "best_portfolio.pkl"
    if pf_path.exists():
        return vbt.Portfolio.load(pf_path)

    # Fallback: first pkl
    pkls = sorted(run_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No portfolio pickle found in: {run_dir}")
    return vbt.Portfolio.load(pkls[0])


def _summarize_kpis(pf: vbt.portfolio.base.Portfolio) -> dict:
    """Return scalar KPIs for a single-column portfolio safely."""
    pf1 = _ensure_single_column_pf(pf)

    # equity & returns based metrics
    total_return = float(np.asarray(pf1.total_return()).item())
    max_dd = float(np.asarray(pf1.max_drawdown()).item())
    sharpe = float(np.asarray(pf1.sharpe_ratio()).item())
    calmar = float(np.asarray(pf1.calmar_ratio()).item()) if hasattr(pf1, "calmar_ratio") else float("nan")

    # trades-based
    t = pf1.trades
    wins = int(np.asarray(t.winning.count()).item())
    tot = int(np.asarray(t.count()).item())
    winrate = float((wins / tot) * 100.0) if tot > 0 else float("nan")
    profits = float(np.asarray(t.winning.pnl.sum()).item())
    losses = float(np.asarray(t.losing.pnl.sum()).item())  # negative
    loss_abs = -losses
    profit_factor = float(profits / loss_abs) if loss_abs > 0 else float("nan")

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "winrate_pct": winrate,
        "profit_factor": profit_factor,
        "trades": tot,
    }


# -------------------------------------------------
# Main report generator
# -------------------------------------------------
def generate_advanced_report(backtest_dir: Path) -> None:
    backtest_dir = backtest_dir.resolve()
    out_plots = backtest_dir / "advanced_plots"
    _safe_make_dir(out_plots)

    # 1) Load portfolio
    pf = _load_best_portfolio(backtest_dir)
    pf_plot = _ensure_single_column_pf(pf)

    # 2) Save KPIs & raw series
    kpis = _summarize_kpis(pf_plot)
    (backtest_dir / "advanced_kpis.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")

    # equity / returns / drawdown series to CSV
    eq = pf_plot.value()
    rets = pf_plot.returns()
    dd = pf_plot.drawdown()
    eq.to_frame("equity").to_csv(backtest_dir / "equity.csv")
    rets.to_frame("returns").to_csv(backtest_dir / "returns.csv")
    dd.to_frame("drawdown").to_csv(backtest_dir / "drawdown.csv")

    # trades to CSV (records table is convenient)
    try:
        pf_plot.trades.records.to_csv(backtest_dir / "trades.csv", index=False)
    except Exception:
        pass  # some versions may differ

    # 3) Figures (Plotly). Try PNG; fallback to HTML.
    # 3.1) Orders/positions over price
    fig_trades = pf_plot.plot()  # plotly Figure with subplots
    _try_write_plotly(
        fig_trades,
        out_plots / "trades_plot.png",
        out_plots / "trades_plot.html",
    )

    # 3.2) PnL distribution
    try:
        fig_pnl = pf_plot.trades.pnl.vbt.hist()
        fig_pnl.update_layout(title="Distribution of Trade PnL")
        _try_write_plotly(
            fig_pnl,
            out_plots / "pnl_distribution.png",
            out_plots / "pnl_distribution.html",
        )
    except Exception:
        # If no trades, skip
        pass

    # 3.3) Monthly returns heatmap
    try:
        fig_month = pf_plot.returns.vbt.plot_monthly_heatmap()
        _try_write_plotly(
            fig_month,
            out_plots / "monthly_returns_heatmap.png",
            out_plots / "monthly_returns_heatmap.html",
        )
    except Exception:
        pass

    # 3.4) Rolling Sharpe (window ~= 4 days for 1h bars -> 96 hours; tweak as needed)
    try:
        # If timeframe is available, you could map to a better window
        fig_roll_sharpe = pf_plot.returns.vbt.rolling_sharpe_ratio(window=96).vbt.plot()
        fig_roll_sharpe.update_layout(title="Rolling Sharpe Ratio (window=96)")
        _try_write_plotly(
            fig_roll_sharpe,
            out_plots / "rolling_sharpe.png",
            out_plots / "rolling_sharpe.html",
        )
    except Exception:
        pass

    # 3.5) Underwater (drawdown) chart
    try:
        fig_underwater = pf_plot.returns.vbt.plot_underwater()
        _try_write_plotly(
            fig_underwater,
            out_plots / "underwater.png",
            out_plots / "underwater.html",
        )
    except Exception:
        pass

    # 3.6) Turnover (optional; may not exist on older versions)
    try:
        fig_turnover = pf_plot.turnover().vbt.plot()
        fig_turnover.update_layout(title="Turnover")
        _try_write_plotly(
            fig_turnover,
            out_plots / "turnover.png",
            out_plots / "turnover.html",
        )
    except Exception:
        pass

    # 4) Tiny HTML index to browse artifacts
    index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Advanced Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 20px; }}
h2 {{ margin-top: 1.2rem; }}
code, pre {{ background: #f6f8fa; padding: 8px; border-radius: 6px; display: block; }}
a {{ text-decoration: none; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
</style>
</head>
<body>
<h1>Advanced Report</h1>

<h2>KPIs</h2>
<pre>{json.dumps(kpis, indent=2)}</pre>

<h2>Artifacts</h2>
<div class="grid">
  <div class="card"><a href="equity.csv">equity.csv</a></div>
  <div class="card"><a href="returns.csv">returns.csv</a></div>
  <div class="card"><a href="drawdown.csv">drawdown.csv</a></div>
  <div class="card"><a href="trades.csv">trades.csv</a></div>
</div>

<h2>Plots</h2>
<div class="grid">
  <div class="card"><a href="advanced_plots/trades_plot.png">trades_plot.png</a> / <a href="advanced_plots/trades_plot.html">html</a></div>
  <div class="card"><a href="advanced_plots/pnl_distribution.png">pnl_distribution.png</a> / <a href="advanced_plots/pnl_distribution.html">html</a></div>
  <div class="card"><a href="advanced_plots/monthly_returns_heatmap.png">monthly_returns_heatmap.png</a> / <a href="advanced_plots/monthly_returns_heatmap.html">html</a></div>
  <div class="card"><a href="advanced_plots/rolling_sharpe.png">rolling_sharpe.png</a> / <a href="advanced_plots/rolling_sharpe.html">html</a></div>
  <div class="card"><a href="advanced_plots/underwater.png">underwater.png</a> / <a href="advanced_plots/underwater.html">html</a></div>
  <div class="card"><a href="advanced_plots/turnover.png">turnover.png</a> / <a href="advanced_plots/turnover.html">html</a></div>
</div>

</body>
</html>
"""
    (backtest_dir / "advanced_index.html").write_text(index_html, encoding="utf-8")
    print(f"Advanced report saved under: {backtest_dir}")
    print("Open advanced_index.html in your browser.")

# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate advanced backtest reports from a run directory.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the backtest output directory (e.g., out/backtests/ema_cross_grid_1h_YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()
    generate_advanced_report(Path(args.dir))


if __name__ == "__main__":
    main()
