# dashboard.py
from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# vectorbt برای رسم نمودارهای ماهانه/شارپ و بارگذاری پورتفولیو
import vectorbt as vbt


# -----------------------------------------------------------
# Configs
# -----------------------------------------------------------
# Resolve repository root so dashboard works regardless of working directory
BACKTEST_ROOT = Path(__file__).resolve().parents[2] / "out" / "backtests"


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_backtest_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    # فقط دایرکتوری‌ها
    return sorted([Path(p) for p in glob.glob(str(root / "*")) if Path(p).is_dir()])

@st.cache_data(show_spinner=False)
def load_grid_results(run_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = run_dir / "grid_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # تلاش برای شفافیت انواع
    for col in ["total_return", "max_drawdown", "sharpe", "winrate", "profit_factor"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df

@st.cache_data(show_spinner=False)
def load_params(run_dir: Path) -> dict:
    p = run_dir / "params.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

@st.cache_data(show_spinner=False)
def load_advanced_kpis(run_dir: Path) -> Optional[dict]:
    p = run_dir / "advanced_kpis.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

@st.cache_resource(show_spinner=False)
def load_best_portfolio(run_dir: Path):
    """برمی‌گرداند vbt.Portfolio یا None."""
    p = run_dir / "best_portfolio.pkl"
    if not p.exists():
        # fallback: اولین pkl
        pkls = sorted(run_dir.glob("*.pkl"))
        if not pkls:
            return None
        p = pkls[0]
    try:
        return vbt.Portfolio.load(p)
    except Exception:
        return None

def ensure_single_col_pf(pf):
    try:
        cols = list(pf.wrapper.columns)
        if len(cols) <= 1:
            return pf
        return pf.select_one(column=cols[0], group_by=False)
    except Exception:
        try:
            return pf.select_one(column=0, group_by=False)
        except Exception:
            return pf

def compute_kpis_from_pf(pf) -> dict:
    pf1 = ensure_single_col_pf(pf)
    total_return = float(np.asarray(pf1.total_return()).item())
    max_dd = float(np.asarray(pf1.max_drawdown()).item())
    sharpe = float(np.asarray(pf1.sharpe_ratio()).item())
    # Trades
    t = pf1.trades
    wins = int(np.asarray(t.winning.count()).item())
    tot = int(np.asarray(t.count()).item())
    winrate = float((wins / tot) * 100.0) if tot > 0 else float("nan")
    profits = float(np.asarray(t.winning.pnl.sum()).item())
    losses = float(np.asarray(t.losing.pnl.sum()).item())  # منفی
    loss_abs = -losses
    profit_factor = float(profits / loss_abs) if loss_abs > 0 else float("nan")
    return dict(
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe=sharpe,
        winrate=winrate,
        profit_factor=profit_factor,
        trades=tot,
    )


# -----------------------------------------------------------
# Page Layout
# -----------------------------------------------------------
st.set_page_config(
    page_title="Crypto Backtest Dashboard",
    layout="wide",
)


st.title("📊 Crypto Backtest Analysis Dashboard")

# Sidebar: select run(s)
st.sidebar.header("Select Backtest Run")
runs = list_backtest_dirs(BACKTEST_ROOT)
if not runs:
    st.info("هیچ پوشه‌ای در out/backtests پیدا نشد. ابتدا یک بک‌تست اجرا کن.")
    st.stop()

run_labels = [f"{p.name}" for p in runs]
selected_label = st.sidebar.selectbox("Backtest run:", run_labels, index=len(run_labels) - 1)
run_dir = runs[run_labels.index(selected_label)]
st.caption(f"Run directory: `{run_dir}`")

tabs = st.tabs(["Overview", "Grid Explorer", "Portfolio & Charts", "Compare Runs"])

# -----------------------------------------------------------
# Tab 1: Overview
# -----------------------------------------------------------
with tabs[0]:
    params = load_params(run_dir)
    kpis = load_advanced_kpis(run_dir)
    pf = load_best_portfolio(run_dir)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Run Parameters")
        if params:
            st.json(params)
        else:
            st.write("params.json یافت نشد.")

    with c2:
        st.subheader("Best KPIs")
        if kpis is None and pf is not None:
            kpis = compute_kpis_from_pf(pf)
        if kpis:
            st.metric("Total Return", f"{kpis.get('total_return', float('nan')):.2%}")
            st.metric("Max Drawdown", f"{kpis.get('max_drawdown', float('nan')):.2%}")
            st.metric("Sharpe", f"{kpis.get('sharpe', float('nan')):.2f}")
            st.metric("Winrate", f"{kpis.get('winrate_pct', kpis.get('winrate', float('nan'))):.2f}%")
            st.metric("Profit Factor", f"{kpis.get('profit_factor', float('nan')):.2f}")
            st.metric("Trades", f"{int(kpis.get('trades', 0))}")
        else:
            st.write("نه KPI ذخیره شده داریم، نه پورتفولیو برای محاسبه.")


# -----------------------------------------------------------
# Tab 2: Grid Explorer
# -----------------------------------------------------------
with tabs[1]:
    st.subheader("Grid Results Explorer")
    df = load_grid_results(run_dir)
    if df is None or df.empty:
        st.warning("grid_results.csv یافت نشد یا خالی است.")
    else:
        # Sidebar-like controls inside tab
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            symbols = ["(All)"] + sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df.columns else ["(All)"]
            sym_sel = st.selectbox("Symbol", symbols, index=0)
        with c2:
            strategies = ["(All)"] + sorted(df["strategy"].dropna().unique().tolist()) if "strategy" in df.columns else ["(All)"]
            strat_sel = st.selectbox("Strategy", strategies, index=0)
        with c3:
            sort_by = st.selectbox("Sort by", ["sharpe", "total_return", "max_drawdown", "winrate", "profit_factor", "trades"])
        with c4:
            ascending = st.checkbox("Ascending", value=False)

        df_f = df.copy()
        if "symbol" in df_f.columns and sym_sel != "(All)":
            df_f = df_f[df_f["symbol"] == sym_sel]
        if "strategy" in df_f.columns and strat_sel != "(All)":
            df_f = df_f[df_f["strategy"] == strat_sel]

        if sort_by in df_f.columns:
            df_f = df_f.sort_values(sort_by, ascending=ascending)

        st.dataframe(df_f, use_container_width=True, height=450)

        # KPI distributions
        st.markdown("#### KPI Distributions")
        kpi_cols = ["sharpe", "total_return", "max_drawdown", "winrate", "profit_factor", "trades"]
        kpi_cols = [c for c in kpi_cols if c in df_f.columns]
        sel_kpi = st.selectbox("KPI", kpi_cols, index=0)

        fig = px.histogram(df_f, x=sel_kpi, nbins=30, title=f"Distribution of {sel_kpi}")
        st.plotly_chart(fig, use_container_width=True)

        # Download filtered CSV
        st.download_button(
            label="⬇️ Download filtered results CSV",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name=f"{run_dir.name}_filtered_grid.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------
# Tab 3: Portfolio & Charts
# -----------------------------------------------------------
with tabs[2]:
    st.subheader("Best Portfolio Charts")
    pf = load_best_portfolio(run_dir)
    if pf is None:
        st.warning("best_portfolio.pkl پیدا نشد (یا امکان بارگذاری نبود).")
    else:
        pf1 = ensure_single_col_pf(pf)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Equity Curve**")
            eq = pf1.value()
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
            fig_eq.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320)
            st.plotly_chart(fig_eq, use_container_width=True)

        with cB:
            st.markdown("**Underwater (Drawdown)**")
            try:
                dd = pf1.returns().vbt.to_drawdown_series()
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
                fig_dd.update_yaxes(tickformat=".0%")
                fig_dd.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320)
                st.plotly_chart(fig_dd, use_container_width=True)
            except Exception as e:
                st.info(f"Drawdown plot skipped: {e}")

        cC, cD = st.columns(2)
        with cC:
            st.markdown("**Rolling Sharpe (window=96)**")
            try:
                roll_sharpe = pf1.returns().vbt.rolling_sharpe_ratio(window=96)
                fig_rs = go.Figure()
                fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, mode="lines", name="Rolling Sharpe"))
                fig_rs.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320)
                st.plotly_chart(fig_rs, use_container_width=True)
            except Exception as e:
                st.info(f"Rolling Sharpe skipped: {e}")

        with cD:
            st.markdown("**Monthly Returns Heatmap**")
            try:
                # از vbt خروجی plotly می‌گیریم
                fig_month = pf1.returns().vbt.plot_monthly_heatmap()
                st.plotly_chart(fig_month, use_container_width=True)
            except Exception as e:
                st.info(f"Monthly heatmap skipped: {e}")

        # Trades PnL distribution
        st.markdown("**Trade PnL Distribution**")
        try:
            pnl = pf1.trades.pnl.to_pd()
            if pnl.size > 0:
                fig_pnl = px.histogram(pnl, nbins=40, title="Trade PnL")
                st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("No trades found.")
        except Exception as e:
            st.info(f"PnL distribution skipped: {e}")

        # Raw tables expanders
        with st.expander("Raw Trades Table"):
            try:
                st.dataframe(pf1.trades.records, use_container_width=True, height=360)
            except Exception:
                st.write("Trades table not available in this vectorbt version.")


# -----------------------------------------------------------
# Tab 4: Compare Runs
# -----------------------------------------------------------
with tabs[3]:
    st.subheader("Compare Multiple Runs")
    labels_multi = st.multiselect(
        "Select runs to compare",
        run_labels,
        default=[selected_label],
    )
    rows = []
    for lbl in labels_multi:
        rd = runs[run_labels.index(lbl)]
        # KPI from advanced file or compute from pf
        k = load_advanced_kpis(rd)
        if k is None:
            pf_cmp = load_best_portfolio(rd)
            if pf_cmp is not None:
                k = compute_kpis_from_pf(pf_cmp)
        if k is None:
            continue
        row = {
            "run": lbl,
            "total_return": k.get("total_return", np.nan),
            "max_drawdown": k.get("max_drawdown", np.nan),
            "sharpe": k.get("sharpe", np.nan),
            "winrate": k.get("winrate_pct", k.get("winrate", np.nan)),
            "profit_factor": k.get("profit_factor", np.nan),
            "trades": k.get("trades", np.nan),
        }
        # enrich with strategy/timeframe/symbols from params if available
        params_i = load_params(rd)
        if params_i:
            row["strategy"] = params_i.get("strategy", "")
            row["timeframe"] = params_i.get("timeframe", "")
            syms = params_i.get("symbols", [])
            row["symbols"] = ", ".join(syms) if isinstance(syms, list) else str(syms)
        rows.append(row)

    if rows:
        df_cmp = pd.DataFrame(rows)
        df_cmp = df_cmp.sort_values("sharpe", ascending=False)
        st.dataframe(df_cmp, use_container_width=True, height=420)

        # small bar chart by Sharpe
        fig_cmp = px.bar(
            df_cmp,
            x="run",
            y="sharpe",
            color="strategy" if "strategy" in df_cmp.columns else None,
            title="Sharpe by Run",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.download_button(
            "⬇️ Download comparison CSV",
            data=df_cmp.to_csv(index=False).encode("utf-8"),
            file_name="runs_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("هیچ ران معتبری برای مقایسه پیدا نشد.")
