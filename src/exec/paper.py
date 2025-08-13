# src/exec/paper.py
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from loguru import logger

from src.utils.io import load_yaml
from src.utils.config import Config


@dataclass
class Trade:
    dt: pd.Timestamp
    day: str
    side: Literal[1, -1]            # 1=long, -1=short (اگر short نداری همیشه 1)
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    cash_after: float


def _ensure_logs():
    logdir = Path("logs")
    logdir.mkdir(parents=True, exist_ok=True)
    logger.add(logdir / "paper_{time}.jsonl", serialize=True, level="INFO", enqueue=True)
    return logdir


def _read_signals_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Signals file not found: {p}")
    df = pd.read_csv(p)

    # انعطاف در اسکیمای سیگنال‌ها:
    # حالت A: ستون‌های entry/exit باینری + close (قیمت)
    # حالت B: ستون‌های side (+1/-1)، price (ورود) و exit_price (خروج)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True)
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError("Signals must contain 'datetime' or 'timestamp' column.")

    df.index = dt
    df = df.sort_index()

    # تلاش برای نرمال‌سازی سیگنال‌ها به (side, entry_price, exit_price)
    if {"side", "price", "exit_price"}.issubset(df.columns):
        # حالت B
        out = pd.DataFrame(index=df.index)
        out["side"] = df["side"].astype(int).clip(-1, 1).replace(0, 1)  # اگر 0 بود، 1 بگیر
        out["entry_price"] = df["price"].astype(float)
        out["exit_price"] = df["exit_price"].astype(float)
        return out

    if {"entry", "exit", "close"}.issubset(df.columns):
        # حالت A: long-only با 1 واحد حجم
        # هر بار entry==True و exit==True را به یک «معاملهٔ یک‌کَندلی» تبدیل می‌کنیم
        out = pd.DataFrame(index=df.index)
        out["side"] = (df["entry"] > 0).astype(int)  # فقط Long
        out["entry_price"] = df["close"].astype(float)
        # اگر exit == True در همان کندل، از همان close به عنوان خروج استفاده می‌کنیم
        out["exit_price"] = out["entry_price"].where(df["exit"] > 0, other=out["entry_price"])
        # فقط ردیف‌هایی که entry==True هستند را نگه داریم
        out = out[out["side"] == 1]
        return out

    raise ValueError("Unknown signals schema. Provide either (side, price, exit_price) or (entry, exit, close).")


def _apply_daily_loss_guard(trades_df: pd.DataFrame, init_cash: float, daily_loss_limit_pct: Optional[float]) -> pd.DataFrame:
    """
    ماسک‌کردن معاملات پس از برخورد با سقف ضرر روزانه:
    - trades_df باید شامل net_pnl و index زمانی باشد.
    - اگر daily_loss_limit_pct=None یا 0، بدون تغییر برمی‌گردیم.
    """
    if not daily_loss_limit_pct or daily_loss_limit_pct <= 0:
        return trades_df

    day = trades_df.index.tz_convert("UTC").normalize()
    daily_cum_loss = trades_df["net_pnl"].where(trades_df["net_pnl"] < 0, 0).groupby(day).cumsum()
    limit = -abs(daily_loss_limit_pct) * init_cash

    # مجاز: تا زمانی که مجموع زیان روزانه > limit نشده
    allowed = daily_cum_loss >= limit  # چون هر دو منفی‌اند، «بزرگ‌تر مساوی» یعنی هنوز نگذشته
    # وقتی از حد گذشت، باقی معاملات همان روز حذف شوند:
    # اجازه بده اولین عبور ثبت شود، بقیه‌ی همان روز False
    # تبدیل allowed به ماسکی که بعد از اولین False همان روز، همه False شود
    masked = []
    current_day = None
    blocked = False
    for ts, ok in allowed.items():
        d = ts.normalize()
        if current_day is None or d != current_day:
            current_day = d
            blocked = False
        if blocked:
            masked.append(False)
        else:
            masked.append(bool(ok))
            if not ok:
                blocked = True
    trades_df = trades_df[masked]
    return trades_df


def main():
    ap = argparse.ArgumentParser(description="Simple paper trading simulator on precomputed signals")
    ap.add_argument("--signals", required=True, help="Path to signals CSV")
    ap.add_argument("--cfg", required=True, help="Path to config.yaml")
    ap.add_argument("--qty", type=float, default=1.0, help="Position size per trade (units)")
    ap.add_argument("--daily_loss_limit_pct", type=float, default=None, help="Stop trading for the day after this % loss of init cash")
    args = ap.parse_args()

    logdir = _ensure_logs()

    # Load config
    raw_cfg = load_yaml(args.cfg)
    cfg = Config.model_validate(raw_cfg)

    # Read & normalize signals
    sig = _read_signals_csv(args.signals)

    # Fees & slippage (both legs)
    fee = cfg.fees_bps / 1e4
    slip = cfg.slippage_bps / 1e4

    # Build trades table (vectorized)
    # اگر side=1 (long)، قیمت ورود = entry_price*(1+slip)، خروج = exit_price*(1 - slip)
    # اگر side=-1 (short)، برعکس
    side = sig["side"].astype(int).clip(-1, 1).replace(0, 1)
    entry_exec = sig["entry_price"] * (1 + slip * side)       # long: +slip, short: -slip
    exit_exec = sig["exit_price"] * (1 - slip * side)         # long: -slip, short: +slip

    qty = float(args.qty)
    gross_pnl = (exit_exec - entry_exec) * side * qty
    # کارمزد دو لبه: fee * (entry + exit) * |qty|
    fees = fee * (entry_exec.abs() + exit_exec.abs()) * abs(qty)
    net_pnl = gross_pnl - fees

    trades_df = pd.DataFrame(
        {
            "side": side,
            "entry_price": entry_exec,
            "exit_price": exit_exec,
            "qty": qty,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
        },
        index=sig.index,
    )

    # اعمال سقف ضرر روزانه (اختیاری)
    trades_df = _apply_daily_loss_guard(trades_df, float(cfg.init_cash), args.daily_loss_limit_pct)

    # شبیه‌سازی cash
    cash = float(cfg.init_cash)
    cash_path = []
    for ts, row in trades_df.iterrows():
        cash += row["net_pnl"]
        cash_path.append((ts, cash))

    # خروجی‌ها
    outdir = Path("out/paper")
    outdir.mkdir(parents=True, exist_ok=True)

    # ذخیره CSV معاملات
    trades_csv = outdir / "trades.csv"
    trades_df.reset_index().rename(columns={"index": "datetime"}).to_csv(trades_csv, index=False)

    # خلاصه
    total_trades = len(trades_df)
    total_net = trades_df["net_pnl"].sum()
    winrate = (trades_df["net_pnl"] > 0).mean() * 100 if total_trades > 0 else 0.0
    logger.info(
        f"Paper summary — trades={total_trades}, net_pnl={total_net:.2f}, "
        f"winrate={winrate:.1f}%, final_cash={cash:.2f}"
    )

    # لاگ JSON همزمان
    logger.bind(summary=True).info(
        {
            "trades": total_trades,
            "net_pnl": float(total_net),
            "winrate_pct": float(winrate),
            "final_cash": float(cash),
            "signals_path": str(args.signals),
            "config": {"fees_bps": cfg.fees_bps, "slippage_bps": cfg.slippage_bps, "init_cash": float(cfg.init_cash)},
        }
    )

    print(f"Saved trades CSV → {trades_csv}")
    print(f"Logs → {logdir}")


if __name__ == "__main__":
    # اجرا از ریشه‌ی پروژه:
    #   python -m src.exec.paper --signals out/signals.csv --cfg config/config.yaml
    main()
