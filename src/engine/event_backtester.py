# src/engine/event_backtester.py
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

StopMode = Literal["percent", "atr"]
Side = Literal["long"]

@dataclass
class Trade:
    entry_dt: pd.Timestamp
    exit_dt: pd.Timestamp
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    fees_entry: float
    fees_exit: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str

class EventBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        *,
        init_cash: float = 10_000.0,
        fees_bps: float = 10.0,
        slippage_bps: float = 2.0,
        risk_pct_per_trade: float = 0.01,
        stop_mode: StopMode = "percent",
        # percent mode
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        # atr mode
        atr_series: Optional[pd.Series] = None,
        atr_mult_sl: Optional[float] = None,
        atr_mult_tp: Optional[float] = None,
        # execution controls
        max_trades_per_day: Optional[int] = None,
        daily_loss_limit_pct: Optional[float] = None,  # fraction of init_cash
        pessimistic_same_bar: bool = True,
    ):
        """
        df: OHLCV DataFrame with columns: open, high, low, close (UTC DateTimeIndex).
        entries/exits: boolean Series aligned on df.index (signals computed on close).
        """
        self.df = df.copy()
        self.entries = entries.reindex(df.index, fill_value=False).astype(bool)
        self.exits = exits.reindex(df.index, fill_value=False).astype(bool)

        self.init_cash = float(init_cash)
        self.cash = float(init_cash)
        self.position_qty = 0.0
        self.position_entry_price = np.nan
        self.position_stop = np.nan
        self.position_tp = np.nan

        self.fees_rate = fees_bps / 1e4
        self.slip_rate = slippage_bps / 1e4
        self.risk_pct = float(risk_pct_per_trade)

        self.stop_mode: StopMode = stop_mode
        self.sl_pct = float(sl_pct) if sl_pct is not None else None
        self.tp_pct = float(tp_pct) if tp_pct is not None else None
        self.atr = atr_series.reindex(df.index) if atr_series is not None else None
        self.atr_mult_sl = float(atr_mult_sl) if atr_mult_sl is not None else None
        self.atr_mult_tp = float(atr_mult_tp) if atr_mult_tp is not None else None

        self.max_trades_per_day = max_trades_per_day
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.pessimistic_same_bar = pessimistic_same_bar

        self.trades: List[Trade] = []
        self.equity_series: List[float] = []
        self.equity_index: List[pd.Timestamp] = []
        self._trade_count_today: Dict[pd.Timestamp.date, int] = {}
        self._realized_pnl_today: Dict[pd.Timestamp.date, float] = {}
        self._blocked_today: Dict[pd.Timestamp.date, bool] = {}

        # runtime vars
        self._pending_entry_for_next_open = False
        self._pending_exit_for_next_open = False
        self._pending_entry_size = 0.0
        self._pending_entry_stop = np.nan
        self._pending_entry_tp = np.nan

        self._last_entry_fee = 0.0
        self._last_entry_fill_px = np.nan
        self._last_entry_dt = pd.NaT

        # safety checks
        if self.stop_mode == "percent" and (self.sl_pct is None or self.sl_pct <= 0):
            raise ValueError("percent stop_mode requires sl_pct > 0")
        if self.stop_mode == "atr":
            if self.atr is None or self.atr_mult_sl is None or self.atr_mult_sl <= 0:
                raise ValueError("atr stop_mode requires atr_series and atr_mult_sl > 0")

    # ----- helpers -----
    def _equity(self, price: float) -> float:
        return self.cash + self.position_qty * price

    def _period_seconds(self) -> float:
        diffs = self.df.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return 0.0
        return diffs.median().total_seconds()

    def _annualization_factor(self) -> float:
        s = self._period_seconds()
        return 31_536_000.0 / s if s > 0 else 1.0  # seconds/year

    def _update_day_counters(self, day):
        self._trade_count_today.setdefault(day, 0)
        self._realized_pnl_today.setdefault(day, 0.0)
        self._blocked_today.setdefault(day, False)

    def _can_trade_today(self, day) -> bool:
        if self._blocked_today.get(day, False):
            return False
        if self.max_trades_per_day is None:
            return True
        return self._trade_count_today.get(day, 0) < self.max_trades_per_day

    # ----- order filling (market) -----
    def _buy_at_open(self, dt: pd.Timestamp, price_open: float, qty: float, stop: float, tp: float):
        fill_px = price_open * (1 + self.slip_rate)  # worse for buy
        cost = fill_px * qty
        fee_in = cost * self.fees_rate
        self.cash -= (cost + fee_in)
        self.position_qty = qty
        self.position_entry_price = fill_px
        self.position_stop = float(stop)
        self.position_tp = float(tp) if tp == tp else np.nan  # keep NaN if NaN
        return fee_in, fill_px

    def _sell_at_price(self, dt: pd.Timestamp, price: float, reason: str) -> Trade:
        fill_px = price * (1 - self.slip_rate)  # worse for sell
        proceeds = fill_px * self.position_qty
        fee_out = proceeds * self.fees_rate
        self.cash += (proceeds - fee_out)
        gross = (fill_px - self.position_entry_price) * self.position_qty
        trade = Trade(
            entry_dt=self._last_entry_dt,
            exit_dt=dt,
            side="long",
            entry_price=self._last_entry_fill_px,
            exit_price=fill_px,
            qty=self.position_qty,
            fees_entry=self._last_entry_fee,
            fees_exit=fee_out,
            gross_pnl=gross,
            net_pnl=gross - self._last_entry_fee - fee_out,
            exit_reason=reason,
        )
        # flat
        self.position_qty = 0.0
        self.position_entry_price = np.nan
        self.position_stop = np.nan
        self.position_tp = np.nan
        return trade

    # ----- main loop -----
    def run(self):
        idx = self.df.index
        o = self.df["open"].astype(float)
        h = self.df["high"].astype(float)
        l = self.df["low"].astype(float)
        c = self.df["close"].astype(float)

        for i in range(len(idx)):
            dt = idx[i]
            day = dt.date()
            self._update_day_counters(day)

            # 1) pending entry at current open
            if self._pending_entry_for_next_open and self.position_qty == 0:
                qty = self._pending_entry_size
                stop = self._pending_entry_stop
                tp = self._pending_entry_tp
                self._pending_entry_for_next_open = False
                if qty > 0:
                    self._last_entry_fee, fill_px = self._buy_at_open(dt, o.iloc[i], qty, stop, tp)
                    self._last_entry_fill_px = fill_px
                    self._last_entry_dt = dt
                    self._trade_count_today[day] = self._trade_count_today.get(day, 0) + 1

            # 2) manage open position intra-bar (pessimistic: SL first)
            if self.position_qty > 0:
                if self.pessimistic_same_bar and l.iloc[i] <= self.position_stop:
                    t = self._sell_at_price(dt, self.position_stop, "SL")
                    self.trades.append(t)
                    self._realized_pnl_today[day] += t.net_pnl
                    if self.daily_loss_limit_pct is not None:
                        if abs(min(0.0, self._realized_pnl_today[day])) > self.daily_loss_limit_pct * self.init_cash:
                            self._blocked_today[day] = True
                elif (self.position_tp == self.position_tp) and (h.iloc[i] >= self.position_tp):
                    t = self._sell_at_price(dt, self.position_tp, "TP")
                    self.trades.append(t)
                    self._realized_pnl_today[day] += t.net_pnl
                else:
                    if self.exits.iloc[i]:
                        self._pending_exit_for_next_open = True

            # 3) exit at next open if queued
            if self._pending_exit_for_next_open and self.position_qty > 0 and i + 1 < len(idx):
                next_dt = idx[i + 1]
                t = self._sell_at_price(next_dt, o.iloc[i + 1], "SignalExitNextOpen")
                self.trades.append(t)
                self._realized_pnl_today[next_dt.date()] = self._realized_pnl_today.get(next_dt.date(), 0.0) + t.net_pnl
                self._pending_exit_for_next_open = False

            # 4) new entry signal -> queue for next open (size by risk)
            if self.position_qty == 0 and self.entries.iloc[i] and (i + 1 < len(idx)) and self._can_trade_today(day):
                entry_next_open = float(o.iloc[i + 1])

                if self.stop_mode == "percent":
                    stop_price = entry_next_open * (1.0 - self.sl_pct)
                    risk_per_unit = entry_next_open - stop_price
                    tp_price = (entry_next_open * (1.0 + self.tp_pct)) if (self.tp_pct and self.tp_pct > 0) else np.nan
                else:  # "atr"
                    a = float(self.atr.iloc[i]) if self.atr is not None else np.nan
                    if not (a == a) or a <= 0:
                        risk_per_unit = np.nan
                        stop_price = np.nan
                        tp_price = np.nan
                    else:
                        stop_price = entry_next_open - self.atr_mult_sl * a
                        risk_per_unit = entry_next_open - stop_price
                        tp_price = entry_next_open + self.atr_mult_tp * a if (self.atr_mult_tp and self.atr_mult_tp > 0) else np.nan

                if risk_per_unit == risk_per_unit and risk_per_unit > 0 and stop_price > 0:
                    risk_dollar = self._equity(c.iloc[i]) * self.risk_pct
                    qty = risk_dollar / risk_per_unit
                    if qty > 0 and np.isfinite(qty):
                        self._pending_entry_for_next_open = True
                        self._pending_entry_size = qty
                        self._pending_entry_stop = stop_price
                        self._pending_entry_tp = tp_price

            # 5) mark-to-market at close
            self.equity_series.append(self._equity(c.iloc[i]))
            self.equity_index.append(dt)

        # 6) force close last position at last close
        if self.position_qty > 0:
            last_dt = self.df.index[-1]
            t = self._sell_at_price(last_dt, float(self.df["close"].iloc[-1]), "ForceCloseEnd")
            self.trades.append(t)

        self.equity_series = pd.Series(self.equity_series, index=pd.DatetimeIndex(self.equity_index, tz="UTC"))

    # ----- KPIs -----
    def kpis(self) -> dict:
        eq = self.equity_series
        total_return = float(eq.iloc[-1] / self.init_cash - 1.0)
        running_max = eq.cummax()
        dd = (eq / running_max - 1.0).fillna(0.0)
        max_drawdown = float(dd.min())

        rets = eq.pct_change().fillna(0.0)
        ann = self._annualization_factor()
        mu = rets.mean()
        sd = rets.std(ddof=0)
        sharpe = float((mu * math.sqrt(ann)) / sd) if sd > 0 else float("nan")

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl < 0]
        winrate = 100.0 * len(wins) / len(self.trades) if self.trades else float("nan")
        profit_factor = (sum(t.net_pnl for t in wins) / abs(sum(t.net_pnl for t in losses))) if losses else float("inf")

        return {
            "init_cash": self.init_cash,
            "final_equity": float(eq.iloc[-1]),
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "trades": len(self.trades),
            "winrate": winrate,
            "profit_factor": float(profit_factor),
        }

    # ----- save -----
    def save_outputs(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(t) for t in self.trades]).to_csv(outdir / "trades.csv", index=False)
        self.equity_series.rename("equity").to_csv(outdir / "equity.csv")

        import json
        (outdir / "summary.json").write_text(json.dumps(self.kpis(), indent=2), encoding="utf-8")

        plt.figure(figsize=(10, 4))
        self.equity_series.plot(title="Equity Curve")
        plt.tight_layout()
        plt.savefig(outdir / "equity.png", dpi=150)
        plt.close()
