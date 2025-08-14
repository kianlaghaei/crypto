# src/backtest/event/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Protocol

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()

class OrderStatus(Enum):
    NEW = auto()
    SUBMITTED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()

@dataclass
class Order:
    id: int
    ts: pd.Timestamp
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW

@dataclass
class Fill:
    order_id: int
    ts: pd.Timestamp
    price: float
    qty: float
    fee: float = 0.0
    slippage: float = 0.0

@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

@dataclass
class PortfolioState:
    cash: float
    pos: Position = field(default_factory=Position)
    equity: float = 0.0
    open_orders: List[Order] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)

class Strategy(Protocol):
    def on_bar(self, bar: Bar, state: PortfolioState) -> List[Order]:
        ...

class PaperBroker:
    def __init__(self, fees_bps: float = 0.0, slippage_bps: float = 0.0):
        self.fees = fees_bps / 1e4
        self.slip = slippage_bps / 1e4
        self._order_id = 0

    def next_id(self) -> int:
        self._order_id += 1
        return self._order_id

    def match(self, orders: List[Order], bar: Bar) -> List[Fill]:
        fills: List[Fill] = []
        for od in orders:
            if od.type != OrderType.MARKET:
                continue
            px = bar.open * (1 + self.slip if od.side == Side.BUY else 1 - self.slip)
            fee = abs(px * od.qty) * self.fees
            fills.append(Fill(order_id=od.id, ts=bar.ts, price=float(px), qty=od.qty, fee=float(fee), slippage=float(self.slip)))
        return fills

class EventEngine:
    def __init__(self, bars: pd.DataFrame, init_cash: float, broker: PaperBroker, strategy: Strategy):
        req_cols = {"open", "high", "low", "close"}
        if not req_cols.issubset(set(map(str.lower, bars.columns))):
            raise ValueError("bars must have columns: open, high, low, close (case-insensitive)")
        self.bars = bars.rename(columns={c: c.lower() for c in bars.columns})
        self.state = PortfolioState(cash=float(init_cash))
        self.broker = broker
        self.strategy = strategy
        self.equity_series: list[tuple[pd.Timestamp, float]] = []

    def _update_equity(self, bar: Bar):
        pos_val = self.state.pos.qty * bar.close
        self.state.equity = self.state.cash + pos_val

    def _apply_fill(self, fill: Fill):
        pos = self.state.pos
        notional = fill.price * fill.qty

        if fill.qty > 0:
            new_qty = pos.qty + fill.qty
            if new_qty == 0:
                pos.avg_price = 0.0
            else:
                pos.avg_price = (pos.avg_price * pos.qty + notional) / new_qty
            pos.qty = new_qty
            self.state.cash -= notional + fill.fee
        else:
            realized = (fill.price - pos.avg_price) * (-fill.qty)
            self.state.cash += (-notional) - fill.fee
            pos.qty += fill.qty
            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0

        self.state.fills.append(fill)

    def run(self):
        for ts, row in self.bars.iterrows():
            bar = Bar(
                ts=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
            intents = self.strategy.on_bar(bar, self.state)
            for od in intents:
                if od.id == 0:
                    od.id = self.broker.next_id()
            fills = self.broker.match(intents, bar)
            for f in fills:
                self._apply_fill(f)
            self._update_equity(bar)
            self.equity_series.append((ts, self.state.equity))

    def equity(self) -> pd.Series:
        if not self.equity_series:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_series)
        return pd.Series(vals, index=pd.Index(idx, name="datetime"), name="equity")
