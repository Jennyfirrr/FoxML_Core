"""
Engine State Management
=======================

Manages trading engine state between cycles.
Provides persistence for positions, trade history, and decision audit trail.

SST Compliance:
- Uses write_atomic_json() for state persistence
- Uses sorted_items() for deterministic dict iteration
- All to_dict() methods use sorted keys
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from TRAINING.common.utils.file_utils import write_atomic_json
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.time_utils import parse_iso
from LIVE_TRADING.common.types import PositionState, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    """
    Complete engine state.

    Tracks portfolio value, positions, cash, and history.
    Can be persisted to disk and reloaded for session continuity.

    Example:
        >>> state = EngineState(portfolio_value=100_000, cash=100_000)
        >>> state.update_position("AAPL", 0.1, 100, 150.0)
        >>> state.save(Path("state.json"))
        >>> loaded = EngineState.load(Path("state.json"))
    """

    portfolio_value: float
    cash: float
    positions: Dict[str, PositionState] = field(default_factory=dict)
    last_update: Optional[datetime] = None
    cycle_count: int = 0
    daily_pnl: float = 0.0
    day_start_value: float = 0.0
    current_date: Optional[str] = None

    # History (kept in memory, can be persisted separately)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize day start value if not set."""
        if self.day_start_value == 0.0:
            self.day_start_value = self.portfolio_value

    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current position weights.

        Returns:
            Dict mapping symbol to weight (sorted by symbol)
        """
        return {s: p.weight for s, p in sorted_items(self.positions)}

    def get_position(self, symbol: str) -> Optional[PositionState]:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            PositionState or None if no position
        """
        return self.positions.get(symbol)

    def update_position(
        self,
        symbol: str,
        weight: float,
        shares: float,
        price: float,
        timestamp: datetime | None = None,
        clock: Clock | None = None,
    ) -> None:
        """
        Update or create position.

        Args:
            symbol: Trading symbol
            weight: Position weight
            shares: Number of shares
            price: Entry/update price
            timestamp: Update time (default: now)
            clock: Clock instance for time (default: global clock)
        """
        if timestamp is None:
            clock = clock or get_clock()
            timestamp = clock.now()

        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.weight = weight
            pos.shares = shares
            pos.current_price = price
            # Note: We don't call update_price here because we already set the weight explicitly
        else:
            self.positions[symbol] = PositionState(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                entry_time=timestamp,
                current_price=price,
                weight=weight,
            )

        logger.debug(f"Position updated: {symbol} weight={weight:.4f} shares={shares}")

    def remove_position(self, symbol: str) -> Optional[PositionState]:
        """
        Remove a position.

        Args:
            symbol: Symbol to remove

        Returns:
            Removed PositionState or None
        """
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            logger.debug(f"Position removed: {symbol}")
            return pos
        return None

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current prices.

        Args:
            prices: Dict mapping symbol to current price
        """
        for symbol, pos in sorted_items(self.positions):
            if symbol in prices:
                pos.update_price(prices[symbol], self.portfolio_value)

    def record_trade(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        horizon: Optional[str] = None,
        order_id: Optional[str] = None,
        clock: Clock | None = None,
    ) -> None:
        """
        Record a trade in history.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            shares: Number of shares
            price: Fill price
            horizon: Trading horizon
            order_id: Order identifier
            clock: Clock instance for time (default: global clock)
        """
        clock = clock or get_clock()
        trade = {
            "fill_price": price,
            "horizon": horizon,
            "order_id": order_id,
            "shares": shares,
            "side": side,
            "symbol": symbol,
            "timestamp": clock.now().isoformat(),
            "value": shares * price,
        }
        self.trade_history.append(trade)
        logger.info(f"Trade recorded: {side} {shares} {symbol} @ {price:.2f}")

    def record_decision(self, decision: TradeDecision) -> None:
        """
        Record a decision in history.

        Args:
            decision: TradeDecision to record
        """
        self.decision_history.append(decision.to_dict())

    def update_daily_tracking(
        self,
        current_time: datetime | None = None,
        clock: Clock | None = None,
    ) -> None:
        """
        Update daily tracking (reset on new day).

        Args:
            current_time: Current timestamp
            clock: Clock instance for time (default: global clock)
        """
        if current_time is None:
            clock = clock or get_clock()
            current_time = clock.now()
        today = current_time.date().isoformat()

        if self.current_date != today:
            # New day - reset daily P&L
            self.day_start_value = self.portfolio_value
            self.daily_pnl = 0.0
            self.current_date = today
            logger.info(f"New trading day: {today}, start value=${self.day_start_value:,.2f}")
        else:
            # Update daily P&L
            if self.day_start_value > 0:
                self.daily_pnl = self.portfolio_value - self.day_start_value

    def increment_cycle(
        self,
        timestamp: datetime | None = None,
        clock: Clock | None = None,
    ) -> None:
        """
        Increment cycle count and update timestamp.

        Args:
            timestamp: Cycle timestamp
            clock: Clock instance for time (default: global clock)
        """
        self.cycle_count += 1
        if timestamp is None:
            clock = clock or get_clock()
            timestamp = clock.now()
        self.last_update = timestamp

    def get_position_value(self) -> float:
        """
        Get total position value.

        Returns:
            Sum of all position values
        """
        return sum(p.shares * p.current_price for p in self.positions.values())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dict with sorted keys.

        Returns:
            Dict representation
        """
        return {
            "cash": self.cash,
            "current_date": self.current_date,
            "cycle_count": self.cycle_count,
            "daily_pnl": self.daily_pnl,
            "day_start_value": self.day_start_value,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "portfolio_value": self.portfolio_value,
            "positions": {s: p.to_dict() for s, p in sorted_items(self.positions)},
        }

    def save(self, path: Path) -> None:
        """
        Save state to file.

        Args:
            path: Path to save state
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_json(path, self.to_dict())
        logger.info(f"State saved to {path}")

    def save_history(self, path: Path) -> None:
        """
        Save trade and decision history to file.

        Args:
            path: Path to save history
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        history = {
            "decision_history": self.decision_history,
            "trade_history": self.trade_history,
        }
        write_atomic_json(path, history)
        logger.info(f"History saved to {path} ({len(self.trade_history)} trades)")

    @classmethod
    def load(cls, path: Path) -> "EngineState":
        """
        Load state from file.

        Args:
            path: Path to load state from

        Returns:
            Loaded EngineState
        """
        with open(path) as f:
            data = json.load(f)

        state = cls(
            portfolio_value=data["portfolio_value"],
            cash=data["cash"],
            cycle_count=data.get("cycle_count", 0),
            daily_pnl=data.get("daily_pnl", 0.0),
            day_start_value=data.get("day_start_value", data["portfolio_value"]),
            current_date=data.get("current_date"),
        )

        if data.get("last_update"):
            state.last_update = parse_iso(data["last_update"])

        for symbol, pos_data in data.get("positions", {}).items():
            state.positions[symbol] = PositionState.from_dict(pos_data)

        logger.info(f"State loaded from {path}: {len(state.positions)} positions")
        return state

    @classmethod
    def load_history(cls, path: Path) -> Dict[str, List]:
        """
        Load history from file.

        Args:
            path: Path to load history from

        Returns:
            Dict with trade_history and decision_history
        """
        with open(path) as f:
            return json.load(f)


@dataclass
class CycleResult:
    """
    Result of a single trading cycle.

    Aggregates all decisions and provides summary statistics.
    """

    cycle_number: int
    timestamp: datetime
    decisions: List[TradeDecision]
    portfolio_value: float
    cash: float
    is_trading_allowed: bool
    kill_switch_reason: Optional[str] = None

    @property
    def num_trades(self) -> int:
        """Number of TRADE decisions."""
        return sum(1 for d in self.decisions if d.decision == "TRADE")

    @property
    def num_holds(self) -> int:
        """Number of HOLD decisions."""
        return sum(1 for d in self.decisions if d.decision == "HOLD")

    @property
    def num_blocked(self) -> int:
        """Number of BLOCKED decisions."""
        return sum(1 for d in self.decisions if d.decision == "BLOCKED")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "cash": self.cash,
            "cycle_number": self.cycle_number,
            "decisions": [d.to_dict() for d in self.decisions],
            "is_trading_allowed": self.is_trading_allowed,
            "kill_switch_reason": self.kill_switch_reason,
            "num_blocked": self.num_blocked,
            "num_holds": self.num_holds,
            "num_trades": self.num_trades,
            "portfolio_value": self.portfolio_value,
            "timestamp": self.timestamp.isoformat(),
        }
