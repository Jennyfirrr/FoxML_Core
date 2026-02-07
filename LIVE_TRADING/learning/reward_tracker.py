"""
Reward Tracker for Bandit Learning
==================================

Tracks realized P&L per arm for bandit feedback.

Maps trades to arms and calculates net reward after:
- Execution fees
- Slippage
- Market impact

SST Compliance:
- Uses Clock abstraction for timestamps
- Uses get_cfg() for configuration
- Deterministic trade ID generation
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)


@dataclass
class PendingTrade:
    """
    A trade awaiting exit for reward calculation.

    Attributes:
        trade_id: Unique trade identifier
        arm: Arm index that generated this trade
        arm_name: Name of the arm (for logging)
        symbol: Trading symbol
        side: "BUY" or "SELL"
        entry_price: Price at entry
        entry_time: Time of entry
        qty: Quantity traded
        expected_horizon_minutes: Expected holding period
    """
    trade_id: str
    arm: int
    arm_name: str
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    qty: float
    expected_horizon_minutes: int = 15


@dataclass
class CompletedTrade:
    """
    A completed trade with realized P&L.

    Attributes:
        trade_id: Unique trade identifier
        arm: Arm index
        arm_name: Name of the arm
        symbol: Trading symbol
        side: "BUY" or "SELL"
        entry_price: Price at entry
        exit_price: Price at exit
        entry_time: Time of entry
        exit_time: Time of exit
        qty: Quantity traded
        gross_pnl_bps: Gross P&L in basis points
        fees_bps: Total fees in basis points
        net_pnl_bps: Net P&L (gross - fees)
        hold_time_minutes: Actual holding time
    """
    trade_id: str
    arm: int
    arm_name: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    qty: float
    gross_pnl_bps: float
    fees_bps: float
    net_pnl_bps: float
    hold_time_minutes: float


class RewardTracker:
    """
    Tracks realized P&L per arm for bandit feedback.

    Maintains pending trades and calculates net rewards after
    accounting for fees and slippage.

    Example:
        >>> tracker = RewardTracker(fee_bps=1.0, slippage_bps=5.0)
        >>> trade_id = tracker.record_entry(
        ...     arm=0, arm_name="5m",
        ...     symbol="AAPL", side="BUY",
        ...     entry_price=150.00, qty=100
        ... )
        >>> reward = tracker.record_exit(
        ...     trade_id=trade_id,
        ...     exit_price=151.00,
        ...     fees=0.50
        ... )
        >>> print(f"Net reward: {reward:.2f} bps")
    """

    def __init__(
        self,
        fee_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        clock: Optional[Clock] = None,
        max_pending_age_hours: float = 24.0,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize reward tracker.

        Args:
            fee_bps: Broker fee in basis points (default: from config)
            slippage_bps: Expected slippage in bps (default: from config)
            clock: Clock instance for timestamps
            max_pending_age_hours: Max age before pending trades are stale
            history_size: Number of completed trades to keep in history
        """
        self._clock = clock or get_clock()

        self._fee_bps = fee_bps if fee_bps is not None else get_cfg(
            "live_trading.reward.fee_bps", default=1.0
        )
        self._slippage_bps = slippage_bps if slippage_bps is not None else get_cfg(
            "live_trading.reward.slippage_bps", default=5.0
        )

        self._max_pending_age = timedelta(hours=max_pending_age_hours)
        self._history_size = history_size

        # Pending trades awaiting exit
        self._pending: Dict[str, PendingTrade] = {}

        # Completed trade history (FIFO, limited size)
        self._completed: List[CompletedTrade] = []

        # Aggregate stats per arm
        self._arm_stats: Dict[int, Dict[str, float]] = {}

        logger.info(
            f"RewardTracker initialized: fee_bps={self._fee_bps}, "
            f"slippage_bps={self._slippage_bps}"
        )

    def record_entry(
        self,
        arm: int,
        arm_name: str,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        expected_horizon_minutes: int = 15,
        trade_id: Optional[str] = None,
        entry_time: Optional[datetime] = None,
    ) -> str:
        """
        Record trade entry.

        Args:
            arm: Arm index that generated this trade
            arm_name: Name of the arm (e.g., "5m", "lightgbm")
            symbol: Trading symbol
            side: "BUY" or "SELL"
            entry_price: Price at entry
            qty: Quantity traded
            expected_horizon_minutes: Expected holding period
            trade_id: Optional custom trade ID (default: UUID)
            entry_time: Entry time (default: now from clock)

        Returns:
            Trade ID for tracking
        """
        trade_id = trade_id or str(uuid.uuid4())[:12]
        entry_time = entry_time or self._clock.now()

        trade = PendingTrade(
            trade_id=trade_id,
            arm=arm,
            arm_name=arm_name,
            symbol=symbol,
            side=side.upper(),
            entry_price=entry_price,
            entry_time=entry_time,
            qty=qty,
            expected_horizon_minutes=expected_horizon_minutes,
        )

        self._pending[trade_id] = trade

        logger.debug(
            f"Recorded entry: {trade_id} arm={arm_name} {side} {qty} {symbol} @ {entry_price}"
        )

        return trade_id

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        fees: float = 0.0,
        exit_time: Optional[datetime] = None,
    ) -> float:
        """
        Record trade exit and calculate net reward.

        Args:
            trade_id: Trade ID from record_entry()
            exit_price: Price at exit
            fees: Actual fees paid (in dollars)
            exit_time: Exit time (default: now from clock)

        Returns:
            Net reward in basis points

        Raises:
            KeyError: If trade_id not found
        """
        if trade_id not in self._pending:
            raise KeyError(f"Trade {trade_id} not found in pending trades")

        exit_time = exit_time or self._clock.now()
        trade = self._pending.pop(trade_id)

        # Calculate gross P&L in bps
        if trade.side == "BUY":
            gross_pnl_bps = (exit_price - trade.entry_price) / trade.entry_price * 10000
        else:  # SELL (short)
            gross_pnl_bps = (trade.entry_price - exit_price) / trade.entry_price * 10000

        # Calculate fees in bps
        notional = trade.entry_price * trade.qty
        fees_bps = (fees / notional * 10000) if notional > 0 else 0

        # Add estimated slippage
        total_fees_bps = self._fee_bps + self._slippage_bps + fees_bps

        # Net P&L
        net_pnl_bps = gross_pnl_bps - total_fees_bps

        # Holding time
        hold_time = (exit_time - trade.entry_time).total_seconds() / 60.0

        # Create completed trade record
        completed = CompletedTrade(
            trade_id=trade_id,
            arm=trade.arm,
            arm_name=trade.arm_name,
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            entry_time=trade.entry_time,
            exit_time=exit_time,
            qty=trade.qty,
            gross_pnl_bps=gross_pnl_bps,
            fees_bps=total_fees_bps,
            net_pnl_bps=net_pnl_bps,
            hold_time_minutes=hold_time,
        )

        # Add to history (FIFO)
        self._completed.append(completed)
        if len(self._completed) > self._history_size:
            self._completed.pop(0)

        # Update arm stats
        self._update_arm_stats(trade.arm, net_pnl_bps)

        logger.info(
            f"Trade exit: {trade_id} {trade.arm_name} {trade.symbol} "
            f"gross={gross_pnl_bps:.2f}bps fees={total_fees_bps:.2f}bps "
            f"net={net_pnl_bps:.2f}bps hold={hold_time:.1f}min"
        )

        return net_pnl_bps

    def _update_arm_stats(self, arm: int, net_pnl_bps: float) -> None:
        """Update aggregate statistics for an arm."""
        if arm not in self._arm_stats:
            self._arm_stats[arm] = {
                "total_trades": 0,
                "total_pnl_bps": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
            }

        stats = self._arm_stats[arm]
        stats["total_trades"] += 1
        stats["total_pnl_bps"] += net_pnl_bps

        if net_pnl_bps > 0:
            stats["winning_trades"] += 1
        elif net_pnl_bps < 0:
            stats["losing_trades"] += 1

    def get_pending_trades(self) -> List[PendingTrade]:
        """
        Get all pending trades.

        Returns:
            List of pending trades (copy)
        """
        return list(self._pending.values())

    def get_pending_for_symbol(self, symbol: str) -> List[PendingTrade]:
        """
        Get pending trades for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of pending trades for the symbol
        """
        return [t for t in self._pending.values() if t.symbol == symbol]

    def get_pending_by_id(self, trade_id: str) -> Optional[PendingTrade]:
        """
        Get a specific pending trade.

        Args:
            trade_id: Trade ID

        Returns:
            PendingTrade or None if not found
        """
        return self._pending.get(trade_id)

    def get_completed_trades(
        self,
        arm: Optional[int] = None,
        limit: int = 100,
    ) -> List[CompletedTrade]:
        """
        Get completed trades from history.

        Args:
            arm: Filter by arm (None for all)
            limit: Maximum number to return

        Returns:
            List of completed trades (most recent first)
        """
        trades = self._completed[::-1]  # Reverse for most recent first

        if arm is not None:
            trades = [t for t in trades if t.arm == arm]

        return trades[:limit]

    def get_arm_stats(self, arm: int) -> Dict[str, float]:
        """
        Get aggregate statistics for an arm.

        Args:
            arm: Arm index

        Returns:
            Dict with stats (total_trades, total_pnl_bps, win_rate, etc.)
        """
        if arm not in self._arm_stats:
            return {
                "total_trades": 0,
                "total_pnl_bps": 0.0,
                "avg_pnl_bps": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
            }

        stats = self._arm_stats[arm].copy()
        total = stats["total_trades"]

        stats["avg_pnl_bps"] = stats["total_pnl_bps"] / total if total > 0 else 0.0
        stats["win_rate"] = stats["winning_trades"] / total if total > 0 else 0.0

        return stats

    def cleanup_stale(self) -> int:
        """
        Remove stale pending trades (older than max_pending_age).

        Returns:
            Number of stale trades removed
        """
        now = self._clock.now()
        stale_ids = []

        for trade_id, trade in self._pending.items():
            age = now - trade.entry_time
            if age > self._max_pending_age:
                stale_ids.append(trade_id)

        for trade_id in stale_ids:
            trade = self._pending.pop(trade_id)
            logger.warning(
                f"Removed stale pending trade: {trade_id} "
                f"{trade.arm_name} {trade.symbol}"
            )

        return len(stale_ids)

    def has_pending_for_symbol(self, symbol: str) -> bool:
        """Check if there are pending trades for a symbol."""
        return any(t.symbol == symbol for t in self._pending.values())

    @property
    def pending_count(self) -> int:
        """Number of pending trades."""
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        """Number of completed trades in history."""
        return len(self._completed)

    def to_dict(self) -> Dict[str, any]:
        """
        Serialize state to dictionary.

        Returns:
            Dict for persistence
        """
        return {
            "fee_bps": self._fee_bps,
            "slippage_bps": self._slippage_bps,
            "pending": [
                {
                    "trade_id": t.trade_id,
                    "arm": t.arm,
                    "arm_name": t.arm_name,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "qty": t.qty,
                    "expected_horizon_minutes": t.expected_horizon_minutes,
                }
                for t in self._pending.values()
            ],
            "arm_stats": self._arm_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any], clock: Optional[Clock] = None) -> "RewardTracker":
        """
        Restore from serialized state.

        Args:
            data: Dict from to_dict()
            clock: Optional clock instance

        Returns:
            Restored RewardTracker
        """
        tracker = cls(
            fee_bps=data["fee_bps"],
            slippage_bps=data["slippage_bps"],
            clock=clock,
        )

        # Restore pending trades
        for p in data.get("pending", []):
            trade = PendingTrade(
                trade_id=p["trade_id"],
                arm=p["arm"],
                arm_name=p["arm_name"],
                symbol=p["symbol"],
                side=p["side"],
                entry_price=p["entry_price"],
                entry_time=datetime.fromisoformat(p["entry_time"]),
                qty=p["qty"],
                expected_horizon_minutes=p.get("expected_horizon_minutes", 15),
            )
            tracker._pending[trade.trade_id] = trade

        # Restore arm stats
        tracker._arm_stats = {int(k): v for k, v in data.get("arm_stats", {}).items()}

        logger.info(
            f"Restored RewardTracker: {tracker.pending_count} pending trades, "
            f"{len(tracker._arm_stats)} arms with stats"
        )

        return tracker

    def reset(self) -> None:
        """Reset tracker state."""
        self._pending.clear()
        self._completed.clear()
        self._arm_stats.clear()
        logger.info("RewardTracker reset")
