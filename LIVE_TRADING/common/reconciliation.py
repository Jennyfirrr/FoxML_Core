"""
Position Reconciliation
=======================

Verify local state matches broker state.

Reconciliation Modes:
    STRICT: Fail on any mismatch
    WARN: Log warnings, continue
    AUTO_SYNC: Automatically update local state to match broker

SST Compliance:
- All discrepancies logged
- Audit trail for corrections
- Configurable via get_cfg()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)


class ReconciliationMode(Enum):
    """How to handle discrepancies."""

    STRICT = auto()  # Fail on mismatch
    WARN = auto()  # Log and continue
    AUTO_SYNC = auto()  # Update local to match broker


class DiscrepancyType(Enum):
    """Type of position discrepancy."""

    QUANTITY_MISMATCH = auto()  # Different share counts
    MISSING_LOCAL = auto()  # Broker has position, local doesn't
    MISSING_BROKER = auto()  # Local has position, broker doesn't
    PRICE_MISMATCH = auto()  # Different average costs (informational)


@dataclass
class PositionDiscrepancy:
    """Record of a position mismatch."""

    symbol: str
    discrepancy_type: DiscrepancyType
    local_qty: float
    broker_qty: float
    local_avg_cost: Optional[float] = None
    broker_avg_cost: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: get_clock().now())
    resolved: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "broker_avg_cost": self.broker_avg_cost,
            "broker_qty": self.broker_qty,
            "local_avg_cost": self.local_avg_cost,
            "local_qty": self.local_qty,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "type": self.discrepancy_type.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionDiscrepancy":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso

        return cls(
            symbol=d["symbol"],
            discrepancy_type=DiscrepancyType[d["type"]],
            local_qty=d["local_qty"],
            broker_qty=d["broker_qty"],
            local_avg_cost=d.get("local_avg_cost"),
            broker_avg_cost=d.get("broker_avg_cost"),
            timestamp=parse_iso(d["timestamp"]),
            resolved=d.get("resolved", False),
            resolution=d.get("resolution"),
        )


@dataclass
class CashDiscrepancy:
    """Record of cash balance mismatch."""

    local_cash: float
    broker_cash: float
    difference: float
    timestamp: datetime = field(default_factory=lambda: get_clock().now())
    resolved: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "broker_cash": self.broker_cash,
            "difference": self.difference,
            "local_cash": self.local_cash,
            "resolution": self.resolution,
            "resolved": self.resolved,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CashDiscrepancy":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso

        return cls(
            local_cash=d["local_cash"],
            broker_cash=d["broker_cash"],
            difference=d["difference"],
            timestamp=parse_iso(d["timestamp"]),
            resolved=d.get("resolved", False),
            resolution=d.get("resolution"),
        )


@dataclass
class ReconciliationResult:
    """Result of reconciliation check."""

    success: bool
    position_discrepancies: List[PositionDiscrepancy]
    cash_discrepancy: Optional[CashDiscrepancy]
    timestamp: datetime
    mode: ReconciliationMode
    corrections_applied: List[str] = field(default_factory=list)

    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies found."""
        return bool(self.position_discrepancies) or self.cash_discrepancy is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cash_discrepancy": (
                self.cash_discrepancy.to_dict() if self.cash_discrepancy else None
            ),
            "corrections_applied": self.corrections_applied,
            "has_discrepancies": self.has_discrepancies,
            "mode": self.mode.name,
            "position_discrepancies": [d.to_dict() for d in self.position_discrepancies],
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReconciliationResult":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso

        cash_disc = None
        if d.get("cash_discrepancy"):
            cash_disc = CashDiscrepancy.from_dict(d["cash_discrepancy"])

        return cls(
            success=d["success"],
            position_discrepancies=[
                PositionDiscrepancy.from_dict(pd)
                for pd in d.get("position_discrepancies", [])
            ],
            cash_discrepancy=cash_disc,
            timestamp=parse_iso(d["timestamp"]),
            mode=ReconciliationMode[d["mode"]],
            corrections_applied=d.get("corrections_applied", []),
        )


class ReconciliationError(Exception):
    """Raised when reconciliation fails in strict mode."""

    def __init__(self, result: ReconciliationResult):
        self.result = result
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with discrepancy details."""
        lines = ["Position reconciliation failed:"]

        for disc in self.result.position_discrepancies:
            lines.append(
                f"  {disc.symbol}: local={disc.local_qty}, "
                f"broker={disc.broker_qty} ({disc.discrepancy_type.name})"
            )

        if self.result.cash_discrepancy:
            c = self.result.cash_discrepancy
            lines.append(
                f"  Cash: local=${c.local_cash:.2f}, broker=${c.broker_cash:.2f}"
            )

        return "\n".join(lines)


class Reconciler:
    """
    Position and cash reconciliation service.

    Verifies that local engine state matches broker state, with configurable
    handling of discrepancies:
    - STRICT: Fail on any mismatch (for critical situations)
    - WARN: Log warnings but continue (default for production)
    - AUTO_SYNC: Automatically update local state to match broker

    Example:
        >>> reconciler = Reconciler(broker, state)
        >>> result = reconciler.reconcile()
        >>> if not result.success:
        ...     print(f"Discrepancies: {result.position_discrepancies}")
    """

    def __init__(
        self,
        broker,  # Broker protocol
        state,  # EngineState
        mode: Optional[ReconciliationMode] = None,
        qty_tolerance: Optional[float] = None,
        cash_tolerance: Optional[float] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize reconciler.

        Args:
            broker: Broker instance for getting positions
            state: Engine state with local positions
            mode: Reconciliation mode (default: from config)
            qty_tolerance: Acceptable qty difference (default: from config)
            cash_tolerance: Acceptable cash difference (default: from config)
            clock: Clock for timestamps
        """
        self._broker = broker
        self._state = state
        self._clock = clock or get_clock()

        # Load config with defaults
        mode_str = get_cfg("live_trading.reconciliation.mode", default="warn")
        self._mode = mode or ReconciliationMode[mode_str.upper()]

        self._qty_tolerance = qty_tolerance if qty_tolerance is not None else get_cfg(
            "live_trading.reconciliation.qty_tolerance",
            default=0.01,  # Allow 0.01 share difference (fractional share rounding)
        )
        self._cash_tolerance = (
            cash_tolerance
            if cash_tolerance is not None
            else get_cfg(
                "live_trading.reconciliation.cash_tolerance",
                default=1.0,  # Allow $1 difference (rounding)
            )
        )

        self._history: List[ReconciliationResult] = []

    @property
    def mode(self) -> ReconciliationMode:
        """Get current reconciliation mode."""
        return self._mode

    @property
    def qty_tolerance(self) -> float:
        """Get quantity tolerance."""
        return self._qty_tolerance

    @property
    def cash_tolerance(self) -> float:
        """Get cash tolerance."""
        return self._cash_tolerance

    def reconcile(self) -> ReconciliationResult:
        """
        Perform full reconciliation.

        Compares local positions and cash against broker state.
        Handles discrepancies according to the configured mode.

        Returns:
            ReconciliationResult with all discrepancies
        """
        now = self._clock.now()
        position_discrepancies: List[PositionDiscrepancy] = []
        corrections: List[str] = []

        try:
            # Get broker positions
            broker_positions = self._broker.get_positions()
            broker_cash = self._broker.get_cash()
        except Exception as e:
            logger.error(f"Failed to get broker state for reconciliation: {e}")
            return ReconciliationResult(
                success=False,
                position_discrepancies=[],
                cash_discrepancy=None,
                timestamp=now,
                mode=self._mode,
            )

        # Get local positions
        local_positions = {
            sym: pos.shares
            for sym, pos in self._state.positions.items()
            if pos.shares != 0
        }
        local_cash = self._state.cash

        # Check positions
        all_symbols = set(broker_positions.keys()) | set(local_positions.keys())

        for symbol in sorted(all_symbols):
            local_qty = local_positions.get(symbol, 0.0)
            broker_qty = broker_positions.get(symbol, 0.0)

            if abs(local_qty - broker_qty) > self._qty_tolerance:
                if local_qty == 0 and broker_qty != 0:
                    disc_type = DiscrepancyType.MISSING_LOCAL
                elif local_qty != 0 and broker_qty == 0:
                    disc_type = DiscrepancyType.MISSING_BROKER
                else:
                    disc_type = DiscrepancyType.QUANTITY_MISMATCH

                discrepancy = PositionDiscrepancy(
                    symbol=symbol,
                    discrepancy_type=disc_type,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    timestamp=now,
                )
                position_discrepancies.append(discrepancy)

                logger.warning(
                    f"Position discrepancy: {symbol} - "
                    f"local={local_qty}, broker={broker_qty}, type={disc_type.name}"
                )

                # Handle based on mode
                if self._mode == ReconciliationMode.AUTO_SYNC:
                    self._sync_position(symbol, broker_qty, discrepancy)
                    corrections.append(f"Synced {symbol}: {local_qty} -> {broker_qty}")

        # Check cash
        cash_discrepancy = None
        if abs(local_cash - broker_cash) > self._cash_tolerance:
            cash_discrepancy = CashDiscrepancy(
                local_cash=local_cash,
                broker_cash=broker_cash,
                difference=broker_cash - local_cash,
                timestamp=now,
            )

            logger.warning(
                f"Cash discrepancy: local=${local_cash:.2f}, "
                f"broker=${broker_cash:.2f}, diff=${cash_discrepancy.difference:.2f}"
            )

            if self._mode == ReconciliationMode.AUTO_SYNC:
                self._state.cash = broker_cash
                cash_discrepancy.resolved = True
                cash_discrepancy.resolution = "Auto-synced to broker value"
                corrections.append(
                    f"Synced cash: ${local_cash:.2f} -> ${broker_cash:.2f}"
                )

        # Determine success based on mode
        has_discrepancies = bool(position_discrepancies) or cash_discrepancy is not None

        if self._mode == ReconciliationMode.STRICT:
            success = not has_discrepancies
        else:
            success = True  # WARN and AUTO_SYNC always "succeed"

        result = ReconciliationResult(
            success=success,
            position_discrepancies=position_discrepancies,
            cash_discrepancy=cash_discrepancy,
            timestamp=now,
            mode=self._mode,
            corrections_applied=corrections,
        )

        self._history.append(result)
        return result

    def _sync_position(
        self,
        symbol: str,
        broker_qty: float,
        discrepancy: PositionDiscrepancy,
    ) -> None:
        """Sync local position to match broker."""
        if broker_qty == 0:
            # Remove position
            if symbol in self._state.positions:
                del self._state.positions[symbol]
                logger.info(f"Removed local position for {symbol}")
        else:
            # Update or create position
            # Get average cost from broker if possible
            try:
                broker_pos = self._broker.get_position(symbol)
                avg_cost = broker_pos.get("avg_cost", 0.0) if broker_pos else 0.0
            except Exception:
                avg_cost = 0.0

            self._state.update_position(
                symbol=symbol,
                weight=0.0,  # Will be recalculated
                shares=broker_qty,
                price=avg_cost,
                timestamp=self._clock.now(),
            )
            logger.info(f"Synced {symbol} to {broker_qty} shares")

        discrepancy.resolved = True
        discrepancy.resolution = "Auto-synced to broker"

    def reconcile_on_startup(self) -> ReconciliationResult:
        """
        Special reconciliation for engine startup.

        More verbose logging and always logs full state comparison.
        """
        logger.info("=" * 60)
        logger.info("STARTUP RECONCILIATION")
        logger.info("=" * 60)

        result = self.reconcile()

        if result.has_discrepancies:
            logger.warning(
                f"Found {len(result.position_discrepancies)} position discrepancies"
            )
            for disc in result.position_discrepancies:
                logger.warning(
                    f"  {disc.symbol}: local={disc.local_qty}, broker={disc.broker_qty}"
                )

            if result.cash_discrepancy:
                logger.warning(
                    f"Cash mismatch: local=${result.cash_discrepancy.local_cash:.2f}, "
                    f"broker=${result.cash_discrepancy.broker_cash:.2f}"
                )
        else:
            logger.info("Reconciliation passed - local state matches broker")

        logger.info("=" * 60)
        return result

    def get_history(self, limit: int = 100) -> List[ReconciliationResult]:
        """Get reconciliation history (most recent first)."""
        return list(reversed(self._history[-limit:]))

    def clear_history(self) -> None:
        """Clear reconciliation history."""
        self._history.clear()
