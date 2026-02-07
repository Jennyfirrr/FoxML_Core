# Plan 0C: Position Reconciliation

## Overview

Implement position reconciliation to verify local state matches broker state. Critical for:
- Startup recovery after crash
- Periodic health checks during trading
- Detecting out-of-band changes (manual trades, dividends, splits)

## Problem Statement

Current code trusts local state blindly:
```python
# In state.py - pure local tracking, no verification
def update_position(self, symbol, weight, shares, price, timestamp):
    self.positions[symbol] = PositionState(...)
    # Never checks if broker agrees!
```

Failure scenarios:
1. Engine crashes with position in AAPL
2. Restart loads stale state (or fresh state if corrupted)
3. Engine thinks it has no positions
4. Submits trades based on wrong portfolio state
5. Over-leveraged or double-positioned

## Files to Create

### 1. `LIVE_TRADING/common/reconciliation.py`

```python
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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)


class ReconciliationMode(Enum):
    """How to handle discrepancies."""
    STRICT = auto()      # Fail on mismatch
    WARN = auto()        # Log and continue
    AUTO_SYNC = auto()   # Update local to match broker


class DiscrepancyType(Enum):
    """Type of position discrepancy."""
    QUANTITY_MISMATCH = auto()     # Different share counts
    MISSING_LOCAL = auto()          # Broker has position, local doesn't
    MISSING_BROKER = auto()         # Local has position, broker doesn't
    PRICE_MISMATCH = auto()         # Different average costs (informational)


@dataclass
class PositionDiscrepancy:
    """Record of a position mismatch."""
    symbol: str
    discrepancy_type: DiscrepancyType
    local_qty: float
    broker_qty: float
    local_avg_cost: Optional[float] = None
    broker_avg_cost: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "type": self.discrepancy_type.name,
            "local_qty": self.local_qty,
            "broker_qty": self.broker_qty,
            "local_avg_cost": self.local_avg_cost,
            "broker_avg_cost": self.broker_avg_cost,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


@dataclass
class CashDiscrepancy:
    """Record of cash balance mismatch."""
    local_cash: float
    broker_cash: float
    difference: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "local_cash": self.local_cash,
            "broker_cash": self.broker_cash,
            "difference": self.difference,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


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
            "success": self.success,
            "has_discrepancies": self.has_discrepancies,
            "position_discrepancies": [d.to_dict() for d in self.position_discrepancies],
            "cash_discrepancy": self.cash_discrepancy.to_dict() if self.cash_discrepancy else None,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode.name,
            "corrections_applied": self.corrections_applied,
        }


class Reconciler:
    """
    Position and cash reconciliation service.

    Example:
        >>> reconciler = Reconciler(broker, state)
        >>> result = reconciler.reconcile()
        >>> if not result.success:
        ...     print(f"Discrepancies: {result.position_discrepancies}")
    """

    def __init__(
        self,
        broker,  # Broker protocol
        state,   # EngineState
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

        self._qty_tolerance = qty_tolerance or get_cfg(
            "live_trading.reconciliation.qty_tolerance",
            default=0.01  # Allow 0.01 share difference (fractional share rounding)
        )
        self._cash_tolerance = cash_tolerance or get_cfg(
            "live_trading.reconciliation.cash_tolerance",
            default=1.0  # Allow $1 difference (rounding)
        )

        self._history: List[ReconciliationResult] = []

    def reconcile(self) -> ReconciliationResult:
        """
        Perform full reconciliation.

        Returns:
            ReconciliationResult with all discrepancies
        """
        now = self._clock.now()
        position_discrepancies = []
        corrections = []

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
                corrections.append(f"Synced cash: ${local_cash:.2f} -> ${broker_cash:.2f}")

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
            logger.warning(f"Found {len(result.position_discrepancies)} position discrepancies")
            for disc in result.position_discrepancies:
                logger.warning(f"  {disc.symbol}: local={disc.local_qty}, broker={disc.broker_qty}")

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
        """Get reconciliation history."""
        return list(reversed(self._history[-limit:]))


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
            lines.append(f"  Cash: local=${c.local_cash:.2f}, broker=${c.broker_cash:.2f}")

        return "\n".join(lines)
```

### 2. Update `LIVE_TRADING/common/__init__.py`

```python
# Add to existing exports
from .reconciliation import (
    Reconciler,
    ReconciliationMode,
    ReconciliationResult,
    ReconciliationError,
    PositionDiscrepancy,
    CashDiscrepancy,
    DiscrepancyType,
)
```

## Files to Modify

### 1. `LIVE_TRADING/engine/trading_engine.py`

Add reconciliation to startup and periodic checks:

```python
from LIVE_TRADING.common.reconciliation import (
    Reconciler,
    ReconciliationMode,
    ReconciliationError,
)

class TradingEngine:
    def __init__(self, ...):
        ...
        self._reconciler: Optional[Reconciler] = None

    def _initialize_reconciler(self) -> None:
        """Initialize reconciler after state is loaded."""
        self._reconciler = Reconciler(
            broker=self.broker,
            state=self._state,
            clock=self._clock,
        )

    def start(self) -> None:
        """Start the trading engine."""
        # Load state
        _ = self.state  # Triggers state loading

        # Initialize reconciler
        self._initialize_reconciler()

        # Reconcile on startup
        result = self._reconciler.reconcile_on_startup()
        if not result.success:
            raise ReconciliationError(result)

        # Continue with normal startup...

    def run_cycle(self, current_time: Optional[datetime] = None) -> List[TradeDecision]:
        """Run a single trading cycle."""
        # Periodic reconciliation (e.g., every 100 cycles)
        if self._cycle_count % 100 == 0:
            result = self._reconciler.reconcile()
            if result.has_discrepancies:
                logger.warning(f"Reconciliation found {len(result.position_discrepancies)} discrepancies")

        # ... rest of cycle
```

## Configuration

Add to `CONFIG/live_trading/live_trading.yaml`:

```yaml
live_trading:
  reconciliation:
    mode: "warn"           # strict, warn, auto_sync
    qty_tolerance: 0.01    # Share count tolerance
    cash_tolerance: 1.0    # Cash tolerance in dollars
    periodic_interval: 100 # Reconcile every N cycles
```

## Tests

### `LIVE_TRADING/tests/test_reconciliation.py`

```python
"""
Reconciliation Tests
====================

Unit tests for position reconciliation.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from LIVE_TRADING.common.reconciliation import (
    Reconciler,
    ReconciliationMode,
    ReconciliationResult,
    ReconciliationError,
    DiscrepancyType,
)
from LIVE_TRADING.common.clock import SimulatedClock


class TestReconciler:
    """Tests for Reconciler class."""

    @pytest.fixture
    def clock(self):
        """Create simulated clock."""
        return SimulatedClock(datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc))

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock()
        broker.get_positions.return_value = {"AAPL": 100.0, "MSFT": 50.0}
        broker.get_cash.return_value = 50000.0
        broker.get_position.return_value = {"avg_cost": 150.0}
        return broker

    @pytest.fixture
    def mock_state(self):
        """Create mock state."""
        state = Mock()
        state.positions = {
            "AAPL": Mock(shares=100.0),
            "MSFT": Mock(shares=50.0),
        }
        state.cash = 50000.0
        return state

    def test_reconcile_no_discrepancies(self, mock_broker, mock_state, clock):
        """Test reconciliation with matching state."""
        reconciler = Reconciler(mock_broker, mock_state, clock=clock)
        result = reconciler.reconcile()

        assert result.success
        assert not result.has_discrepancies
        assert len(result.position_discrepancies) == 0
        assert result.cash_discrepancy is None

    def test_reconcile_quantity_mismatch(self, mock_broker, mock_state, clock):
        """Test detection of quantity mismatch."""
        mock_state.positions["AAPL"].shares = 90.0  # Local has 90, broker has 100

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.WARN, clock=clock)
        result = reconciler.reconcile()

        assert result.success  # WARN mode succeeds
        assert result.has_discrepancies
        assert len(result.position_discrepancies) == 1

        disc = result.position_discrepancies[0]
        assert disc.symbol == "AAPL"
        assert disc.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        assert disc.local_qty == 90.0
        assert disc.broker_qty == 100.0

    def test_reconcile_missing_local(self, mock_broker, mock_state, clock):
        """Test detection of position missing locally."""
        mock_broker.get_positions.return_value = {"AAPL": 100.0, "MSFT": 50.0, "GOOG": 25.0}
        # GOOG exists in broker but not in local state

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.WARN, clock=clock)
        result = reconciler.reconcile()

        assert result.has_discrepancies
        goog_disc = [d for d in result.position_discrepancies if d.symbol == "GOOG"][0]
        assert goog_disc.discrepancy_type == DiscrepancyType.MISSING_LOCAL

    def test_reconcile_missing_broker(self, mock_broker, mock_state, clock):
        """Test detection of position missing in broker."""
        mock_state.positions["NVDA"] = Mock(shares=30.0)
        # NVDA exists locally but not in broker

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.WARN, clock=clock)
        result = reconciler.reconcile()

        assert result.has_discrepancies
        nvda_disc = [d for d in result.position_discrepancies if d.symbol == "NVDA"][0]
        assert nvda_disc.discrepancy_type == DiscrepancyType.MISSING_BROKER

    def test_reconcile_cash_mismatch(self, mock_broker, mock_state, clock):
        """Test detection of cash mismatch."""
        mock_state.cash = 49000.0  # Local has less cash

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.WARN, clock=clock)
        result = reconciler.reconcile()

        assert result.has_discrepancies
        assert result.cash_discrepancy is not None
        assert result.cash_discrepancy.local_cash == 49000.0
        assert result.cash_discrepancy.broker_cash == 50000.0
        assert result.cash_discrepancy.difference == 1000.0

    def test_strict_mode_fails_on_discrepancy(self, mock_broker, mock_state, clock):
        """Test that strict mode returns failure on discrepancy."""
        mock_state.positions["AAPL"].shares = 90.0

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.STRICT, clock=clock)
        result = reconciler.reconcile()

        assert not result.success

    def test_auto_sync_corrects_position(self, mock_broker, mock_state, clock):
        """Test that auto_sync mode corrects positions."""
        mock_state.positions["AAPL"].shares = 90.0
        mock_state.update_position = Mock()

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.AUTO_SYNC, clock=clock)
        result = reconciler.reconcile()

        assert result.success
        assert len(result.corrections_applied) > 0
        mock_state.update_position.assert_called()

    def test_auto_sync_corrects_cash(self, mock_broker, mock_state, clock):
        """Test that auto_sync mode corrects cash."""
        mock_state.cash = 49000.0

        reconciler = Reconciler(mock_broker, mock_state, mode=ReconciliationMode.AUTO_SYNC, clock=clock)
        result = reconciler.reconcile()

        assert mock_state.cash == 50000.0  # Should be corrected
        assert result.cash_discrepancy.resolved

    def test_tolerance_within_bounds(self, mock_broker, mock_state, clock):
        """Test that small differences within tolerance are ignored."""
        mock_state.positions["AAPL"].shares = 100.005  # Within 0.01 tolerance

        reconciler = Reconciler(mock_broker, mock_state, qty_tolerance=0.01, clock=clock)
        result = reconciler.reconcile()

        assert not result.has_discrepancies

    def test_history_tracking(self, mock_broker, mock_state, clock):
        """Test that reconciliation history is tracked."""
        reconciler = Reconciler(mock_broker, mock_state, clock=clock)

        reconciler.reconcile()
        clock.advance(minutes=5)
        reconciler.reconcile()
        clock.advance(minutes=5)
        reconciler.reconcile()

        history = reconciler.get_history()
        assert len(history) == 3


class TestReconciliationError:
    """Tests for ReconciliationError."""

    def test_error_message_includes_discrepancies(self):
        """Test that error message includes discrepancy details."""
        from LIVE_TRADING.common.reconciliation import PositionDiscrepancy

        result = ReconciliationResult(
            success=False,
            position_discrepancies=[
                PositionDiscrepancy(
                    symbol="AAPL",
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    local_qty=90.0,
                    broker_qty=100.0,
                    timestamp=datetime.now(timezone.utc),
                )
            ],
            cash_discrepancy=None,
            timestamp=datetime.now(timezone.utc),
            mode=ReconciliationMode.STRICT,
        )

        error = ReconciliationError(result)
        assert "AAPL" in str(error)
        assert "90" in str(error)
        assert "100" in str(error)
```

## SST Compliance

- [x] Configuration via get_cfg()
- [x] Comprehensive audit logging
- [x] Timezone-aware timestamps
- [x] Serializable results for persistence

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `common/reconciliation.py` | 350 |
| `tests/test_reconciliation.py` | 200 |
| Modifications to trading_engine.py | ~40 |
| **Total** | ~590 |
