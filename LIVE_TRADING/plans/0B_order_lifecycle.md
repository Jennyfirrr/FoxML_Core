# Plan 0B: Order Lifecycle Management

## Overview

Implement proper order state machine to handle the full lifecycle of orders: submission, partial fills, complete fills, rejections, and cancellations.

## Problem Statement

Current code assumes instant, complete fills:
```python
# In trading_engine.py:567-575
result = self.broker.submit_order(...)
fill_price = result.get("fill_price", 0.0)  # Assumes complete fill
# Updates position as if entire order filled
```

Production brokers have:
- Partial fills (order for 1000 shares, only 500 available)
- Pending orders (limit orders waiting to fill)
- Rejected orders (insufficient funds, market closed)
- Order timeouts (GTC orders that never fill)

## Files to Create

### 1. `LIVE_TRADING/common/order.py`

```python
"""
Order Management
================

Order state machine and tracking for live trading.

Order States:
    PENDING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
                        \-> REJECTED
                        \-> CANCELLED
                        \-> EXPIRED

SST Compliance:
- Immutable order records
- State transitions logged
- Timezone-aware timestamps
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import logging

from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = auto()       # Created, not yet submitted
    SUBMITTED = auto()     # Sent to broker
    PARTIALLY_FILLED = auto()  # Some quantity filled
    FILLED = auto()        # Fully filled
    REJECTED = auto()      # Broker rejected
    CANCELLED = auto()     # User cancelled
    EXPIRED = auto()       # Time-in-force expired


class OrderSide(Enum):
    """Order direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(Enum):
    """Order time-in-force."""
    DAY = "DAY"       # Cancel at market close
    GTC = "GTC"       # Good til cancelled
    IOC = "IOC"       # Immediate or cancel
    FOK = "FOK"       # Fill or kill


@dataclass
class OrderFill:
    """Record of a single fill."""
    fill_id: str
    qty: float
    price: float
    timestamp: datetime
    commission: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "qty": self.qty,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission,
        }


@dataclass
class OrderStateTransition:
    """Record of state transition."""
    from_status: OrderStatus
    to_status: OrderStatus
    timestamp: datetime
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from": self.from_status.name,
            "to": self.to_status.name,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }


@dataclass
class Order:
    """
    Order with full lifecycle tracking.

    Example:
        >>> order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        >>> order.submit("broker_order_123")
        >>> order.add_fill(OrderFill("fill_1", 50, 150.0, now))
        >>> order.add_fill(OrderFill("fill_2", 50, 150.05, now))
        >>> order.status
        OrderStatus.FILLED
    """

    # Identity
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType

    # Quantities
    qty: float                    # Requested quantity
    filled_qty: float = 0.0       # Quantity filled so far
    remaining_qty: float = 0.0    # Quantity remaining

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.DAY

    # Broker reference
    broker_order_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # History
    fills: List[OrderFill] = field(default_factory=list)
    transitions: List[OrderStateTransition] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize remaining quantity."""
        self.remaining_qty = self.qty - self.filled_qty

    @classmethod
    def create(
        cls,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        order_id: Optional[str] = None,
        clock: Optional[Clock] = None,
        **metadata,
    ) -> "Order":
        """
        Factory method to create an order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            qty: Quantity to trade
            order_type: MARKET, LIMIT, etc.
            limit_price: Limit price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            time_in_force: Order duration
            order_id: Custom order ID (auto-generated if not provided)
            clock: Clock for timestamps
            **metadata: Additional order metadata
        """
        clock = clock or get_clock()
        now = clock.now()

        if order_id is None:
            import uuid
            order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"

        return cls(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            created_at=now,
            metadata=metadata,
        )

    def _transition(
        self,
        new_status: OrderStatus,
        reason: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        """Record state transition."""
        clock = clock or get_clock()
        transition = OrderStateTransition(
            from_status=self.status,
            to_status=new_status,
            timestamp=clock.now(),
            reason=reason,
        )
        self.transitions.append(transition)
        self.status = new_status
        logger.debug(f"Order {self.order_id}: {transition.from_status.name} -> {new_status.name}")

    def submit(
        self,
        broker_order_id: str,
        clock: Optional[Clock] = None,
    ) -> None:
        """
        Mark order as submitted to broker.

        Args:
            broker_order_id: ID assigned by broker
            clock: Clock for timestamp
        """
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in {self.status.name} state")

        clock = clock or get_clock()
        self.broker_order_id = broker_order_id
        self.submitted_at = clock.now()
        self._transition(OrderStatus.SUBMITTED, clock=clock)

    def add_fill(
        self,
        fill: OrderFill,
        clock: Optional[Clock] = None,
    ) -> None:
        """
        Add a fill to the order.

        Args:
            fill: Fill record
            clock: Clock for timestamp
        """
        if self.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED):
            raise ValueError(f"Cannot fill order in {self.status.name} state")

        clock = clock or get_clock()

        # Record fill
        self.fills.append(fill)

        # Update quantities
        self.filled_qty += fill.qty
        self.remaining_qty = self.qty - self.filled_qty

        # Update average fill price
        total_value = sum(f.qty * f.price for f in self.fills)
        self.avg_fill_price = total_value / self.filled_qty

        # Transition state
        if self.remaining_qty <= 0:
            self.filled_at = clock.now()
            self._transition(OrderStatus.FILLED, clock=clock)
        elif self.status == OrderStatus.SUBMITTED:
            self._transition(OrderStatus.PARTIALLY_FILLED, clock=clock)

    def reject(
        self,
        reason: str,
        clock: Optional[Clock] = None,
    ) -> None:
        """
        Mark order as rejected.

        Args:
            reason: Rejection reason from broker
            clock: Clock for timestamp
        """
        if self.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            raise ValueError(f"Cannot reject order in {self.status.name} state")

        self._transition(OrderStatus.REJECTED, reason=reason, clock=clock)

    def cancel(
        self,
        reason: Optional[str] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        """
        Mark order as cancelled.

        Args:
            reason: Cancellation reason
            clock: Clock for timestamp
        """
        if self.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            raise ValueError(f"Cannot cancel order in {self.status.name} state")

        self._transition(OrderStatus.CANCELLED, reason=reason, clock=clock)

    def expire(
        self,
        clock: Optional[Clock] = None,
    ) -> None:
        """Mark order as expired."""
        if self.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            raise ValueError(f"Cannot expire order in {self.status.name} state")

        self._transition(OrderStatus.EXPIRED, reason="Time in force expired", clock=clock)

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        """Check if order is active (submitted and not terminal)."""
        return self.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)

    @property
    def fill_ratio(self) -> float:
        """Get fill ratio (0.0 to 1.0)."""
        return self.filled_qty / self.qty if self.qty > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "remaining_qty": self.remaining_qty,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.name,
            "time_in_force": self.time_in_force.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "fills": [f.to_dict() for f in self.fills],
            "transitions": [t.to_dict() for t in self.transitions],
            "metadata": self.metadata,
        }


class OrderBook:
    """
    Track active and historical orders.

    Example:
        >>> book = OrderBook()
        >>> order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        >>> book.add(order)
        >>> book.get_active_orders()
        [Order(...)]
    """

    def __init__(self):
        """Initialize order book."""
        self._orders: Dict[str, Order] = {}
        self._by_broker_id: Dict[str, str] = {}  # broker_order_id -> order_id

    def add(self, order: Order) -> None:
        """Add order to book."""
        self._orders[order.order_id] = order
        if order.broker_order_id:
            self._by_broker_id[order.broker_order_id] = order.order_id

    def get(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        """Get order by broker order ID."""
        order_id = self._by_broker_id.get(broker_order_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def update_broker_id(self, order_id: str, broker_order_id: str) -> None:
        """Update broker order ID mapping."""
        self._by_broker_id[broker_order_id] = order_id
        order = self._orders.get(order_id)
        if order:
            order.broker_order_id = broker_order_id

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        orders = [o for o in self._orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_pending_orders(self) -> List[Order]:
        """Get orders pending submission."""
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status."""
        return [o for o in self._orders.values() if o.status == status]

    def get_orders_for_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    def cancel_all_active(
        self,
        symbol: Optional[str] = None,
        reason: str = "Bulk cancel",
        clock: Optional[Clock] = None,
    ) -> List[Order]:
        """
        Cancel all active orders.

        Returns list of cancelled orders.
        """
        cancelled = []
        for order in self.get_active_orders(symbol):
            order.cancel(reason=reason, clock=clock)
            cancelled.append(order)
        return cancelled

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "orders": {oid: o.to_dict() for oid, o in self._orders.items()},
            "active_count": len(self.get_active_orders()),
            "total_count": len(self._orders),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBook":
        """Reconstruct from dictionary."""
        # Implementation for state restoration
        book = cls()
        # TODO: Deserialize orders
        return book
```

### 2. Update `LIVE_TRADING/common/__init__.py`

```python
# Add to existing exports
from .order import (
    Order,
    OrderBook,
    OrderStatus,
    OrderSide,
    OrderType,
    OrderFill,
    TimeInForce,
)
```

## Files to Modify

### 1. `LIVE_TRADING/engine/trading_engine.py`

Add order book and proper order handling:

```python
# Add to __init__
from LIVE_TRADING.common.order import Order, OrderBook, OrderStatus, OrderSide, OrderType, OrderFill

class TradingEngine:
    def __init__(self, ...):
        ...
        self._order_book = OrderBook()

    def _execute_decision(self, decision: TradeDecision) -> Optional[Dict[str, Any]]:
        """Execute a trade decision with proper order lifecycle."""
        # Create order
        order = Order.create(
            symbol=decision.symbol,
            side=OrderSide.BUY if decision.direction == "long" else OrderSide.SELL,
            qty=abs(decision.shares),
            order_type=OrderType.MARKET,
            clock=self._clock,
            decision_id=decision.decision_id,
        )
        self._order_book.add(order)

        try:
            # Submit to broker
            result = self.broker.submit_order(
                symbol=order.symbol,
                side=order.side.value,
                qty=order.qty,
                order_type=order.order_type.value.lower(),
            )

            # Update order with broker response
            broker_order_id = result.get("order_id")
            if broker_order_id:
                order.submit(broker_order_id, clock=self._clock)
                self._order_book.update_broker_id(order.order_id, broker_order_id)

            # Handle fills
            filled_qty = result.get("filled_qty", 0)
            fill_price = result.get("fill_price")

            if filled_qty > 0 and fill_price:
                fill = OrderFill(
                    fill_id=f"{broker_order_id}-1",
                    qty=filled_qty,
                    price=fill_price,
                    timestamp=self._clock.now(),
                )
                order.add_fill(fill, clock=self._clock)

            # Check for rejection
            if result.get("status") == "rejected":
                order.reject(result.get("reject_reason", "Unknown"), clock=self._clock)

            return order.to_dict()

        except Exception as e:
            order.reject(str(e), clock=self._clock)
            raise
```

### 2. `LIVE_TRADING/engine/state.py`

Add order book to state:

```python
@dataclass
class EngineState:
    ...
    order_book: OrderBook = field(default_factory=OrderBook)

    def to_dict(self) -> Dict[str, Any]:
        return {
            ...
            "order_book": self.order_book.to_dict(),
        }
```

## Tests

### `LIVE_TRADING/tests/test_order.py`

```python
"""
Order Management Tests
======================

Unit tests for order lifecycle.
"""

import pytest
from datetime import datetime, timezone

from LIVE_TRADING.common.order import (
    Order,
    OrderBook,
    OrderStatus,
    OrderSide,
    OrderType,
    OrderFill,
    TimeInForce,
)
from LIVE_TRADING.common.clock import SimulatedClock


class TestOrder:
    """Tests for Order class."""

    @pytest.fixture
    def clock(self):
        """Create simulated clock."""
        return SimulatedClock(datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc))

    def test_create_market_order(self, clock):
        """Test creating a market order."""
        order = Order.create(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MARKET,
            clock=clock,
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.qty == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.remaining_qty == 100

    def test_create_limit_order(self, clock):
        """Test creating a limit order."""
        order = Order.create(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=50,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            clock=clock,
        )

        assert order.limit_price == 150.0

    def test_submit_order(self, clock):
        """Test submitting order to broker."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)

        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == "BROKER-123"
        assert order.submitted_at is not None

    def test_full_fill(self, clock):
        """Test order fully filled."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)

        fill = OrderFill("FILL-1", 100, 150.0, clock.now())
        order.add_fill(fill, clock=clock)

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.remaining_qty == 0
        assert order.avg_fill_price == 150.0

    def test_partial_fill(self, clock):
        """Test partial fill."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)

        fill1 = OrderFill("FILL-1", 60, 150.0, clock.now())
        order.add_fill(fill1, clock=clock)

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_qty == 60
        assert order.remaining_qty == 40
        assert order.fill_ratio == 0.6

    def test_multiple_fills_to_complete(self, clock):
        """Test multiple partial fills completing order."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)

        order.add_fill(OrderFill("FILL-1", 30, 149.0, clock.now()), clock=clock)
        order.add_fill(OrderFill("FILL-2", 40, 150.0, clock.now()), clock=clock)
        order.add_fill(OrderFill("FILL-3", 30, 151.0, clock.now()), clock=clock)

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        # Weighted average: (30*149 + 40*150 + 30*151) / 100 = 150.0
        assert abs(order.avg_fill_price - 150.0) < 0.01

    def test_reject_order(self, clock):
        """Test order rejection."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)
        order.reject("Insufficient funds", clock=clock)

        assert order.status == OrderStatus.REJECTED
        assert len(order.transitions) == 2  # PENDING->SUBMITTED->REJECTED

    def test_cancel_order(self, clock):
        """Test order cancellation."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)
        order.cancel("User requested", clock=clock)

        assert order.status == OrderStatus.CANCELLED

    def test_cancel_partially_filled(self, clock):
        """Test cancelling partially filled order."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)
        order.add_fill(OrderFill("FILL-1", 30, 150.0, clock.now()), clock=clock)
        order.cancel("Market volatile", clock=clock)

        assert order.status == OrderStatus.CANCELLED
        assert order.filled_qty == 30  # Keep record of partial fill

    def test_cannot_fill_rejected_order(self, clock):
        """Test that rejected orders cannot be filled."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)
        order.reject("Bad symbol", clock=clock)

        with pytest.raises(ValueError, match="Cannot fill"):
            order.add_fill(OrderFill("FILL-1", 100, 150.0, clock.now()), clock=clock)

    def test_is_terminal(self, clock):
        """Test terminal state detection."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        assert not order.is_terminal

        order.submit("BROKER-123", clock=clock)
        assert not order.is_terminal

        order.add_fill(OrderFill("FILL-1", 100, 150.0, clock.now()), clock=clock)
        assert order.is_terminal

    def test_to_dict(self, clock):
        """Test serialization to dict."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)

        d = order.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["side"] == "BUY"
        assert d["status"] == "SUBMITTED"


class TestOrderBook:
    """Tests for OrderBook class."""

    @pytest.fixture
    def clock(self):
        """Create simulated clock."""
        return SimulatedClock(datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc))

    @pytest.fixture
    def book(self):
        """Create empty order book."""
        return OrderBook()

    def test_add_and_get(self, book, clock):
        """Test adding and retrieving orders."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        book.add(order)

        assert book.get(order.order_id) is order

    def test_get_by_broker_id(self, book, clock):
        """Test retrieval by broker order ID."""
        order = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        order.submit("BROKER-123", clock=clock)
        book.add(order)

        assert book.get_by_broker_id("BROKER-123") is order

    def test_get_active_orders(self, book, clock):
        """Test getting active orders."""
        o1 = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        o1.submit("B-1", clock=clock)

        o2 = Order.create("MSFT", OrderSide.BUY, 50, OrderType.MARKET, clock=clock)
        o2.submit("B-2", clock=clock)
        o2.add_fill(OrderFill("F-1", 50, 300.0, clock.now()), clock=clock)

        o3 = Order.create("GOOG", OrderSide.BUY, 25, OrderType.MARKET, clock=clock)
        # Not submitted

        book.add(o1)
        book.add(o2)
        book.add(o3)

        active = book.get_active_orders()
        assert len(active) == 1
        assert active[0].symbol == "AAPL"

    def test_get_active_orders_by_symbol(self, book, clock):
        """Test getting active orders filtered by symbol."""
        o1 = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        o1.submit("B-1", clock=clock)

        o2 = Order.create("AAPL", OrderSide.SELL, 50, OrderType.MARKET, clock=clock)
        o2.submit("B-2", clock=clock)

        o3 = Order.create("MSFT", OrderSide.BUY, 25, OrderType.MARKET, clock=clock)
        o3.submit("B-3", clock=clock)

        book.add(o1)
        book.add(o2)
        book.add(o3)

        active = book.get_active_orders("AAPL")
        assert len(active) == 2

    def test_cancel_all_active(self, book, clock):
        """Test bulk cancellation."""
        o1 = Order.create("AAPL", OrderSide.BUY, 100, OrderType.MARKET, clock=clock)
        o1.submit("B-1", clock=clock)

        o2 = Order.create("MSFT", OrderSide.BUY, 50, OrderType.MARKET, clock=clock)
        o2.submit("B-2", clock=clock)

        book.add(o1)
        book.add(o2)

        cancelled = book.cancel_all_active(reason="Kill switch", clock=clock)

        assert len(cancelled) == 2
        assert all(o.status == OrderStatus.CANCELLED for o in cancelled)
```

## SST Compliance

- [x] Immutable order records (append-only fills/transitions)
- [x] Timezone-aware timestamps
- [x] Serializable to dict for persistence
- [x] Protocol-compatible with clock abstraction

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `common/order.py` | 400 |
| `tests/test_order.py` | 250 |
| Modifications to trading_engine.py | ~80 |
| **Total** | ~730 |
