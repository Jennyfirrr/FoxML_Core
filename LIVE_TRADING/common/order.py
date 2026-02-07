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

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from LIVE_TRADING.common.clock import Clock, get_clock

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states."""
    PENDING = auto()           # Created, not yet submitted
    SUBMITTED = auto()         # Sent to broker
    PARTIALLY_FILLED = auto()  # Some quantity filled
    FILLED = auto()            # Fully filled
    REJECTED = auto()          # Broker rejected
    CANCELLED = auto()         # User cancelled
    EXPIRED = auto()           # Time-in-force expired


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
            "commission": self.commission,
            "fill_id": self.fill_id,
            "price": self.price,
            "qty": self.qty,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderFill":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso
        return cls(
            fill_id=d["fill_id"],
            qty=d["qty"],
            price=d["price"],
            timestamp=parse_iso(d["timestamp"]),
            commission=d.get("commission", 0.0),
        )


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
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "to": self.to_status.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderStateTransition":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso
        return cls(
            from_status=OrderStatus[d["from"]],
            to_status=OrderStatus[d["to"]],
            timestamp=parse_iso(d["timestamp"]),
            reason=d.get("reason"),
        )


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
        if self.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED):
            raise ValueError(f"Cannot cancel order in {self.status.name} state")

        self._transition(OrderStatus.CANCELLED, reason=reason, clock=clock)

    def expire(
        self,
        clock: Optional[Clock] = None,
    ) -> None:
        """Mark order as expired."""
        if self.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED):
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
            "avg_fill_price": self.avg_fill_price,
            "broker_order_id": self.broker_order_id,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_qty": self.filled_qty,
            "fills": [f.to_dict() for f in self.fills],
            "limit_price": self.limit_price,
            "metadata": self.metadata,
            "order_id": self.order_id,
            "order_type": self.order_type.value,
            "qty": self.qty,
            "remaining_qty": self.remaining_qty,
            "side": self.side.value,
            "status": self.status.name,
            "stop_price": self.stop_price,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "symbol": self.symbol,
            "time_in_force": self.time_in_force.value,
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Order":
        """Create from dictionary."""
        from LIVE_TRADING.common.time_utils import parse_iso

        order = cls(
            order_id=d["order_id"],
            symbol=d["symbol"],
            side=OrderSide(d["side"]),
            order_type=OrderType(d["order_type"]),
            qty=d["qty"],
            filled_qty=d.get("filled_qty", 0.0),
            limit_price=d.get("limit_price"),
            stop_price=d.get("stop_price"),
            avg_fill_price=d.get("avg_fill_price"),
            status=OrderStatus[d["status"]],
            time_in_force=TimeInForce(d.get("time_in_force", "DAY")),
            broker_order_id=d.get("broker_order_id"),
            created_at=parse_iso(d["created_at"]),
            submitted_at=parse_iso(d["submitted_at"]) if d.get("submitted_at") else None,
            filled_at=parse_iso(d["filled_at"]) if d.get("filled_at") else None,
            fills=[OrderFill.from_dict(f) for f in d.get("fills", [])],
            transitions=[OrderStateTransition.from_dict(t) for t in d.get("transitions", [])],
            metadata=d.get("metadata", {}),
        )
        return order


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
            "active_count": len(self.get_active_orders()),
            "orders": {oid: o.to_dict() for oid, o in self._orders.items()},
            "total_count": len(self._orders),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBook":
        """Reconstruct from dictionary."""
        book = cls()
        for order_id, order_data in data.get("orders", {}).items():
            order = Order.from_dict(order_data)
            book.add(order)
        return book

    def __len__(self) -> int:
        """Get total number of orders."""
        return len(self._orders)
