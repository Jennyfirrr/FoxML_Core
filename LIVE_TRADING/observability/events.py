"""
Event Bus
=========

Publish/subscribe event system for system-wide notifications.

Events are used for:
- Alerting triggers
- Audit logging
- Monitoring dashboards
- Integration with external systems

SST Compliance:
- Thread-safe
- UTC timestamps
- Configurable via get_cfg()
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the trading system."""

    # Trading events
    CYCLE_START = auto()
    CYCLE_END = auto()
    TRADE_SUBMITTED = auto()
    TRADE_FILLED = auto()
    TRADE_REJECTED = auto()
    TRADE_CANCELLED = auto()
    DECISION_MADE = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()

    # Risk events
    KILL_SWITCH_TRIGGERED = auto()
    DAILY_LOSS_WARNING = auto()
    DRAWDOWN_WARNING = auto()
    POSITION_LIMIT_WARNING = auto()
    EXPOSURE_WARNING = auto()
    GATE_BLOCKED = auto()

    # System events
    ENGINE_START = auto()
    ENGINE_STOP = auto()
    ENGINE_PAUSE = auto()
    ENGINE_RESUME = auto()
    ERROR = auto()
    WARNING = auto()
    HEARTBEAT = auto()

    # Connection events
    CONNECTION_LOST = auto()
    CONNECTION_RESTORED = auto()
    BROKER_CONNECTED = auto()
    BROKER_DISCONNECTED = auto()

    # Data events
    QUOTE_STALE = auto()
    QUOTE_RECEIVED = auto()
    DATA_ERROR = auto()
    DATA_GAP_DETECTED = auto()

    # Reconciliation events
    RECONCILIATION_START = auto()
    RECONCILIATION_END = auto()
    RECONCILIATION_MISMATCH = auto()

    # Backtest events
    BACKTEST_START = auto()
    BACKTEST_END = auto()

    # Pipeline stage events
    STAGE_CHANGE = auto()


class Severity(Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    """
    Event with metadata.

    Example:
        >>> event = Event(
        ...     event_type=EventType.TRADE_FILLED,
        ...     data={"symbol": "AAPL", "qty": 100},
        ...     severity="info",
        ... )
        >>> event.to_dict()
    """

    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "live_trading"
    severity: str = "info"  # debug, info, warning, error, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        """Human-readable string."""
        return f"[{self.severity.upper()}] {self.event_type.name}: {self.data}"


class EventBus:
    """
    Publish/subscribe event bus.

    Allows components to publish events and subscribe to specific
    event types or all events.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe(EventType.TRADE_FILLED, lambda e: print(e))
        >>> bus.publish(Event(EventType.TRADE_FILLED, data={"symbol": "AAPL"}))
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize event bus.

        Args:
            buffer_size: Maximum events to keep in history buffer
        """
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._global_subscribers: List[Callable[[Event], None]] = []
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._enabled = True

    def subscribe(
        self,
        event_type: Optional[EventType],
        callback: Callable[[Event], None],
    ) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Event type to subscribe to (None for all events)
            callback: Function to call when event fires
        """
        with self._lock:
            if event_type is None:
                self._global_subscribers.append(callback)
            else:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: Optional[EventType],
        callback: Callable[[Event], None],
    ) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from (None for global)
            callback: Callback to remove
        """
        with self._lock:
            if event_type is None:
                if callback in self._global_subscribers:
                    self._global_subscribers.remove(callback)
            elif event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish
        """
        if not self._enabled:
            return

        # Buffer event
        with self._lock:
            self._buffer.append(event)

        # Get callbacks (copy to avoid holding lock during callbacks)
        with self._lock:
            callbacks = list(self._global_subscribers)
            if event.event_type in self._subscribers:
                callbacks.extend(self._subscribers[event.event_type])

        # Call subscribers
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error for {event.event_type}: {e}")

    def get_recent(self, count: int = 100) -> List[Event]:
        """
        Get recent events from buffer.

        Args:
            count: Maximum events to return

        Returns:
            List of recent events (newest last)
        """
        with self._lock:
            return list(self._buffer)[-count:]

    def get_by_type(
        self,
        event_type: EventType,
        count: int = 100,
    ) -> List[Event]:
        """
        Get recent events of a specific type.

        Args:
            event_type: Event type to filter
            count: Maximum events to return

        Returns:
            List of matching events
        """
        with self._lock:
            matching = [e for e in self._buffer if e.event_type == event_type]
            return matching[-count:]

    def get_by_severity(
        self,
        severity: str,
        count: int = 100,
    ) -> List[Event]:
        """
        Get recent events by severity.

        Args:
            severity: Severity level (info, warning, error, critical)
            count: Maximum events to return

        Returns:
            List of matching events
        """
        with self._lock:
            matching = [e for e in self._buffer if e.severity == severity]
            return matching[-count:]

    def clear_buffer(self) -> None:
        """Clear event buffer."""
        with self._lock:
            self._buffer.clear()

    def enable(self) -> None:
        """Enable event publishing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event publishing."""
        self._enabled = False

    @property
    def subscriber_count(self) -> int:
        """Get total subscriber count."""
        with self._lock:
            count = len(self._global_subscribers)
            for subs in self._subscribers.values():
                count += len(subs)
            return count


# Global event bus
events = EventBus(
    buffer_size=get_cfg("live_trading.observability.events.buffer_size", default=1000)
)


# =============================================================================
# Helper Functions for Common Events
# =============================================================================


def emit_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    order_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Emit trade filled event.

    Args:
        symbol: Trading symbol
        side: BUY or SELL
        qty: Quantity traded
        price: Fill price
        order_id: Optional order ID
        **kwargs: Additional data
    """
    events.publish(
        Event(
            event_type=EventType.TRADE_FILLED,
            data={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "order_id": order_id,
                **kwargs,
            },
        )
    )


def emit_decision(
    symbol: str,
    decision: str,
    alpha: float,
    horizon: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Emit trading decision event.

    Args:
        symbol: Trading symbol
        decision: TRADE, HOLD, or BLOCKED
        alpha: Alpha/signal strength
        horizon: Selected horizon
        **kwargs: Additional data
    """
    events.publish(
        Event(
            event_type=EventType.DECISION_MADE,
            data={
                "symbol": symbol,
                "decision": decision,
                "alpha": alpha,
                "horizon": horizon,
                **kwargs,
            },
        )
    )


def emit_kill_switch(
    reason: str,
    value: float,
    limit: float,
    **kwargs: Any,
) -> None:
    """
    Emit kill switch triggered event.

    Args:
        reason: Kill switch reason
        value: Current value
        limit: Limit that was exceeded
        **kwargs: Additional data
    """
    events.publish(
        Event(
            event_type=EventType.KILL_SWITCH_TRIGGERED,
            severity="critical",
            data={
                "reason": reason,
                "value": value,
                "limit": limit,
                **kwargs,
            },
        )
    )


def emit_error(
    error: str,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
) -> None:
    """
    Emit error event.

    Args:
        error: Error message
        context: Optional context dict
        exception: Optional exception instance
    """
    data = {"error": error, "context": context or {}}
    if exception:
        data["exception_type"] = type(exception).__name__
        data["exception_str"] = str(exception)

    events.publish(
        Event(
            event_type=EventType.ERROR,
            severity="error",
            data=data,
        )
    )


def emit_warning(
    message: str,
    category: str = "general",
    **kwargs: Any,
) -> None:
    """
    Emit warning event.

    Args:
        message: Warning message
        category: Warning category
        **kwargs: Additional data
    """
    events.publish(
        Event(
            event_type=EventType.WARNING,
            severity="warning",
            data={"message": message, "category": category, **kwargs},
        )
    )


def emit_cycle_start(cycle_id: int, symbols: List[str]) -> None:
    """Emit cycle start event."""
    events.publish(
        Event(
            event_type=EventType.CYCLE_START,
            data={"cycle_id": cycle_id, "symbols": symbols},
        )
    )


def emit_cycle_end(
    cycle_id: int,
    duration_seconds: float,
    decisions_count: int,
    trades_count: int,
) -> None:
    """Emit cycle end event."""
    events.publish(
        Event(
            event_type=EventType.CYCLE_END,
            data={
                "cycle_id": cycle_id,
                "duration_seconds": duration_seconds,
                "decisions_count": decisions_count,
                "trades_count": trades_count,
            },
        )
    )


def emit_heartbeat() -> None:
    """Emit heartbeat event."""
    events.publish(
        Event(
            event_type=EventType.HEARTBEAT,
            severity="debug",
            data={},
        )
    )


def emit_stage_change(
    stage: str,
    symbol: Optional[str] = None,
    cycle_id: Optional[int] = None,
) -> None:
    """
    Emit pipeline stage change event.

    Args:
        stage: New stage name (idle, prediction, blending, arbitration, gating, sizing, risk, execution)
        symbol: Optional symbol being processed
        cycle_id: Optional cycle ID
    """
    events.publish(
        Event(
            event_type=EventType.STAGE_CHANGE,
            severity="debug",
            data={
                "stage": stage,
                "symbol": symbol,
                "cycle_id": cycle_id,
            },
        )
    )
