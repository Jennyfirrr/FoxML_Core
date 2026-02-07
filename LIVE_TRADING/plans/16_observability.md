# Plan 16: Observability

## Overview

Metrics hooks and event system for monitoring and future dashboard integration.

## Design Goals

1. **Non-Intrusive**: Minimal performance overhead
2. **Extensible**: Easy to add new metrics and exporters
3. **Decoupled**: Core engine doesn't depend on specific backends
4. **Complete**: Capture all significant events

## Files to Create

### 1. `LIVE_TRADING/observability/__init__.py`

```python
"""
Observability Module
====================

Metrics and event system for monitoring.

Components:
- MetricsRegistry: Central metrics storage
- EventBus: Publish/subscribe event system
- Exporters: Future Prometheus/StatsD support
"""

from .metrics import MetricsRegistry, Counter, Gauge, Histogram, metrics
from .events import EventBus, Event, EventType, events

__all__ = [
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "metrics",
    "EventBus",
    "Event",
    "EventType",
    "events",
]
```

### 2. `LIVE_TRADING/observability/metrics.py`

```python
"""
Metrics Registry
================

Central metrics collection for monitoring.

Provides:
- Counter: Monotonically increasing values
- Gauge: Point-in-time values
- Histogram: Distribution of values

Designed for easy integration with Prometheus, StatsD, etc.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp."""

    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter metric (monotonically increasing).

    Example:
        >>> counter = Counter("trades_total", "Total trades executed")
        >>> counter.inc()
        >>> counter.inc(5, labels={"symbol": "AAPL"})
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = {}  # label_key -> value
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all values for export."""
        now = datetime.now(timezone.utc)
        result = []
        for label_key, value in self._values.items():
            labels = self._key_to_labels(label_key)
            result.append(MetricValue(value=value, timestamp=now, labels=labels))
        return result

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert key back to labels."""
        if not key:
            return {}
        return dict(kv.split("=") for kv in key.split(","))


class Gauge:
    """
    Gauge metric (point-in-time value).

    Example:
        >>> gauge = Gauge("portfolio_value", "Current portfolio value")
        >>> gauge.set(100000.0)
        >>> gauge.set(50000.0, labels={"account": "paper"})
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge."""
        self.inc(-amount, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0)

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def collect(self) -> List[MetricValue]:
        """Collect all values for export."""
        now = datetime.now(timezone.utc)
        result = []
        for label_key, value in self._values.items():
            labels = dict(kv.split("=") for kv in label_key.split(",")) if label_key else {}
            result.append(MetricValue(value=value, timestamp=now, labels=labels))
        return result


class Histogram:
    """
    Histogram metric (distribution).

    Example:
        >>> hist = Histogram("latency_ms", "Request latency", buckets=[10, 50, 100, 500])
        >>> hist.observe(45.0)
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[str, Dict[float, int]] = {}  # label_key -> bucket -> count
        self._sums: Dict[str, float] = {}  # label_key -> sum
        self._totals: Dict[str, int] = {}  # label_key -> total count
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            # Initialize if needed
            if label_key not in self._counts:
                self._counts[label_key] = {b: 0 for b in self.buckets}
                self._counts[label_key][float("inf")] = 0
                self._sums[label_key] = 0
                self._totals[label_key] = 0

            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
            self._counts[label_key][float("inf")] += 1

            # Update sum and total
            self._sums[label_key] += value
            self._totals[label_key] += 1

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsRegistry:
    """
    Central registry for all metrics.

    Example:
        >>> registry = MetricsRegistry(prefix="live_trading")
        >>> trades = registry.counter("trades_total", "Total trades")
        >>> trades.inc()
    """

    def __init__(self, prefix: str = "live_trading"):
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, description)
            return self._metrics[full_name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(full_name, description)
            return self._metrics[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, description, buckets)
            return self._metrics[full_name]

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metrics for export."""
        result = {}
        for name, metric in self._metrics.items():
            if hasattr(metric, "collect"):
                result[name] = metric.collect()
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dict (for JSON serialization)."""
        result = {}
        for name, metric in self._metrics.items():
            if isinstance(metric, (Counter, Gauge)):
                result[name] = metric.get()
        return result

    def register_callback(self, callback: Callable) -> None:
        """Register callback for metric updates."""
        self._callbacks.append(callback)


# Global metrics registry
metrics = MetricsRegistry(
    prefix=get_cfg("live_trading.observability.metrics.prefix", default="live_trading")
)

# Pre-defined metrics
trades_total = metrics.counter("trades_total", "Total trades executed")
trades_value = metrics.counter("trades_value_usd", "Total trade value in USD")
decisions_total = metrics.counter("decisions_total", "Total trading decisions")
cycles_total = metrics.counter("cycles_total", "Total trading cycles")
errors_total = metrics.counter("errors_total", "Total errors")

portfolio_value = metrics.gauge("portfolio_value_usd", "Current portfolio value")
cash_balance = metrics.gauge("cash_balance_usd", "Current cash balance")
positions_count = metrics.gauge("positions_count", "Number of open positions")
daily_pnl = metrics.gauge("daily_pnl_usd", "Daily P&L")

cycle_duration = metrics.histogram(
    "cycle_duration_seconds",
    "Trading cycle duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
order_latency = metrics.histogram(
    "order_latency_seconds",
    "Order submission latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)
```

### 3. `LIVE_TRADING/observability/events.py`

```python
"""
Event Bus
=========

Publish/subscribe event system for system-wide notifications.

Events are used for:
- Alerting triggers
- Audit logging
- Monitoring dashboards
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from collections import deque

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types."""

    # Trading events
    CYCLE_START = auto()
    CYCLE_END = auto()
    TRADE_SUBMITTED = auto()
    TRADE_FILLED = auto()
    TRADE_REJECTED = auto()
    DECISION_MADE = auto()

    # Risk events
    KILL_SWITCH_TRIGGERED = auto()
    DAILY_LOSS_WARNING = auto()
    DRAWDOWN_WARNING = auto()
    POSITION_LIMIT_WARNING = auto()

    # System events
    ENGINE_START = auto()
    ENGINE_STOP = auto()
    ERROR = auto()
    CONNECTION_LOST = auto()
    CONNECTION_RESTORED = auto()

    # Data events
    QUOTE_STALE = auto()
    DATA_ERROR = auto()


@dataclass
class Event:
    """Event with metadata."""

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


class EventBus:
    """
    Publish/subscribe event bus.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe(EventType.TRADE_FILLED, lambda e: print(e))
        >>> bus.publish(Event(EventType.TRADE_FILLED, data={"symbol": "AAPL"}))
    """

    def __init__(self, buffer_size: int = 1000):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._global_subscribers: List[Callable[[Event], None]] = []
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: Optional[EventType],
        callback: Callable[[Event], None],
    ) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Event type to subscribe to (None for all)
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
        """Unsubscribe from events."""
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
        # Buffer event
        with self._lock:
            self._buffer.append(event)

        # Call subscribers
        callbacks = list(self._global_subscribers)
        if event.event_type in self._subscribers:
            callbacks.extend(self._subscribers[event.event_type])

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_recent(self, count: int = 100) -> List[Event]:
        """Get recent events from buffer."""
        with self._lock:
            return list(self._buffer)[-count:]

    def get_by_type(self, event_type: EventType, count: int = 100) -> List[Event]:
        """Get recent events of a specific type."""
        with self._lock:
            matching = [e for e in self._buffer if e.event_type == event_type]
            return matching[-count:]


# Global event bus
events = EventBus(
    buffer_size=get_cfg("live_trading.observability.events.buffer_size", default=1000)
)


# Helper functions for common events
def emit_trade(symbol: str, side: str, qty: float, price: float, **kwargs) -> None:
    """Emit trade event."""
    events.publish(Event(
        event_type=EventType.TRADE_FILLED,
        data={"symbol": symbol, "side": side, "qty": qty, "price": price, **kwargs},
    ))


def emit_kill_switch(reason: str, value: float, limit: float) -> None:
    """Emit kill switch event."""
    events.publish(Event(
        event_type=EventType.KILL_SWITCH_TRIGGERED,
        severity="critical",
        data={"reason": reason, "value": value, "limit": limit},
    ))


def emit_error(error: str, context: Optional[Dict] = None) -> None:
    """Emit error event."""
    events.publish(Event(
        event_type=EventType.ERROR,
        severity="error",
        data={"error": error, "context": context or {}},
    ))
```

## Integration Points

### Trading Engine Integration

Add metric/event hooks to `TradingEngine`:

```python
# In trading_engine.py

from LIVE_TRADING.observability import metrics, events, EventType, Event

class TradingEngine:
    def run_cycle(self, symbols, current_time=None):
        # Emit cycle start
        events.publish(Event(EventType.CYCLE_START))
        metrics.cycles_total.inc()

        start = time.time()
        try:
            # ... existing cycle logic ...

            # Update metrics
            metrics.portfolio_value.set(result.portfolio_value)
            metrics.cash_balance.set(result.cash)
            metrics.decisions_total.inc(len(result.decisions))

            for decision in result.decisions:
                if decision.decision == DECISION_TRADE:
                    metrics.trades_total.inc(labels={"symbol": decision.symbol})

        finally:
            duration = time.time() - start
            metrics.cycle_duration.observe(duration)
            events.publish(Event(EventType.CYCLE_END, data={"duration": duration}))
```

## Tests

### `LIVE_TRADING/tests/test_observability.py`

```python
"""Tests for observability module."""

import pytest
from LIVE_TRADING.observability import (
    MetricsRegistry, Counter, Gauge, Histogram,
    EventBus, Event, EventType,
)


class TestCounter:
    def test_inc(self):
        counter = Counter("test")
        counter.inc()
        assert counter.get() == 1
        counter.inc(5)
        assert counter.get() == 6

    def test_labels(self):
        counter = Counter("test")
        counter.inc(labels={"symbol": "AAPL"})
        counter.inc(labels={"symbol": "MSFT"})
        assert counter.get(labels={"symbol": "AAPL"}) == 1
        assert counter.get(labels={"symbol": "MSFT"}) == 1


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []

        bus.subscribe(EventType.TRADE_FILLED, lambda e: received.append(e))
        bus.publish(Event(EventType.TRADE_FILLED, data={"symbol": "AAPL"}))

        assert len(received) == 1
        assert received[0].data["symbol"] == "AAPL"
```

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `observability/__init__.py` | 20 |
| `observability/metrics.py` | 250 |
| `observability/events.py` | 180 |
| `tests/test_observability.py` | 100 |
| **Total** | ~550 |
