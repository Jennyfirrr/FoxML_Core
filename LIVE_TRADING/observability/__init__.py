"""
Observability Module
====================

Metrics and event system for monitoring live trading.

Components:
- MetricsRegistry: Central metrics storage with Counter, Gauge, Histogram
- EventBus: Publish/subscribe event system

Usage:
    >>> from LIVE_TRADING.observability import metrics, events
    >>> from LIVE_TRADING.observability import EventType, Event

    # Record a metric
    >>> metrics.trades_total.inc()
    >>> metrics.portfolio_value.set(100_000.0)

    # Subscribe to events
    >>> def on_trade(event):
    ...     print(f"Trade: {event.data}")
    >>> events.subscribe(EventType.TRADE_FILLED, on_trade)

    # Emit events
    >>> from LIVE_TRADING.observability import emit_trade
    >>> emit_trade("AAPL", "BUY", 100, 150.0)
"""

from LIVE_TRADING.observability.metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    MetricValue,
    metrics,
    # Pre-defined metrics
    trades_total,
    trades_value,
    decisions_total,
    cycles_total,
    errors_total,
    portfolio_value,
    cash_balance,
    positions_count,
    daily_pnl,
    cycle_duration,
    order_latency,
)
from LIVE_TRADING.observability.events import (
    EventBus,
    Event,
    EventType,
    Severity,
    events,
    # Helper functions
    emit_trade,
    emit_decision,
    emit_kill_switch,
    emit_error,
    emit_warning,
    emit_cycle_start,
    emit_cycle_end,
    emit_heartbeat,
)

__all__ = [
    # Metrics
    "MetricsRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricValue",
    "metrics",
    # Pre-defined metrics
    "trades_total",
    "trades_value",
    "decisions_total",
    "cycles_total",
    "errors_total",
    "portfolio_value",
    "cash_balance",
    "positions_count",
    "daily_pnl",
    "cycle_duration",
    "order_latency",
    # Events
    "EventBus",
    "Event",
    "EventType",
    "Severity",
    "events",
    # Event helpers
    "emit_trade",
    "emit_decision",
    "emit_kill_switch",
    "emit_error",
    "emit_warning",
    "emit_cycle_start",
    "emit_cycle_end",
    "emit_heartbeat",
]
