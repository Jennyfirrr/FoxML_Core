"""
Metrics Registry
================

Central metrics collection for monitoring.

Provides:
- Counter: Monotonically increasing values
- Gauge: Point-in-time values
- Histogram: Distribution of values

Designed for easy integration with Prometheus, StatsD, etc.

SST Compliance:
- Thread-safe
- Configurable prefix via get_cfg()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with timestamp and labels."""

    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter metric (monotonically increasing).

    Counters track cumulative values that only go up, like
    total requests, trades executed, or errors.

    Example:
        >>> counter = Counter("trades_total", "Total trades executed")
        >>> counter.inc()
        >>> counter.inc(5, labels={"symbol": "AAPL"})
        >>> counter.get()
        1
        >>> counter.get(labels={"symbol": "AAPL"})
        5
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize counter.

        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._values: Dict[str, float] = {}  # label_key -> value
        self._lock = threading.Lock()

    def inc(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment counter.

        Args:
            amount: Amount to increment (default 1)
            labels: Optional label dict for multi-dimensional metrics
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented")

        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0) + amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get current value.

        Args:
            labels: Label dict to query

        Returns:
            Current counter value
        """
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0)

    def reset(self, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Reset counter (use with caution).

        Args:
            labels: Label to reset (None = all)
        """
        with self._lock:
            if labels is None:
                self._values.clear()
            else:
                label_key = self._labels_to_key(labels)
                self._values.pop(label_key, None)

    def collect(self) -> List[MetricValue]:
        """
        Collect all values for export.

        Returns:
            List of MetricValue instances
        """
        now = datetime.now(timezone.utc)
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = self._key_to_labels(label_key)
                result.append(MetricValue(value=value, timestamp=now, labels=labels))
        return result

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert key back to labels."""
        if not key:
            return {}
        return dict(kv.split("=") for kv in key.split(","))


class Gauge:
    """
    Gauge metric (point-in-time value).

    Gauges track values that can go up or down, like
    portfolio value, active connections, or queue depth.

    Example:
        >>> gauge = Gauge("portfolio_value", "Current portfolio value")
        >>> gauge.set(100000.0)
        >>> gauge.inc(5000.0)
        >>> gauge.dec(1000.0)
        >>> gauge.get()
        104000.0
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize gauge.

        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set gauge value.

        Args:
            value: New value
            labels: Optional label dict
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment gauge.

        Args:
            amount: Amount to increment
            labels: Optional label dict
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0) + amount

    def dec(
        self,
        amount: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Decrement gauge.

        Args:
            amount: Amount to decrement
            labels: Optional label dict
        """
        self.inc(-amount, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get current value.

        Args:
            labels: Label dict to query

        Returns:
            Current gauge value
        """
        label_key = self._labels_to_key(labels)
        return self._values.get(label_key, 0)

    def collect(self) -> List[MetricValue]:
        """
        Collect all values for export.

        Returns:
            List of MetricValue instances
        """
        now = datetime.now(timezone.utc)
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = self._key_to_labels(label_key)
                result.append(MetricValue(value=value, timestamp=now, labels=labels))
        return result

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert key back to labels."""
        if not key:
            return {}
        return dict(kv.split("=") for kv in key.split(","))


class Histogram:
    """
    Histogram metric (distribution).

    Histograms track the distribution of values, useful for
    latencies, durations, and sizes.

    Example:
        >>> hist = Histogram("latency_ms", "Request latency", buckets=[10, 50, 100, 500])
        >>> hist.observe(45.0)
        >>> hist.observe(150.0)
    """

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ):
        """
        Initialize histogram.

        Args:
            name: Metric name
            description: Human-readable description
            buckets: Bucket boundaries (sorted)
        """
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[str, Dict[float, int]] = {}  # label_key -> bucket -> count
        self._sums: Dict[str, float] = {}  # label_key -> sum
        self._totals: Dict[str, int] = {}  # label_key -> total count
        self._lock = threading.Lock()

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record an observation.

        Args:
            value: Observed value
            labels: Optional label dict
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            # Initialize if needed
            if label_key not in self._counts:
                self._counts[label_key] = {b: 0 for b in self.buckets}
                self._counts[label_key][float("inf")] = 0
                self._sums[label_key] = 0
                self._totals[label_key] = 0

            # Update buckets (cumulative)
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
            self._counts[label_key][float("inf")] += 1

            # Update sum and total
            self._sums[label_key] += value
            self._totals[label_key] += 1

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get total observation count."""
        label_key = self._labels_to_key(labels)
        return self._totals.get(label_key, 0)

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get sum of observations."""
        label_key = self._labels_to_key(labels)
        return self._sums.get(label_key, 0)

    def get_mean(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get mean of observations."""
        count = self.get_count(labels)
        if count == 0:
            return 0
        return self.get_sum(labels) / count

    def get_buckets(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[float, int]:
        """Get bucket counts."""
        label_key = self._labels_to_key(labels)
        return dict(self._counts.get(label_key, {}))

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsRegistry:
    """
    Central registry for all metrics.

    Provides a single point for creating and accessing metrics,
    with automatic prefixing and collection.

    Example:
        >>> registry = MetricsRegistry(prefix="live_trading")
        >>> trades = registry.counter("trades_total", "Total trades")
        >>> trades.inc()
        >>> registry.to_dict()
        {'live_trading_trades_total': 1}
    """

    def __init__(self, prefix: str = "live_trading"):
        """
        Initialize registry.

        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def counter(self, name: str, description: str = "") -> Counter:
        """
        Get or create a counter.

        Args:
            name: Metric name (will be prefixed)
            description: Human-readable description

        Returns:
            Counter instance
        """
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(full_name, description)
            return self._metrics[full_name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """
        Get or create a gauge.

        Args:
            name: Metric name (will be prefixed)
            description: Human-readable description

        Returns:
            Gauge instance
        """
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
        """
        Get or create a histogram.

        Args:
            name: Metric name (will be prefixed)
            description: Human-readable description
            buckets: Bucket boundaries

        Returns:
            Histogram instance
        """
        full_name = f"{self.prefix}_{name}"
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(full_name, description, buckets)
            return self._metrics[full_name]

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """
        Collect all metrics for export.

        Returns:
            Dict mapping metric name to list of values
        """
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if hasattr(metric, "collect"):
                    result[name] = metric.collect()
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Export metrics as dict (for JSON serialization).

        Returns:
            Dict mapping metric name to current value
        """
        result = {}
        with self._lock:
            for name, metric in self._metrics.items():
                if isinstance(metric, (Counter, Gauge)):
                    result[name] = metric.get()
                elif isinstance(metric, Histogram):
                    result[f"{name}_count"] = metric.get_count()
                    result[f"{name}_sum"] = metric.get_sum()
        return result

    def register_callback(self, callback: Callable[[], None]) -> None:
        """
        Register callback for metric collection.

        Callbacks are invoked before collection to update dynamic metrics.

        Args:
            callback: Function to call before collection
        """
        self._callbacks.append(callback)

    def reset_all(self) -> None:
        """Reset all metrics (use with caution)."""
        with self._lock:
            self._metrics.clear()


# Global metrics registry
metrics = MetricsRegistry(
    prefix=get_cfg("live_trading.observability.metrics.prefix", default="live_trading")
)

# Pre-defined counters
trades_total = metrics.counter("trades_total", "Total trades executed")
trades_value = metrics.counter("trades_value_usd", "Total trade value in USD")
decisions_total = metrics.counter("decisions_total", "Total trading decisions")
cycles_total = metrics.counter("cycles_total", "Total trading cycles")
errors_total = metrics.counter("errors_total", "Total errors")

# Pre-defined gauges
portfolio_value = metrics.gauge("portfolio_value_usd", "Current portfolio value")
cash_balance = metrics.gauge("cash_balance_usd", "Current cash balance")
positions_count = metrics.gauge("positions_count", "Number of open positions")
daily_pnl = metrics.gauge("daily_pnl_usd", "Daily P&L")

# Pre-defined histograms
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
