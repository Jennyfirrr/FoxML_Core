"""
Alert Manager
=============

Central alert coordination with batching and filtering.

SST Compliance:
- Configuration via get_cfg()
- Severity filtering from config
- Rate limiting to prevent alert spam
- Thread-safe alert history
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.alerting.channels import AlertChannel, LogChannel

logger = logging.getLogger(__name__)


class AlertSeverity(IntEnum):
    """
    Alert severity levels.

    Higher values indicate more severe alerts.
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, s: str) -> "AlertSeverity":
        """
        Parse severity from string.

        Args:
            s: Severity string (case-insensitive)

        Returns:
            AlertSeverity enum value (defaults to INFO if unknown)
        """
        mapping = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
        }
        return mapping.get(s.lower(), cls.INFO)

    def to_string(self) -> str:
        """Convert to lowercase string."""
        return self.name.lower()


@dataclass
class Alert:
    """
    Alert data structure.

    Contains the alert message, severity, timestamp, and metadata.
    """

    message: str
    severity: AlertSeverity
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "message": self.message,
            "severity": self.severity.to_string(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "sent_to": self.sent_to,
        }


class AlertManager:
    """
    Central alert manager.

    Features:
    - Multiple channels (webhook, log, console, etc.)
    - Severity filtering
    - Rate limiting / batching
    - Alert history

    Example:
        >>> manager = AlertManager()
        >>> manager.add_channel(WebhookChannel("https://..."))
        >>> manager.alert("Position opened", severity="info", symbol="AAPL", qty=100)

        # Convenience methods
        >>> manager.trade_executed("AAPL", "BUY", 100, 150.0)
        >>> manager.kill_switch_triggered("Daily loss limit exceeded")
    """

    def __init__(
        self,
        min_severity: Optional[str] = None,
        rate_limit_seconds: Optional[float] = None,
        max_history: int = 1000,
        add_log_channel: bool = True,
    ):
        """
        Initialize alert manager.

        Args:
            min_severity: Minimum severity to send (default: from config)
            rate_limit_seconds: Min seconds between similar alerts (default: from config)
            max_history: Maximum alerts to keep in history
            add_log_channel: Add default log channel
        """
        self._channels: Dict[str, AlertChannel] = {}
        self._history: List[Alert] = []
        self._max_history = max_history
        self._lock = threading.Lock()

        # Load from config with defaults
        self._min_severity = AlertSeverity.from_string(
            min_severity
            or get_cfg("live_trading.alerting.min_severity", default="info")
        )
        self._rate_limit = (
            rate_limit_seconds
            if rate_limit_seconds is not None
            else get_cfg("live_trading.alerting.rate_limit_seconds", default=1.0)
        )
        self._last_alert_time: Dict[str, float] = {}

        # Add default log channel
        if add_log_channel:
            self.add_channel(LogChannel())

        logger.info(
            f"AlertManager initialized: min_severity={self._min_severity.name}, "
            f"rate_limit={self._rate_limit}s"
        )

    @property
    def min_severity(self) -> AlertSeverity:
        """Get minimum severity threshold."""
        return self._min_severity

    @min_severity.setter
    def min_severity(self, value: str | AlertSeverity) -> None:
        """Set minimum severity threshold."""
        if isinstance(value, str):
            self._min_severity = AlertSeverity.from_string(value)
        else:
            self._min_severity = value

    @property
    def rate_limit(self) -> float:
        """Get rate limit in seconds."""
        return self._rate_limit

    @rate_limit.setter
    def rate_limit(self, value: float) -> None:
        """Set rate limit in seconds."""
        self._rate_limit = value

    def add_channel(self, channel: AlertChannel) -> None:
        """
        Add an alert channel.

        Args:
            channel: Channel instance to add
        """
        self._channels[channel.name] = channel
        logger.debug(f"Added alert channel: {channel.name}")

    def remove_channel(self, name: str) -> None:
        """
        Remove an alert channel.

        Args:
            name: Channel name to remove
        """
        if name in self._channels:
            del self._channels[name]
            logger.debug(f"Removed alert channel: {name}")

    def get_channel(self, name: str) -> Optional[AlertChannel]:
        """
        Get a channel by name.

        Args:
            name: Channel name

        Returns:
            Channel instance or None
        """
        return self._channels.get(name)

    @property
    def channel_names(self) -> List[str]:
        """Get list of registered channel names."""
        return list(self._channels.keys())

    def alert(
        self,
        message: str,
        severity: str = "info",
        **metadata: Any,
    ) -> Optional[Alert]:
        """
        Send an alert to all channels.

        Args:
            message: Alert message
            severity: Severity level (debug, info, warning, error, critical)
            **metadata: Additional context (symbol, qty, price, etc.)

        Returns:
            Alert object if sent, None if filtered/rate-limited
        """
        sev = AlertSeverity.from_string(severity)

        # Filter by severity
        if sev < self._min_severity:
            logger.debug(
                f"Alert filtered (severity {severity} < {self._min_severity.name})"
            )
            return None

        # Rate limiting
        if self._rate_limit > 0:
            alert_key = f"{sev}:{message[:50]}"
            now = time.time()
            with self._lock:
                last = self._last_alert_time.get(alert_key, 0)
                if now - last < self._rate_limit:
                    logger.debug(f"Alert rate-limited: {message[:50]}")
                    return None
                self._last_alert_time[alert_key] = now

        # Create alert
        alert = Alert(
            message=message,
            severity=sev,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata,
        )

        # Send to all channels
        for name, channel in self._channels.items():
            try:
                if channel.send(message, severity, metadata):
                    alert.sent_to.append(name)
            except Exception as e:
                logger.error(f"Channel '{name}' failed: {e}")

        # Store in history
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        return alert

    def get_history(
        self,
        min_severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            min_severity: Filter by minimum severity
            limit: Maximum alerts to return

        Returns:
            List of recent alerts (newest first)
        """
        with self._lock:
            alerts = list(reversed(self._history))

        if min_severity:
            min_sev = AlertSeverity.from_string(min_severity)
            alerts = [a for a in alerts if a.severity >= min_sev]

        return alerts[:limit]

    def clear_history(self) -> None:
        """Clear alert history."""
        with self._lock:
            self._history.clear()
            self._last_alert_time.clear()

    # =========================================================================
    # Convenience Methods for Common Alerts
    # =========================================================================

    def trade_executed(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for trade execution.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            qty: Quantity traded
            price: Execution price
            **kwargs: Additional metadata
        """
        return self.alert(
            f"Trade executed: {side} {qty} {symbol} @ ${price:.2f}",
            severity="info",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            **kwargs,
        )

    def position_opened(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for new position.

        Args:
            symbol: Trading symbol
            qty: Position size
            entry_price: Entry price
            **kwargs: Additional metadata
        """
        return self.alert(
            f"Position opened: {qty} {symbol} @ ${entry_price:.2f}",
            severity="info",
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            **kwargs,
        )

    def position_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for closed position.

        Args:
            symbol: Trading symbol
            pnl: Profit/loss in dollars
            pnl_pct: Profit/loss percentage
            **kwargs: Additional metadata
        """
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        return self.alert(
            f"Position closed: {symbol} {pnl_str} ({pnl_pct:+.2f}%)",
            severity="info" if pnl >= 0 else "warning",
            symbol=symbol,
            pnl=pnl,
            pnl_pct=pnl_pct,
            **kwargs,
        )

    def risk_warning(
        self,
        message: str,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for risk warnings.

        Args:
            message: Warning message
            **kwargs: Additional metadata
        """
        return self.alert(
            message,
            severity="warning",
            alert_type="risk",
            **kwargs,
        )

    def risk_breach(
        self,
        message: str,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for risk limit breaches.

        Args:
            message: Breach description
            **kwargs: Additional metadata
        """
        return self.alert(
            message,
            severity="error",
            alert_type="risk_breach",
            **kwargs,
        )

    def kill_switch_triggered(
        self,
        reason: str,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for kill switch activation.

        Args:
            reason: Kill switch reason
            **kwargs: Additional metadata (value, limit, etc.)
        """
        return self.alert(
            f"KILL SWITCH TRIGGERED: {reason}",
            severity="critical",
            alert_type="kill_switch",
            reason=reason,
            **kwargs,
        )

    def system_error(
        self,
        error: str,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Alert for system errors.

        Args:
            error: Error message
            **kwargs: Additional metadata
        """
        return self.alert(
            f"System error: {error}",
            severity="error",
            alert_type="system",
            **kwargs,
        )

    def daily_summary(
        self,
        date: str,
        pnl: float,
        trades: int,
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        Daily performance summary alert.

        Args:
            date: Trading date
            pnl: Daily P&L
            trades: Number of trades
            **kwargs: Additional metrics
        """
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        return self.alert(
            f"Daily Summary ({date}): {pnl_str}, {trades} trades",
            severity="info",
            alert_type="summary",
            date=date,
            pnl=pnl,
            trades=trades,
            **kwargs,
        )

    def heartbeat(
        self,
        status: str = "healthy",
        **kwargs: Any,
    ) -> Optional[Alert]:
        """
        System heartbeat/health check alert.

        Args:
            status: System status
            **kwargs: Additional health metrics
        """
        return self.alert(
            f"System heartbeat: {status}",
            severity="debug",
            alert_type="heartbeat",
            status=status,
            **kwargs,
        )


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get or create global alert manager.

    Returns:
        Global AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def alert(message: str, severity: str = "info", **metadata: Any) -> Optional[Alert]:
    """
    Send an alert using the global manager.

    Convenience function for quick alerts.

    Args:
        message: Alert message
        severity: Severity level
        **metadata: Additional context

    Returns:
        Alert if sent, None if filtered
    """
    return get_alert_manager().alert(message, severity, **metadata)
