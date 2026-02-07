# Plan 17: Alerting System

## Overview

Webhook-based alerting with extensible channel architecture for Discord, Slack, email, etc.

## Design Philosophy

- **Modular**: Easy to add new channels without touching core logic
- **Channel Protocol**: All channels implement same interface
- **Batching**: Coalesce rapid alerts to prevent spam
- **Severity Levels**: Filter alerts by importance
- **Async-first**: Non-blocking alert delivery

## Files to Create

### 1. `LIVE_TRADING/alerting/__init__.py`

```python
"""
Alerting System
===============

Modular alerting with pluggable channels.

Example:
    >>> from LIVE_TRADING.alerting import AlertManager, WebhookChannel
    >>> manager = AlertManager()
    >>> manager.add_channel(WebhookChannel("https://my-webhook.com"))
    >>> manager.alert("Trade executed", severity="info")
"""

from .channels import AlertChannel, WebhookChannel, LogChannel
from .manager import AlertManager, AlertSeverity, Alert

__all__ = [
    "AlertChannel",
    "WebhookChannel",
    "LogChannel",
    "AlertManager",
    "AlertSeverity",
    "Alert",
]
```

### 2. `LIVE_TRADING/alerting/channels.py`

```python
"""
Alert Channels
==============

Pluggable alert delivery channels.

To add a new channel:
1. Implement AlertChannel Protocol
2. Define send() method
3. Register with AlertManager

SST Compliance:
- Configuration via get_cfg()
- No hardcoded URLs or credentials
"""

import logging
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertChannel(Protocol):
    """Protocol for alert channels."""

    @property
    def name(self) -> str:
        """Channel identifier."""
        ...

    def send(self, message: str, severity: str, metadata: Dict[str, Any]) -> bool:
        """
        Send an alert.

        Args:
            message: Alert message
            severity: Alert severity (debug, info, warning, error, critical)
            metadata: Additional context

        Returns:
            True if sent successfully
        """
        ...


class WebhookChannel:
    """
    Generic webhook channel.

    Sends JSON payloads to any webhook URL. Compatible with:
    - Discord webhooks (via discord_format=True)
    - Slack incoming webhooks (via slack_format=True)
    - Custom endpoints

    Example:
        >>> channel = WebhookChannel(
        ...     url="https://discord.com/api/webhooks/xxx",
        ...     discord_format=True
        ... )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        name: str = "webhook",
        timeout: float = 10.0,
        discord_format: bool = False,
        slack_format: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize webhook channel.

        Args:
            url: Webhook URL (default: from config)
            name: Channel name for identification
            timeout: Request timeout in seconds
            discord_format: Format payload for Discord
            slack_format: Format payload for Slack
            headers: Additional HTTP headers
        """
        self._url = url or get_cfg("live_trading.alerting.webhook.url")
        self._name = name
        self._timeout = timeout
        self._discord_format = discord_format
        self._slack_format = slack_format
        self._headers = headers or {}

        if not self._url:
            logger.warning(f"WebhookChannel '{name}' has no URL configured")

    @property
    def name(self) -> str:
        """Channel identifier."""
        return self._name

    def send(self, message: str, severity: str, metadata: Dict[str, Any]) -> bool:
        """Send alert via webhook."""
        if not self._url:
            logger.debug(f"Webhook '{self._name}' skipped (no URL)")
            return False

        try:
            payload = self._format_payload(message, severity, metadata)
            return self._post(payload)
        except Exception as e:
            logger.error(f"Webhook '{self._name}' failed: {e}")
            return False

    def _format_payload(
        self, message: str, severity: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format payload based on channel type."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if self._discord_format:
            # Discord webhook format
            color = self._severity_color_discord(severity)
            return {
                "embeds": [{
                    "title": f"[{severity.upper()}] Trading Alert",
                    "description": message,
                    "color": color,
                    "timestamp": timestamp,
                    "fields": [
                        {"name": k, "value": str(v), "inline": True}
                        for k, v in list(metadata.items())[:25]  # Discord limit
                    ],
                }]
            }

        if self._slack_format:
            # Slack webhook format
            emoji = self._severity_emoji(severity)
            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} Trading Alert"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{severity.upper()}*: {message}"}
                },
            ]
            if metadata:
                fields = [
                    {"type": "mrkdwn", "text": f"*{k}*: {v}"}
                    for k, v in list(metadata.items())[:10]
                ]
                blocks.append({"type": "section", "fields": fields})
            return {"blocks": blocks}

        # Generic JSON payload
        return {
            "severity": severity,
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata,
        }

    def _post(self, payload: Dict[str, Any]) -> bool:
        """POST JSON to webhook URL."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        req = urllib.request.Request(self._url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            logger.error(f"Webhook HTTP error: {e.code} {e.reason}")
            return False
        except urllib.error.URLError as e:
            logger.error(f"Webhook URL error: {e.reason}")
            return False

    def _severity_color_discord(self, severity: str) -> int:
        """Get Discord embed color for severity."""
        colors = {
            "debug": 0x808080,    # Gray
            "info": 0x3498DB,     # Blue
            "warning": 0xF39C12,  # Orange
            "error": 0xE74C3C,    # Red
            "critical": 0x8E44AD, # Purple
        }
        return colors.get(severity.lower(), 0x95A5A6)

    def _severity_emoji(self, severity: str) -> str:
        """Get emoji for severity."""
        emojis = {
            "debug": ":beetle:",
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        return emojis.get(severity.lower(), ":bell:")


class LogChannel:
    """
    Log-based alert channel for development/testing.

    Writes alerts to Python logging system.
    """

    def __init__(self, name: str = "log", logger_name: Optional[str] = None):
        """
        Initialize log channel.

        Args:
            name: Channel name
            logger_name: Logger name (default: LIVE_TRADING.alerting)
        """
        self._name = name
        self._logger = logging.getLogger(logger_name or "LIVE_TRADING.alerting")

    @property
    def name(self) -> str:
        """Channel identifier."""
        return self._name

    def send(self, message: str, severity: str, metadata: Dict[str, Any]) -> bool:
        """Log the alert."""
        level = getattr(logging, severity.upper(), logging.INFO)
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        self._logger.log(level, f"[ALERT] {message} | {meta_str}")
        return True


# Future channel stubs for extensibility
class DiscordChannel(WebhookChannel):
    """Discord-specific channel (convenience wrapper)."""

    def __init__(self, webhook_url: Optional[str] = None, name: str = "discord"):
        url = webhook_url or get_cfg("live_trading.alerting.discord.webhook_url")
        super().__init__(url=url, name=name, discord_format=True)


class SlackChannel(WebhookChannel):
    """Slack-specific channel (convenience wrapper)."""

    def __init__(self, webhook_url: Optional[str] = None, name: str = "slack"):
        url = webhook_url or get_cfg("live_trading.alerting.slack.webhook_url")
        super().__init__(url=url, name=name, slack_format=True)
```

### 3. `LIVE_TRADING/alerting/manager.py`

```python
"""
Alert Manager
=============

Central alert coordination with batching and filtering.

SST Compliance:
- Configuration via get_cfg()
- Severity filtering from config
- Batching to prevent alert spam
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from .channels import AlertChannel, LogChannel

logger = logging.getLogger(__name__)


class AlertSeverity(IntEnum):
    """Alert severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, s: str) -> "AlertSeverity":
        """Parse severity from string."""
        mapping = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
        }
        return mapping.get(s.lower(), cls.INFO)


@dataclass
class Alert:
    """Alert data structure."""
    message: str
    severity: AlertSeverity
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_to: List[str] = field(default_factory=list)


class AlertManager:
    """
    Central alert manager.

    Features:
    - Multiple channels
    - Severity filtering
    - Rate limiting / batching
    - Alert history

    Example:
        >>> manager = AlertManager()
        >>> manager.add_channel(WebhookChannel("https://..."))
        >>> manager.alert("Position opened", severity="info", symbol="AAPL", qty=100)
    """

    def __init__(
        self,
        min_severity: Optional[str] = None,
        rate_limit_seconds: Optional[float] = None,
        max_history: int = 1000,
    ):
        """
        Initialize alert manager.

        Args:
            min_severity: Minimum severity to send (default: from config)
            rate_limit_seconds: Min seconds between alerts (default: from config)
            max_history: Maximum alerts to keep in history
        """
        self._channels: Dict[str, AlertChannel] = {}
        self._history: List[Alert] = []
        self._max_history = max_history
        self._lock = threading.Lock()

        # Load from config with defaults
        self._min_severity = AlertSeverity.from_string(
            min_severity or get_cfg(
                "live_trading.alerting.min_severity",
                default="info"
            )
        )
        self._rate_limit = rate_limit_seconds or get_cfg(
            "live_trading.alerting.rate_limit_seconds",
            default=1.0
        )
        self._last_alert_time: Dict[str, float] = {}

        # Add default log channel
        self.add_channel(LogChannel())

        logger.info(
            f"AlertManager initialized: min_severity={self._min_severity.name}, "
            f"rate_limit={self._rate_limit}s"
        )

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self._channels[channel.name] = channel
        logger.debug(f"Added alert channel: {channel.name}")

    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        if name in self._channels:
            del self._channels[name]
            logger.debug(f"Removed alert channel: {name}")

    def alert(
        self,
        message: str,
        severity: str = "info",
        **metadata,
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
            logger.debug(f"Alert filtered (severity {severity} < {self._min_severity.name})")
            return None

        # Rate limiting
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
                self._history = self._history[-self._max_history:]

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

    # Convenience methods for common alerts
    def trade_executed(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        **kwargs,
    ) -> Optional[Alert]:
        """Alert for trade execution."""
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
        **kwargs,
    ) -> Optional[Alert]:
        """Alert for new position."""
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
        **kwargs,
    ) -> Optional[Alert]:
        """Alert for closed position."""
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        return self.alert(
            f"Position closed: {symbol} {pnl_str} ({pnl_pct:+.2f}%)",
            severity="info" if pnl >= 0 else "warning",
            symbol=symbol,
            pnl=pnl,
            pnl_pct=pnl_pct,
            **kwargs,
        )

    def risk_warning(self, message: str, **kwargs) -> Optional[Alert]:
        """Alert for risk warnings."""
        return self.alert(message, severity="warning", alert_type="risk", **kwargs)

    def risk_breach(self, message: str, **kwargs) -> Optional[Alert]:
        """Alert for risk breaches."""
        return self.alert(message, severity="error", alert_type="risk_breach", **kwargs)

    def kill_switch_triggered(self, reason: str, **kwargs) -> Optional[Alert]:
        """Alert for kill switch activation."""
        return self.alert(
            f"KILL SWITCH TRIGGERED: {reason}",
            severity="critical",
            alert_type="kill_switch",
            **kwargs,
        )

    def system_error(self, error: str, **kwargs) -> Optional[Alert]:
        """Alert for system errors."""
        return self.alert(
            f"System error: {error}",
            severity="error",
            alert_type="system",
            **kwargs,
        )
```

## Tests

### `LIVE_TRADING/tests/test_alerting.py`

```python
"""
Alerting System Tests
=====================

Unit tests for alert channels and manager.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import json

from LIVE_TRADING.alerting import (
    AlertManager,
    AlertSeverity,
    Alert,
    WebhookChannel,
    LogChannel,
    AlertChannel,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_from_string_valid(self):
        """Test parsing valid severity strings."""
        assert AlertSeverity.from_string("debug") == AlertSeverity.DEBUG
        assert AlertSeverity.from_string("INFO") == AlertSeverity.INFO
        assert AlertSeverity.from_string("Warning") == AlertSeverity.WARNING
        assert AlertSeverity.from_string("ERROR") == AlertSeverity.ERROR
        assert AlertSeverity.from_string("critical") == AlertSeverity.CRITICAL

    def test_from_string_invalid(self):
        """Test default for invalid severity."""
        assert AlertSeverity.from_string("unknown") == AlertSeverity.INFO

    def test_severity_ordering(self):
        """Test severity comparison."""
        assert AlertSeverity.DEBUG < AlertSeverity.INFO
        assert AlertSeverity.INFO < AlertSeverity.WARNING
        assert AlertSeverity.WARNING < AlertSeverity.ERROR
        assert AlertSeverity.ERROR < AlertSeverity.CRITICAL


class TestLogChannel:
    """Tests for LogChannel."""

    def test_send_logs_message(self, caplog):
        """Test that send logs the message."""
        channel = LogChannel(name="test")
        result = channel.send("Test alert", "info", {"key": "value"})

        assert result is True
        assert "Test alert" in caplog.text
        assert "key=value" in caplog.text

    def test_name_property(self):
        """Test name property."""
        channel = LogChannel(name="custom")
        assert channel.name == "custom"


class TestWebhookChannel:
    """Tests for WebhookChannel."""

    def test_init_with_url(self):
        """Test initialization with URL."""
        channel = WebhookChannel(url="https://test.com/hook", name="test")
        assert channel.name == "test"
        assert channel._url == "https://test.com/hook"

    def test_send_no_url(self):
        """Test send with no URL configured."""
        channel = WebhookChannel(url=None, name="test")
        result = channel.send("Test", "info", {})
        assert result is False

    @patch("urllib.request.urlopen")
    def test_send_success(self, mock_urlopen):
        """Test successful webhook send."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(url="https://test.com/hook")
        result = channel.send("Test message", "info", {"key": "value"})

        assert result is True
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_discord_format(self, mock_urlopen):
        """Test Discord payload formatting."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(
            url="https://discord.com/api/webhooks/xxx",
            discord_format=True
        )
        channel.send("Test", "error", {"symbol": "AAPL"})

        # Check the request was made
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data)

        assert "embeds" in payload
        assert payload["embeds"][0]["title"] == "[ERROR] Trading Alert"

    @patch("urllib.request.urlopen")
    def test_slack_format(self, mock_urlopen):
        """Test Slack payload formatting."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookChannel(
            url="https://hooks.slack.com/xxx",
            slack_format=True
        )
        channel.send("Test", "warning", {})

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        payload = json.loads(request.data)

        assert "blocks" in payload


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create alert manager for testing."""
        return AlertManager(min_severity="debug", rate_limit_seconds=0)

    def test_init_default_log_channel(self, manager):
        """Test that log channel is added by default."""
        assert "log" in manager._channels

    def test_add_channel(self, manager):
        """Test adding a channel."""
        channel = LogChannel(name="test")
        manager.add_channel(channel)
        assert "test" in manager._channels

    def test_remove_channel(self, manager):
        """Test removing a channel."""
        channel = LogChannel(name="test")
        manager.add_channel(channel)
        manager.remove_channel("test")
        assert "test" not in manager._channels

    def test_alert_returns_alert_object(self, manager):
        """Test that alert returns Alert object."""
        alert = manager.alert("Test message", severity="info", key="value")

        assert isinstance(alert, Alert)
        assert alert.message == "Test message"
        assert alert.severity == AlertSeverity.INFO
        assert alert.metadata == {"key": "value"}

    def test_alert_severity_filtering(self):
        """Test that alerts below min_severity are filtered."""
        manager = AlertManager(min_severity="warning", rate_limit_seconds=0)

        debug_alert = manager.alert("Debug", severity="debug")
        info_alert = manager.alert("Info", severity="info")
        warning_alert = manager.alert("Warning", severity="warning")

        assert debug_alert is None
        assert info_alert is None
        assert warning_alert is not None

    def test_alert_rate_limiting(self):
        """Test rate limiting of duplicate alerts."""
        manager = AlertManager(min_severity="debug", rate_limit_seconds=10)

        alert1 = manager.alert("Same message", severity="info")
        alert2 = manager.alert("Same message", severity="info")

        assert alert1 is not None
        assert alert2 is None  # Rate limited

    def test_get_history(self, manager):
        """Test getting alert history."""
        manager.alert("First", severity="info")
        manager.alert("Second", severity="warning")
        manager.alert("Third", severity="error")

        history = manager.get_history(limit=10)

        assert len(history) == 3
        # Newest first
        assert history[0].message == "Third"
        assert history[1].message == "Second"
        assert history[2].message == "First"

    def test_get_history_with_severity_filter(self, manager):
        """Test history filtering by severity."""
        manager.alert("Info", severity="info")
        manager.alert("Warning", severity="warning")
        manager.alert("Error", severity="error")

        history = manager.get_history(min_severity="warning")

        assert len(history) == 2
        assert all(a.severity >= AlertSeverity.WARNING for a in history)

    def test_clear_history(self, manager):
        """Test clearing history."""
        manager.alert("Test", severity="info")
        assert len(manager.get_history()) > 0

        manager.clear_history()
        assert len(manager.get_history()) == 0

    # Convenience method tests
    def test_trade_executed(self, manager):
        """Test trade_executed convenience method."""
        alert = manager.trade_executed("AAPL", "BUY", 100, 150.0)

        assert alert is not None
        assert "AAPL" in alert.message
        assert "BUY" in alert.message
        assert alert.metadata["symbol"] == "AAPL"

    def test_position_opened(self, manager):
        """Test position_opened convenience method."""
        alert = manager.position_opened("MSFT", 50, 300.0)

        assert alert is not None
        assert "MSFT" in alert.message

    def test_position_closed_profit(self, manager):
        """Test position_closed with profit."""
        alert = manager.position_closed("GOOG", 500.0, 2.5)

        assert alert is not None
        assert alert.severity == AlertSeverity.INFO
        assert "+$500" in alert.message

    def test_position_closed_loss(self, manager):
        """Test position_closed with loss."""
        alert = manager.position_closed("GOOG", -200.0, -1.5)

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_kill_switch_triggered(self, manager):
        """Test kill_switch_triggered."""
        alert = manager.kill_switch_triggered("Daily loss limit exceeded")

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert "KILL SWITCH" in alert.message


class TestAlertChannelProtocol:
    """Test that channels implement the Protocol."""

    def test_log_channel_is_alert_channel(self):
        """Test LogChannel implements AlertChannel."""
        channel = LogChannel()
        assert isinstance(channel, AlertChannel)

    def test_webhook_channel_is_alert_channel(self):
        """Test WebhookChannel implements AlertChannel."""
        channel = WebhookChannel(url="https://test.com")
        assert isinstance(channel, AlertChannel)
```

## Configuration

Add to `CONFIG/live_trading/live_trading.yaml`:

```yaml
live_trading:
  alerting:
    min_severity: "info"           # debug, info, warning, error, critical
    rate_limit_seconds: 1.0        # Min seconds between duplicate alerts

    # Webhook configuration
    webhook:
      url: null                    # Generic webhook URL

    # Discord (optional)
    discord:
      webhook_url: null            # Discord webhook URL

    # Slack (optional)
    slack:
      webhook_url: null            # Slack incoming webhook URL
```

## Integration with Trading Engine

The AlertManager integrates with the trading engine via events:

```python
# In TradingEngine
from LIVE_TRADING.alerting import AlertManager, WebhookChannel

class TradingEngine:
    def __init__(self, ...):
        self._alert_manager = AlertManager()

        # Optionally add webhook channel
        webhook_url = get_cfg("live_trading.alerting.webhook.url")
        if webhook_url:
            self._alert_manager.add_channel(WebhookChannel(url=webhook_url))

    def _on_fill(self, fill: Dict[str, Any]) -> None:
        self._alert_manager.trade_executed(
            symbol=fill["symbol"],
            side=fill["side"],
            qty=fill["qty"],
            price=fill["price"],
        )

    def _check_risk(self) -> None:
        if self._guardrails.daily_loss_exceeded():
            self._alert_manager.kill_switch_triggered(
                "Daily loss limit exceeded",
                daily_pnl=self._state.daily_pnl,
                limit=self._guardrails.max_daily_loss,
            )
```

## SST Compliance

- [ ] Configuration via get_cfg()
- [ ] No hardcoded URLs or credentials
- [ ] Protocol-based channel interface
- [ ] Timezone-aware timestamps
- [ ] Thread-safe alert history

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `alerting/__init__.py` | 20 |
| `alerting/channels.py` | 200 |
| `alerting/manager.py` | 250 |
| `tests/test_alerting.py` | 200 |
| **Total** | ~670 |
