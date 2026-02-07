"""
Alerting System
===============

Modular alerting with pluggable channels.

Channels:
- WebhookChannel: Generic webhook (JSON)
- DiscordChannel: Discord webhook formatting
- SlackChannel: Slack webhook formatting
- LogChannel: Python logging
- ConsoleChannel: Console output

Usage:
    >>> from LIVE_TRADING.alerting import AlertManager, WebhookChannel
    >>> manager = AlertManager()
    >>> manager.add_channel(WebhookChannel("https://my-webhook.com"))
    >>> manager.alert("Trade executed", severity="info", symbol="AAPL")

    # Convenience methods
    >>> manager.trade_executed("AAPL", "BUY", 100, 150.0)
    >>> manager.kill_switch_triggered("Daily loss exceeded")

Quick alert (using global manager):
    >>> from LIVE_TRADING.alerting import alert
    >>> alert("Position opened", severity="info", symbol="MSFT")
"""

from LIVE_TRADING.alerting.channels import (
    AlertChannel,
    WebhookChannel,
    DiscordChannel,
    SlackChannel,
    LogChannel,
    ConsoleChannel,
)
from LIVE_TRADING.alerting.manager import (
    AlertManager,
    AlertSeverity,
    Alert,
    get_alert_manager,
    alert,
)

__all__ = [
    # Protocol
    "AlertChannel",
    # Channels
    "WebhookChannel",
    "DiscordChannel",
    "SlackChannel",
    "LogChannel",
    "ConsoleChannel",
    # Manager
    "AlertManager",
    "AlertSeverity",
    "Alert",
    # Convenience
    "get_alert_manager",
    "alert",
]
