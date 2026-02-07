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

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from CONFIG.config_loader import get_cfg

logger = logging.getLogger(__name__)


@runtime_checkable
class AlertChannel(Protocol):
    """
    Protocol for alert channels.

    All alert channels must implement this interface.
    """

    @property
    def name(self) -> str:
        """Channel identifier."""
        ...

    def send(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> bool:
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
        >>> channel.send("Alert!", "warning", {"symbol": "AAPL"})
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
        self._url = url or get_cfg("live_trading.alerting.webhook.url", default=None)
        self._name = name
        self._timeout = timeout
        self._discord_format = discord_format
        self._slack_format = slack_format
        self._headers = headers or {}

        if not self._url:
            logger.debug(f"WebhookChannel '{name}' has no URL configured")

    @property
    def name(self) -> str:
        """Channel identifier."""
        return self._name

    @property
    def url(self) -> Optional[str]:
        """Get webhook URL."""
        return self._url

    def send(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Send alert via webhook.

        Args:
            message: Alert message
            severity: Severity level
            metadata: Additional context

        Returns:
            True if successful
        """
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
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format payload based on channel type."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if self._discord_format:
            return self._format_discord(message, severity, metadata, timestamp)

        if self._slack_format:
            return self._format_slack(message, severity, metadata, timestamp)

        # Generic JSON payload
        return {
            "severity": severity,
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata,
        }

    def _format_discord(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Format payload for Discord webhook."""
        color = self._severity_color_discord(severity)
        fields = [
            {"name": k, "value": str(v), "inline": True}
            for k, v in list(metadata.items())[:25]  # Discord limit
        ]

        return {
            "embeds": [
                {
                    "title": f"[{severity.upper()}] Trading Alert",
                    "description": message,
                    "color": color,
                    "timestamp": timestamp,
                    "fields": fields,
                }
            ]
        }

    def _format_slack(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Format payload for Slack webhook."""
        emoji = self._severity_emoji(severity)
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Trading Alert"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{severity.upper()}*: {message}",
                },
            },
        ]

        if metadata:
            fields = [
                {"type": "mrkdwn", "text": f"*{k}*: {v}"}
                for k, v in list(metadata.items())[:10]
            ]
            blocks.append({"type": "section", "fields": fields})

        return {"blocks": blocks}

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
            "debug": 0x808080,  # Gray
            "info": 0x3498DB,  # Blue
            "warning": 0xF39C12,  # Orange
            "error": 0xE74C3C,  # Red
            "critical": 0x8E44AD,  # Purple
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

    Example:
        >>> channel = LogChannel()
        >>> channel.send("Test alert", "info", {"key": "value"})
    """

    def __init__(
        self,
        name: str = "log",
        logger_name: Optional[str] = None,
    ):
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

    def send(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Log the alert.

        Args:
            message: Alert message
            severity: Severity level
            metadata: Additional context

        Returns:
            Always True (logging doesn't fail)
        """
        level = getattr(logging, severity.upper(), logging.INFO)
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        log_message = f"[ALERT] {message}"
        if meta_str:
            log_message += f" | {meta_str}"
        self._logger.log(level, log_message)
        return True


class DiscordChannel(WebhookChannel):
    """
    Discord-specific channel (convenience wrapper).

    Example:
        >>> channel = DiscordChannel()  # Uses config
        >>> # or
        >>> channel = DiscordChannel("https://discord.com/api/webhooks/xxx")
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        name: str = "discord",
    ):
        """
        Initialize Discord channel.

        Args:
            webhook_url: Discord webhook URL (default: from config)
            name: Channel name
        """
        url = webhook_url or get_cfg(
            "live_trading.alerting.discord.webhook_url", default=None
        )
        super().__init__(url=url, name=name, discord_format=True)


class SlackChannel(WebhookChannel):
    """
    Slack-specific channel (convenience wrapper).

    Example:
        >>> channel = SlackChannel()  # Uses config
        >>> # or
        >>> channel = SlackChannel("https://hooks.slack.com/xxx")
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        name: str = "slack",
    ):
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack webhook URL (default: from config)
            name: Channel name
        """
        url = webhook_url or get_cfg(
            "live_trading.alerting.slack.webhook_url", default=None
        )
        super().__init__(url=url, name=name, slack_format=True)


class ConsoleChannel:
    """
    Console output channel for development.

    Prints alerts to stdout with color coding.
    """

    COLORS = {
        "debug": "\033[90m",  # Gray
        "info": "\033[94m",  # Blue
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m",  # Red
        "critical": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, name: str = "console", use_color: bool = True):
        """
        Initialize console channel.

        Args:
            name: Channel name
            use_color: Use ANSI color codes
        """
        self._name = name
        self._use_color = use_color

    @property
    def name(self) -> str:
        """Channel identifier."""
        return self._name

    def send(
        self,
        message: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Print alert to console."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if self._use_color:
            color = self.COLORS.get(severity.lower(), "")
            prefix = f"{color}[{severity.upper()}]{self.RESET}"
        else:
            prefix = f"[{severity.upper()}]"

        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        output = f"{timestamp} {prefix} {message}"
        if meta_str:
            output += f" | {meta_str}"

        print(output)
        return True
