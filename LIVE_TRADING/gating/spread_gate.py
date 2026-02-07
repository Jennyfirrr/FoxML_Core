"""
Spread Gate
===========

Gates based on bid-ask spread and quote freshness thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import SpreadTooWideError, StaleDataError

logger = logging.getLogger(__name__)


@dataclass
class SpreadGateResult:
    """Result of spread gate evaluation."""

    allowed: bool
    spread_bps: float
    reason: str
    quote_age_ms: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "spread_bps": self.spread_bps,
            "reason": self.reason,
            "quote_age_ms": self.quote_age_ms,
        }


class SpreadGate:
    """
    Gates trades based on spread and data freshness.

    This gate ensures we only trade when:
    1. Spread is within acceptable bounds
    2. Quote data is fresh enough
    """

    def __init__(
        self,
        max_spread_bps: float | None = None,
        max_quote_age_ms: float | None = None,
    ):
        """
        Initialize spread gate.

        Args:
            max_spread_bps: Maximum allowed spread in basis points
            max_quote_age_ms: Maximum quote age in milliseconds
        """
        self.max_spread_bps = max_spread_bps if max_spread_bps is not None else get_cfg(
            "live_trading.risk.spread_max_bps",
            default=DEFAULT_CONFIG["spread_max_bps"],
        )
        self.max_quote_age_ms = max_quote_age_ms if max_quote_age_ms is not None else get_cfg(
            "live_trading.risk.quote_age_max_ms",
            default=DEFAULT_CONFIG["quote_age_max_ms"],
        )

        logger.info(
            f"SpreadGate: max_spread={self.max_spread_bps}bps, "
            f"max_age={self.max_quote_age_ms}ms"
        )

    def evaluate(
        self,
        spread_bps: float,
        quote_timestamp: datetime | None = None,
        current_time: datetime | None = None,
    ) -> SpreadGateResult:
        """
        Evaluate spread gate.

        Args:
            spread_bps: Current spread in basis points
            quote_timestamp: Quote timestamp for freshness check
            current_time: Current time (defaults to now)

        Returns:
            SpreadGateResult
        """
        # Check spread
        if spread_bps > self.max_spread_bps:
            return SpreadGateResult(
                allowed=False,
                spread_bps=spread_bps,
                reason=f"spread {spread_bps:.1f}bps > max {self.max_spread_bps}bps",
            )

        # Check quote freshness
        quote_age_ms = None
        if quote_timestamp is not None:
            now = current_time or datetime.now(timezone.utc)
            age_ms = (now - quote_timestamp).total_seconds() * 1000
            quote_age_ms = age_ms

            if age_ms > self.max_quote_age_ms:
                return SpreadGateResult(
                    allowed=False,
                    spread_bps=spread_bps,
                    reason=f"quote age {age_ms:.0f}ms > max {self.max_quote_age_ms}ms",
                    quote_age_ms=quote_age_ms,
                )

        return SpreadGateResult(
            allowed=True,
            spread_bps=spread_bps,
            reason="spread_ok",
            quote_age_ms=quote_age_ms,
        )

    def validate_or_raise(
        self,
        symbol: str,
        spread_bps: float,
        quote_timestamp: datetime | None = None,
        current_time: datetime | None = None,
    ) -> None:
        """
        Validate and raise exception if gate fails.

        Use this when you want to enforce hard limits with exceptions.

        Args:
            symbol: Trading symbol
            spread_bps: Current spread
            quote_timestamp: Quote timestamp
            current_time: Current time (defaults to now)

        Raises:
            SpreadTooWideError: If spread exceeds maximum
            StaleDataError: If quote is too old
        """
        if spread_bps > self.max_spread_bps:
            raise SpreadTooWideError(symbol, spread_bps, self.max_spread_bps)

        if quote_timestamp is not None:
            now = current_time or datetime.now(timezone.utc)
            age_ms = (now - quote_timestamp).total_seconds() * 1000
            if age_ms > self.max_quote_age_ms:
                raise StaleDataError(symbol, age_ms, self.max_quote_age_ms)

    def calculate_spread_bps(
        self,
        bid: float,
        ask: float,
    ) -> float:
        """
        Calculate spread in basis points.

        Args:
            bid: Bid price
            ask: Ask price

        Returns:
            Spread in basis points
        """
        if bid <= 0 or ask <= 0:
            return float("inf")

        mid = (bid + ask) / 2
        spread = ask - bid
        return (spread / mid) * 10000

    def get_effective_spread_bps(
        self,
        bid: float,
        ask: float,
        last_price: float,
    ) -> float:
        """
        Calculate effective spread using last trade price.

        Effective spread = 2 × |trade_price - mid_price| / mid_price

        Args:
            bid: Bid price
            ask: Ask price
            last_price: Last trade price

        Returns:
            Effective spread in basis points
        """
        if bid <= 0 or ask <= 0:
            return float("inf")

        mid = (bid + ask) / 2
        effective = 2 * abs(last_price - mid)
        return (effective / mid) * 10000

    def is_tradeable(
        self,
        spread_bps: float,
        quote_timestamp: datetime | None = None,
    ) -> bool:
        """
        Quick check if conditions allow trading.

        Args:
            spread_bps: Current spread
            quote_timestamp: Quote timestamp

        Returns:
            True if trading is allowed
        """
        result = self.evaluate(spread_bps, quote_timestamp)
        return result.allowed

    def get_spread_analysis(
        self,
        bid: float,
        ask: float,
        last_price: float | None = None,
        quote_timestamp: datetime | None = None,
    ) -> Dict[str, Any]:
        """
        Get detailed spread analysis.

        Args:
            bid: Bid price
            ask: Ask price
            last_price: Optional last trade price
            quote_timestamp: Quote timestamp

        Returns:
            Analysis dict
        """
        spread_bps = self.calculate_spread_bps(bid, ask)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        analysis = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "spread_ok": spread_bps <= self.max_spread_bps,
        }

        if last_price is not None:
            analysis["effective_spread_bps"] = self.get_effective_spread_bps(
                bid, ask, last_price
            )
            analysis["last_vs_mid_bps"] = abs(last_price - mid) / mid * 10000 if mid > 0 else 0

        if quote_timestamp is not None:
            age_ms = (datetime.now(timezone.utc) - quote_timestamp).total_seconds() * 1000
            analysis["quote_age_ms"] = age_ms
            analysis["quote_fresh"] = age_ms <= self.max_quote_age_ms

        return analysis
