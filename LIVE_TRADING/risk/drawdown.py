"""
Drawdown Monitoring
===================

Tracks portfolio drawdown from peak and triggers alerts/kill switches
when drawdown exceeds configured thresholds.

SST Compliance:
- Uses get_cfg() for configuration
- Deterministic behavior (no random elements)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Tuple

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import MaxDrawdownExceeded

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """
    Current drawdown state.

    Captures the relationship between peak value and current value
    to understand portfolio drawdown.
    """

    peak_value: float
    current_value: float
    drawdown_pct: float  # As percentage (e.g., 5.0 for 5%)
    peak_date: Optional[datetime]
    is_exceeded: bool  # True if drawdown exceeds limit

    @property
    def drawdown_decimal(self) -> float:
        """Drawdown as decimal (e.g., 0.05 for 5%)."""
        return self.drawdown_pct / 100

    @property
    def recovery_needed_pct(self) -> float:
        """Percentage gain needed to recover to peak."""
        if self.current_value <= 0:
            return float("inf")
        return (self.peak_value / self.current_value - 1) * 100


class DrawdownMonitor:
    """
    Monitors portfolio drawdown from peak.

    Tracks the high-water mark and calculates current drawdown.
    Can trigger kill switch when drawdown exceeds limit.

    Example:
        >>> monitor = DrawdownMonitor(max_drawdown_pct=10.0)
        >>> monitor.update(100_000)  # Initial value
        >>> state = monitor.update(95_000)  # 5% drawdown
        >>> state.drawdown_pct
        5.0
        >>> state.is_exceeded
        False
    """

    def __init__(
        self,
        max_drawdown_pct: float | None = None,
        history_size: int = 252,
        clock: Clock | None = None,
    ) -> None:
        """
        Initialize drawdown monitor.

        Args:
            max_drawdown_pct: Maximum allowed drawdown percentage.
                              Default from config or 10%.
            history_size: Number of historical values to keep (default: 252 = 1 year)
            clock: Clock instance for time (default: system clock)
        """
        self._clock = clock or get_clock()
        self.max_drawdown_pct = max_drawdown_pct if max_drawdown_pct is not None else get_cfg(
            "live_trading.risk.max_drawdown_pct",
            default=DEFAULT_CONFIG["max_drawdown_pct"],
        )
        self.history_size = history_size

        # State
        self._peak_value: float = 0.0
        self._peak_date: Optional[datetime] = None
        self._value_history: Deque[Tuple[datetime, float]] = deque(maxlen=history_size)
        self._current_drawdown: float = 0.0

        logger.info(f"DrawdownMonitor: max_drawdown={self.max_drawdown_pct}%")

    def update(
        self,
        portfolio_value: float,
        timestamp: datetime | None = None,
    ) -> DrawdownState:
        """
        Update with new portfolio value.

        Args:
            portfolio_value: Current portfolio value
            timestamp: Current timestamp (default: now)

        Returns:
            Current DrawdownState
        """
        timestamp = timestamp or self._clock.now()
        self._value_history.append((timestamp, portfolio_value))

        # Update peak (high-water mark)
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._peak_date = timestamp

        # Calculate drawdown percentage
        drawdown = 0.0
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value * 100
        self._current_drawdown = drawdown

        is_exceeded = drawdown > self.max_drawdown_pct

        if is_exceeded:
            logger.warning(
                f"Max drawdown exceeded: {drawdown:.2f}% > {self.max_drawdown_pct}%"
            )

        return DrawdownState(
            peak_value=self._peak_value,
            current_value=portfolio_value,
            drawdown_pct=drawdown,
            peak_date=self._peak_date,
            is_exceeded=is_exceeded,
        )

    def check_or_raise(self, portfolio_value: float) -> None:
        """
        Check drawdown and raise exception if exceeded.

        Args:
            portfolio_value: Current portfolio value

        Raises:
            MaxDrawdownExceeded: If drawdown exceeds limit
        """
        state = self.update(portfolio_value)
        if state.is_exceeded:
            raise MaxDrawdownExceeded(state.drawdown_pct, self.max_drawdown_pct)

    def get_current_state(self) -> Optional[DrawdownState]:
        """
        Get current state without updating.

        Returns:
            Current DrawdownState or None if no values recorded
        """
        if not self._value_history:
            return None

        _, last_value = self._value_history[-1]
        return DrawdownState(
            peak_value=self._peak_value,
            current_value=last_value,
            drawdown_pct=self._current_drawdown,
            peak_date=self._peak_date,
            is_exceeded=self._current_drawdown > self.max_drawdown_pct,
        )

    def get_max_drawdown_in_window(self) -> float:
        """
        Calculate maximum drawdown in the history window.

        Returns:
            Maximum drawdown percentage observed in history
        """
        if len(self._value_history) < 2:
            return 0.0

        max_dd = 0.0
        peak = 0.0

        for _, value in self._value_history:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)

        return max_dd

    def get_history(self) -> List[Tuple[datetime, float]]:
        """
        Get value history.

        Returns:
            List of (timestamp, value) tuples
        """
        return list(self._value_history)

    def reset(self, initial_value: float | None = None) -> None:
        """
        Reset monitor state.

        Args:
            initial_value: New initial value (optional)
        """
        self._peak_value = initial_value or 0.0
        self._peak_date = self._clock.now() if initial_value else None
        self._value_history.clear()
        self._current_drawdown = 0.0

        if initial_value:
            self._value_history.append((self._clock.now(), initial_value))

        logger.info(f"DrawdownMonitor reset: initial_value=${initial_value or 0:,.2f}")

    @property
    def peak_value(self) -> float:
        """Get current peak (high-water mark)."""
        return self._peak_value

    @property
    def peak_date(self) -> Optional[datetime]:
        """Get date of peak."""
        return self._peak_date

    @property
    def current_drawdown_pct(self) -> float:
        """Get current drawdown percentage."""
        return self._current_drawdown
