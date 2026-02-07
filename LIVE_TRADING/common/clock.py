"""
Clock Abstraction
=================

Protocol-based time abstraction for live and simulated trading.

SST Compliance:
- All time access goes through Clock protocol
- No direct datetime.now() calls in trading code
"""

from abc import abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Protocol, runtime_checkable
import time as time_module


@runtime_checkable
class Clock(Protocol):
    """Protocol for time access."""

    @abstractmethod
    def now(self) -> datetime:
        """Get current time (always UTC, timezone-aware)."""
        ...

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep for duration (real or simulated)."""
        ...

    @abstractmethod
    def timestamp(self) -> float:
        """Get Unix timestamp."""
        ...


class SystemClock:
    """
    Real system clock for live trading.

    Example:
        >>> clock = SystemClock()
        >>> clock.now()
        datetime.datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    """

    def now(self) -> datetime:
        """Get current UTC time."""
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        """Sleep for real duration."""
        time_module.sleep(seconds)

    def timestamp(self) -> float:
        """Get Unix timestamp."""
        return time_module.time()


class SimulatedClock:
    """
    Simulated clock for backtesting.

    Time only advances when explicitly told to via advance() or set_time().

    Example:
        >>> clock = SimulatedClock(datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc))
        >>> clock.now()
        datetime.datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        >>> clock.advance(minutes=5)
        >>> clock.now()
        datetime.datetime(2024, 1, 1, 9, 35, 0, tzinfo=timezone.utc)
    """

    def __init__(self, start_time: datetime):
        """
        Initialize simulated clock.

        Args:
            start_time: Initial time (must be timezone-aware UTC)
        """
        if start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware (use timezone.utc)")
        self._current_time = start_time
        self._sleep_calls: list[float] = []  # Track sleep calls for verification

    def now(self) -> datetime:
        """Get current simulated time."""
        return self._current_time

    def sleep(self, seconds: float) -> None:
        """
        Record sleep call (doesn't actually sleep).

        In backtesting, we don't want real delays.
        """
        self._sleep_calls.append(seconds)

    def timestamp(self) -> float:
        """Get Unix timestamp of simulated time."""
        return self._current_time.timestamp()

    def advance(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
    ) -> None:
        """
        Advance simulated time.

        Args:
            seconds: Seconds to advance
            minutes: Minutes to advance
            hours: Hours to advance
            days: Days to advance
        """
        total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400
        self._current_time += timedelta(seconds=total_seconds)

    def set_time(self, new_time: datetime) -> None:
        """
        Set simulated time to specific value.

        Args:
            new_time: New time (must be timezone-aware UTC)
        """
        if new_time.tzinfo is None:
            raise ValueError("new_time must be timezone-aware (use timezone.utc)")
        self._current_time = new_time

    def get_sleep_calls(self) -> list[float]:
        """Get list of sleep durations requested (for test verification)."""
        return self._sleep_calls.copy()

    def clear_sleep_calls(self) -> None:
        """Clear recorded sleep calls."""
        self._sleep_calls.clear()


class OffsetClock:
    """
    Clock with fixed offset from system time.

    Useful for testing time-sensitive logic without full simulation.

    Example:
        >>> clock = OffsetClock(hours=-24)  # Yesterday
        >>> clock.now()  # Returns current time minus 24 hours
    """

    def __init__(
        self,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
    ):
        """
        Initialize offset clock.

        Args:
            seconds: Seconds offset (negative = past)
            minutes: Minutes offset
            hours: Hours offset
            days: Days offset
        """
        total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400
        self._offset = timedelta(seconds=total_seconds)

    def now(self) -> datetime:
        """Get current time with offset."""
        return datetime.now(timezone.utc) + self._offset

    def sleep(self, seconds: float) -> None:
        """Sleep for real duration."""
        time_module.sleep(seconds)

    def timestamp(self) -> float:
        """Get Unix timestamp with offset."""
        return (datetime.now(timezone.utc) + self._offset).timestamp()


# Default clock instance (can be replaced for testing)
_default_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """Get the current clock instance."""
    return _default_clock


def set_clock(clock: Clock) -> None:
    """
    Set the global clock instance.

    WARNING: Only use in tests or at application startup.
    """
    global _default_clock
    _default_clock = clock


def reset_clock() -> None:
    """Reset to system clock."""
    global _default_clock
    _default_clock = SystemClock()
