# Plan 0A: Clock Abstraction

## Overview

Introduce a `Clock` protocol to abstract time throughout the trading system. This enables:
- Backtesting with simulated time
- Deterministic testing
- Time-travel debugging

## Problem Statement

Current code has `datetime.now(timezone.utc)` hardcoded in 15+ locations:
- `trading_engine.py:232` - cycle timestamp
- `state.py:177,199,221` - trade/position timestamps
- `paper.py:200,316` - fill timestamps
- `data_provider.py:144,224` - quote timestamps

This makes backtesting impossible because all timestamps use wall-clock time.

## Files to Create

### 1. `LIVE_TRADING/common/clock.py`

```python
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
from typing import Protocol, runtime_checkable, Optional
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
```

### 2. Update `LIVE_TRADING/common/__init__.py`

```python
# Add to existing exports
from .clock import (
    Clock,
    SystemClock,
    SimulatedClock,
    OffsetClock,
    get_clock,
    set_clock,
    reset_clock,
)
```

## Files to Modify

### 1. `LIVE_TRADING/engine/trading_engine.py`

**Change:** Accept clock in constructor, use throughout.

```python
# Before (line ~85)
class TradingEngine:
    def __init__(
        self,
        config: TradingConfig,
        broker: Broker,
        ...
    ):

# After
from LIVE_TRADING.common.clock import Clock, get_clock

class TradingEngine:
    def __init__(
        self,
        config: TradingConfig,
        broker: Broker,
        ...,
        clock: Optional[Clock] = None,
    ):
        self._clock = clock or get_clock()
```

**Change:** Replace all `datetime.now(timezone.utc)` with `self._clock.now()`:

```python
# Before (line ~232)
current_time = current_time or datetime.now(timezone.utc)

# After
current_time = current_time or self._clock.now()
```

**Change:** Replace `time.sleep()` with `self._clock.sleep()`:

```python
# Before
time.sleep(self.config.cycle_interval)

# After
self._clock.sleep(self.config.cycle_interval)
```

### 2. `LIVE_TRADING/engine/state.py`

**Change:** Accept clock in methods, use for timestamps.

```python
# Before (line ~103)
def update_position(self, symbol, weight, shares, price, timestamp=None):
    timestamp = timestamp or datetime.now()

# After
from LIVE_TRADING.common.clock import Clock, get_clock

def update_position(self, symbol, weight, shares, price, timestamp=None, clock: Optional[Clock] = None):
    clock = clock or get_clock()
    timestamp = timestamp or clock.now()
```

### 3. `LIVE_TRADING/brokers/paper.py`

**Change:** Accept clock in constructor.

```python
# Before (line ~50)
class PaperBroker:
    def __init__(self, initial_cash: float = 100000.0, ...):

# After
from LIVE_TRADING.common.clock import Clock, get_clock

class PaperBroker:
    def __init__(
        self,
        initial_cash: float = 100000.0,
        ...,
        clock: Optional[Clock] = None,
    ):
        self._clock = clock or get_clock()
```

**Change:** Replace `datetime.now(timezone.utc)` with `self._clock.now()`.

### 4. `LIVE_TRADING/engine/data_provider.py`

**Change:** Accept clock in constructor, use for quote timestamps.

```python
# Before
class ConfigurableDataProvider:
    def __init__(self, quotes: Optional[Dict] = None, ...):

# After
from LIVE_TRADING.common.clock import Clock, get_clock

class ConfigurableDataProvider:
    def __init__(
        self,
        quotes: Optional[Dict] = None,
        ...,
        clock: Optional[Clock] = None,
    ):
        self._clock = clock or get_clock()
```

## Locations to Update

Here's the complete list of `datetime.now` calls to replace:

| File | Line | Current Code | Replacement |
|------|------|--------------|-------------|
| `trading_engine.py` | 232 | `datetime.now(timezone.utc)` | `self._clock.now()` |
| `trading_engine.py` | 304 | `datetime.now(timezone.utc)` | `self._clock.now()` |
| `state.py` | 103 | `datetime.now()` | `clock.now()` |
| `state.py` | 177 | `datetime.now().isoformat()` | `clock.now().isoformat()` |
| `state.py` | 199 | `datetime.now()` | `clock.now()` |
| `state.py` | 221 | `datetime.now()` | `clock.now()` |
| `paper.py` | 200 | `datetime.now(timezone.utc)` | `self._clock.now()` |
| `paper.py` | 316 | `datetime.now(timezone.utc)` | `self._clock.now()` |
| `data_provider.py` | 144 | `datetime.now(timezone.utc)` | `self._clock.now()` |
| `data_provider.py` | 224 | `datetime.now(timezone.utc)` | `self._clock.now()` |

## Tests

### `LIVE_TRADING/tests/test_clock.py`

```python
"""
Clock Abstraction Tests
=======================

Unit tests for clock implementations.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from LIVE_TRADING.common.clock import (
    Clock,
    SystemClock,
    SimulatedClock,
    OffsetClock,
    get_clock,
    set_clock,
    reset_clock,
)


class TestSystemClock:
    """Tests for SystemClock."""

    def test_now_returns_utc(self):
        """Test that now() returns UTC time."""
        clock = SystemClock()
        now = clock.now()
        assert now.tzinfo == timezone.utc

    def test_now_is_current(self):
        """Test that now() returns approximately current time."""
        clock = SystemClock()
        before = datetime.now(timezone.utc)
        now = clock.now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after

    def test_timestamp_matches_now(self):
        """Test that timestamp() matches now()."""
        clock = SystemClock()
        now = clock.now()
        ts = clock.timestamp()
        assert abs(now.timestamp() - ts) < 0.1

    def test_implements_protocol(self):
        """Test that SystemClock implements Clock protocol."""
        clock = SystemClock()
        assert isinstance(clock, Clock)


class TestSimulatedClock:
    """Tests for SimulatedClock."""

    @pytest.fixture
    def clock(self):
        """Create simulated clock at known time."""
        return SimulatedClock(datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc))

    def test_now_returns_start_time(self, clock):
        """Test that now() returns start time initially."""
        assert clock.now() == datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)

    def test_advance_seconds(self, clock):
        """Test advancing by seconds."""
        clock.advance(seconds=30)
        assert clock.now() == datetime(2024, 1, 15, 9, 30, 30, tzinfo=timezone.utc)

    def test_advance_minutes(self, clock):
        """Test advancing by minutes."""
        clock.advance(minutes=5)
        assert clock.now() == datetime(2024, 1, 15, 9, 35, tzinfo=timezone.utc)

    def test_advance_hours(self, clock):
        """Test advancing by hours."""
        clock.advance(hours=2)
        assert clock.now() == datetime(2024, 1, 15, 11, 30, tzinfo=timezone.utc)

    def test_advance_combined(self, clock):
        """Test advancing by multiple units."""
        clock.advance(hours=1, minutes=30, seconds=45)
        assert clock.now() == datetime(2024, 1, 15, 11, 0, 45, tzinfo=timezone.utc)

    def test_set_time(self, clock):
        """Test setting time directly."""
        new_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        clock.set_time(new_time)
        assert clock.now() == new_time

    def test_sleep_records_duration(self, clock):
        """Test that sleep records duration without blocking."""
        import time
        start = time.time()
        clock.sleep(10.0)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be instant
        assert clock.get_sleep_calls() == [10.0]

    def test_sleep_multiple_calls(self, clock):
        """Test multiple sleep calls are recorded."""
        clock.sleep(1.0)
        clock.sleep(2.0)
        clock.sleep(3.0)
        assert clock.get_sleep_calls() == [1.0, 2.0, 3.0]

    def test_clear_sleep_calls(self, clock):
        """Test clearing sleep call history."""
        clock.sleep(5.0)
        clock.clear_sleep_calls()
        assert clock.get_sleep_calls() == []

    def test_timestamp_matches_simulated_time(self, clock):
        """Test that timestamp() reflects simulated time."""
        expected_ts = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc).timestamp()
        assert clock.timestamp() == expected_ts

    def test_requires_timezone_aware_start(self):
        """Test that naive datetime raises error."""
        with pytest.raises(ValueError, match="timezone-aware"):
            SimulatedClock(datetime(2024, 1, 15, 9, 30))  # No timezone

    def test_set_time_requires_timezone(self):
        """Test that set_time requires timezone-aware datetime."""
        clock = SimulatedClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(ValueError, match="timezone-aware"):
            clock.set_time(datetime(2024, 1, 2))  # No timezone

    def test_implements_protocol(self):
        """Test that SimulatedClock implements Clock protocol."""
        clock = SimulatedClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert isinstance(clock, Clock)


class TestOffsetClock:
    """Tests for OffsetClock."""

    def test_negative_offset(self):
        """Test clock with negative offset (past)."""
        clock = OffsetClock(hours=-1)
        system_now = datetime.now(timezone.utc)
        clock_now = clock.now()

        diff = abs((system_now - clock_now).total_seconds() - 3600)
        assert diff < 1  # Within 1 second

    def test_positive_offset(self):
        """Test clock with positive offset (future)."""
        clock = OffsetClock(hours=1)
        system_now = datetime.now(timezone.utc)
        clock_now = clock.now()

        diff = abs((clock_now - system_now).total_seconds() - 3600)
        assert diff < 1

    def test_combined_offset(self):
        """Test clock with combined offsets."""
        clock = OffsetClock(days=-1, hours=-2, minutes=-30)
        expected_offset = timedelta(days=1, hours=2, minutes=30)

        system_now = datetime.now(timezone.utc)
        clock_now = clock.now()

        diff = abs((system_now - clock_now).total_seconds() - expected_offset.total_seconds())
        assert diff < 1

    def test_implements_protocol(self):
        """Test that OffsetClock implements Clock protocol."""
        clock = OffsetClock()
        assert isinstance(clock, Clock)


class TestGlobalClock:
    """Tests for global clock functions."""

    def teardown_method(self):
        """Reset clock after each test."""
        reset_clock()

    def test_default_is_system_clock(self):
        """Test that default clock is SystemClock."""
        clock = get_clock()
        assert isinstance(clock, SystemClock)

    def test_set_clock(self):
        """Test setting global clock."""
        sim_clock = SimulatedClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        set_clock(sim_clock)

        assert get_clock() is sim_clock
        assert get_clock().now() == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_reset_clock(self):
        """Test resetting to system clock."""
        sim_clock = SimulatedClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        set_clock(sim_clock)
        reset_clock()

        assert isinstance(get_clock(), SystemClock)


class TestClockIntegration:
    """Integration tests for clock with trading components."""

    def test_simulated_clock_enables_backtesting(self):
        """Test that simulated clock allows time control for backtesting."""
        clock = SimulatedClock(datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc))

        # Simulate a trading day
        timestamps = []
        for _ in range(10):
            timestamps.append(clock.now())
            clock.advance(minutes=5)

        # Verify time progression
        assert timestamps[0] == datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
        assert timestamps[-1] == datetime(2024, 1, 2, 10, 15, tzinfo=timezone.utc)

        # Verify consistent 5-minute intervals
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            assert delta == 300  # 5 minutes
```

## SST Compliance

- [x] Protocol-based design
- [x] No direct datetime.now() in trading code
- [x] Timezone-aware throughout (UTC only)
- [x] Testable via dependency injection

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `common/clock.py` | 180 |
| `tests/test_clock.py` | 200 |
| Modifications to existing files | ~50 |
| **Total** | ~430 |
