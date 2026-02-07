# Plan 0E: Timezone Standardization

## Overview

Standardize all datetime handling to use UTC timezone-aware datetimes throughout the codebase.

## Problem Statement

Current code mixes timezone-aware and naive datetimes:
```python
# In state.py:103
timestamp = timestamp or datetime.now()  # NAIVE! No timezone

# In trading_engine.py:232
current_time = current_time or datetime.now(timezone.utc)  # UTC aware

# These cannot be compared - will raise TypeError in Python 3.x
```

This causes:
- Comparison failures between datetimes
- Incorrect timestamps in logs/artifacts
- Backtesting time travel bugs

## Files to Create

### 1. `LIVE_TRADING/common/time_utils.py`

```python
"""
Time Utilities
==============

Standardized time handling - all times are UTC timezone-aware.

Rules:
1. All datetimes must be timezone-aware (UTC)
2. Storage format: ISO 8601 with timezone
3. Never use naive datetimes in trading code

SST Compliance:
- Consistent timestamp format
- Deterministic serialization
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """
    Get current UTC time (timezone-aware).

    This is the ONLY approved way to get current time in trading code.
    Use Clock abstraction for testability.

    Returns:
        Current UTC datetime (timezone-aware)
    """
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure datetime is UTC timezone-aware.

    Args:
        dt: Datetime to convert (can be naive or aware)

    Returns:
        UTC timezone-aware datetime, or None if input was None
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # Naive datetime - assume it's already UTC and add timezone
        logger.debug(f"Converting naive datetime to UTC: {dt}")
        return dt.replace(tzinfo=timezone.utc)

    if dt.tzinfo == timezone.utc:
        return dt

    # Convert from other timezone to UTC
    return dt.astimezone(timezone.utc)


def require_utc(dt: datetime) -> datetime:
    """
    Require datetime to be UTC timezone-aware.

    Args:
        dt: Datetime that must be UTC

    Returns:
        The datetime if valid

    Raises:
        ValueError: If datetime is naive or not UTC
    """
    if dt.tzinfo is None:
        raise ValueError(f"Datetime must be timezone-aware, got naive: {dt}")

    if dt.tzinfo != timezone.utc:
        raise ValueError(f"Datetime must be UTC, got {dt.tzinfo}: {dt}")

    return dt


def parse_iso(s: str) -> datetime:
    """
    Parse ISO 8601 datetime string to UTC datetime.

    Handles:
    - "2024-01-15T09:30:00Z"
    - "2024-01-15T09:30:00+00:00"
    - "2024-01-15T09:30:00" (assumes UTC)

    Args:
        s: ISO 8601 datetime string

    Returns:
        UTC timezone-aware datetime
    """
    # Handle Z suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)

    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


def to_iso(dt: datetime) -> str:
    """
    Convert datetime to ISO 8601 string (UTC).

    Args:
        dt: Datetime to convert

    Returns:
        ISO 8601 string with timezone
    """
    utc_dt = ensure_utc(dt)
    return utc_dt.isoformat()


def to_iso_z(dt: datetime) -> str:
    """
    Convert datetime to ISO 8601 string with Z suffix.

    Args:
        dt: Datetime to convert

    Returns:
        ISO 8601 string ending in Z (e.g., "2024-01-15T09:30:00Z")
    """
    utc_dt = ensure_utc(dt)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_ms(dt: Optional[datetime] = None) -> int:
    """
    Get Unix timestamp in milliseconds.

    Args:
        dt: Datetime to convert (default: now)

    Returns:
        Unix timestamp in milliseconds
    """
    if dt is None:
        dt = utc_now()
    else:
        dt = ensure_utc(dt)

    return int(dt.timestamp() * 1000)


def from_timestamp_ms(ts_ms: int) -> datetime:
    """
    Convert Unix timestamp in milliseconds to UTC datetime.

    Args:
        ts_ms: Unix timestamp in milliseconds

    Returns:
        UTC timezone-aware datetime
    """
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def trading_day(dt: Optional[datetime] = None) -> str:
    """
    Get trading day string (YYYY-MM-DD) for a datetime.

    Note: Uses UTC date. For market-specific dates, use market calendar.

    Args:
        dt: Datetime (default: now)

    Returns:
        Date string in YYYY-MM-DD format
    """
    if dt is None:
        dt = utc_now()
    else:
        dt = ensure_utc(dt)

    return dt.strftime("%Y-%m-%d")


def is_same_trading_day(dt1: datetime, dt2: datetime) -> bool:
    """
    Check if two datetimes are on the same trading day.

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        True if same UTC date
    """
    return trading_day(dt1) == trading_day(dt2)


def market_open_utc(date: Optional[datetime] = None) -> datetime:
    """
    Get US market open time in UTC (9:30 AM ET = 14:30 UTC in EST).

    Note: This is simplified. Real implementation should use market calendar
    that accounts for DST.

    Args:
        date: Date to get market open for (default: today)

    Returns:
        Market open time in UTC
    """
    if date is None:
        date = utc_now()
    else:
        date = ensure_utc(date)

    # Simplified: assume EST (UTC-5)
    # Real implementation would check DST
    return date.replace(hour=14, minute=30, second=0, microsecond=0)


def market_close_utc(date: Optional[datetime] = None) -> datetime:
    """
    Get US market close time in UTC (4:00 PM ET = 21:00 UTC in EST).

    Note: This is simplified. Real implementation should use market calendar.

    Args:
        date: Date to get market close for (default: today)

    Returns:
        Market close time in UTC
    """
    if date is None:
        date = utc_now()
    else:
        date = ensure_utc(date)

    return date.replace(hour=21, minute=0, second=0, microsecond=0)


# Validation decorator for functions that require UTC datetimes
def validate_utc_params(*param_names):
    """
    Decorator to validate that datetime parameters are UTC.

    Usage:
        @validate_utc_params("timestamp", "created_at")
        def process_trade(timestamp: datetime, created_at: datetime):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in param_names:
                if name in bound.arguments:
                    value = bound.arguments[name]
                    if value is not None:
                        if not isinstance(value, datetime):
                            raise TypeError(f"{name} must be datetime, got {type(value)}")
                        if value.tzinfo is None:
                            raise ValueError(f"{name} must be timezone-aware")
                        if value.tzinfo != timezone.utc:
                            # Auto-convert to UTC
                            bound.arguments[name] = value.astimezone(timezone.utc)

            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator
```

### 2. Update `LIVE_TRADING/common/__init__.py`

```python
# Add to existing exports
from .time_utils import (
    utc_now,
    ensure_utc,
    require_utc,
    parse_iso,
    to_iso,
    to_iso_z,
    timestamp_ms,
    from_timestamp_ms,
    trading_day,
    is_same_trading_day,
    validate_utc_params,
)
```

## Files to Modify

Replace all `datetime.now()` (naive) with proper UTC handling:

### 1. `LIVE_TRADING/engine/state.py`

```python
# Before
timestamp = timestamp or datetime.now()

# After
from LIVE_TRADING.common.time_utils import ensure_utc
from LIVE_TRADING.common.clock import get_clock

timestamp = timestamp or get_clock().now()
timestamp = ensure_utc(timestamp)  # Safety check
```

### 2. All serialization points

```python
# Before
"timestamp": datetime.now().isoformat()

# After
from LIVE_TRADING.common.time_utils import to_iso
"timestamp": to_iso(clock.now())
```

### 3. All deserialization points

```python
# Before
timestamp = datetime.fromisoformat(data["timestamp"])

# After
from LIVE_TRADING.common.time_utils import parse_iso
timestamp = parse_iso(data["timestamp"])
```

## Locations to Update

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `state.py` | 103 | `datetime.now()` naive | Use `clock.now()` |
| `state.py` | 177 | `.isoformat()` | Use `to_iso()` |
| `state.py` | 199 | `datetime.now()` naive | Use `clock.now()` |
| `state.py` | 221 | `datetime.now()` naive | Use `clock.now()` |
| `state.py` | 288 | `fromisoformat()` | Use `parse_iso()` |
| `paper.py` | 200 | OK (uses UTC) | Keep |
| `paper.py` | 316 | OK (uses UTC) | Keep |
| `types.py` | Various | Check serialization | Use `to_iso()` |

## Tests

### `LIVE_TRADING/tests/test_time_utils.py`

```python
"""
Time Utilities Tests
====================

Unit tests for timezone-aware time handling.
"""

import pytest
from datetime import datetime, timezone, timedelta

from LIVE_TRADING.common.time_utils import (
    utc_now,
    ensure_utc,
    require_utc,
    parse_iso,
    to_iso,
    to_iso_z,
    timestamp_ms,
    from_timestamp_ms,
    trading_day,
    is_same_trading_day,
    validate_utc_params,
)


class TestUtcNow:
    """Tests for utc_now."""

    def test_returns_utc(self):
        """Test that utc_now returns UTC timezone."""
        now = utc_now()
        assert now.tzinfo == timezone.utc

    def test_is_current(self):
        """Test that utc_now is approximately current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)

        assert before <= now <= after


class TestEnsureUtc:
    """Tests for ensure_utc."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert ensure_utc(None) is None

    def test_naive_datetime_converted(self):
        """Test naive datetime gets UTC timezone added."""
        naive = datetime(2024, 1, 15, 9, 30)
        result = ensure_utc(naive)

        assert result.tzinfo == timezone.utc
        assert result.hour == 9  # Same time, just with timezone

    def test_utc_datetime_unchanged(self):
        """Test UTC datetime is returned unchanged."""
        utc = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        result = ensure_utc(utc)

        assert result == utc

    def test_other_timezone_converted(self):
        """Test other timezones are converted to UTC."""
        # Create datetime in EST (UTC-5)
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=est)

        result = ensure_utc(dt)

        assert result.tzinfo == timezone.utc
        assert result.hour == 14  # 9:30 EST = 14:30 UTC


class TestRequireUtc:
    """Tests for require_utc."""

    def test_utc_passes(self):
        """Test UTC datetime passes validation."""
        utc = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        assert require_utc(utc) == utc

    def test_naive_raises(self):
        """Test naive datetime raises ValueError."""
        naive = datetime(2024, 1, 15, 9, 30)
        with pytest.raises(ValueError, match="timezone-aware"):
            require_utc(naive)

    def test_other_timezone_raises(self):
        """Test non-UTC timezone raises ValueError."""
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=est)

        with pytest.raises(ValueError, match="UTC"):
            require_utc(dt)


class TestParseIso:
    """Tests for parse_iso."""

    def test_parse_z_suffix(self):
        """Test parsing ISO string with Z suffix."""
        result = parse_iso("2024-01-15T09:30:00Z")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 9
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_parse_offset_suffix(self):
        """Test parsing ISO string with +00:00 offset."""
        result = parse_iso("2024-01-15T09:30:00+00:00")

        assert result.tzinfo == timezone.utc
        assert result.hour == 9

    def test_parse_no_timezone(self):
        """Test parsing ISO string without timezone assumes UTC."""
        result = parse_iso("2024-01-15T09:30:00")

        assert result.tzinfo == timezone.utc

    def test_parse_other_offset(self):
        """Test parsing ISO string with non-UTC offset converts to UTC."""
        result = parse_iso("2024-01-15T09:30:00-05:00")

        assert result.tzinfo == timezone.utc
        assert result.hour == 14  # 9:30 - 5 = 14:30 UTC


class TestToIso:
    """Tests for to_iso."""

    def test_utc_datetime(self):
        """Test serializing UTC datetime."""
        dt = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        result = to_iso(dt)

        assert "2024-01-15" in result
        assert "09:30:00" in result
        assert "+00:00" in result

    def test_naive_datetime_treated_as_utc(self):
        """Test naive datetime is treated as UTC."""
        dt = datetime(2024, 1, 15, 9, 30, 0)
        result = to_iso(dt)

        assert "+00:00" in result


class TestToIsoZ:
    """Tests for to_iso_z."""

    def test_z_suffix(self):
        """Test Z suffix format."""
        dt = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        result = to_iso_z(dt)

        assert result == "2024-01-15T09:30:00Z"


class TestTimestampMs:
    """Tests for timestamp_ms."""

    def test_known_timestamp(self):
        """Test converting known datetime to timestamp."""
        dt = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = timestamp_ms(dt)

        # 2024-01-15 00:00:00 UTC
        expected = 1705276800000
        assert result == expected

    def test_from_timestamp_ms(self):
        """Test converting timestamp back to datetime."""
        ts = 1705276800000
        result = from_timestamp_ms(ts)

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc


class TestTradingDay:
    """Tests for trading_day."""

    def test_returns_date_string(self):
        """Test trading_day returns YYYY-MM-DD format."""
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        result = trading_day(dt)

        assert result == "2024-01-15"

    def test_is_same_trading_day_true(self):
        """Test same day returns True."""
        dt1 = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)

        assert is_same_trading_day(dt1, dt2)

    def test_is_same_trading_day_false(self):
        """Test different days returns False."""
        dt1 = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 16, 9, 30, tzinfo=timezone.utc)

        assert not is_same_trading_day(dt1, dt2)


class TestValidateUtcParams:
    """Tests for validate_utc_params decorator."""

    def test_valid_utc_params(self):
        """Test valid UTC params pass."""
        @validate_utc_params("ts")
        def func(ts: datetime) -> datetime:
            return ts

        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        assert func(dt) == dt

    def test_naive_param_raises(self):
        """Test naive param raises ValueError."""
        @validate_utc_params("ts")
        def func(ts: datetime) -> datetime:
            return ts

        dt = datetime(2024, 1, 15)  # Naive
        with pytest.raises(ValueError, match="timezone-aware"):
            func(dt)

    def test_auto_converts_other_timezone(self):
        """Test non-UTC timezone is auto-converted."""
        @validate_utc_params("ts")
        def func(ts: datetime) -> datetime:
            return ts

        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=est)

        result = func(dt)
        assert result.tzinfo == timezone.utc
        assert result.hour == 14
```

## Audit Script

Create a script to find remaining timezone issues:

### `bin/audit_timezones.py`

```python
#!/usr/bin/env python3
"""Audit codebase for timezone issues."""

import ast
import sys
from pathlib import Path


class TimezoneAuditor(ast.NodeVisitor):
    """Find datetime.now() calls without timezone."""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues = []

    def visit_Call(self, node):
        # Check for datetime.now()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "now":
                # Check if it's datetime.now
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "datetime":
                        # Check if timezone.utc is passed
                        if not node.args and not node.keywords:
                            self.issues.append({
                                "line": node.lineno,
                                "issue": "datetime.now() without timezone",
                            })

        # Check for fromisoformat without ensure_utc
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "fromisoformat":
                self.issues.append({
                    "line": node.lineno,
                    "issue": "fromisoformat() - ensure result is UTC",
                })

        self.generic_visit(node)


def audit_file(path: Path) -> list:
    """Audit a single file."""
    try:
        with open(path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        auditor = TimezoneAuditor(str(path))
        auditor.visit(tree)
        return auditor.issues
    except Exception as e:
        return [{"line": 0, "issue": f"Parse error: {e}"}]


def main():
    live_trading_dir = Path("LIVE_TRADING")
    issues = []

    for py_file in live_trading_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        file_issues = audit_file(py_file)
        if file_issues:
            issues.append((py_file, file_issues))

    if issues:
        print("Timezone issues found:")
        for path, file_issues in issues:
            print(f"\n{path}:")
            for issue in file_issues:
                print(f"  Line {issue['line']}: {issue['issue']}")
        sys.exit(1)
    else:
        print("No timezone issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

## SST Compliance

- [x] All datetimes are UTC timezone-aware
- [x] Consistent serialization format (ISO 8601)
- [x] Validation utilities for enforcement
- [x] Audit script for verification

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `common/time_utils.py` | 250 |
| `tests/test_time_utils.py` | 220 |
| `bin/audit_timezones.py` | 80 |
| Modifications to existing files | ~40 |
| **Total** | ~590 |
