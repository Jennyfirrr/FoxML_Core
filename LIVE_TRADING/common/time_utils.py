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
from typing import Optional
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
