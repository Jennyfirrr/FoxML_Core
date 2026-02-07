"""
Date Parsing Utilities
======================

Shared date/time parsing functions for data providers and backtesting.

This module consolidates duplicate parsing logic that was spread across:
- LIVE_TRADING/data/alpaca.py
- LIVE_TRADING/data/polygon.py
- LIVE_TRADING/data/simulated.py
- LIVE_TRADING/backtest/data_loader.py

SST Compliance:
- Uses get_cfg() for default values
- All datetime operations are timezone-aware (UTC)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

from CONFIG.config_loader import get_cfg


def parse_period(period: str, end: datetime) -> datetime:
    """
    Parse period string to calculate start datetime.

    Supported formats:
    - "Nd" or "Nday" or "Ndays": N days ago
    - "Nw" or "Nweek" or "Nweeks": N weeks ago
    - "Nmo" or "Nmonth" or "Nmonths": N months ago (approx 30 days)
    - "Ny" or "Nyear" or "Nyears": N years ago (approx 365 days)

    Args:
        period: Period string (e.g., "1mo", "30d", "1y")
        end: End datetime (usually now)

    Returns:
        Start datetime

    Example:
        >>> end = datetime(2024, 1, 15, tzinfo=timezone.utc)
        >>> parse_period("30d", end)
        datetime(2023, 12, 16, tzinfo=timezone.utc)
    """
    period = period.lower().strip()

    # Days
    if period.endswith("days"):
        days = int(period[:-4])
        return end - timedelta(days=days)
    if period.endswith("day"):
        days = int(period[:-3])
        return end - timedelta(days=days)
    if period.endswith("d"):
        days = int(period[:-1])
        return end - timedelta(days=days)

    # Weeks
    if period.endswith("weeks"):
        weeks = int(period[:-5])
        return end - timedelta(weeks=weeks)
    if period.endswith("week"):
        weeks = int(period[:-4])
        return end - timedelta(weeks=weeks)
    if period.endswith("w"):
        weeks = int(period[:-1])
        return end - timedelta(weeks=weeks)

    # Months (approximate as 30 days)
    if period.endswith("months"):
        months = int(period[:-6])
        return end - timedelta(days=months * 30)
    if period.endswith("month"):
        months = int(period[:-5])
        return end - timedelta(days=months * 30)
    if period.endswith("mo"):
        months = int(period[:-2])
        return end - timedelta(days=months * 30)

    # Years (approximate as 365 days)
    if period.endswith("years"):
        years = int(period[:-5])
        return end - timedelta(days=years * 365)
    if period.endswith("year"):
        years = int(period[:-4])
        return end - timedelta(days=years * 365)
    if period.endswith("y"):
        years = int(period[:-1])
        return end - timedelta(days=years * 365)

    # Default: try to parse as days
    try:
        days = int(period)
        return end - timedelta(days=days)
    except ValueError:
        raise ValueError(f"Unknown period format: {period}")


def parse_interval(interval: str) -> Tuple[str, int]:
    """
    Parse interval string to unit and multiplier.

    Supported formats:
    - "Nm" or "Nmin": N minutes
    - "Nh" or "Nhour": N hours
    - "Nd" or "Nday": N days
    - "Nw" or "Nweek": N weeks

    Args:
        interval: Interval string (e.g., "1m", "5min", "1h", "1d")

    Returns:
        Tuple of (unit, multiplier) where unit is one of:
        - "minute"
        - "hour"
        - "day"
        - "week"

    Example:
        >>> parse_interval("5m")
        ("minute", 5)
        >>> parse_interval("1h")
        ("hour", 1)
    """
    interval = interval.lower().strip()

    # Minutes
    if interval.endswith("min"):
        return ("minute", int(interval[:-3]))
    if interval.endswith("m"):
        return ("minute", int(interval[:-1]))

    # Hours
    if interval.endswith("hour"):
        return ("hour", int(interval[:-4]))
    if interval.endswith("h"):
        return ("hour", int(interval[:-1]))

    # Days
    if interval.endswith("day"):
        return ("day", int(interval[:-3]))
    if interval.endswith("d"):
        return ("day", int(interval[:-1]))

    # Weeks
    if interval.endswith("week"):
        return ("week", int(interval[:-4]))
    if interval.endswith("w"):
        return ("week", int(interval[:-1]))

    raise ValueError(f"Unknown interval format: {interval}")


def interval_to_minutes(interval: str) -> int:
    """
    Convert interval string to total minutes.

    Args:
        interval: Interval string (e.g., "1m", "5min", "1h", "1d")

    Returns:
        Total minutes

    Example:
        >>> interval_to_minutes("5m")
        5
        >>> interval_to_minutes("1h")
        60
        >>> interval_to_minutes("1d")
        1440
    """
    unit, multiplier = parse_interval(interval)

    if unit == "minute":
        return multiplier
    elif unit == "hour":
        return multiplier * 60
    elif unit == "day":
        return multiplier * 1440
    elif unit == "week":
        return multiplier * 1440 * 7
    else:
        raise ValueError(f"Unknown interval unit: {unit}")


def is_cache_valid(
    cache_time: datetime,
    ttl_seconds: float,
    now: datetime | None = None,
) -> bool:
    """
    Check if a cached value is still valid.

    Args:
        cache_time: When the value was cached
        ttl_seconds: Time-to-live in seconds
        now: Current time (default: UTC now)

    Returns:
        True if cache is still valid

    Example:
        >>> cache_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        >>> is_cache_valid(cache_time, ttl_seconds=10.0)
        True
    """
    if now is None:
        now = datetime.now(timezone.utc)

    age_seconds = (now - cache_time).total_seconds()
    return age_seconds < ttl_seconds


def get_default_historical_period() -> str:
    """
    Get default historical data period from config.

    Returns:
        Period string (default: "1mo")
    """
    return get_cfg("live_trading.data.default_period", default="1mo")


def get_default_interval() -> str:
    """
    Get default data interval from config.

    Returns:
        Interval string (default: "1d")
    """
    return get_cfg("live_trading.data.default_interval", default="1d")
