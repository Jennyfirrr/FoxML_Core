# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Interval Specification and Conversion Utilities

Single source of truth for interval handling. All bar↔minute conversions
must use these utilities to ensure consistent rounding policy.

Core invariant: Store time (minutes), derive bars with ceil() rounding.

Usage:
    from TRAINING.common.interval import (
        IntervalSpec,
        IntervalSource,
        minutes_to_bars,
        bars_to_minutes,
        validate_interval_target_compatibility,
    )

    # Create interval spec
    spec = IntervalSpec.from_minutes(5)
    spec = IntervalSpec.from_string("5m")

    # Convert time to bars (always use ceil to prevent under-lookback)
    bars = minutes_to_bars(70, interval_minutes=5)  # Returns 14

    # Validate interval/target compatibility
    is_valid, warnings = validate_interval_target_compatibility(
        interval_minutes=5,
        target_horizon_minutes=60,
    )

    # Enable audit logging for migration debugging
    from TRAINING.common.interval import set_audit_mode, get_conversion_audit_log
    set_audit_mode(True)
    # ... run conversions ...
    audit_log = get_conversion_audit_log()
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

from TRAINING.common.utils.duration_parser import Duration, parse_duration

logger = logging.getLogger(__name__)


# =============================================================================
# Audit Logging for Migration
# =============================================================================

@dataclass
class ConversionAuditEntry:
    """Record of a single bar/minute conversion for audit."""
    function: str
    input_value: float
    interval_minutes: int
    rounding: str
    result: int
    caller: str = ""  # Optional caller context


@dataclass
class ConversionAuditLog:
    """Accumulated audit log of conversions."""
    entries: list[ConversionAuditEntry] = field(default_factory=list)

    def add(self, entry: ConversionAuditEntry) -> None:
        """Add an entry to the log."""
        self.entries.append(entry)

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()

    def summary(self) -> dict:
        """Get summary statistics."""
        if not self.entries:
            return {"total": 0}

        by_function = {}
        by_rounding = {}
        for e in self.entries:
            by_function[e.function] = by_function.get(e.function, 0) + 1
            by_rounding[e.rounding] = by_rounding.get(e.rounding, 0) + 1

        return {
            "total": len(self.entries),
            "by_function": by_function,
            "by_rounding": by_rounding,
        }


# Module-level audit state
_audit_mode: bool = False
_audit_log: ConversionAuditLog = ConversionAuditLog()


def set_audit_mode(enabled: bool) -> None:
    """Enable or disable conversion audit logging."""
    global _audit_mode
    _audit_mode = enabled
    if enabled:
        logger.info("Interval conversion audit mode ENABLED")
    else:
        logger.info("Interval conversion audit mode DISABLED")


def get_audit_mode() -> bool:
    """Check if audit mode is enabled."""
    return _audit_mode


def get_conversion_audit_log() -> ConversionAuditLog:
    """Get the current audit log."""
    return _audit_log


def clear_conversion_audit_log() -> None:
    """Clear the audit log."""
    _audit_log.clear()


@contextmanager
def audit_conversions() -> Iterator[ConversionAuditLog]:
    """
    Context manager for temporarily enabling audit mode.

    Usage:
        with audit_conversions() as log:
            # ... run code that does conversions ...
            print(log.summary())
    """
    global _audit_mode
    prev_mode = _audit_mode
    _audit_log.clear()
    _audit_mode = True
    try:
        yield _audit_log
    finally:
        _audit_mode = prev_mode


def _maybe_audit(
    function: str,
    input_value: float,
    interval_minutes: int,
    rounding: str,
    result: int,
    caller: str = "",
) -> None:
    """Record conversion if audit mode is enabled."""
    if _audit_mode:
        entry = ConversionAuditEntry(
            function=function,
            input_value=input_value,
            interval_minutes=interval_minutes,
            rounding=rounding,
            result=result,
            caller=caller,
        )
        _audit_log.add(entry)
        logger.debug(
            f"[AUDIT] {function}({input_value}, {interval_minutes}, {rounding}) = {result}"
        )


class IntervalSource(str, Enum):
    """How the interval was determined."""

    CONFIG = "config"  # Explicit in config file
    DETECTED = "detected"  # Auto-detected from timestamps
    EXPLICIT = "explicit"  # Passed as function argument
    UNKNOWN = "unknown"  # Legacy/untracked


@dataclass(frozen=True)
class IntervalSpec:
    """
    Canonical interval specification with provenance tracking.

    Attributes:
        duration: Interval as Duration (canonical representation)
        source: How the interval was determined
        confidence: Detection confidence (1.0 = certain, <1.0 = inferred)
        is_uniform: Whether timestamps are uniformly spaced
    """

    duration: Duration
    source: IntervalSource
    confidence: float = 1.0
    is_uniform: bool = True

    def __post_init__(self) -> None:
        if self.duration.to_minutes() <= 0:
            raise ValueError(
                f"IntervalSpec.duration must be positive, got {self.duration}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"IntervalSpec.confidence must be in [0, 1], got {self.confidence}"
            )

    @property
    def minutes(self) -> int:
        """Interval in minutes (integer, for bar calculations)."""
        return int(self.duration.to_minutes())

    @property
    def seconds(self) -> float:
        """Interval in seconds."""
        return self.duration.to_seconds()

    @staticmethod
    def from_minutes(
        minutes: int,
        source: IntervalSource = IntervalSource.EXPLICIT,
        confidence: float = 1.0,
        is_uniform: bool = True,
    ) -> "IntervalSpec":
        """Create IntervalSpec from minutes."""
        return IntervalSpec(
            duration=Duration.from_seconds(minutes * 60),
            source=source,
            confidence=confidence,
            is_uniform=is_uniform,
        )

    @staticmethod
    def from_string(
        interval_str: str,
        source: IntervalSource = IntervalSource.CONFIG,
        confidence: float = 1.0,
        is_uniform: bool = True,
    ) -> "IntervalSpec":
        """Create IntervalSpec from string like '5m', '1h'."""
        duration = parse_duration(interval_str)
        return IntervalSpec(
            duration=duration,
            source=source,
            confidence=confidence,
            is_uniform=is_uniform,
        )

    def to_dict(self) -> dict:
        """Serialize for artifacts."""
        return {
            "minutes": self.minutes,
            "seconds": self.seconds,
            "source": self.source.value,
            "confidence": self.confidence,
            "is_uniform": self.is_uniform,
        }


# =============================================================================
# Conversion Utilities (Single Source of Truth)
# =============================================================================


def minutes_to_bars(
    lookback_minutes: float,
    interval_minutes: int,
    rounding: str = "ceil",
) -> int:
    """
    Convert time (minutes) to bars with explicit rounding policy.

    CRITICAL: Always use ceil() to avoid under-lookback (data leakage).

    Args:
        lookback_minutes: Lookback period in minutes
        interval_minutes: Bar interval in minutes
        rounding: "ceil" (default, safe) or "floor" (only if you know what you're doing)

    Returns:
        Number of bars

    Raises:
        ValueError: If inputs are invalid

    Examples:
        >>> minutes_to_bars(100, 5)  # Exact: 100/5 = 20
        20
        >>> minutes_to_bars(101, 5)  # Ceil: 101/5 = 20.2 → 21
        21
        >>> minutes_to_bars(101, 5, rounding="floor")  # Floor: 101/5 = 20.2 → 20
        20
    """
    if lookback_minutes < 0:
        raise ValueError(f"lookback_minutes must be >= 0, got {lookback_minutes}")
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")

    if lookback_minutes == 0:
        result = 0
    elif rounding == "ceil":
        result = int(math.ceil(lookback_minutes / interval_minutes))
    elif rounding == "floor":
        result = int(lookback_minutes // interval_minutes)
    else:
        raise ValueError(f"Unknown rounding policy: {rounding}. Use 'ceil' or 'floor'.")

    # Audit logging for migration debugging
    _maybe_audit(
        function="minutes_to_bars",
        input_value=lookback_minutes,
        interval_minutes=interval_minutes,
        rounding=rounding,
        result=result,
    )

    return result


def bars_to_minutes(bars: int, interval_minutes: int) -> int:
    """
    Convert bars to minutes.

    Args:
        bars: Number of bars
        interval_minutes: Bar interval in minutes

    Returns:
        Time in minutes

    Raises:
        ValueError: If inputs are invalid

    Examples:
        >>> bars_to_minutes(14, 5)
        70
        >>> bars_to_minutes(60, 1)
        60
    """
    if bars < 0:
        raise ValueError(f"bars must be >= 0, got {bars}")
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")

    result = bars * interval_minutes

    # Audit logging for migration debugging
    _maybe_audit(
        function="bars_to_minutes",
        input_value=float(bars),
        interval_minutes=interval_minutes,
        rounding="exact",  # bars_to_minutes has no rounding
        result=result,
    )

    return result


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_interval_target_compatibility(
    *,
    interval_minutes: int,
    target_horizon_minutes: int,
    strict: bool = True,
    min_forward_bars: int = 1,
) -> tuple[bool, list[str]]:
    """
    Validate that data interval is compatible with target horizon.

    Rules:
    1. interval <= horizon (can't predict finer than data resolution)
    2. horizon / interval >= min_forward_bars (enough temporal structure)

    Args:
        interval_minutes: Data bar interval
        target_horizon_minutes: Target prediction horizon
        strict: If True, raise on violation; if False, return warnings
        min_forward_bars: Minimum bars forward (default: 1)

    Returns:
        (is_valid, warnings) tuple

    Raises:
        ValueError: If strict=True and validation fails

    Examples:
        >>> validate_interval_target_compatibility(interval_minutes=5, target_horizon_minutes=60)
        (True, [])
        >>> validate_interval_target_compatibility(interval_minutes=15, target_horizon_minutes=10, strict=False)
        (False, ['Invalid: interval (15m) > horizon (10m)...'])
    """
    warnings: list[str] = []
    is_valid = True

    # Rule 1: interval <= horizon
    if interval_minutes > target_horizon_minutes:
        msg = (
            f"Invalid: interval ({interval_minutes}m) > horizon ({target_horizon_minutes}m). "
            f"Cannot predict {target_horizon_minutes}m forward with {interval_minutes}m bars."
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
        is_valid = False

    # Rule 2: enough forward bars
    if is_valid:
        forward_bars = minutes_to_bars(target_horizon_minutes, interval_minutes)
        if forward_bars < min_forward_bars:
            msg = f"Only {forward_bars} forward bar(s); recommended minimum is {min_forward_bars}."
            warnings.append(msg)
            # Not a hard failure, just a warning

    # Non-aligned warning
    if is_valid and (target_horizon_minutes % interval_minutes != 0):
        warnings.append(
            f"Horizon ({target_horizon_minutes}m) not evenly divisible by interval ({interval_minutes}m). "
            f"Will use ceil() rounding."
        )

    return is_valid, warnings


# =============================================================================
# Interval Fallback Helpers (Priority 3 Remediation)
# =============================================================================

# Module-level fallback state
_INTERVAL_FALLBACK_MINUTES: float = 5.0
_INTERVAL_FALLBACK_WARNED: bool = False


def get_interval_fallback() -> float:
    """
    Get fallback interval with warning (logs warning once per session).

    This should only be used when:
    - Data interval detection fails
    - Config doesn't specify interval
    - No CLI override provided

    Returns:
        5.0 (minutes) with logged warning

    Example:
        >>> interval = detected_interval or get_interval_fallback()
    """
    global _INTERVAL_FALLBACK_WARNED

    if not _INTERVAL_FALLBACK_WARNED:
        logger.warning(
            f"Using fallback interval ({_INTERVAL_FALLBACK_MINUTES}m). "
            "For production, specify interval via: "
            "1) Data directory naming (interval=Xm), "
            "2) Config (pipeline.data.interval_minutes), or "
            "3) CLI (--interval-minutes)"
        )
        _INTERVAL_FALLBACK_WARNED = True

    return _INTERVAL_FALLBACK_MINUTES


def reset_interval_fallback_warning() -> None:
    """Reset the fallback warning flag (useful for testing)."""
    global _INTERVAL_FALLBACK_WARNED
    _INTERVAL_FALLBACK_WARNED = False


def get_interval_strict(interval_minutes: Optional[float]) -> float:
    """
    Strict mode: require explicit interval (errors if not provided).

    Use this in strict mode pipelines where missing interval should fail fast.

    Args:
        interval_minutes: The interval value (may be None)

    Returns:
        The interval if provided

    Raises:
        ValueError: If interval_minutes is None

    Example:
        >>> interval = get_interval_strict(config.get("interval_minutes"))
    """
    if interval_minutes is None:
        raise ValueError(
            "Interval not specified. In strict mode, interval must be explicit. "
            "Set pipeline.data.interval_minutes in config or pass --interval-minutes."
        )
    return float(interval_minutes)


def get_interval_with_fallback(
    interval_minutes: Optional[float],
    strict: bool = False,
) -> float:
    """
    Get interval with configurable strictness.

    Combines get_interval_strict and get_interval_fallback based on mode.

    Args:
        interval_minutes: The interval value (may be None)
        strict: If True, error on missing interval; if False, use fallback

    Returns:
        The interval (from parameter or fallback)

    Raises:
        ValueError: If strict=True and interval_minutes is None

    Example:
        >>> interval = get_interval_with_fallback(
        ...     config.get("interval_minutes"),
        ...     strict=determinism_mode
        ... )
    """
    if interval_minutes is not None:
        return float(interval_minutes)

    if strict:
        return get_interval_strict(interval_minutes)

    return get_interval_fallback()
