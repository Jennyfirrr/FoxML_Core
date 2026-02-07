# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Horizon Conversion Utilities

Centralized conversion logic for converting horizon_minutes to horizon_bars.
This is the SINGLE SOURCE OF TRUTH for horizon conversion - use everywhere.
"""

from typing import Optional


def horizon_minutes_to_bars(
    horizon_minutes: Optional[float],
    interval_minutes: Optional[float],
    epsilon: float = 1e-6
) -> Optional[int]:
    """
    Convert horizon_minutes to horizon_bars with strict validation.
    
    CRITICAL: Returns None (not 0) if conversion is not exact (fail-closed).
    Rounded values would pollute registry metadata.
    
    This is the SINGLE SOURCE OF TRUTH for horizon conversion - use everywhere.
    
    Args:
        horizon_minutes: Target horizon in minutes (None if unknown)
        interval_minutes: Data bar interval in minutes (None if unknown)
        epsilon: Tolerance for "effectively integer" check
    
    Returns:
        Horizon in bars (int) if exactly convertible, None otherwise
    
    Raises:
        ValueError: If interval_minutes <= 0 (when provided)
    """
    # Handle None inputs (fail-closed)
    if horizon_minutes is None or interval_minutes is None:
        return None
    
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")
    
    if horizon_minutes <= 0:
        return None
    
    # Ratio-based check (handles floats correctly, not modulo)
    ratio = horizon_minutes / interval_minutes
    bars_rounded = round(ratio)
    
    # Check if ratio is "effectively integer" (not modulo, which fails for floats)
    if abs(ratio - bars_rounded) > epsilon:
        # Not exactly convertible - return None (fail-closed)
        return None
    
    return int(bars_rounded)


def is_effectively_integer(value: Optional[float], epsilon: float = 1e-6) -> Optional[int]:
    """
    Check if value is effectively integer, return int if so, None otherwise.
    
    Uses epsilon-based check (consistent with horizon_minutes_to_bars pattern).
    This is the SINGLE SOURCE OF TRUTH for "effectively integer" checks.
    
    Args:
        value: Value to check (None if unknown)
        epsilon: Tolerance for "effectively integer" check
    
    Returns:
        int if effectively integer, None otherwise
    """
    if value is None:
        return None
    rounded = round(value)
    if abs(value - rounded) <= epsilon:
        return int(rounded)
    return None
