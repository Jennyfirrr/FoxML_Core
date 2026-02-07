# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Purge and Embargo Specification

Single source of truth for temporal safety calculations.
All purge/embargo values are stored in MINUTES, converted to bars at boundaries.

This module replaces hardcoded purge_overlap values (e.g., purge_overlap=17) with
time-based specifications that scale correctly across different data intervals.

Usage:
    from TRAINING.ranking.utils.purge import (
        PurgeSpec,
        compute_purge_minutes,
        make_purge_spec,
    )

    # Create purge spec for 60m horizon with 5m buffer
    spec = make_purge_spec(target_horizon_minutes=60, buffer_minutes=5)

    # Convert to bars for CV split at runtime
    purge_bars = spec.purge_bars(interval_minutes=5)  # Returns 13

    # Get pandas Timedelta for PurgedTimeSeriesSplit
    purge_td = spec.to_timedelta()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from TRAINING.common.interval import minutes_to_bars

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class PurgeSpec:
    """
    Temporal safety specification for CV splits.

    Stored in minutes (time semantics), converted to bars only at CV split creation.

    Attributes:
        purge_minutes: Time to purge after test fold (prevents train seeing test's past)
        embargo_minutes: Time to embargo before test fold (prevents test seeing train's future)
        buffer_minutes: Additional safety buffer
        rounding_policy: How to round when converting to bars ("ceil" recommended)

    Example:
        For a 60m target horizon with 5m buffer at 5m intervals:
        - purge_minutes = 65 (60 + 5)
        - purge_bars(5) = ceil(65/5) = 13 bars

        Same spec at 1m intervals:
        - purge_bars(1) = ceil(65/1) = 65 bars
    """

    purge_minutes: float
    embargo_minutes: float = 0.0
    buffer_minutes: float = 0.0
    rounding_policy: str = "ceil"

    def __post_init__(self) -> None:
        if self.purge_minutes < 0:
            raise ValueError(f"purge_minutes must be >= 0, got {self.purge_minutes}")
        if self.embargo_minutes < 0:
            raise ValueError(f"embargo_minutes must be >= 0, got {self.embargo_minutes}")
        if self.buffer_minutes < 0:
            raise ValueError(f"buffer_minutes must be >= 0, got {self.buffer_minutes}")
        if self.rounding_policy not in ("ceil", "floor"):
            raise ValueError(
                f"rounding_policy must be 'ceil' or 'floor', got {self.rounding_policy}"
            )

    def purge_bars(self, interval_minutes: int) -> int:
        """
        Convert purge to bars for CV split.

        Args:
            interval_minutes: Data bar interval

        Returns:
            Number of bars to purge

        Example:
            >>> spec = PurgeSpec(purge_minutes=65)
            >>> spec.purge_bars(5)  # ceil(65/5) = 13
            13
            >>> spec.purge_bars(1)  # ceil(65/1) = 65
            65
        """
        return minutes_to_bars(
            self.purge_minutes, interval_minutes, self.rounding_policy
        )

    def embargo_bars(self, interval_minutes: int) -> int:
        """
        Convert embargo to bars for CV split.

        Args:
            interval_minutes: Data bar interval

        Returns:
            Number of bars to embargo
        """
        return minutes_to_bars(
            self.embargo_minutes, interval_minutes, self.rounding_policy
        )

    def total_exclusion_minutes(self) -> float:
        """Total time excluded around test fold."""
        return self.purge_minutes + self.embargo_minutes

    def total_exclusion_bars(self, interval_minutes: int) -> int:
        """Total bars excluded around test fold."""
        return self.purge_bars(interval_minutes) + self.embargo_bars(interval_minutes)

    def to_dict(self) -> dict:
        """Serialize for artifacts."""
        return {
            "purge_minutes": self.purge_minutes,
            "embargo_minutes": self.embargo_minutes,
            "buffer_minutes": self.buffer_minutes,
            "rounding_policy": self.rounding_policy,
        }

    @staticmethod
    def from_dict(data: dict) -> "PurgeSpec":
        """Deserialize from artifacts."""
        return PurgeSpec(
            purge_minutes=data["purge_minutes"],
            embargo_minutes=data.get("embargo_minutes", 0.0),
            buffer_minutes=data.get("buffer_minutes", 0.0),
            rounding_policy=data.get("rounding_policy", "ceil"),
        )

    def to_timedelta(self) -> "pd.Timedelta":
        """
        Convert purge to pandas Timedelta for PurgedTimeSeriesSplit.

        This integrates with the existing PurgedTimeSeriesSplit.purge_overlap_time
        parameter for time-based CV splitting.

        Returns:
            pandas Timedelta representing purge duration
        """
        import pandas as pd

        return pd.Timedelta(minutes=self.purge_minutes)


def compute_purge_minutes(
    *,
    target_horizon_minutes: float,
    buffer_minutes: float = 5.0,
    max_feature_lookback_minutes: Optional[float] = None,
    include_feature_lookback: bool = False,
) -> float:
    """
    Compute purge window in minutes.

    Base formula: purge = horizon + buffer

    If include_feature_lookback=True, also consider max feature lookback
    (features looking back N minutes could see test data).

    Args:
        target_horizon_minutes: Target prediction horizon
        buffer_minutes: Additional safety buffer (default: 5 minutes)
        max_feature_lookback_minutes: Maximum feature lookback (if known)
        include_feature_lookback: Whether to include feature lookback in purge

    Returns:
        Purge window in minutes

    Raises:
        ValueError: If inputs are invalid

    Examples:
        >>> compute_purge_minutes(target_horizon_minutes=60)
        65.0
        >>> compute_purge_minutes(target_horizon_minutes=60, buffer_minutes=10)
        70.0
        >>> compute_purge_minutes(
        ...     target_horizon_minutes=60,
        ...     max_feature_lookback_minutes=100,
        ...     include_feature_lookback=True
        ... )
        105.0
    """
    if target_horizon_minutes <= 0:
        raise ValueError(
            f"target_horizon_minutes must be > 0, got {target_horizon_minutes}"
        )
    if buffer_minutes < 0:
        raise ValueError(f"buffer_minutes must be >= 0, got {buffer_minutes}")

    base_purge = target_horizon_minutes + buffer_minutes

    if include_feature_lookback:
        if max_feature_lookback_minutes is None:
            raise ValueError(
                "include_feature_lookback=True requires max_feature_lookback_minutes"
            )
        if max_feature_lookback_minutes < 0:
            raise ValueError(
                f"max_feature_lookback_minutes must be >= 0, got {max_feature_lookback_minutes}"
            )

        # Purge must cover both horizon and max feature lookback
        feature_purge = max_feature_lookback_minutes + buffer_minutes
        return max(base_purge, feature_purge)

    return base_purge


def make_purge_spec(
    *,
    target_horizon_minutes: float,
    buffer_minutes: float = 5.0,
    embargo_minutes: float = 0.0,
    max_feature_lookback_minutes: Optional[float] = None,
    include_feature_lookback: bool = False,
    rounding_policy: str = "ceil",
) -> PurgeSpec:
    """
    Create a complete PurgeSpec.

    Convenience function that computes purge_minutes and wraps in PurgeSpec.

    Args:
        target_horizon_minutes: Target prediction horizon
        buffer_minutes: Additional safety buffer (default: 5 minutes)
        embargo_minutes: Time to embargo before test fold (default: 0)
        max_feature_lookback_minutes: Maximum feature lookback (if known)
        include_feature_lookback: Whether to include feature lookback in purge
        rounding_policy: How to round when converting to bars ("ceil" recommended)

    Returns:
        PurgeSpec configured for the given parameters

    Examples:
        >>> spec = make_purge_spec(target_horizon_minutes=60)
        >>> spec.purge_minutes
        65.0
        >>> spec.purge_bars(5)
        13

        >>> spec = make_purge_spec(
        ...     target_horizon_minutes=60,
        ...     max_feature_lookback_minutes=100,
        ...     include_feature_lookback=True
        ... )
        >>> spec.purge_minutes
        105.0
    """
    purge_minutes = compute_purge_minutes(
        target_horizon_minutes=target_horizon_minutes,
        buffer_minutes=buffer_minutes,
        max_feature_lookback_minutes=max_feature_lookback_minutes,
        include_feature_lookback=include_feature_lookback,
    )

    return PurgeSpec(
        purge_minutes=purge_minutes,
        embargo_minutes=embargo_minutes,
        buffer_minutes=buffer_minutes,
        rounding_policy=rounding_policy,
    )


# =============================================================================
# Legacy Compatibility
# =============================================================================


def legacy_purge_overlap_to_spec(
    purge_overlap: int,
    interval_minutes: int = 5,
) -> PurgeSpec:
    """
    Convert legacy purge_overlap (bars) to PurgeSpec (minutes).

    This is a migration helper for converting existing hardcoded values
    like purge_overlap=17 to the new time-based system.

    Args:
        purge_overlap: Legacy purge overlap in bars
        interval_minutes: Assumed interval (default: 5m, the historical assumption)

    Returns:
        PurgeSpec equivalent

    Example:
        >>> spec = legacy_purge_overlap_to_spec(17, 5)
        >>> spec.purge_minutes
        85.0
        >>> spec.purge_bars(5)  # Should round-trip
        17
    """
    purge_minutes = float(purge_overlap * interval_minutes)
    return PurgeSpec(purge_minutes=purge_minutes, rounding_policy="ceil")


# =============================================================================
# Convenience Functions for CV Splitting
# =============================================================================


# Default purge configuration (used when target horizon is unknown)
DEFAULT_PURGE_HORIZON_MINUTES = 60  # Assume 60m horizon if unknown
DEFAULT_PURGE_BUFFER_MINUTES = 25   # 5 bar buffer at 5m = 25 minutes
DEFAULT_INTERVAL_MINUTES = 5        # Historical default


def get_purge_overlap_bars(
    target_horizon_minutes: Optional[float] = None,
    interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
    buffer_minutes: float = DEFAULT_PURGE_BUFFER_MINUTES,
) -> int:
    """
    Get purge_overlap in bars for CV splitting.

    This is the SST function to replace hardcoded `purge_overlap = 17`.
    It computes purge correctly for any interval and target horizon.

    Args:
        target_horizon_minutes: Target prediction horizon in minutes.
            If None, uses DEFAULT_PURGE_HORIZON_MINUTES (60m).
        interval_minutes: Data bar interval in minutes (default: 5).
        buffer_minutes: Additional safety buffer in minutes (default: 25m = 5 bars at 5m).

    Returns:
        purge_overlap in bars (suitable for PurgedTimeSeriesSplit)

    Examples:
        >>> get_purge_overlap_bars()  # Default: 60m + 25m buffer at 5m
        17
        >>> get_purge_overlap_bars(target_horizon_minutes=30)  # 30m + 25m at 5m
        11
        >>> get_purge_overlap_bars(target_horizon_minutes=60, interval_minutes=1)  # 60m + 25m at 1m
        85
        >>> get_purge_overlap_bars(target_horizon_minutes=60, interval_minutes=15)  # 60m + 25m at 15m
        6

    Migration note:
        Replace `purge_overlap = 17` with:
        ```python
        from TRAINING.ranking.utils.purge import get_purge_overlap_bars
        purge_overlap = get_purge_overlap_bars(target_horizon_minutes, interval_minutes)
        ```
    """
    # Use default horizon if not provided
    effective_horizon = target_horizon_minutes if target_horizon_minutes is not None else DEFAULT_PURGE_HORIZON_MINUTES

    # Create PurgeSpec and get bars
    spec = make_purge_spec(
        target_horizon_minutes=effective_horizon,
        buffer_minutes=buffer_minutes,
    )

    return spec.purge_bars(interval_minutes)


def get_purge_spec_for_target(
    target_horizon_minutes: Optional[float] = None,
    buffer_minutes: float = DEFAULT_PURGE_BUFFER_MINUTES,
) -> PurgeSpec:
    """
    Get a PurgeSpec for a target.

    Args:
        target_horizon_minutes: Target prediction horizon in minutes.
            If None, uses DEFAULT_PURGE_HORIZON_MINUTES (60m).
        buffer_minutes: Additional safety buffer in minutes.

    Returns:
        PurgeSpec configured for the target

    Example:
        >>> spec = get_purge_spec_for_target(60)
        >>> spec.purge_bars(5)
        17
        >>> spec.purge_bars(1)
        85
    """
    effective_horizon = target_horizon_minutes if target_horizon_minutes is not None else DEFAULT_PURGE_HORIZON_MINUTES

    return make_purge_spec(
        target_horizon_minutes=effective_horizon,
        buffer_minutes=buffer_minutes,
    )
