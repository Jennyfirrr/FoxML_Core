"""
Exposure Tracking
=================

Tracks gross and net portfolio exposure for risk management.
Enforces position size limits and gross exposure limits.

SST Compliance:
- Uses get_cfg() for configuration
- Uses sorted_items() for deterministic dict iteration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.common.exceptions import PositionLimitExceeded

logger = logging.getLogger(__name__)


@dataclass
class ExposureState:
    """
    Current exposure state.

    Captures gross, net, and directional exposure metrics.
    """

    gross_exposure: float  # Sum of |weights|
    net_exposure: float  # Sum of weights (can be negative)
    long_exposure: float  # Sum of positive weights
    short_exposure: float  # Sum of |negative weights|
    position_count: int  # Number of positions

    @property
    def is_net_long(self) -> bool:
        """True if net exposure is positive."""
        return self.net_exposure > 0

    @property
    def is_net_short(self) -> bool:
        """True if net exposure is negative."""
        return self.net_exposure < 0

    @property
    def leverage_ratio(self) -> float:
        """Gross exposure (1.0 = 100% invested, 2.0 = 2x leveraged)."""
        return self.gross_exposure


class ExposureTracker:
    """
    Tracks portfolio exposure and enforces limits.

    Monitors gross exposure, net exposure, and individual position sizes.
    Can validate trades before execution.

    Example:
        >>> tracker = ExposureTracker(max_position_pct=20.0)
        >>> weights = {"AAPL": 0.15, "MSFT": 0.10, "SPY": -0.05}
        >>> state = tracker.calculate_exposure(weights)
        >>> state.gross_exposure
        0.30
        >>> tracker.validate_position("GOOG", 0.25)
        False  # Exceeds 20% limit
    """

    def __init__(
        self,
        max_gross_exposure: float | None = None,
        max_position_pct: float | None = None,
    ) -> None:
        """
        Initialize exposure tracker.

        Args:
            max_gross_exposure: Maximum gross exposure as decimal (1.0 = 100%).
                               Default from config or 1.0.
            max_position_pct: Maximum single position as percentage.
                             Default from config or 20%.
        """
        self.max_gross = max_gross_exposure if max_gross_exposure is not None else get_cfg(
            "live_trading.risk.max_gross_exposure",
            default=DEFAULT_CONFIG["max_gross_exposure"],
        )

        max_pos_pct = max_position_pct if max_position_pct is not None else get_cfg(
            "live_trading.risk.max_position_pct",
            default=DEFAULT_CONFIG["max_position_pct"],
        )
        self.max_position = max_pos_pct / 100  # Convert percentage to decimal

        logger.info(
            f"ExposureTracker: max_gross={self.max_gross:.0%}, "
            f"max_position={self.max_position:.0%}"
        )

    def calculate_exposure(self, weights: Dict[str, float]) -> ExposureState:
        """
        Calculate current exposure from position weights.

        Args:
            weights: Dict mapping symbol to weight (as decimal, e.g., 0.05 for 5%)

        Returns:
            ExposureState with all exposure metrics
        """
        gross = 0.0
        net = 0.0
        long_exp = 0.0
        short_exp = 0.0

        # Use sorted_items for deterministic iteration
        for symbol, weight in sorted_items(weights):
            gross += abs(weight)
            net += weight
            if weight > 0:
                long_exp += weight
            else:
                short_exp += abs(weight)

        return ExposureState(
            gross_exposure=gross,
            net_exposure=net,
            long_exposure=long_exp,
            short_exposure=short_exp,
            position_count=len(weights),
        )

    def validate_position(
        self,
        symbol: str,
        weight: float,
        raise_on_violation: bool = False,
    ) -> bool:
        """
        Check if a position weight is within limits.

        Args:
            symbol: Position symbol
            weight: Position weight as decimal
            raise_on_violation: If True, raise exception on violation

        Returns:
            True if position is valid

        Raises:
            PositionLimitExceeded: If raise_on_violation=True and limit exceeded
        """
        if abs(weight) > self.max_position:
            logger.warning(
                f"Position {symbol} weight {weight:.2%} exceeds "
                f"max {self.max_position:.2%}"
            )
            if raise_on_violation:
                raise PositionLimitExceeded(symbol, weight, self.max_position)
            return False
        return True

    def validate_portfolio(
        self,
        weights: Dict[str, float],
        raise_on_violation: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all positions and gross exposure.

        Args:
            weights: Dict mapping symbol to weight
            raise_on_violation: If True, raise exception on first violation

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations: List[str] = []

        # Check gross exposure
        exposure = self.calculate_exposure(weights)
        if exposure.gross_exposure > self.max_gross:
            violation = (
                f"Gross exposure {exposure.gross_exposure:.2%} exceeds "
                f"max {self.max_gross:.2%}"
            )
            logger.warning(violation)
            violations.append(violation)

        # Check individual positions
        for symbol, weight in sorted_items(weights):
            if abs(weight) > self.max_position:
                violation = (
                    f"Position {symbol} weight {weight:.2%} exceeds "
                    f"max {self.max_position:.2%}"
                )
                logger.warning(violation)
                violations.append(violation)

                if raise_on_violation:
                    raise PositionLimitExceeded(symbol, weight, self.max_position)

        return len(violations) == 0, violations

    def calculate_available_capacity(
        self,
        current_weights: Dict[str, float],
    ) -> float:
        """
        Calculate remaining gross exposure capacity.

        Args:
            current_weights: Current position weights

        Returns:
            Available capacity as decimal (e.g., 0.2 for 20%)
        """
        exposure = self.calculate_exposure(current_weights)
        return max(0.0, self.max_gross - exposure.gross_exposure)

    def scale_weights_to_limit(
        self,
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Scale weights down to fit within gross exposure limit.

        Args:
            target_weights: Desired position weights

        Returns:
            Scaled weights that satisfy gross exposure limit
        """
        exposure = self.calculate_exposure(target_weights)

        if exposure.gross_exposure <= self.max_gross:
            return dict(target_weights)

        # Calculate scale factor
        scale = self.max_gross / exposure.gross_exposure

        # Apply scale factor
        scaled = {}
        for symbol, weight in sorted_items(target_weights):
            scaled[symbol] = weight * scale

        logger.info(
            f"Scaled weights by {scale:.2f}x to meet gross exposure limit"
        )

        return scaled

    def clip_position_to_limit(
        self,
        symbol: str,
        target_weight: float,
    ) -> float:
        """
        Clip a position weight to maximum allowed.

        Args:
            symbol: Position symbol
            target_weight: Desired weight

        Returns:
            Clipped weight within limits
        """
        if abs(target_weight) <= self.max_position:
            return target_weight

        clipped = self.max_position if target_weight > 0 else -self.max_position
        logger.info(
            f"Clipped {symbol} weight from {target_weight:.2%} to {clipped:.2%}"
        )
        return clipped
