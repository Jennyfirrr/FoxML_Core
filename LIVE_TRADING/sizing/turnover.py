"""
Turnover Management
===================

Manages position turnover to reduce trading costs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class TurnoverManager:
    """
    Manages turnover using no-trade bands.

    The no-trade band prevents excessive trading when position
    changes are small. If the delta between target and current
    weight is smaller than the band, we keep the current position.
    """

    def __init__(
        self,
        no_trade_band: float | None = None,
    ):
        """
        Initialize turnover manager.

        Args:
            no_trade_band: Minimum weight delta to trigger trade (e.g., 0.008 = 80 bps)
        """
        self.no_trade_band = no_trade_band if no_trade_band is not None else get_cfg(
            "live_trading.sizing.no_trade_band",
            default=DEFAULT_CONFIG["no_trade_band"],
        )
        logger.info(f"TurnoverManager: no_trade_band={self.no_trade_band}")

    def apply_no_trade_band(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply no-trade band to target weights.

        If the change is smaller than the no-trade band,
        keep the current position.

        Args:
            target_weights: Target position weights
            current_weights: Current position weights

        Returns:
            Adjusted target weights
        """
        adjusted = {}

        for symbol, target in sorted_items(target_weights):
            current = current_weights.get(symbol, 0.0)
            delta = abs(target - current)

            if delta < self.no_trade_band:
                adjusted[symbol] = current  # Keep current
            else:
                adjusted[symbol] = target

        return adjusted

    def calculate_turnover(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> float:
        """
        Calculate total turnover.

        Turnover = Σ |target_i - current_i|

        Args:
            target_weights: Target position weights
            current_weights: Current position weights

        Returns:
            Total turnover (sum of absolute weight changes)
        """
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        turnover = 0.0

        for symbol in all_symbols:
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            turnover += abs(target - current)

        return turnover

    def calculate_two_way_turnover(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> float:
        """
        Calculate two-way turnover (buys + sells).

        This is the same as regular turnover but explicitly
        named to indicate it counts both sides.

        Args:
            target_weights: Target position weights
            current_weights: Current position weights

        Returns:
            Two-way turnover
        """
        return self.calculate_turnover(target_weights, current_weights)

    def get_trades(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> List[Tuple[str, float, str]]:
        """
        Get list of trades to execute.

        Returns trades that exceed the no-trade band.

        Args:
            target_weights: Target weights
            current_weights: Current weights

        Returns:
            List of (symbol, delta, side) tuples
        """
        trades = []
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())

        for symbol in sorted(all_symbols):
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            delta = target - current

            if abs(delta) >= self.no_trade_band:
                side = "BUY" if delta > 0 else "SELL"
                trades.append((symbol, delta, side))

        return trades

    def estimate_cost(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        cost_per_trade_bps: float = 10.0,
    ) -> float:
        """
        Estimate turnover cost in basis points.

        Args:
            target_weights: Target weights
            current_weights: Current weights
            cost_per_trade_bps: Cost per unit of turnover

        Returns:
            Estimated cost in basis points
        """
        turnover = self.calculate_turnover(target_weights, current_weights)
        return turnover * cost_per_trade_bps * 100  # Convert to bps

    def get_analysis(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Get detailed turnover analysis.

        Args:
            target_weights: Target weights
            current_weights: Current weights

        Returns:
            Analysis dict
        """
        adjusted = self.apply_no_trade_band(target_weights, current_weights)
        trades = self.get_trades(target_weights, current_weights)

        raw_turnover = self.calculate_turnover(target_weights, current_weights)
        actual_turnover = self.calculate_turnover(adjusted, current_weights)

        return {
            "raw_turnover": raw_turnover,
            "actual_turnover": actual_turnover,
            "turnover_saved": raw_turnover - actual_turnover,
            "no_trade_band": self.no_trade_band,
            "num_trades": len(trades),
            "trades": trades,
            "positions_unchanged": len(target_weights) - len(trades),
        }
