"""
Cost Model
==========

Estimates trading costs including spread, timing, and market impact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG, HORIZON_MINUTES

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Breakdown of trading costs."""

    spread_cost: float  # Spread in bps
    timing_cost: float  # Volatility timing cost
    impact_cost: float  # Market impact cost
    total_cost: float  # Sum of all costs
    horizon: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread_cost": self.spread_cost,
            "timing_cost": self.timing_cost,
            "impact_cost": self.impact_cost,
            "total_cost": self.total_cost,
            "horizon": self.horizon,
        }


class CostModel:
    """
    Estimates trading costs for cost-aware arbitration.

    cost = k₁×spread + k₂×σ×√(h/5) + k₃×impact(q)

    Where:
    - k₁ = spread penalty coefficient
    - k₂ = volatility timing coefficient
    - k₃ = market impact coefficient
    - spread = bid-ask spread in bps
    - σ = volatility estimate
    - h = horizon in minutes
    - impact(q) = market impact of order size q
    """

    def __init__(
        self,
        k1_spread: float | None = None,
        k2_volatility: float | None = None,
        k3_impact: float | None = None,
    ):
        """
        Initialize cost model.

        Args:
            k1_spread: Spread penalty coefficient
            k2_volatility: Volatility timing coefficient
            k3_impact: Market impact coefficient
        """
        self.k1 = k1_spread if k1_spread is not None else get_cfg(
            "live_trading.cost_model.k1",
            default=DEFAULT_CONFIG["k1_spread"],
        )
        self.k2 = k2_volatility if k2_volatility is not None else get_cfg(
            "live_trading.cost_model.k2",
            default=DEFAULT_CONFIG["k2_volatility"],
        )
        self.k3 = k3_impact if k3_impact is not None else get_cfg(
            "live_trading.cost_model.k3",
            default=DEFAULT_CONFIG["k3_impact"],
        )

        logger.info(f"CostModel: k1={self.k1}, k2={self.k2}, k3={self.k3}")

    def estimate_costs(
        self,
        horizon: str,
        spread_bps: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = float("inf"),
    ) -> TradingCosts:
        """
        Estimate total trading costs.

        Args:
            horizon: Horizon string (e.g., "5m")
            spread_bps: Current bid-ask spread in basis points
            volatility: Volatility estimate (e.g., daily vol as decimal)
            order_size: Order size in dollars
            adv: Average daily volume in dollars

        Returns:
            TradingCosts breakdown
        """
        h_minutes = HORIZON_MINUTES.get(horizon, 5)

        # Spread cost (constant, proportional to spread)
        spread_cost = self.k1 * spread_bps

        # Volatility timing cost (increases with sqrt of horizon)
        # This represents the uncertainty of entry/exit over the horizon
        timing_cost = self.k2 * volatility * 10000 * math.sqrt(h_minutes / 5)

        # Market impact cost
        impact_cost = self._calculate_impact(order_size, adv)

        total = spread_cost + timing_cost + impact_cost

        return TradingCosts(
            spread_cost=spread_cost,
            timing_cost=timing_cost,
            impact_cost=impact_cost,
            total_cost=total,
            horizon=horizon,
        )

    def _calculate_impact(
        self,
        order_size: float,
        adv: float,
    ) -> float:
        """
        Calculate market impact cost using square-root model.

        impact ∝ √(q / ADV)

        This is a standard market microstructure model where impact
        scales with the square root of participation rate.

        Args:
            order_size: Order size in dollars
            adv: Average daily volume in dollars

        Returns:
            Impact cost in bps
        """
        if order_size <= 0 or adv <= 0 or not math.isfinite(adv):
            return 0.0

        participation = order_size / adv

        # Square-root impact model
        # Calibrated to ~10 bps at 1% participation
        impact_bps = self.k3 * 10.0 * math.sqrt(participation / 0.01)

        return impact_bps

    def estimate_all_horizons(
        self,
        horizons: List[str],
        spread_bps: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = float("inf"),
    ) -> Dict[str, TradingCosts]:
        """
        Estimate costs for all horizons.

        Args:
            horizons: List of horizon strings
            spread_bps: Current spread
            volatility: Volatility estimate
            order_size: Order size
            adv: Average daily volume

        Returns:
            Dict mapping horizon to TradingCosts
        """
        return {
            h: self.estimate_costs(h, spread_bps, volatility, order_size, adv)
            for h in horizons
        }

    def estimate_breakeven_alpha(
        self,
        horizon: str,
        spread_bps: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = float("inf"),
    ) -> float:
        """
        Calculate the minimum alpha needed to break even.

        Args:
            horizon: Horizon string
            spread_bps: Current spread
            volatility: Volatility estimate
            order_size: Order size
            adv: Average daily volume

        Returns:
            Breakeven alpha in decimal (e.g., 0.001 = 10 bps)
        """
        costs = self.estimate_costs(horizon, spread_bps, volatility, order_size, adv)
        # Convert from bps to decimal return
        return costs.total_cost / 10000
