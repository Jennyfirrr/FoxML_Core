"""
Horizon Arbiter
===============

Selects optimal horizon based on cost-adjusted scores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    HORIZON_MINUTES,
    DECISION_TRADE,
    DECISION_HOLD,
)
from LIVE_TRADING.blending.horizon_blender import BlendedAlpha
from .cost_model import CostModel, TradingCosts

logger = logging.getLogger(__name__)


@dataclass
class ArbitrationResult:
    """Result of horizon arbitration."""

    decision: str  # "TRADE" or "HOLD"
    selected_horizon: Optional[str]
    net_score: float  # Score after costs
    alpha: float  # Alpha of selected horizon
    costs: Optional[TradingCosts]
    horizon_scores: Dict[str, float]  # All horizon scores
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision,
            "selected_horizon": self.selected_horizon,
            "net_score": self.net_score,
            "alpha": self.alpha,
            "costs": self.costs.to_dict() if self.costs else None,
            "horizon_scores": dict(self.horizon_scores),
            "reason": self.reason,
        }


class HorizonArbiter:
    """
    Selects optimal horizon based on cost-adjusted net scores.

    net_h = α_h - costs
    score_h = net_h / √(h/5)

    The horizon penalty √(h/5) favors shorter horizons when
    trading in a 5-minute loop, as longer horizons have more
    timing uncertainty.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        entry_threshold_bps: float | None = None,
        reserve_bps: float = 0.5,
    ):
        """
        Initialize horizon arbiter.

        Args:
            cost_model: CostModel instance (created if not provided)
            entry_threshold_bps: Minimum score to enter trade
            reserve_bps: Safety margin multiplier for spread
        """
        self.cost_model = cost_model or CostModel()
        self.reserve_bps = reserve_bps

        # Entry threshold = base + reserve × spread
        self._base_threshold = entry_threshold_bps if entry_threshold_bps is not None else get_cfg(
            "live_trading.arbitration.entry_threshold_bps",
            default=2.0,  # 2 bps base threshold
        )

        logger.info(f"HorizonArbiter: threshold={self._base_threshold}bps")

    def arbitrate(
        self,
        blended_alphas: Dict[str, BlendedAlpha],
        spread_bps: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = float("inf"),
    ) -> ArbitrationResult:
        """
        Select optimal horizon.

        Args:
            blended_alphas: Dict mapping horizon to BlendedAlpha
            spread_bps: Current bid-ask spread in bps
            volatility: Volatility estimate
            order_size: Planned order size in dollars
            adv: Average daily volume in dollars

        Returns:
            ArbitrationResult with selected horizon
        """
        if not blended_alphas:
            return ArbitrationResult(
                decision=DECISION_HOLD,
                selected_horizon=None,
                net_score=0.0,
                alpha=0.0,
                costs=None,
                horizon_scores={},
                reason="no_alphas",
            )

        # Calculate net scores for all horizons
        horizon_scores: Dict[str, float] = {}
        horizon_costs: Dict[str, TradingCosts] = {}

        for horizon, blended in sorted_items(blended_alphas):
            # Estimate costs for this horizon
            costs = self.cost_model.estimate_costs(
                horizon=horizon,
                spread_bps=spread_bps,
                volatility=volatility,
                order_size=order_size,
                adv=adv,
            )
            horizon_costs[horizon] = costs

            # Net score = alpha (in bps) - total_cost (in bps)
            alpha_bps = blended.alpha * 10000
            net = alpha_bps - costs.total_cost

            # Score with horizon penalty: penalize longer horizons
            h_minutes = HORIZON_MINUTES.get(horizon, 5)
            score = net / math.sqrt(h_minutes / 5)
            horizon_scores[horizon] = score

        # Select best horizon by score
        best_horizon = max(horizon_scores, key=lambda h: horizon_scores[h])
        best_score = horizon_scores[best_horizon]
        best_alpha = blended_alphas[best_horizon].alpha
        best_costs = horizon_costs[best_horizon]

        # Calculate dynamic entry threshold
        entry_threshold = self._calculate_threshold(spread_bps)

        # Trade/no-trade decision
        if best_score >= entry_threshold:
            decision = DECISION_TRADE
            reason = f"score {best_score:.2f}bps >= threshold {entry_threshold:.2f}bps"
        else:
            decision = DECISION_HOLD
            reason = f"score {best_score:.2f}bps < threshold {entry_threshold:.2f}bps"

        return ArbitrationResult(
            decision=decision,
            selected_horizon=best_horizon if decision == DECISION_TRADE else None,
            net_score=best_score,
            alpha=best_alpha,
            costs=best_costs,
            horizon_scores=horizon_scores,
            reason=reason,
        )

    def _calculate_threshold(self, spread_bps: float) -> float:
        """
        Calculate dynamic entry threshold.

        threshold = base_threshold + reserve × spread

        Args:
            spread_bps: Current spread

        Returns:
            Entry threshold in bps
        """
        return self._base_threshold + self.reserve_bps * spread_bps

    def get_horizon_breakdown(
        self,
        blended_alphas: Dict[str, BlendedAlpha],
        spread_bps: float,
        volatility: float,
        order_size: float = 0.0,
        adv: float = float("inf"),
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed breakdown for all horizons.

        Args:
            blended_alphas: Blended alphas
            spread_bps: Current spread
            volatility: Volatility
            order_size: Order size
            adv: Average daily volume

        Returns:
            Dict with detailed breakdown per horizon
        """
        breakdown = {}

        for horizon, blended in sorted_items(blended_alphas):
            costs = self.cost_model.estimate_costs(
                horizon=horizon,
                spread_bps=spread_bps,
                volatility=volatility,
                order_size=order_size,
                adv=adv,
            )

            alpha_bps = blended.alpha * 10000
            net = alpha_bps - costs.total_cost
            h_minutes = HORIZON_MINUTES.get(horizon, 5)
            score = net / math.sqrt(h_minutes / 5)

            breakdown[horizon] = {
                "alpha": blended.alpha,
                "alpha_bps": alpha_bps,
                "spread_cost": costs.spread_cost,
                "timing_cost": costs.timing_cost,
                "impact_cost": costs.impact_cost,
                "total_cost": costs.total_cost,
                "net_bps": net,
                "horizon_penalty": math.sqrt(h_minutes / 5),
                "score": score,
                "confidence": blended.confidence,
                "weights": blended.weights,
            }

        return breakdown

    def rank_horizons(
        self,
        blended_alphas: Dict[str, BlendedAlpha],
        spread_bps: float,
        volatility: float,
    ) -> List[tuple]:
        """
        Rank horizons by score.

        Args:
            blended_alphas: Blended alphas
            spread_bps: Current spread
            volatility: Volatility

        Returns:
            List of (horizon, score) tuples sorted by score descending
        """
        result = self.arbitrate(blended_alphas, spread_bps, volatility)
        return sorted(
            result.horizon_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
