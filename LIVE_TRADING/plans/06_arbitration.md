# Plan 06: Cost Arbitration

## Overview

Cost-aware horizon selection that accounts for trading costs when choosing the optimal horizon. Selects the horizon with the best net score after deducting spread, volatility timing, and market impact costs.

## Mathematical Foundation

### Net Score Calculation
```
net_h = α_h - k₁×spread_bps - k₂×σ×√(h/5) - k₃×impact(q)
```
- `α_h` = horizon alpha from blending
- `k₁` = spread penalty coefficient (1.0)
- `k₂` = volatility timing penalty (0.15)
- `k₃` = market impact penalty (1.0)
- `spread_bps` = bid-ask spread in basis points
- `σ` = volatility estimate
- `h` = horizon in minutes
- `impact(q)` = market impact function of order size

### Horizon Score with Penalty
```
score_h = net_h / √(h/5)
```
- Penalizes longer horizons when deciding entries in a 5m loop
- Favors shorter horizons for intraday trading

### Trade/No-Trade Gate
```
score_{h*} ≥ θ_enter
θ_enter = cost_bps + reserve_bps
```
- Only trade when expected alpha exceeds costs plus safety margin

## Files to Create

### 1. `LIVE_TRADING/arbitration/__init__.py`

```python
from .cost_model import CostModel, TradingCosts
from .horizon_arbiter import HorizonArbiter, ArbitrationResult

__all__ = ["CostModel", "TradingCosts", "HorizonArbiter", "ArbitrationResult"]
```

### 2. `LIVE_TRADING/arbitration/cost_model.py`
**Purpose:** Trading cost estimation

```python
"""
Cost Model
==========

Estimates trading costs including spread, timing, and market impact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG, HORIZON_MINUTES

logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """Breakdown of trading costs."""
    spread_cost: float  # Spread in bps
    timing_cost: float  # Volatility timing cost
    impact_cost: float  # Market impact cost
    total_cost: float   # Sum of all costs
    horizon: str


class CostModel:
    """
    Estimates trading costs for cost-aware arbitration.

    cost = k₁×spread + k₂×σ×√(h/5) + k₃×impact(q)
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
        self.k1 = k1_spread or get_cfg(
            "live_trading.cost_model.k1",
            default=DEFAULT_CONFIG["k1_spread"],
        )
        self.k2 = k2_volatility or get_cfg(
            "live_trading.cost_model.k2",
            default=DEFAULT_CONFIG["k2_volatility"],
        )
        self.k3 = k3_impact or get_cfg(
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
            volatility: Volatility estimate (annualized or per-period)
            order_size: Order size in dollars
            adv: Average daily volume in dollars

        Returns:
            TradingCosts breakdown
        """
        h_minutes = HORIZON_MINUTES.get(horizon, 5)

        # Spread cost (constant)
        spread_cost = self.k1 * spread_bps

        # Volatility timing cost (increases with sqrt of horizon)
        timing_cost = self.k2 * volatility * math.sqrt(h_minutes / 5)

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
        Calculate market impact cost.

        Uses square-root impact model: impact ∝ √(q / ADV)

        Args:
            order_size: Order size in dollars
            adv: Average daily volume in dollars

        Returns:
            Impact cost in bps
        """
        if order_size <= 0 or adv <= 0:
            return 0.0

        participation = order_size / adv

        # Square-root impact model
        # Typical coefficient: 10 bps at 1% participation
        impact_bps = self.k3 * 10.0 * math.sqrt(participation / 0.01)

        return impact_bps

    def estimate_all_horizons(
        self,
        horizons: list[str],
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
```

### 3. `LIVE_TRADING/arbitration/horizon_arbiter.py`
**Purpose:** Cost-aware horizon selection

```python
"""
Horizon Arbiter
===============

Selects optimal horizon based on cost-adjusted scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    HORIZONS,
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
    net_score: float
    alpha: float  # Alpha of selected horizon
    costs: Optional[TradingCosts]
    horizon_scores: Dict[str, float]  # All horizon scores
    reason: str


class HorizonArbiter:
    """
    Selects optimal horizon based on cost-adjusted net scores.

    net_h = α_h - costs
    score_h = net_h / √(h/5)
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
            reserve_bps: Safety margin in bps
        """
        self.cost_model = cost_model or CostModel()
        self.reserve_bps = reserve_bps

        # Entry threshold = cost + reserve
        self._base_threshold = entry_threshold_bps or get_cfg(
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
            spread_bps: Current bid-ask spread
            volatility: Volatility estimate
            order_size: Planned order size
            adv: Average daily volume

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
        horizon_nets: Dict[str, float] = {}
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

            # Net score = alpha - total_cost
            # Convert alpha to bps (assume alpha is in decimal returns)
            alpha_bps = blended.alpha * 10000
            net = alpha_bps - costs.total_cost
            horizon_nets[horizon] = net

            # Score with horizon penalty
            h_minutes = HORIZON_MINUTES.get(horizon, 5)
            score = net / math.sqrt(h_minutes / 5)
            horizon_scores[horizon] = score

        # Select best horizon
        best_horizon = max(horizon_scores, key=horizon_scores.get)
        best_score = horizon_scores[best_horizon]
        best_alpha = blended_alphas[best_horizon].alpha
        best_costs = horizon_costs[best_horizon]

        # Calculate dynamic entry threshold
        entry_threshold = self._calculate_threshold(spread_bps)

        # Trade/no-trade decision
        if best_score >= entry_threshold:
            decision = DECISION_TRADE
            reason = f"score {best_score:.2f} >= threshold {entry_threshold:.2f}"
        else:
            decision = DECISION_HOLD
            reason = f"score {best_score:.2f} < threshold {entry_threshold:.2f}"

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
    ) -> Dict[str, Dict]:
        """
        Get detailed breakdown for all horizons.

        Args:
            blended_alphas: Blended alphas
            spread_bps: Current spread
            volatility: Volatility

        Returns:
            Dict with detailed breakdown per horizon
        """
        breakdown = {}

        for horizon, blended in sorted_items(blended_alphas):
            costs = self.cost_model.estimate_costs(
                horizon=horizon,
                spread_bps=spread_bps,
                volatility=volatility,
            )

            alpha_bps = blended.alpha * 10000
            net = alpha_bps - costs.total_cost
            h_minutes = HORIZON_MINUTES.get(horizon, 5)
            score = net / math.sqrt(h_minutes / 5)

            breakdown[horizon] = {
                "alpha_bps": alpha_bps,
                "spread_cost": costs.spread_cost,
                "timing_cost": costs.timing_cost,
                "impact_cost": costs.impact_cost,
                "total_cost": costs.total_cost,
                "net": net,
                "score": score,
                "confidence": blended.confidence,
            }

        return breakdown
```

## Tests

### `LIVE_TRADING/tests/test_arbitration.py`

```python
"""Tests for cost arbitration."""

import pytest
from unittest.mock import Mock

from LIVE_TRADING.arbitration.cost_model import CostModel, TradingCosts
from LIVE_TRADING.arbitration.horizon_arbiter import HorizonArbiter, ArbitrationResult
from LIVE_TRADING.common.constants import DECISION_TRADE, DECISION_HOLD


class TestCostModel:
    def test_init_defaults(self):
        model = CostModel()
        assert model.k1 > 0
        assert model.k2 > 0
        assert model.k3 > 0

    def test_spread_cost(self):
        model = CostModel(k1_spread=1.0, k2_volatility=0.0, k3_impact=0.0)

        costs = model.estimate_costs(
            horizon="5m",
            spread_bps=5.0,
            volatility=0.0,
        )

        assert costs.spread_cost == 5.0
        assert costs.timing_cost == 0.0
        assert costs.total_cost == 5.0

    def test_timing_cost_increases_with_horizon(self):
        model = CostModel(k1_spread=0.0, k2_volatility=0.15, k3_impact=0.0)

        costs_5m = model.estimate_costs("5m", spread_bps=0, volatility=0.01)
        costs_30m = model.estimate_costs("30m", spread_bps=0, volatility=0.01)

        assert costs_30m.timing_cost > costs_5m.timing_cost

    def test_impact_cost(self):
        model = CostModel(k1_spread=0.0, k2_volatility=0.0, k3_impact=1.0)

        costs = model.estimate_costs(
            horizon="5m",
            spread_bps=0,
            volatility=0,
            order_size=100_000,
            adv=10_000_000,
        )

        assert costs.impact_cost > 0


class TestHorizonArbiter:
    @pytest.fixture
    def arbiter(self):
        return HorizonArbiter(entry_threshold_bps=2.0, reserve_bps=0.5)

    def test_select_best_horizon(self, arbiter):
        # Mock blended alphas - 10m has highest alpha
        alphas = {
            "5m": Mock(alpha=0.001, confidence=0.8),  # 10 bps
            "10m": Mock(alpha=0.002, confidence=0.8),  # 20 bps
            "15m": Mock(alpha=0.0015, confidence=0.8),  # 15 bps
        }

        result = arbiter.arbitrate(
            blended_alphas=alphas,
            spread_bps=2.0,
            volatility=0.01,
        )

        # Should select a horizon (depends on cost tradeoffs)
        assert result.selected_horizon is not None
        assert result.horizon_scores  # Should have scores for all

    def test_hold_when_alpha_too_low(self, arbiter):
        # Very low alphas
        alphas = {
            "5m": Mock(alpha=0.00001, confidence=0.8),  # 0.1 bps
        }

        result = arbiter.arbitrate(
            blended_alphas=alphas,
            spread_bps=10.0,  # Wide spread
            volatility=0.02,
        )

        assert result.decision == DECISION_HOLD

    def test_trade_when_alpha_sufficient(self, arbiter):
        # High alpha
        alphas = {
            "5m": Mock(alpha=0.005, confidence=0.8),  # 50 bps
        }

        result = arbiter.arbitrate(
            blended_alphas=alphas,
            spread_bps=2.0,
            volatility=0.01,
        )

        assert result.decision == DECISION_TRADE

    def test_empty_alphas_returns_hold(self, arbiter):
        result = arbiter.arbitrate(
            blended_alphas={},
            spread_bps=2.0,
            volatility=0.01,
        )

        assert result.decision == DECISION_HOLD
        assert result.reason == "no_alphas"
```

## SST Compliance Checklist

- [ ] Uses `get_cfg()` for configuration
- [ ] Uses `sorted_items()` for dict iteration
- [ ] Proper dataclass definitions
- [ ] No magic numbers - all from config or constants

## Dependencies

- `CONFIG.config_loader.get_cfg`
- `TRAINING.common.utils.determinism_ordering.sorted_items`
- `LIVE_TRADING.blending.horizon_blender.BlendedAlpha`
- `LIVE_TRADING.common.constants`

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `cost_model.py` | 180 |
| `horizon_arbiter.py` | 230 |
| `tests/test_arbitration.py` | 120 |
| **Total** | ~540 |
