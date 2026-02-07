# Plan 08: Position Sizing

## Overview

Volatility-scaled position sizing that converts alpha signals to portfolio weights while respecting risk limits and turnover constraints.

## Mathematical Foundation

### Volatility Scaling
```
z = clip(α / σ, -z_max, z_max)
weight = z × (max_weight / z_max)
```
- `α` = alpha signal from arbitration
- `σ` = volatility estimate
- `z_max` = maximum z-score (3.0)
- `max_weight` = maximum position weight (5%)

### No-Trade Band
```
if |target_weight - current_weight| < no_trade_band:
    keep current position
```
- Prevents excessive turnover from small signals

### Gross Exposure Target
```
gross_target = 0.5  # 50% of capital
```

## Files to Create

### 1. `LIVE_TRADING/sizing/__init__.py`

```python
from .position_sizer import PositionSizer, SizingResult
from .vol_scaling import VolatilityScaler
from .turnover import TurnoverManager

__all__ = ["PositionSizer", "SizingResult", "VolatilityScaler", "TurnoverManager"]
```

### 2. `LIVE_TRADING/sizing/vol_scaling.py`

```python
"""
Volatility Scaling
==================

Scales positions based on volatility estimates.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class VolatilityScaler:
    """
    Scales alpha to position weight using volatility.

    z = clip(α / σ, -z_max, z_max)
    weight = z × (max_weight / z_max)
    """

    def __init__(
        self,
        z_max: float | None = None,
        max_weight: float | None = None,
    ):
        self.z_max = z_max or get_cfg(
            "live_trading.sizing.z_max",
            default=DEFAULT_CONFIG["z_max"],
        )
        self.max_weight = max_weight or get_cfg(
            "live_trading.sizing.max_weight",
            default=DEFAULT_CONFIG["max_weight"],
        )
        logger.info(f"VolatilityScaler: z_max={self.z_max}, max_weight={self.max_weight}")

    def scale(self, alpha: float, volatility: float) -> float:
        """
        Scale alpha to position weight.

        Args:
            alpha: Alpha signal (decimal return)
            volatility: Volatility estimate (annualized std)

        Returns:
            Target weight (-max_weight to +max_weight)
        """
        if volatility <= 0:
            return 0.0

        z = alpha / volatility
        z_clipped = np.clip(z, -self.z_max, self.z_max)
        weight = z_clipped * (self.max_weight / self.z_max)

        return float(weight)

    def scale_batch(
        self,
        alphas: Dict[str, float],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """Scale multiple symbols."""
        return {
            symbol: self.scale(alpha, volatilities.get(symbol, 0.0))
            for symbol, alpha in alphas.items()
        }
```

### 3. `LIVE_TRADING/sizing/turnover.py`

```python
"""
Turnover Management
===================

Manages position turnover to reduce trading costs.
"""

from __future__ import annotations

import logging
from typing import Dict

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items
from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class TurnoverManager:
    """
    Manages turnover using no-trade bands.
    """

    def __init__(
        self,
        no_trade_band: float | None = None,
    ):
        self.no_trade_band = no_trade_band or get_cfg(
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
        """Calculate total turnover."""
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        turnover = 0.0

        for symbol in all_symbols:
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            turnover += abs(target - current)

        return turnover
```

### 4. `LIVE_TRADING/sizing/position_sizer.py`

```python
"""
Position Sizer
==============

Main position sizing engine combining all components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG
from LIVE_TRADING.gating.barrier_gate import GateResult
from LIVE_TRADING.arbitration.horizon_arbiter import ArbitrationResult
from .vol_scaling import VolatilityScaler
from .turnover import TurnoverManager

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing."""
    symbol: str
    target_weight: float
    current_weight: float
    trade_weight: float  # Actual weight change
    shares: int
    notional: float
    reason: str


class PositionSizer:
    """
    Main position sizing engine.

    Pipeline:
    1. Scale alpha by volatility
    2. Apply gate reduction
    3. Apply no-trade band
    4. Normalize to gross target
    5. Calculate shares
    """

    def __init__(
        self,
        gross_target: float | None = None,
        max_position_pct: float | None = None,
    ):
        self.vol_scaler = VolatilityScaler()
        self.turnover_manager = TurnoverManager()

        self.gross_target = gross_target or get_cfg(
            "live_trading.sizing.gross_target",
            default=DEFAULT_CONFIG["gross_target"],
        )
        self.max_position_pct = max_position_pct or get_cfg(
            "live_trading.risk.max_position_pct",
            default=DEFAULT_CONFIG["max_position_pct"],
        ) / 100

        logger.info(f"PositionSizer: gross_target={self.gross_target}")

    def calculate_target_weight(
        self,
        alpha: float,
        volatility: float,
        gate_result: Optional[GateResult] = None,
    ) -> float:
        """
        Calculate target weight for a symbol.

        Args:
            alpha: Alpha signal
            volatility: Volatility estimate
            gate_result: Optional gate result for reduction

        Returns:
            Target weight
        """
        # Vol-scaled weight
        weight = self.vol_scaler.scale(alpha, volatility)

        # Apply gate reduction
        if gate_result is not None:
            weight *= gate_result.gate_value

        # Cap at max position
        weight = max(-self.max_position_pct, min(self.max_position_pct, weight))

        return weight

    def size_portfolio(
        self,
        alphas: Dict[str, float],
        volatilities: Dict[str, float],
        current_weights: Dict[str, float],
        gates: Dict[str, GateResult] | None = None,
        portfolio_value: float = 100_000.0,
        prices: Dict[str, float] | None = None,
    ) -> Dict[str, SizingResult]:
        """
        Size all positions in portfolio.

        Args:
            alphas: Alpha signals per symbol
            volatilities: Volatilities per symbol
            current_weights: Current position weights
            gates: Gate results per symbol
            portfolio_value: Total portfolio value
            prices: Current prices per symbol

        Returns:
            Dict mapping symbol to SizingResult
        """
        gates = gates or {}
        prices = prices or {}

        # Calculate raw target weights
        raw_targets = {}
        for symbol, alpha in sorted_items(alphas):
            vol = volatilities.get(symbol, 0.01)
            gate = gates.get(symbol)
            raw_targets[symbol] = self.calculate_target_weight(alpha, vol, gate)

        # Apply no-trade band
        targets = self.turnover_manager.apply_no_trade_band(
            raw_targets, current_weights
        )

        # Normalize to gross target
        targets = self._normalize_to_gross(targets)

        # Calculate results
        results = {}
        for symbol, target in sorted_items(targets):
            current = current_weights.get(symbol, 0.0)
            trade = target - current
            price = prices.get(symbol, 100.0)

            notional = abs(trade) * portfolio_value
            shares = int(notional / price) if price > 0 else 0

            results[symbol] = SizingResult(
                symbol=symbol,
                target_weight=target,
                current_weight=current,
                trade_weight=trade,
                shares=shares,
                notional=notional,
                reason="sized",
            )

        return results

    def _normalize_to_gross(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to gross target."""
        gross = sum(abs(w) for w in weights.values())

        if gross <= 0 or gross <= self.gross_target:
            return weights

        scale = self.gross_target / gross
        return {s: w * scale for s, w in sorted_items(weights)}
```

## Tests

### `LIVE_TRADING/tests/test_sizing.py`

```python
"""Tests for position sizing."""

import pytest
from unittest.mock import Mock

from LIVE_TRADING.sizing.vol_scaling import VolatilityScaler
from LIVE_TRADING.sizing.turnover import TurnoverManager
from LIVE_TRADING.sizing.position_sizer import PositionSizer, SizingResult


class TestVolatilityScaler:
    def test_scale_positive_alpha(self):
        scaler = VolatilityScaler(z_max=3.0, max_weight=0.05)
        weight = scaler.scale(alpha=0.01, volatility=0.01)
        assert weight > 0
        assert weight <= 0.05

    def test_scale_clipping(self):
        scaler = VolatilityScaler(z_max=3.0, max_weight=0.05)
        # Very high alpha should clip to max
        weight = scaler.scale(alpha=0.1, volatility=0.01)
        assert weight == pytest.approx(0.05)

    def test_zero_volatility(self):
        scaler = VolatilityScaler()
        assert scaler.scale(alpha=0.01, volatility=0.0) == 0.0


class TestTurnoverManager:
    def test_no_trade_band_keeps_position(self):
        mgr = TurnoverManager(no_trade_band=0.01)
        target = {"AAPL": 0.05}
        current = {"AAPL": 0.055}  # 0.5% difference
        adjusted = mgr.apply_no_trade_band(target, current)
        assert adjusted["AAPL"] == 0.055  # Kept current

    def test_outside_band_updates(self):
        mgr = TurnoverManager(no_trade_band=0.01)
        target = {"AAPL": 0.05}
        current = {"AAPL": 0.08}  # 3% difference
        adjusted = mgr.apply_no_trade_band(target, current)
        assert adjusted["AAPL"] == 0.05  # Updated to target


class TestPositionSizer:
    def test_size_portfolio(self):
        sizer = PositionSizer(gross_target=0.5, max_position_pct=10.0)

        alphas = {"AAPL": 0.001, "MSFT": 0.0005}
        vols = {"AAPL": 0.02, "MSFT": 0.015}
        current = {}

        results = sizer.size_portfolio(
            alphas=alphas,
            volatilities=vols,
            current_weights=current,
            portfolio_value=100_000,
            prices={"AAPL": 150, "MSFT": 300},
        )

        assert "AAPL" in results
        assert isinstance(results["AAPL"], SizingResult)
```

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 10 |
| `vol_scaling.py` | 90 |
| `turnover.py` | 100 |
| `position_sizer.py` | 180 |
| `tests/test_sizing.py` | 80 |
| **Total** | ~460 |
