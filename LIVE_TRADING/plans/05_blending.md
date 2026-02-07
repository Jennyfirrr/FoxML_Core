# Plan 05: Blending

## Overview

Per-horizon model blending using Ridge risk-parity weights with temperature compression. Combines predictions from multiple model families into a single horizon alpha.

## Mathematical Foundation

### Ridge Risk-Parity Weights
```
w_h ∝ (Σ_h + λI)^{-1} μ_h
w_h ← clip(w_h, 0, ∞)
∑w_h = 1
```
- `Σ_h` = correlation matrix of standardized scores across families
- `λ` = ridge regularization (default 0.15)
- `μ_h` = target vector of net IC after costs
- Clipping ensures non-negative weights
- Weights sum to 1

### Net IC After Costs
```
μ_{m,h} = IC_{m,h} - λ_c × cost_share_{m,h}
```
- `IC_{m,h}` = Information Coefficient for model m, horizon h
- `λ_c` = cost penalty parameter (0.5)
- `cost_share` = expected_cost / expected_|alpha|

### Temperature Compression
```
w_h^{(T)} ∝ w_h^{1/T}
```
- `T_{5m} = 0.75` - more conservative for shortest horizon
- `T_{10m} = 0.85`
- `T_{15m+} = 1.0` - no compression

### Horizon Alpha
```
α_h = ∑_m w_{m,h}^{(T)} × s̃_{m,h}
```
- Final blended prediction per horizon

## Files to Create

### 1. `LIVE_TRADING/blending/__init__.py`

```python
from .ridge_weights import RidgeWeightCalculator, calculate_ridge_weights
from .temperature import TemperatureCompressor
from .horizon_blender import HorizonBlender, BlendedAlpha

__all__ = [
    "RidgeWeightCalculator",
    "calculate_ridge_weights",
    "TemperatureCompressor",
    "HorizonBlender",
    "BlendedAlpha",
]
```

### 2. `LIVE_TRADING/blending/ridge_weights.py`
**Purpose:** Ridge risk-parity weight calculation

```python
"""
Ridge Risk-Parity Weights
=========================

Calculates optimal blending weights using ridge regression on
the correlation structure of model predictions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import inv

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG, MIN_IC_THRESHOLD

logger = logging.getLogger(__name__)


class RidgeWeightCalculator:
    """
    Calculates Ridge risk-parity weights for model ensemble.

    w ∝ (Σ + λI)^{-1} μ
    """

    def __init__(
        self,
        ridge_lambda: float | None = None,
        cost_penalty: float = 0.5,
        min_weight: float = 0.0,
    ):
        """
        Initialize weight calculator.

        Args:
            ridge_lambda: Ridge regularization parameter
            cost_penalty: Cost penalty multiplier (λ_c)
            min_weight: Minimum weight threshold (below = 0)
        """
        self.ridge_lambda = ridge_lambda or get_cfg(
            "live_trading.blending.ridge_lambda",
            default=DEFAULT_CONFIG["ridge_lambda"],
        )
        self.cost_penalty = cost_penalty
        self.min_weight = min_weight

        logger.info(f"RidgeWeightCalculator: λ={self.ridge_lambda}")

    def calculate_weights(
        self,
        ic_values: Dict[str, float],
        correlation_matrix: np.ndarray,
        cost_shares: Dict[str, float] | None = None,
        model_order: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Calculate Ridge risk-parity weights.

        Args:
            ic_values: Dict mapping model name to IC value
            correlation_matrix: (M x M) correlation matrix of predictions
            cost_shares: Dict mapping model name to cost share
            model_order: Order of models (matches correlation matrix)

        Returns:
            Dict mapping model name to weight (sums to 1)
        """
        if not ic_values:
            return {}

        # Ensure consistent ordering
        if model_order is None:
            model_order = sorted(ic_values.keys())

        n_models = len(model_order)

        # Build target vector μ (net IC after costs)
        mu = np.zeros(n_models)
        for i, model in enumerate(model_order):
            ic = ic_values.get(model, 0.0)
            cost = (cost_shares or {}).get(model, 0.0)
            mu[i] = max(ic - self.cost_penalty * cost, MIN_IC_THRESHOLD)

        # Ensure correlation matrix is the right size
        if correlation_matrix.shape != (n_models, n_models):
            logger.warning(
                f"Correlation matrix shape mismatch: {correlation_matrix.shape} vs {n_models}"
            )
            # Fall back to identity matrix
            correlation_matrix = np.eye(n_models)

        # Ridge regression: w ∝ (Σ + λI)^{-1} μ
        regularized = correlation_matrix + self.ridge_lambda * np.eye(n_models)

        try:
            inv_matrix = inv(regularized)
            raw_weights = inv_matrix @ mu
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, using equal weights")
            raw_weights = np.ones(n_models) / n_models

        # Clip to non-negative
        raw_weights = np.clip(raw_weights, 0, None)

        # Apply minimum threshold
        raw_weights[raw_weights < self.min_weight] = 0.0

        # Normalize to sum to 1
        total = raw_weights.sum()
        if total > 0:
            normalized = raw_weights / total
        else:
            # Fall back to equal weights
            normalized = np.ones(n_models) / n_models

        # Build result dict
        return {model: float(normalized[i]) for i, model in enumerate(model_order)}

    def calculate_from_predictions(
        self,
        prediction_history: Dict[str, List[float]],
        ic_values: Dict[str, float],
        cost_shares: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Calculate weights from prediction history.

        Args:
            prediction_history: Dict mapping model to list of predictions
            ic_values: Current IC values
            cost_shares: Cost shares per model

        Returns:
            Weight dict
        """
        if not prediction_history:
            return {}

        model_order = sorted(prediction_history.keys())
        n_models = len(model_order)

        # Build prediction matrix
        min_len = min(len(v) for v in prediction_history.values())
        if min_len < 3:
            # Not enough history, use equal weights
            return {m: 1.0 / n_models for m in model_order}

        pred_matrix = np.zeros((min_len, n_models))
        for i, model in enumerate(model_order):
            pred_matrix[:, i] = prediction_history[model][-min_len:]

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(pred_matrix.T)

        return self.calculate_weights(
            ic_values=ic_values,
            correlation_matrix=correlation_matrix,
            cost_shares=cost_shares,
            model_order=model_order,
        )


def calculate_ridge_weights(
    ic_values: Dict[str, float],
    correlation_matrix: np.ndarray,
    ridge_lambda: float = 0.15,
) -> Dict[str, float]:
    """
    Convenience function for ridge weight calculation.

    Args:
        ic_values: Model IC values
        correlation_matrix: Correlation matrix
        ridge_lambda: Regularization parameter

    Returns:
        Weight dict
    """
    calculator = RidgeWeightCalculator(ridge_lambda=ridge_lambda)
    return calculator.calculate_weights(ic_values, correlation_matrix)
```

### 3. `LIVE_TRADING/blending/temperature.py`
**Purpose:** Temperature compression for shorter horizons

```python
"""
Temperature Compression
=======================

Applies temperature compression to model weights for shorter horizons.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class TemperatureCompressor:
    """
    Applies temperature compression to model weights.

    w^{(T)} ∝ w^{1/T}

    Lower temperature = more conservative (flatter) weights.
    """

    def __init__(self, temperatures: Dict[str, float] | None = None):
        """
        Initialize temperature compressor.

        Args:
            temperatures: Dict mapping horizon to temperature value
        """
        self.temperatures = temperatures or get_cfg(
            "live_trading.blending.temperature",
            default=DEFAULT_CONFIG["temperature"],
        )
        logger.info(f"TemperatureCompressor: {self.temperatures}")

    def get_temperature(self, horizon: str) -> float:
        """Get temperature for a horizon."""
        return self.temperatures.get(horizon, 1.0)

    def compress(
        self,
        weights: Dict[str, float],
        horizon: str,
    ) -> Dict[str, float]:
        """
        Apply temperature compression to weights.

        Args:
            weights: Dict mapping model to weight
            horizon: Horizon for temperature lookup

        Returns:
            Temperature-compressed weights (normalized to sum to 1)
        """
        if not weights:
            return {}

        temperature = self.get_temperature(horizon)

        if temperature >= 1.0:
            # No compression needed
            return dict(weights)

        # Apply compression: w^{1/T}
        compressed = {}
        for model, weight in sorted_items(weights):
            if weight > 0:
                compressed[model] = weight ** (1.0 / temperature)
            else:
                compressed[model] = 0.0

        # Normalize
        total = sum(compressed.values())
        if total > 0:
            return {m: w / total for m, w in sorted_items(compressed)}
        else:
            return dict(weights)

    def analyze_compression(
        self,
        weights: Dict[str, float],
        horizon: str,
    ) -> Dict[str, any]:
        """
        Analyze the effect of compression.

        Returns:
            Analysis dict with before/after entropy, etc.
        """
        original = weights
        compressed = self.compress(weights, horizon)

        # Calculate entropy
        def entropy(w_dict):
            vals = np.array(list(w_dict.values()))
            vals = vals[vals > 0]
            if len(vals) == 0:
                return 0.0
            p = vals / vals.sum()
            return float(-np.sum(p * np.log(p + 1e-10)))

        return {
            "horizon": horizon,
            "temperature": self.get_temperature(horizon),
            "entropy_before": entropy(original),
            "entropy_after": entropy(compressed),
            "max_weight_before": max(original.values()) if original else 0,
            "max_weight_after": max(compressed.values()) if compressed else 0,
        }
```

### 4. `LIVE_TRADING/blending/horizon_blender.py`
**Purpose:** Per-horizon blending orchestrator

```python
"""
Horizon Blender
===============

Combines model predictions within a horizon using Ridge weights
and temperature compression.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import HORIZONS
from LIVE_TRADING.prediction.predictor import HorizonPredictions
from .ridge_weights import RidgeWeightCalculator
from .temperature import TemperatureCompressor

logger = logging.getLogger(__name__)


@dataclass
class BlendedAlpha:
    """Result of horizon blending."""
    horizon: str
    alpha: float  # Blended prediction
    weights: Dict[str, float]  # Model weights used
    temperature: float  # Temperature applied
    confidence: float  # Aggregate confidence


class HorizonBlender:
    """
    Blends model predictions within a single horizon.

    Pipeline:
    1. Extract calibrated predictions
    2. Calculate Ridge risk-parity weights
    3. Apply temperature compression
    4. Compute weighted sum (alpha)
    """

    def __init__(
        self,
        ridge_lambda: float | None = None,
        prediction_history_size: int = 50,
    ):
        """
        Initialize horizon blender.

        Args:
            ridge_lambda: Ridge regularization parameter
            prediction_history_size: History size for correlation estimation
        """
        self.weight_calculator = RidgeWeightCalculator(ridge_lambda=ridge_lambda)
        self.temperature_compressor = TemperatureCompressor()
        self.history_size = prediction_history_size

        # Prediction history per (horizon, model) for correlation
        self._prediction_history: Dict[str, Dict[str, deque]] = {}

        # Weight history for analysis
        self._weight_history: Dict[str, List[Dict[str, float]]] = {}

        logger.info("HorizonBlender initialized")

    def blend(
        self,
        horizon_predictions: HorizonPredictions,
        ic_values: Dict[str, float] | None = None,
        cost_shares: Dict[str, float] | None = None,
    ) -> BlendedAlpha:
        """
        Blend predictions for a horizon.

        Args:
            horizon_predictions: Predictions from all models for this horizon
            ic_values: IC values per model (optional, uses confidence if not provided)
            cost_shares: Cost shares per model (optional)

        Returns:
            BlendedAlpha with weighted prediction
        """
        horizon = horizon_predictions.horizon
        predictions = horizon_predictions.predictions

        if not predictions:
            return BlendedAlpha(
                horizon=horizon,
                alpha=0.0,
                weights={},
                temperature=1.0,
                confidence=0.0,
            )

        # Extract calibrated predictions and IC values
        calibrated = {}
        ics = ic_values or {}
        confidences = {}

        for family, pred in sorted_items(predictions):
            calibrated[family] = pred.calibrated
            confidences[family] = pred.confidence.overall

            # Use IC from confidence if not provided
            if family not in ics:
                ics[family] = pred.confidence.ic

        # Update prediction history
        self._update_history(horizon, calibrated)

        # Calculate weights
        weights = self._calculate_weights(horizon, ics, cost_shares)

        # Apply temperature compression
        temp_weights = self.temperature_compressor.compress(weights, horizon)

        # Calculate blended alpha
        alpha = 0.0
        for family, weight in sorted_items(temp_weights):
            if family in calibrated:
                alpha += weight * calibrated[family]

        # Aggregate confidence (weighted average)
        total_conf = 0.0
        for family, weight in sorted_items(temp_weights):
            if family in confidences:
                total_conf += weight * confidences[family]

        # Store weight history
        if horizon not in self._weight_history:
            self._weight_history[horizon] = []
        self._weight_history[horizon].append(temp_weights)

        return BlendedAlpha(
            horizon=horizon,
            alpha=float(alpha),
            weights=temp_weights,
            temperature=self.temperature_compressor.get_temperature(horizon),
            confidence=float(total_conf),
        )

    def _update_history(
        self,
        horizon: str,
        predictions: Dict[str, float],
    ) -> None:
        """Update prediction history for correlation estimation."""
        if horizon not in self._prediction_history:
            self._prediction_history[horizon] = {}

        for family, pred in sorted_items(predictions):
            if family not in self._prediction_history[horizon]:
                self._prediction_history[horizon][family] = deque(
                    maxlen=self.history_size
                )
            self._prediction_history[horizon][family].append(pred)

    def _calculate_weights(
        self,
        horizon: str,
        ic_values: Dict[str, float],
        cost_shares: Dict[str, float] | None,
    ) -> Dict[str, float]:
        """Calculate Ridge weights from history."""
        history = self._prediction_history.get(horizon, {})

        if not history or all(len(v) < 5 for v in history.values()):
            # Not enough history, use equal weights
            return {m: 1.0 / len(ic_values) for m in ic_values}

        # Build prediction history dict
        pred_history = {
            m: list(v) for m, v in sorted_items(history) if len(v) >= 5
        }

        return self.weight_calculator.calculate_from_predictions(
            prediction_history=pred_history,
            ic_values=ic_values,
            cost_shares=cost_shares,
        )

    def blend_all_horizons(
        self,
        all_predictions: Dict[str, HorizonPredictions],
        ic_values: Dict[str, Dict[str, float]] | None = None,
    ) -> Dict[str, BlendedAlpha]:
        """
        Blend predictions for all horizons.

        Args:
            all_predictions: Dict mapping horizon to HorizonPredictions
            ic_values: Dict mapping horizon to model IC values

        Returns:
            Dict mapping horizon to BlendedAlpha
        """
        results = {}
        for horizon in sorted(all_predictions.keys()):
            horizon_ics = (ic_values or {}).get(horizon)
            results[horizon] = self.blend(
                all_predictions[horizon],
                ic_values=horizon_ics,
            )
        return results

    def get_weight_history(self, horizon: str) -> List[Dict[str, float]]:
        """Get weight history for analysis."""
        return self._weight_history.get(horizon, [])
```

## Tests

### `LIVE_TRADING/tests/test_blending.py`

```python
"""Tests for blending components."""

import pytest
import numpy as np

from LIVE_TRADING.blending.ridge_weights import (
    RidgeWeightCalculator,
    calculate_ridge_weights,
)
from LIVE_TRADING.blending.temperature import TemperatureCompressor
from LIVE_TRADING.blending.horizon_blender import HorizonBlender, BlendedAlpha


class TestRidgeWeightCalculator:
    def test_equal_ic_equal_weights(self):
        calc = RidgeWeightCalculator(ridge_lambda=0.15)

        ic_values = {"LightGBM": 0.1, "XGBoost": 0.1}
        corr = np.eye(2)  # Uncorrelated

        weights = calc.calculate_weights(ic_values, corr)

        assert len(weights) == 2
        assert abs(weights["LightGBM"] - weights["XGBoost"]) < 0.01
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_higher_ic_higher_weight(self):
        calc = RidgeWeightCalculator(ridge_lambda=0.15)

        ic_values = {"LightGBM": 0.2, "XGBoost": 0.1}
        corr = np.eye(2)

        weights = calc.calculate_weights(ic_values, corr)

        assert weights["LightGBM"] > weights["XGBoost"]

    def test_correlated_models_diversified(self):
        calc = RidgeWeightCalculator(ridge_lambda=0.15)

        ic_values = {"A": 0.1, "B": 0.1, "C": 0.1}
        # A and B highly correlated, C uncorrelated
        corr = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])

        weights = calc.calculate_weights(ic_values, corr, model_order=["A", "B", "C"])

        # C should get higher weight due to diversification
        assert weights["C"] > weights["A"]

    def test_empty_input(self):
        calc = RidgeWeightCalculator()
        assert calc.calculate_weights({}, np.eye(0)) == {}


class TestTemperatureCompressor:
    def test_no_compression_at_t1(self):
        comp = TemperatureCompressor({"5m": 1.0})

        weights = {"A": 0.7, "B": 0.3}
        result = comp.compress(weights, "5m")

        assert result["A"] == pytest.approx(0.7)
        assert result["B"] == pytest.approx(0.3)

    def test_compression_flattens_weights(self):
        comp = TemperatureCompressor({"5m": 0.75})

        weights = {"A": 0.8, "B": 0.2}
        result = comp.compress(weights, "5m")

        # After compression, weights should be more equal
        assert result["A"] < 0.8
        assert result["B"] > 0.2
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_unknown_horizon_no_compression(self):
        comp = TemperatureCompressor({"5m": 0.75})

        weights = {"A": 0.8, "B": 0.2}
        result = comp.compress(weights, "unknown")

        assert result == weights


class TestHorizonBlender:
    def test_blend_single_model(self):
        blender = HorizonBlender()

        # Mock HorizonPredictions
        from unittest.mock import Mock
        hp = Mock()
        hp.horizon = "5m"
        hp.predictions = {
            "LightGBM": Mock(calibrated=0.5, confidence=Mock(ic=0.1, overall=0.8))
        }

        result = blender.blend(hp)

        assert isinstance(result, BlendedAlpha)
        assert result.horizon == "5m"
        assert result.alpha == pytest.approx(0.5)

    def test_blend_multiple_models(self):
        blender = HorizonBlender()

        from unittest.mock import Mock
        hp = Mock()
        hp.horizon = "5m"
        hp.predictions = {
            "LightGBM": Mock(calibrated=0.6, confidence=Mock(ic=0.1, overall=0.8)),
            "XGBoost": Mock(calibrated=0.4, confidence=Mock(ic=0.1, overall=0.7)),
        }

        result = blender.blend(hp)

        # Alpha should be between individual predictions
        assert 0.4 <= result.alpha <= 0.6
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
```

## SST Compliance Checklist

- [ ] Uses `get_cfg()` for all configuration
- [ ] Uses `sorted_items()` for all dict iteration
- [ ] Proper numerical stability (checks for division by zero, etc.)
- [ ] Weight normalization ensures sum = 1
- [ ] History buffers use `deque` with `maxlen`

## Dependencies

- `CONFIG.config_loader.get_cfg`
- `TRAINING.common.utils.determinism_ordering.sorted_items`
- `LIVE_TRADING.prediction.predictor.HorizonPredictions`
- External: `numpy`, `numpy.linalg.inv`

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `__init__.py` | 15 |
| `ridge_weights.py` | 200 |
| `temperature.py` | 120 |
| `horizon_blender.py` | 250 |
| `tests/test_blending.py` | 150 |
| **Total** | ~735 |
