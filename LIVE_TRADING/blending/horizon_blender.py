"""
Horizon Blender
===============

Combines model predictions within a horizon using Ridge weights
and temperature compression.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon": self.horizon,
            "alpha": self.alpha,
            "weights": dict(self.weights),
            "temperature": self.temperature,
            "confidence": self.confidence,
        }


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
        ics = ic_values.copy() if ic_values else {}
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
            n = len(ic_values)
            if n == 0:
                return {}
            return {m: 1.0 / n for m in ic_values}

        # Build prediction history dict
        pred_history = {
            m: list(v) for m, v in sorted_items(history) if len(v) >= 5
        }

        if not pred_history:
            n = len(ic_values)
            if n == 0:
                return {}
            return {m: 1.0 / n for m in ic_values}

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
        """
        Get weight history for analysis.

        Args:
            horizon: Horizon name

        Returns:
            List of weight dicts over time
        """
        return self._weight_history.get(horizon, [])

    def get_prediction_history(
        self,
        horizon: str,
        family: str | None = None,
    ) -> Dict[str, List[float]] | List[float] | None:
        """
        Get prediction history for analysis.

        Args:
            horizon: Horizon name
            family: Optional specific family

        Returns:
            Prediction history (dict if family is None, list otherwise)
        """
        if horizon not in self._prediction_history:
            return None

        history = self._prediction_history[horizon]

        if family is not None:
            if family in history:
                return list(history[family])
            return None

        return {f: list(v) for f, v in sorted_items(history)}

    def reset(self, horizon: str | None = None) -> None:
        """
        Reset blender state.

        Args:
            horizon: Specific horizon to reset (None = all)
        """
        if horizon is None:
            self._prediction_history.clear()
            self._weight_history.clear()
            logger.debug("Reset all blender history")
        else:
            if horizon in self._prediction_history:
                self._prediction_history[horizon].clear()
            if horizon in self._weight_history:
                self._weight_history[horizon].clear()
            logger.debug(f"Reset blender history for {horizon}")
