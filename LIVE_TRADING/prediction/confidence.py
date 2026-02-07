"""
Confidence Scoring
==================

Calculates confidence scores for model predictions based on
IC, freshness, capacity, and stability.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from scipy import stats

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import (
    FRESHNESS_TAU,
    CAPACITY_KAPPA,
    MIN_IC_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceComponents:
    """Components of confidence score."""

    ic: float
    freshness: float
    capacity: float
    stability: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "ic": self.ic,
            "freshness": self.freshness,
            "capacity": self.capacity,
            "stability": self.stability,
            "overall": self.overall,
        }


class ConfidenceScorer:
    """
    Calculates confidence scores for model predictions.

    Confidence = IC × freshness × capacity × stability
    """

    def __init__(
        self,
        ic_window: int = 20,
        stability_window: int = 10,
    ):
        """
        Initialize confidence scorer.

        Args:
            ic_window: Window for IC calculation
            stability_window: Window for stability calculation
        """
        self.ic_window = ic_window
        self.stability_window = stability_window

        # Rolling buffers for IC calculation: (prediction, actual_return)
        self._prediction_buffers: Dict[str, deque] = {}
        self._return_buffers: Dict[str, deque] = {}

        # Calibration RMSE buffers
        self._rmse_buffers: Dict[str, deque] = {}

        logger.info(f"ConfidenceScorer initialized: ic_window={ic_window}")

    def _get_key(self, model: str, horizon: str) -> str:
        return f"{model}:{horizon}"

    def update_with_actual(
        self,
        model: str,
        horizon: str,
        prediction: float,
        actual_return: float,
    ) -> None:
        """
        Update with actual return for IC calculation.

        Args:
            model: Model family name
            horizon: Horizon
            prediction: Previous prediction
            actual_return: Actual realized return
        """
        key = self._get_key(model, horizon)

        # Initialize buffers
        if key not in self._prediction_buffers:
            self._prediction_buffers[key] = deque(maxlen=self.ic_window)
            self._return_buffers[key] = deque(maxlen=self.ic_window)
            self._rmse_buffers[key] = deque(maxlen=self.stability_window)

        # Add to buffers
        self._prediction_buffers[key].append(prediction)
        self._return_buffers[key].append(actual_return)

        # Calculate squared error for stability
        error_sq = (prediction - actual_return) ** 2
        self._rmse_buffers[key].append(error_sq)

    def calculate_ic(self, model: str, horizon: str) -> float:
        """
        Calculate Information Coefficient (Spearman correlation).

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            IC value (0 if insufficient data)
        """
        key = self._get_key(model, horizon)

        if key not in self._prediction_buffers:
            return 0.0

        preds = np.array(self._prediction_buffers[key])
        rets = np.array(self._return_buffers[key])

        if len(preds) < 5:
            return 0.0

        # Spearman correlation
        try:
            corr, _ = stats.spearmanr(preds, rets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def calculate_freshness(
        self,
        horizon: str,
        data_timestamp: datetime,
        current_time: datetime | None = None,
    ) -> float:
        """
        Calculate freshness factor based on data age.

        freshness = e^{-Δt/τ_h}

        Args:
            horizon: Horizon
            data_timestamp: Timestamp of data
            current_time: Current time (default: now)

        Returns:
            Freshness factor in [0, 1]
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure both timestamps are comparable
        if data_timestamp.tzinfo is None:
            # Assume UTC for naive timestamps
            data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Time difference in seconds
        delta_t = (current_time - data_timestamp).total_seconds()

        # Don't penalize for future timestamps (clock skew)
        if delta_t < 0:
            return 1.0

        # Get tau for horizon
        tau = FRESHNESS_TAU.get(horizon, 300.0)

        # Exponential decay
        freshness = math.exp(-delta_t / tau)

        return max(0.0, min(1.0, freshness))

    def calculate_capacity(
        self,
        adv: float,
        planned_dollars: float,
        kappa: float | None = None,
    ) -> float:
        """
        Calculate capacity factor.

        capacity = min(1, κ × ADV / planned_dollars)

        Args:
            adv: Average daily volume in dollars
            planned_dollars: Planned trade size in dollars
            kappa: Participation rate (default from config)

        Returns:
            Capacity factor in [0, 1]
        """
        if kappa is None:
            kappa = get_cfg("live_trading.capacity_kappa", default=CAPACITY_KAPPA)

        if planned_dollars <= 0:
            return 1.0

        if adv <= 0 or not np.isfinite(adv):
            return 1.0

        capacity = kappa * adv / planned_dollars
        return min(1.0, capacity)

    def calculate_stability(self, model: str, horizon: str) -> float:
        """
        Calculate stability factor based on calibration RMSE.

        stability = 1 / (1 + rolling_RMSE)

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            Stability factor in [0, 1]
        """
        key = self._get_key(model, horizon)

        if key not in self._rmse_buffers or len(self._rmse_buffers[key]) < 3:
            return 0.5  # Neutral during warmup

        rmse = math.sqrt(np.mean(self._rmse_buffers[key]))
        stability = 1.0 / (1.0 + rmse)

        return stability

    def calculate_confidence(
        self,
        model: str,
        horizon: str,
        data_timestamp: datetime,
        adv: float = float("inf"),
        planned_dollars: float = 0.0,
    ) -> ConfidenceComponents:
        """
        Calculate overall confidence score.

        Args:
            model: Model family name
            horizon: Horizon
            data_timestamp: Data timestamp
            adv: Average daily volume (optional)
            planned_dollars: Planned trade size (optional)

        Returns:
            ConfidenceComponents with all factors
        """
        ic = self.calculate_ic(model, horizon)
        freshness = self.calculate_freshness(horizon, data_timestamp)
        capacity = self.calculate_capacity(adv, planned_dollars)
        stability = self.calculate_stability(model, horizon)

        # Combined confidence
        # Use max(IC, threshold) to avoid negative multiplication
        ic_adjusted = max(ic, MIN_IC_THRESHOLD)
        overall = ic_adjusted * freshness * capacity * stability

        return ConfidenceComponents(
            ic=ic,
            freshness=freshness,
            capacity=capacity,
            stability=stability,
            overall=overall,
        )

    def apply_confidence(
        self,
        standardized_prediction: float,
        confidence: float,
    ) -> float:
        """
        Apply confidence weighting to prediction.

        calibrated = standardized × confidence

        Args:
            standardized_prediction: Z-score standardized prediction
            confidence: Confidence score

        Returns:
            Calibrated prediction
        """
        return standardized_prediction * confidence

    def reset(self, model: str | None = None, horizon: str | None = None) -> None:
        """
        Reset buffers.

        Args:
            model: Model to reset (None = all)
            horizon: Horizon to reset (None = all)
        """
        if model is None and horizon is None:
            self._prediction_buffers.clear()
            self._return_buffers.clear()
            self._rmse_buffers.clear()
            logger.debug("Reset all confidence buffers")
        elif model is not None and horizon is not None:
            key = self._get_key(model, horizon)
            for buf_dict in [self._prediction_buffers, self._return_buffers, self._rmse_buffers]:
                if key in buf_dict:
                    buf_dict[key].clear()
            logger.debug(f"Reset confidence buffers for {key}")

    def get_ic_history(self, model: str, horizon: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get IC calculation history for debugging.

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            Dict with predictions and returns arrays, or None
        """
        key = self._get_key(model, horizon)

        if key not in self._prediction_buffers:
            return None

        return {
            "predictions": np.array(self._prediction_buffers[key]),
            "returns": np.array(self._return_buffers[key]),
        }
