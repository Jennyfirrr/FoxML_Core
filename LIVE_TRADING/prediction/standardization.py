"""
Z-Score Standardization
=======================

Standardizes model predictions to comparable scales using rolling z-scores.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.common.constants import (
    ZSCORE_CLIP_MIN,
    ZSCORE_CLIP_MAX,
    STANDARDIZATION_WINDOW,
)

logger = logging.getLogger(__name__)


@dataclass
class StandardizationStats:
    """Statistics for standardization."""

    mean: float
    std: float
    count: int
    last_raw: float
    last_standardized: float


class ZScoreStandardizer:
    """
    Rolling Z-score standardizer for model predictions.

    Maintains separate statistics for each (model, horizon) pair.
    """

    def __init__(
        self,
        window_size: int | None = None,
        clip_min: float = ZSCORE_CLIP_MIN,
        clip_max: float = ZSCORE_CLIP_MAX,
    ):
        """
        Initialize standardizer.

        Args:
            window_size: Rolling window size in observations
            clip_min: Minimum clipped z-score
            clip_max: Maximum clipped z-score
        """
        self.window_size = window_size or get_cfg(
            "live_trading.standardization.window_size",
            default=STANDARDIZATION_WINDOW,
        )
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Rolling buffers per (model, horizon) key
        self._buffers: Dict[str, deque] = {}

        logger.info(f"ZScoreStandardizer initialized: window={self.window_size}")

    def _get_key(self, model: str, horizon: str) -> str:
        """Get buffer key for model/horizon pair."""
        return f"{model}:{horizon}"

    def standardize(
        self,
        raw_prediction: float,
        model: str,
        horizon: str,
    ) -> float:
        """
        Standardize a prediction using rolling z-score.

        Args:
            raw_prediction: Raw model prediction
            model: Model family name
            horizon: Horizon (e.g., "5m")

        Returns:
            Standardized prediction clipped to [-3, 3]
        """
        key = self._get_key(model, horizon)

        # Initialize buffer if needed
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.window_size)

        buffer = self._buffers[key]

        # Add to buffer
        buffer.append(raw_prediction)

        # Need minimum samples for meaningful standardization
        if len(buffer) < 3:
            # Return clipped raw value during warmup
            return float(np.clip(raw_prediction, self.clip_min, self.clip_max))

        # Calculate rolling statistics
        arr = np.array(buffer)
        mean = np.mean(arr)
        std = np.std(arr)

        # Avoid division by zero
        if std < 1e-9:
            return 0.0

        # Z-score
        z = (raw_prediction - mean) / std

        # Clip
        z_clipped = np.clip(z, self.clip_min, self.clip_max)

        return float(z_clipped)

    def standardize_batch(
        self,
        predictions: Dict[str, float],
        horizon: str,
    ) -> Dict[str, float]:
        """
        Standardize predictions from multiple models.

        Args:
            predictions: Dict mapping model name to raw prediction
            horizon: Horizon for all predictions

        Returns:
            Dict mapping model name to standardized prediction
        """
        return {
            model: self.standardize(pred, model, horizon)
            for model, pred in predictions.items()
        }

    def get_stats(self, model: str, horizon: str) -> Optional[StandardizationStats]:
        """
        Get current statistics for a model/horizon.

        Args:
            model: Model family name
            horizon: Horizon

        Returns:
            Statistics or None if not enough data
        """
        key = self._get_key(model, horizon)
        if key not in self._buffers or len(self._buffers[key]) < 2:
            return None

        buffer = self._buffers[key]
        arr = np.array(buffer)

        last_raw = float(arr[-1])
        # Note: standardize will add to buffer again, so we compute inline
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-9:
            last_std = 0.0
        else:
            last_std = float(np.clip((last_raw - mean) / std, self.clip_min, self.clip_max))

        return StandardizationStats(
            mean=mean,
            std=std,
            count=len(buffer),
            last_raw=last_raw,
            last_standardized=last_std,
        )

    def reset(self, model: str | None = None, horizon: str | None = None) -> None:
        """
        Reset buffers.

        Args:
            model: Model to reset (None = all)
            horizon: Horizon to reset (None = all)
        """
        if model is None and horizon is None:
            self._buffers.clear()
            logger.debug("Reset all standardization buffers")
        elif model is not None and horizon is not None:
            key = self._get_key(model, horizon)
            if key in self._buffers:
                self._buffers[key].clear()
                logger.debug(f"Reset standardization buffer for {key}")
        elif model is not None:
            # Reset all horizons for this model
            to_remove = [k for k in self._buffers.keys() if k.startswith(f"{model}:")]
            for k in to_remove:
                self._buffers[k].clear()
        elif horizon is not None:
            # Reset all models for this horizon
            to_remove = [k for k in self._buffers.keys() if k.endswith(f":{horizon}")]
            for k in to_remove:
                self._buffers[k].clear()
