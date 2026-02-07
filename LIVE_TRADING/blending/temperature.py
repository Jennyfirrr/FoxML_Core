"""
Temperature Compression
=======================

Applies temperature compression to model weights for shorter horizons.
Lower temperature = more conservative (flatter) weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

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
    - T=1.0: No compression (use raw weights)
    - T=0.75: Moderate compression (shorter horizons)
    - T→0: Maximum compression (equal weights)
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
        """
        Get temperature for a horizon.

        Args:
            horizon: Horizon name (e.g., "5m", "1d")

        Returns:
            Temperature value (default 1.0 if not specified)
        """
        return self.temperatures.get(horizon, 1.0)

    def compress(
        self,
        weights: Dict[str, float],
        horizon: str,
    ) -> Dict[str, float]:
        """
        Apply temperature compression to weights.

        w^{(T)} ∝ w^{1/T}

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

        # Avoid division by zero
        if temperature <= 0:
            # Maximum compression = equal weights
            n = len(weights)
            return {m: 1.0 / n for m in weights}

        # Apply compression: w^T (T<1 flattens the distribution)
        # This is softmax-style temperature where lower T = more uniform
        compressed = {}
        for model, weight in sorted_items(weights):
            if weight > 0:
                compressed[model] = weight ** temperature
            else:
                compressed[model] = 0.0

        # Normalize
        total = sum(compressed.values())
        if total > 0:
            return {m: w / total for m, w in sorted_items(compressed)}
        else:
            # Fallback to original weights
            return dict(weights)

    def analyze_compression(
        self,
        weights: Dict[str, float],
        horizon: str,
    ) -> Dict[str, Any]:
        """
        Analyze the effect of compression.

        Args:
            weights: Original weights
            horizon: Horizon

        Returns:
            Analysis dict with before/after entropy, etc.
        """
        compressed = self.compress(weights, horizon)

        # Calculate entropy
        def entropy(w_dict: Dict[str, float]) -> float:
            vals = np.array(list(w_dict.values()))
            vals = vals[vals > 0]
            if len(vals) == 0:
                return 0.0
            p = vals / vals.sum()
            return float(-np.sum(p * np.log(p + 1e-10)))

        return {
            "horizon": horizon,
            "temperature": self.get_temperature(horizon),
            "entropy_before": entropy(weights),
            "entropy_after": entropy(compressed),
            "max_weight_before": max(weights.values()) if weights else 0,
            "max_weight_after": max(compressed.values()) if compressed else 0,
            "weight_count": len(weights),
        }

    def set_temperature(self, horizon: str, temperature: float) -> None:
        """
        Set temperature for a specific horizon.

        Args:
            horizon: Horizon name
            temperature: Temperature value (0 < T <= 1 typically)
        """
        self.temperatures[horizon] = temperature
        logger.debug(f"Set temperature for {horizon}: {temperature}")
