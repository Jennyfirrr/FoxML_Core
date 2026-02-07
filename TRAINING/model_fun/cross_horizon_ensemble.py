# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Horizon Ensemble

Trains a meta-learner that blends predictions across multiple horizons.
Uses ridge regression to learn optimal weights based on prediction correlations.

This mirrors the live trading HorizonBlender but for training-time ensemble.

Usage:
    from TRAINING.model_fun.cross_horizon_ensemble import (
        CrossHorizonEnsemble,
        calculate_horizon_weights,
    )

    # Train ensemble from horizon predictions
    ensemble = CrossHorizonEnsemble(config)
    result = ensemble.fit(
        horizon_predictions={
            "fwd_ret_5m": pred_5m,
            "fwd_ret_15m": pred_15m,
            "fwd_ret_60m": pred_60m,
        },
        y_true=actual_returns,
    )

    # Blend new predictions
    blended = ensemble.blend(new_predictions)
"""

from __future__ import annotations

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first (after __future__)

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import inv, LinAlgError

from CONFIG.config_loader import get_cfg
from TRAINING.common.horizon_bundle import parse_horizon_from_target

logger = logging.getLogger(__name__)


# SST: Load cross-horizon config from CONFIG/models/ensemble.yaml
def _get_cross_horizon_config() -> Dict[str, Any]:
    """Load cross-horizon ensemble config from CONFIG."""
    return {
        "ridge_lambda": get_cfg("cross_horizon.ridge_lambda", default=0.15, config_name="ensemble"),
        "horizon_decay_enabled": get_cfg("cross_horizon.decay_function", default="exponential", config_name="ensemble") != "none",
        "horizon_decay_half_life_minutes": get_cfg("cross_horizon.decay_half_life_minutes", default=30.0, config_name="ensemble"),
        "min_weight": get_cfg("cross_horizon.min_weight", default=0.01, config_name="ensemble"),
        "ic_lookback_samples": get_cfg("cross_horizon.ic_lookback_samples", default=500, config_name="ensemble"),
        "use_ic_weighting": get_cfg("cross_horizon.use_ic_weighting", default=True, config_name="ensemble"),
    }


# For backward compatibility
DEFAULT_CROSS_HORIZON_CONFIG = _get_cross_horizon_config()


@dataclass
class EnsembleWeights:
    """Learned ensemble weights with metadata."""

    weights: Dict[str, float]
    horizons: List[str]
    ridge_lambda: float
    correlation_matrix: Optional[np.ndarray] = None
    ic_values: Optional[Dict[str, float]] = None
    decay_applied: bool = False

    @property
    def n_horizons(self) -> int:
        """Number of horizons."""
        return len(self.horizons)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "weights": self.weights,
            "horizons": self.horizons,
            "ridge_lambda": self.ridge_lambda,
            "correlation_matrix": (
                self.correlation_matrix.tolist()
                if self.correlation_matrix is not None
                else None
            ),
            "ic_values": self.ic_values,
            "decay_applied": self.decay_applied,
            "n_horizons": self.n_horizons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleWeights":
        """Deserialize from dict."""
        corr = data.get("correlation_matrix")
        return cls(
            weights=data["weights"],
            horizons=data["horizons"],
            ridge_lambda=data["ridge_lambda"],
            correlation_matrix=np.array(corr) if corr else None,
            ic_values=data.get("ic_values"),
            decay_applied=data.get("decay_applied", False),
        )


@dataclass
class BlendedPrediction:
    """Blended prediction result."""

    value: float
    horizon_contributions: Dict[str, float]
    weights_used: Dict[str, float]
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "value": self.value,
            "horizon_contributions": self.horizon_contributions,
            "weights_used": self.weights_used,
            "timestamp": self.timestamp,
        }


class CrossHorizonEnsemble:
    """
    Learns optimal blending weights across prediction horizons.

    The ensemble uses ridge regression on the correlation structure
    of horizon predictions to determine optimal weights:

        w ∝ (Σ + λI)^{-1} μ

    Where:
        Σ = correlation matrix of predictions across horizons
        λ = ridge regularization (stabilizes matrix inversion)
        μ = IC (information coefficient) per horizon

    Additionally supports horizon decay to weight shorter horizons higher
    (useful for live trading where shorter-term signals are more actionable).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble.

        Args:
            config: Configuration dict with keys:
                - ridge_lambda: Ridge regularization (default: 0.15)
                - horizon_decay_enabled: Apply horizon decay (default: True)
                - horizon_decay_half_life_minutes: Decay half-life (default: 30.0)
                - min_weight: Minimum weight threshold (default: 0.01)
                - ic_lookback_samples: Samples for IC calculation (default: 500)
                - use_ic_weighting: Weight by IC (default: True)
        """
        self.config = {**DEFAULT_CROSS_HORIZON_CONFIG, **(config or {})}

        # Load from centralized config if available
        self.ridge_lambda = self.config.get(
            "ridge_lambda",
            get_cfg("ensemble.cross_horizon.ridge_lambda", default=0.15),
        )
        self.horizon_decay_enabled = self.config.get(
            "horizon_decay_enabled",
            get_cfg("ensemble.cross_horizon.decay_function", default=True),
        )
        self.horizon_decay_half_life = self.config.get(
            "horizon_decay_half_life_minutes",
            get_cfg("ensemble.cross_horizon.decay_half_life_minutes", default=30.0),
        )
        self.min_weight = self.config.get("min_weight", 0.01)
        self.use_ic_weighting = self.config.get("use_ic_weighting", True)

        # Learned state
        self.weights: Optional[EnsembleWeights] = None
        self.is_fitted = False
        self._horizon_order: List[str] = []
        self._horizon_minutes: Dict[str, int] = {}

    def fit(
        self,
        horizon_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit ensemble weights from horizon predictions.

        Args:
            horizon_predictions: Dict mapping horizon name to predictions (N,)
            y_true: True target values (N,)
            sample_weights: Optional sample weights (N,)

        Returns:
            Fit results dict with weights, ICs, and metrics
        """
        if not horizon_predictions:
            raise ValueError("No horizon predictions provided")

        # Establish horizon ordering (sorted by horizon minutes for determinism)
        self._horizon_order = sorted(
            horizon_predictions.keys(),
            key=lambda h: self._extract_horizon_minutes(h),
        )
        self._horizon_minutes = {
            h: self._extract_horizon_minutes(h) for h in self._horizon_order
        }

        n_horizons = len(self._horizon_order)
        logger.info(f"Fitting cross-horizon ensemble with {n_horizons} horizons")

        # Align all predictions to same length
        min_len = min(len(v) for v in horizon_predictions.values())
        min_len = min(min_len, len(y_true))

        pred_matrix = np.zeros((min_len, n_horizons))
        for i, horizon in enumerate(self._horizon_order):
            pred_matrix[:, i] = horizon_predictions[horizon][:min_len]

        y_aligned = y_true[:min_len]
        weights_aligned = sample_weights[:min_len] if sample_weights is not None else None

        # Calculate IC per horizon
        ic_values = self._calculate_ic_values(pred_matrix, y_aligned, weights_aligned)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(pred_matrix)

        # Calculate ridge weights
        raw_weights = self._calculate_ridge_weights(ic_values, correlation_matrix)

        # Apply horizon decay if enabled
        if self.horizon_decay_enabled:
            decay_factors = self._calculate_horizon_decay()
            raw_weights = raw_weights * decay_factors

        # Normalize
        raw_weights = np.clip(raw_weights, 0, None)
        raw_weights[raw_weights < self.min_weight] = 0.0
        total = raw_weights.sum()
        if total > 0:
            normalized_weights = raw_weights / total
        else:
            normalized_weights = np.ones(n_horizons) / n_horizons

        # Build weight dict
        weight_dict = {
            h: float(normalized_weights[i]) for i, h in enumerate(self._horizon_order)
        }
        ic_dict = {h: float(ic_values[i]) for i, h in enumerate(self._horizon_order)}

        self.weights = EnsembleWeights(
            weights=weight_dict,
            horizons=self._horizon_order,
            ridge_lambda=self.ridge_lambda,
            correlation_matrix=correlation_matrix,
            ic_values=ic_dict,
            decay_applied=self.horizon_decay_enabled,
        )
        self.is_fitted = True

        # Calculate blended predictions for metrics
        blended_preds = pred_matrix @ normalized_weights
        blend_ic = self._correlation(blended_preds, y_aligned, weights_aligned)

        result = {
            "weights": weight_dict,
            "ic_values": ic_dict,
            "correlation_matrix": correlation_matrix.tolist(),
            "blended_ic": float(blend_ic),
            "n_horizons": n_horizons,
            "n_samples": min_len,
            "decay_applied": self.horizon_decay_enabled,
            "ridge_lambda": self.ridge_lambda,
        }

        logger.info(
            f"Fitted ensemble: blended_ic={blend_ic:.4f}, weights={weight_dict}"
        )
        return result

    def blend(
        self,
        horizon_predictions: Dict[str, float | np.ndarray],
    ) -> BlendedPrediction | np.ndarray:
        """
        Blend predictions using learned weights.

        Args:
            horizon_predictions: Dict mapping horizon name to prediction(s).
                - If values are floats: single prediction mode
                - If values are np.ndarray: batch prediction mode

        Returns:
            - BlendedPrediction: For single values (includes metadata like contributions)
            - np.ndarray: For batch mode (raw blended values for efficiency)

        Note:
            Use isinstance() to narrow the return type if needed:
            ```
            result = ensemble.blend(preds)
            if isinstance(result, BlendedPrediction):
                print(result.value, result.horizon_contributions)
            else:
                print(result)  # np.ndarray
            ```
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Check if batch or single
        first_val = next(iter(horizon_predictions.values()))
        is_batch = isinstance(first_val, np.ndarray)

        if is_batch:
            return self._blend_batch(horizon_predictions)
        else:
            return self._blend_single(horizon_predictions)

    def _blend_single(
        self,
        horizon_predictions: Dict[str, float],
    ) -> BlendedPrediction:
        """Blend single predictions."""
        contributions = {}
        total = 0.0

        for horizon in self._horizon_order:
            if horizon in horizon_predictions and horizon in self.weights.weights:
                pred = horizon_predictions[horizon]
                weight = self.weights.weights[horizon]
                contribution = pred * weight
                contributions[horizon] = contribution
                total += contribution

        return BlendedPrediction(
            value=total,
            horizon_contributions=contributions,
            weights_used=self.weights.weights,
        )

    def _blend_batch(
        self,
        horizon_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Blend batch predictions."""
        # Determine output size
        first_arr = next(iter(horizon_predictions.values()))
        n_samples = len(first_arr)

        result = np.zeros(n_samples)

        for horizon in self._horizon_order:
            if horizon in horizon_predictions and horizon in self.weights.weights:
                weight = self.weights.weights[horizon]
                result += horizon_predictions[horizon] * weight

        return result

    def _calculate_ic_values(
        self,
        pred_matrix: np.ndarray,
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """Calculate IC (rank correlation) for each horizon."""
        n_horizons = pred_matrix.shape[1]
        ics = np.zeros(n_horizons)

        for i in range(n_horizons):
            ics[i] = self._correlation(pred_matrix[:, i], y_true, sample_weights)

        return ics

    def _correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate weighted Pearson correlation."""
        if weights is None:
            # Simple Pearson correlation
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 3:
                return 0.0
            return float(np.corrcoef(x[valid], y[valid])[0, 1])

        # Weighted correlation
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(weights))
        if valid.sum() < 3:
            return 0.0

        x_v = x[valid]
        y_v = y[valid]
        w_v = weights[valid]
        w_sum = w_v.sum()
        if w_sum < 1e-12:  # Avoid division by near-zero
            return 0.0
        w_v = w_v / w_sum

        x_mean = np.sum(w_v * x_v)
        y_mean = np.sum(w_v * y_v)

        cov_xy = np.sum(w_v * (x_v - x_mean) * (y_v - y_mean))
        var_x = np.sum(w_v * (x_v - x_mean) ** 2)
        var_y = np.sum(w_v * (y_v - y_mean) ** 2)

        denom = np.sqrt(var_x * var_y)
        if denom < 1e-12:  # Avoid division by near-zero
            return 0.0

        return float(cov_xy / denom)

    def _calculate_correlation_matrix(self, pred_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix of horizon predictions."""
        n_horizons = pred_matrix.shape[1]

        # Handle NaN values
        valid_rows = ~np.any(np.isnan(pred_matrix), axis=1)
        if valid_rows.sum() < 3:
            return np.eye(n_horizons)

        pred_clean = pred_matrix[valid_rows]

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(pred_clean.T)

        # Handle NaN (from constant predictions)
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        return corr

    def _calculate_ridge_weights(
        self,
        ic_values: np.ndarray,
        correlation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate ridge regression weights."""
        n = len(ic_values)

        if n == 0:
            return np.array([])

        if n == 1:
            return np.array([1.0])

        # Build target vector (use IC if enabled, else equal)
        if self.use_ic_weighting:
            mu = np.clip(ic_values, 0.001, None)  # Ensure positive
        else:
            mu = np.ones(n)

        # Ridge regression: w ∝ (Σ + λI)^{-1} μ
        regularized = correlation_matrix + self.ridge_lambda * np.eye(n)

        try:
            inv_matrix = inv(regularized)
            weights = inv_matrix @ mu
        except LinAlgError:
            logger.warning("Matrix inversion failed, using equal weights")
            weights = np.ones(n) / n

        return weights

    def _calculate_horizon_decay(self) -> np.ndarray:
        """Calculate horizon decay factors."""
        if not self._horizon_order:
            return np.array([])

        # Find primary horizon (middle one)
        sorted_minutes = sorted(self._horizon_minutes.values())
        primary_idx = len(sorted_minutes) // 2
        primary_minutes = sorted_minutes[primary_idx]

        decay_factors = np.zeros(len(self._horizon_order))
        for i, horizon in enumerate(self._horizon_order):
            horizon_min = self._horizon_minutes.get(horizon, 0)
            distance = abs(horizon_min - primary_minutes)
            # Exponential decay
            decay_factors[i] = np.exp(
                -np.log(2) * distance / self.horizon_decay_half_life
            )

        return decay_factors

    def _extract_horizon_minutes(self, target: str) -> int:
        """Extract horizon in minutes from target name."""
        _, horizon = parse_horizon_from_target(target)
        return horizon if horizon else 0

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        import os
        from pathlib import Path
        from TRAINING.common.utils.file_utils import write_atomic_json

        os.makedirs(path, exist_ok=True)
        path_obj = Path(path)

        if self.weights:
            write_atomic_json(path_obj / "weights.json", self.weights.to_dict())

        write_atomic_json(path_obj / "config.json", self.config)

        write_atomic_json(
            path_obj / "metadata.json",
            {
                "is_fitted": self.is_fitted,
                "horizon_order": self._horizon_order,
                "horizon_minutes": self._horizon_minutes,
            },
        )

        logger.info(f"Saved cross-horizon ensemble to {path}")

    @classmethod
    def load(cls, path: str) -> "CrossHorizonEnsemble":
        """Load ensemble from disk."""
        import json
        import os

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        ensemble = cls(config)

        weights_path = os.path.join(path, "weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights_data = json.load(f)
            ensemble.weights = EnsembleWeights.from_dict(weights_data)

        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            ensemble.is_fitted = metadata.get("is_fitted", False)
            ensemble._horizon_order = metadata.get("horizon_order", [])
            ensemble._horizon_minutes = metadata.get("horizon_minutes", {})

        logger.info(f"Loaded cross-horizon ensemble from {path}")
        return ensemble


def calculate_horizon_weights(
    horizon_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    ridge_lambda: float = 0.15,
    apply_decay: bool = True,
    decay_half_life_minutes: float = 30.0,
) -> Dict[str, float]:
    """
    Convenience function to calculate cross-horizon weights.

    Args:
        horizon_predictions: Dict mapping horizon to predictions
        y_true: True target values
        ridge_lambda: Ridge regularization
        apply_decay: Whether to apply horizon decay
        decay_half_life_minutes: Decay half-life in minutes

    Returns:
        Dict mapping horizon to weight
    """
    config = {
        "ridge_lambda": ridge_lambda,
        "horizon_decay_enabled": apply_decay,
        "horizon_decay_half_life_minutes": decay_half_life_minutes,
    }

    ensemble = CrossHorizonEnsemble(config)
    result = ensemble.fit(horizon_predictions, y_true)
    return result["weights"]


def blend_horizon_predictions(
    horizon_predictions: Dict[str, float | np.ndarray],
    weights: Dict[str, float],
) -> float | np.ndarray:
    """
    Blend predictions using pre-computed weights.

    Args:
        horizon_predictions: Predictions per horizon
        weights: Weight per horizon

    Returns:
        Blended prediction(s)
    """
    first_val = next(iter(horizon_predictions.values()))
    is_batch = isinstance(first_val, np.ndarray)

    if is_batch:
        n_samples = len(first_val)
        from TRAINING.common.utils.determinism_ordering import sorted_items
        result = np.zeros(n_samples)
        for horizon, pred in sorted_items(horizon_predictions):
            if horizon in weights:
                result += pred * weights[horizon]
        return result
    else:
        from TRAINING.common.utils.determinism_ordering import sorted_items
        total = 0.0
        for horizon, pred in sorted_items(horizon_predictions):
            if horizon in weights:
                total += pred * weights[horizon]
        return total
