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
from numpy.linalg import inv, LinAlgError

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.constants import DEFAULT_CONFIG, MIN_IC_THRESHOLD

logger = logging.getLogger(__name__)


class RidgeWeightCalculator:
    """
    Calculates Ridge risk-parity weights for model ensemble.

    w ∝ (Σ + λI)^{-1} μ

    Where:
    - Σ = correlation matrix of standardized scores across families
    - λ = ridge regularization (default 0.15)
    - μ = target vector of net IC after costs
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

        if n_models == 0:
            return {}

        if n_models == 1:
            return {model_order[0]: 1.0}

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

        # Handle NaN values in correlation matrix
        if np.any(np.isnan(correlation_matrix)):
            logger.warning("NaN in correlation matrix, using identity")
            correlation_matrix = np.eye(n_models)

        # Ridge regression: w ∝ (Σ + λI)^{-1} μ
        regularized = correlation_matrix + self.ridge_lambda * np.eye(n_models)

        try:
            inv_matrix = inv(regularized)
            raw_weights = inv_matrix @ mu
        except LinAlgError:
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

        if n_models == 0:
            return {}

        if n_models == 1:
            return {model_order[0]: 1.0}

        # Build prediction matrix
        min_len = min(len(v) for v in prediction_history.values())
        if min_len < 3:
            # Not enough history, use equal weights
            return {m: 1.0 / n_models for m in model_order}

        pred_matrix = np.zeros((min_len, n_models))
        for i, model in enumerate(model_order):
            pred_matrix[:, i] = prediction_history[model][-min_len:]

        # Calculate correlation matrix
        # Handle case where predictions have no variance
        with np.errstate(invalid='ignore'):
            correlation_matrix = np.corrcoef(pred_matrix.T)

        # Replace NaN with identity (when a model has constant predictions)
        if np.any(np.isnan(correlation_matrix)):
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            np.fill_diagonal(correlation_matrix, 1.0)

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
