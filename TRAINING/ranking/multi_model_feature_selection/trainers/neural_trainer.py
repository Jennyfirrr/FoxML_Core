# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Neural network model trainer for multi-model feature selection.

Handles:
- Lookahead bias prevention with CV-based normalization
- NaN imputation (neural networks can't handle NaN)
- StandardScaler normalization
- Thread management for MKL/OMP
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import (
    TrainerResult,
    register_trainer,
    clean_model_config,
)

logger = logging.getLogger(__name__)


@register_trainer('neural_network')
def train_neural_network(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    X_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    **kwargs
) -> TrainerResult:
    """Train an MLP neural network model.

    This trainer handles lookahead bias prevention by:
    - Fitting imputer/scaler on training data only (if X_train/X_test provided)
    - Transforming test data using training statistics
    - Falling back to full-dataset normalization if no split provided

    Args:
        model_config: Model hyperparameters
        X: Full feature matrix
        y: Full target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        X_train: Optional training features (for CV-based normalization)
        X_test: Optional test features (for CV-based normalization)
        y_train: Optional training targets
        y_test: Optional test targets
        **kwargs: Additional arguments (ignored)

    Returns:
        TrainerResult with trained model and score
    """
    try:
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Load lookahead bias fix config
        normalize_inside_cv = False
        try:
            from TRAINING.ranking.utils.lookahead_bias_config import get_lookahead_bias_fix_config
            fix_config = get_lookahead_bias_fix_config()
            normalize_inside_cv = fix_config.get('normalize_inside_cv', False)
        except Exception as e:
            from TRAINING.common.determinism import is_strict_mode
            if is_strict_mode():
                from TRAINING.common.exceptions import ConfigError
                raise ConfigError(f"Failed to load lookahead_bias_fix_config in strict mode: {e}") from e

        # Determine task type
        unique_vals = np.unique(y[~np.isnan(y)])
        is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
        is_classification = is_binary or len(unique_vals) <= 10

        # CV-based normalization if train/test split provided
        if X_train is not None and X_test is not None and normalize_inside_cv:
            # Fit imputer and scaler on training data only
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            X_scaled = X_train_scaled
            y_for_training = y_train if y_train is not None else y

            logger.debug("  Neural Network: Using CV-based normalization (fit on train, transform test)")
        else:
            # Fallback: Full-dataset normalization
            if normalize_inside_cv and X_train is None:
                logger.warning(
                    "normalize_inside_cv=True but no train/test split provided. "
                    "Using full-dataset normalization (acceptable for feature selection)."
                )

            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            y_for_training = y

        # Determine estimator class
        est_cls = MLPClassifier if is_classification else MLPRegressor

        # Clean config
        extra = {"random_state": model_seed}
        nn_config = clean_model_config(est_cls, model_config, extra, "neural_network")

        # Instantiate model
        model = est_cls(**nn_config, **extra)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('MLP', total_threads=default_threads())
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_scaled, y_for_training)
                train_score = model.score(X_scaled, y_for_training)
        except ImportError:
            model.fit(X_scaled, y_for_training)
            train_score = model.score(X_scaled, y_for_training)

        metric_name = 'accuracy' if is_classification else 'R²'
        logger.debug(f"    neural_network: {metric_name}={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except (ValueError, TypeError) as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ['least populated class', 'too few', 'invalid classes']):
            logger.debug("    Neural Network: Target too imbalanced")
            return TrainerResult(model=None, train_score=0.0, error="Target too imbalanced")
        logger.error(f"Neural Network failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))

    except Exception as e:
        logger.error(f"Neural Network failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))
