# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Random Forest model trainer for multi-model feature selection.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .base import (
    TrainerResult,
    TaskType,
    detect_task_type,
    register_trainer,
    clean_model_config,
)

logger = logging.getLogger(__name__)


@register_trainer('random_forest')
def train_random_forest(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train a Random Forest model.

    Args:
        model_config: Model hyperparameters
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments (ignored)

    Returns:
        TrainerResult with trained model and score
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Determine estimator class
        est_cls = RandomForestClassifier if is_classification else RandomForestRegressor

        # Clean config using systematic helper
        extra = {"random_state": model_seed}
        rf_config = clean_model_config(est_cls, model_config, extra, "random_forest")

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('RandomForest', total_threads=default_threads())
            rf_config['n_jobs'] = plan['OMP']

            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model = est_cls(**rf_config, **extra)
                model.fit(X, y)
        except ImportError:
            model = est_cls(**rf_config, **extra)
            model.fit(X, y)

        train_score = model.score(X, y)

        metric_name = 'accuracy' if is_classification else 'R²'
        logger.debug(f"    random_forest: metric={metric_name}, score={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Random Forest failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))
