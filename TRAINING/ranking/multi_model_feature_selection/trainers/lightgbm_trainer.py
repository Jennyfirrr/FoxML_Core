# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
LightGBM model trainer for multi-model feature selection.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import (
    TrainerResult,
    TaskType,
    detect_task_type,
    register_trainer,
    get_gpu_config,
    clean_model_config,
)

logger = logging.getLogger(__name__)


@register_trainer('lightgbm')
def train_lightgbm(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train a LightGBM model.

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
        from lightgbm import LGBMRegressor, LGBMClassifier

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Get GPU config (empty in strict mode)
        gpu_params = get_gpu_config('lightgbm')

        # Clean config: remove params that don't apply to sklearn wrapper
        lgb_config = model_config.copy()
        for key in ['boosting_type', 'device', 'gpu_device_id', 'gpu_platform_id',
                    'early_stopping_rounds', 'early_stopping_round', 'callbacks',
                    'threads', 'min_samples_split']:
            lgb_config.pop(key, None)

        # Determine estimator class
        est_cls = LGBMClassifier if is_classification else LGBMRegressor

        # Clean config using systematic helper
        extra = {"random_seed": model_seed}
        lgb_config = clean_model_config(est_cls, lgb_config, extra, "lightgbm")

        # Remove verbose (will use default)
        lgb_config.pop('verbose', None)

        # Force deterministic mode for reproducibility
        lgb_config['deterministic'] = True
        lgb_config['force_row_wise'] = True

        # Add GPU params if available
        lgb_config.update(gpu_params)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('LightGBM', total_threads=default_threads())
            lgb_config['num_threads'] = plan['OMP']

            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model = est_cls(**lgb_config, **extra)
                model.fit(X, y)
        except ImportError:
            # Fallback: no thread management
            model = est_cls(**lgb_config, **extra)
            model.fit(X, y)

        train_score = model.score(X, y)

        metric_name = 'accuracy' if is_classification else 'R²'
        logger.debug(f"    lightgbm: metric={metric_name}, score={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"LightGBM failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))
