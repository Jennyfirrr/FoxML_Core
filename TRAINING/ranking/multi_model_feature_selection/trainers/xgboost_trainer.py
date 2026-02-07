# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
XGBoost model trainer for multi-model feature selection.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import (
    TrainerResult,
    TaskType,
    detect_task_type,
    register_trainer,
    clean_model_config,
)

logger = logging.getLogger(__name__)


@register_trainer('xgboost')
def train_xgboost(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train an XGBoost model.

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
        import xgboost as xgb

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # GPU settings
        gpu_params = _get_xgboost_gpu_params()

        # Remove early stopping params (requires eval_set)
        xgb_config = model_config.copy()
        for key in ['early_stopping_rounds', 'early_stopping_round', 'callbacks',
                    'eval_set', 'eval_metric', 'tree_method', 'device', 'gpu_id']:
            xgb_config.pop(key, None)

        # Determine estimator class
        est_cls = xgb.XGBClassifier if is_classification else xgb.XGBRegressor

        # Clean config using systematic helper
        extra = {"random_state": model_seed}
        xgb_config = clean_model_config(est_cls, xgb_config, extra, "xgboost")

        # Add GPU params if available
        xgb_config.update(gpu_params)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('XGBoost', total_threads=default_threads())
            xgb_config['n_jobs'] = plan['OMP']

            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model = est_cls(**xgb_config, **extra)
                model.fit(X, y)
        except ImportError:
            model = est_cls(**xgb_config, **extra)
            model.fit(X, y)

        train_score = model.score(X, y)

        metric_name = 'accuracy' if is_classification else 'R²'
        logger.debug(f"    xgboost: metric={metric_name}, score={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except (ValueError, TypeError) as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ['invalid classes', 'expected', 'too few']):
            logger.debug("    XGBoost: Target degenerate")
            return TrainerResult(model=None, train_score=0.0, error="Target degenerate")
        logger.error(f"XGBoost failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))

    except ImportError:
        logger.error("XGBoost not available")
        return TrainerResult(model=None, train_score=0.0, error="XGBoost not available")

    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


def _get_xgboost_gpu_params() -> Dict[str, Any]:
    """Get XGBoost GPU configuration."""
    from TRAINING.common.determinism import is_strict_mode

    if is_strict_mode():
        return {}

    try:
        from CONFIG.config_loader import get_cfg

        xgb_device = get_cfg('gpu.xgboost.device', default='cpu', config_name='gpu_config')
        xgb_tree_method = get_cfg('gpu.xgboost.tree_method', default='hist', config_name='gpu_config')
        test_enabled = get_cfg('gpu.xgboost.test_enabled', default=True, config_name='gpu_config')
        test_n_estimators = get_cfg('gpu.xgboost.test_n_estimators', default=1, config_name='gpu_config')
        test_samples = get_cfg('gpu.xgboost.test_samples', default=10, config_name='gpu_config')
        test_features = get_cfg('gpu.xgboost.test_features', default=5, config_name='gpu_config')

        if xgb_device == 'cuda':
            import xgboost as xgb

            if test_enabled:
                # Test CUDA availability
                try:
                    test_model = xgb.XGBRegressor(
                        tree_method='hist', device='cuda',
                        n_estimators=test_n_estimators, verbosity=0
                    )
                    test_model.fit(
                        np.random.rand(test_samples, test_features),
                        np.random.rand(test_samples)
                    )
                    return {'tree_method': xgb_tree_method, 'device': 'cuda'}
                except Exception:
                    # Try legacy API
                    try:
                        test_model = xgb.XGBRegressor(
                            tree_method='gpu_hist',
                            n_estimators=test_n_estimators, verbosity=0
                        )
                        test_model.fit(
                            np.random.rand(test_samples, test_features),
                            np.random.rand(test_samples)
                        )
                        return {'tree_method': 'gpu_hist'}
                    except Exception:
                        pass
            else:
                return {'tree_method': xgb_tree_method, 'device': 'cuda'}
    except Exception:
        pass

    return {}
