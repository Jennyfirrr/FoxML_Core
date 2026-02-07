# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Base utilities for model family trainers.

Provides common functionality used across all model trainers:
- Task type detection (binary, multiclass, regression)
- Training target validation
- Trainer registry for dispatch
- Common result type
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Type of ML task determined from target values."""
    BINARY = auto()
    MULTICLASS = auto()
    REGRESSION = auto()


@dataclass
class TrainerResult:
    """Result from a model trainer.

    Attributes:
        model: The trained model (or None if training failed)
        train_score: Score on training data (R² for regression, accuracy for classification)
        error: Error message if training failed
    """
    model: Optional[Any]
    train_score: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """True if training succeeded."""
        return self.model is not None and self.error is None


def detect_task_type(y: np.ndarray) -> TaskType:
    """Detect task type from target values.

    Args:
        y: Target array (may contain NaN)

    Returns:
        TaskType indicating binary, multiclass, or regression
    """
    # Filter out NaN values
    y_valid = y[~np.isnan(y)]
    unique_vals = np.unique(y_valid)
    n_unique = len(unique_vals)

    # Binary classification: exactly 2 classes that are 0/1
    is_binary = (
        n_unique == 2 and
        set(unique_vals).issubset({0, 1, 0.0, 1.0})
    )

    if is_binary:
        return TaskType.BINARY

    # Multiclass: up to 10 integer-like classes
    is_multiclass = (
        n_unique <= 10 and
        all(
            isinstance(v, (int, np.integer)) or
            (isinstance(v, float) and v.is_integer())
            for v in unique_vals
        )
    )

    if is_multiclass:
        return TaskType.MULTICLASS

    return TaskType.REGRESSION


def validate_training_target(
    y: np.ndarray,
    min_samples: int = 10,
    min_class_samples: int = 2
) -> Tuple[bool, Optional[str]]:
    """Validate target array for training.

    Args:
        y: Target array
        min_samples: Minimum total samples required
        min_class_samples: Minimum samples per class for classification

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from TRAINING.ranking.utils.target_validation import validate_target
        return validate_target(y, min_samples=min_samples, min_class_samples=min_class_samples)
    except ImportError:
        # Fallback validation
        y_valid = y[~np.isnan(y)]

        if len(y_valid) < min_samples:
            return False, f"Insufficient samples: {len(y_valid)} < {min_samples}"

        unique_vals = np.unique(y_valid)
        if len(unique_vals) < 2:
            return False, f"Target has only {len(unique_vals)} unique value(s)"

        return True, None


# Trainer registry: maps family name to trainer function
# Each trainer function signature:
#   trainer(model_config, X, y, feature_names, model_seed, **kwargs) -> TrainerResult
TRAINER_REGISTRY: Dict[str, Callable] = {}


def register_trainer(family_name: str):
    """Decorator to register a trainer function.

    Usage:
        @register_trainer('lightgbm')
        def train_lightgbm(model_config, X, y, feature_names, model_seed, **kwargs):
            ...
            return TrainerResult(model=model, train_score=score)
    """
    def decorator(func: Callable) -> Callable:
        TRAINER_REGISTRY[family_name] = func
        return func
    return decorator


def get_trainer(family_name: str) -> Optional[Callable]:
    """Get trainer function for a model family.

    Args:
        family_name: Name of the model family

    Returns:
        Trainer function or None if not found
    """
    return TRAINER_REGISTRY.get(family_name)


def get_gpu_config(family: str) -> Dict[str, Any]:
    """Get GPU configuration for a model family.

    Args:
        family: Model family name (lightgbm, xgboost, catboost)

    Returns:
        Dict of GPU parameters (empty if GPU not available or in strict mode)
    """
    from TRAINING.common.determinism import is_strict_mode

    if is_strict_mode():
        logger.debug(f"  Strict mode: forcing CPU for {family} (GPU disabled for determinism)")
        return {}

    try:
        from CONFIG.config_loader import get_cfg
        gpu_enabled = get_cfg(f'gpu.{family}.enabled', default=False, config_name='gpu_config')
        if not gpu_enabled:
            return {}

        # Family-specific GPU config
        if family == 'lightgbm':
            return _get_lightgbm_gpu_config()
        elif family == 'xgboost':
            return _get_xgboost_gpu_config()
        elif family == 'catboost':
            return _get_catboost_gpu_config()

    except Exception as e:
        logger.debug(f"GPU config unavailable for {family}: {e}")

    return {}


def _get_lightgbm_gpu_config() -> Dict[str, Any]:
    """Get LightGBM GPU configuration."""
    try:
        from CONFIG.config_loader import get_cfg
        from lightgbm import LGBMRegressor

        test_enabled = get_cfg('gpu.lightgbm.test_enabled', default=True, config_name='gpu_config')
        test_n_estimators = get_cfg('gpu.lightgbm.test_n_estimators', default=1, config_name='gpu_config')
        test_samples = get_cfg('gpu.lightgbm.test_samples', default=10, config_name='gpu_config')
        test_features = get_cfg('gpu.lightgbm.test_features', default=5, config_name='gpu_config')
        gpu_device_id = get_cfg('gpu.lightgbm.gpu_device_id', default=0, config_name='gpu_config')
        gpu_platform_id = get_cfg('gpu.lightgbm.gpu_platform_id', default=0, config_name='gpu_config')
        try_cuda_first = get_cfg('gpu.lightgbm.try_cuda_first', default=True, config_name='gpu_config')

        if test_enabled and try_cuda_first:
            # Try CUDA first
            try:
                test_model = LGBMRegressor(
                    device='cuda', n_estimators=test_n_estimators,
                    gpu_device_id=gpu_device_id, verbose=-1
                )
                test_model.fit(
                    np.random.rand(test_samples, test_features),
                    np.random.rand(test_samples)
                )
                return {'device': 'cuda', 'gpu_device_id': gpu_device_id}
            except Exception:
                # Try OpenCL
                try:
                    test_model = LGBMRegressor(
                        device='gpu', n_estimators=test_n_estimators,
                        gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id,
                        verbose=-1
                    )
                    test_model.fit(
                        np.random.rand(test_samples, test_features),
                        np.random.rand(test_samples)
                    )
                    return {
                        'device': 'gpu',
                        'gpu_platform_id': gpu_platform_id,
                        'gpu_device_id': gpu_device_id
                    }
                except Exception:
                    pass
    except Exception:
        pass

    return {}


def _get_xgboost_gpu_config() -> Dict[str, Any]:
    """Get XGBoost GPU configuration."""
    try:
        from CONFIG.config_loader import get_cfg

        test_enabled = get_cfg('gpu.xgboost.test_enabled', default=True, config_name='gpu_config')
        gpu_id = get_cfg('gpu.xgboost.gpu_id', default=0, config_name='gpu_config')

        if test_enabled:
            import xgboost as xgb
            try:
                # Test GPU availability
                test_model = xgb.XGBRegressor(
                    tree_method='hist', device='cuda',
                    n_estimators=1, verbosity=0
                )
                test_model.fit(
                    np.random.rand(10, 5),
                    np.random.rand(10)
                )
                return {'tree_method': 'hist', 'device': 'cuda', 'gpu_id': gpu_id}
            except Exception:
                pass
    except Exception:
        pass

    return {}


def _get_catboost_gpu_config() -> Dict[str, Any]:
    """Get CatBoost GPU configuration."""
    try:
        from CONFIG.config_loader import get_cfg

        test_enabled = get_cfg('gpu.catboost.test_enabled', default=True, config_name='gpu_config')
        gpu_id = get_cfg('gpu.catboost.devices', default='0', config_name='gpu_config')

        if test_enabled:
            from catboost import CatBoostRegressor
            try:
                test_model = CatBoostRegressor(
                    task_type='GPU', devices=gpu_id,
                    iterations=1, verbose=False
                )
                test_model.fit(
                    np.random.rand(10, 5),
                    np.random.rand(10)
                )
                return {'task_type': 'GPU', 'devices': gpu_id}
            except Exception:
                pass
    except Exception:
        pass

    return {}


def clean_model_config(
    estimator_class: type,
    config: Dict[str, Any],
    extra: Dict[str, Any],
    family: str
) -> Dict[str, Any]:
    """Clean model configuration for an estimator.

    Args:
        estimator_class: The model class to instantiate
        config: Raw configuration dictionary
        extra: Extra parameters to add (e.g., random_seed)
        family: Model family name for logging

    Returns:
        Cleaned configuration dictionary
    """
    try:
        from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
        return clean_config_for_estimator(estimator_class, config, extra, family)
    except ImportError:
        # Fallback: just merge configs
        result = config.copy()
        result.update(extra)
        return result
