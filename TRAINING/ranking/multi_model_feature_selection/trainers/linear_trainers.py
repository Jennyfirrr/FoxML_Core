# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Linear model trainers for multi-model feature selection.

Includes: Lasso, Ridge, ElasticNet, LogisticRegression
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


def _make_dense_array(X: np.ndarray, feature_names: List[str]):
    """Convert sparse/nan array to dense for sklearn linear models."""
    try:
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        return make_sklearn_dense_X(X, feature_names)
    except ImportError:
        # Fallback: simple nan handling
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        return imputer.fit_transform(X), feature_names


@register_trainer('lasso')
def train_lasso(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train a Lasso model."""
    try:
        from sklearn.linear_model import Lasso

        # Lasso doesn't handle NaNs
        X_dense, feature_names_dense = _make_dense_array(X, feature_names)

        # Clean config (Lasso is deterministic, no random_state)
        lasso_config = clean_model_config(Lasso, model_config, {}, "lasso")

        model = Lasso(**lasso_config)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('Lasso', total_threads=default_threads())
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_dense, y)
        except ImportError:
            model.fit(X_dense, y)

        train_score = model.score(X_dense, y)
        logger.debug(f"    lasso: R²={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Lasso failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('ridge')
def train_ridge(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train a Ridge model."""
    try:
        from sklearn.linear_model import Ridge, RidgeClassifier

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Ridge doesn't handle NaNs
        X_dense, _ = _make_dense_array(X, feature_names)

        # Select estimator
        est_cls = RidgeClassifier if is_classification else Ridge

        # Clean config
        extra = {"random_state": model_seed}
        ridge_config = clean_model_config(est_cls, model_config, extra, "ridge")

        model = est_cls(**ridge_config)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('Ridge', total_threads=default_threads())
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_dense, y)
        except ImportError:
            model.fit(X_dense, y)

        train_score = model.score(X_dense, y)
        metric_name = 'accuracy' if is_classification else 'R²'
        logger.debug(f"    ridge: {metric_name}={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Ridge failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('elastic_net')
def train_elastic_net(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train an ElasticNet model."""
    try:
        from sklearn.linear_model import ElasticNet

        # ElasticNet doesn't handle NaNs
        X_dense, _ = _make_dense_array(X, feature_names)

        # Clean config
        extra = {"random_state": model_seed}
        en_config = clean_model_config(ElasticNet, model_config, extra, "elastic_net")

        model = ElasticNet(**en_config)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('ElasticNet', total_threads=default_threads())
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_dense, y)
        except ImportError:
            model.fit(X_dense, y)

        train_score = model.score(X_dense, y)
        logger.debug(f"    elastic_net: R²={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"ElasticNet failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('logistic_regression')
def train_logistic_regression(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Train a Logistic Regression model."""
    try:
        from sklearn.linear_model import LogisticRegression

        # LogisticRegression doesn't handle NaNs
        X_dense, _ = _make_dense_array(X, feature_names)

        # Clean config
        extra = {"random_state": model_seed}
        lr_config = clean_model_config(LogisticRegression, model_config, extra, "logistic_regression")

        model = LogisticRegression(**lr_config)

        # Train with thread management
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('LogisticRegression', total_threads=default_threads())
            lr_config['n_jobs'] = plan['OMP']
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                model.fit(X_dense, y)
        except ImportError:
            model.fit(X_dense, y)

        train_score = model.score(X_dense, y)
        logger.debug(f"    logistic_regression: accuracy={train_score:.4f}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"LogisticRegression failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))
