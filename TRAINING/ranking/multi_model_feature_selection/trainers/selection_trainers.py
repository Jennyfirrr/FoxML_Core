# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature selection method trainers for multi-model feature selection.

These are not traditional models but feature selection methods that
produce importance scores for the multi-model aggregation.

Includes: Mutual Information, Univariate Selection, RFE, Boruta, Stability Selection
"""

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import (
    TrainerResult,
    TaskType,
    detect_task_type,
    register_trainer,
)

logger = logging.getLogger(__name__)


def _boruta_to_importance(
    support: np.ndarray,
    support_weak=None,
    ranking=None,
    n_features=None,
):
    """
    Build a robust importance vector from Boruta outputs.

    Returns values: confirmed=1.0, tentative=0.3, rejected=0.0
    Returns None if truly no signal (no confirmed, no tentative).
    """
    support = np.asarray(support, dtype=bool)
    if n_features is None:
        n_features = support.shape[0]

    if support_weak is None:
        support_weak = np.zeros_like(support, dtype=bool)
    else:
        support_weak = np.asarray(support_weak, dtype=bool)

    # Defensive: ensure correct length
    if support.shape[0] != n_features:
        raise ValueError(f"support length {support.shape[0]} != n_features {n_features}")
    if support_weak.shape[0] != n_features:
        raise ValueError(f"support_weak length {support_weak.shape[0]} != n_features {n_features}")

    has_confirmed = support.any()
    has_tentative = support_weak.any()

    # Nothing selected → no signal
    if not has_confirmed and not has_tentative:
        return None

    # Initialize with zeros (rejected)
    importance = np.zeros(n_features, dtype=float)

    # Assign importance: confirmed=1.0, tentative=0.3
    importance[support] = 1.0
    importance[support_weak] = 0.3

    # Handle overlap (confirmed takes precedence)
    overlap = support & support_weak
    if overlap.any():
        importance[overlap] = 1.0

    if importance.sum() <= 0:
        return None

    return importance


def _normalize_importance_local(
    raw_importance,
    n_features: int,
    family: str,
    feature_names=None
):
    """
    Local helper to normalize importance vectors with fallback handling.

    Handles None, NaN, inf, shape mismatches, and all-zero vectors.
    Returns (normalized_importance, fallback_reason).
    """
    uniform_importance = 1e-6

    # Handle None / empty
    if raw_importance is None:
        importance = np.full(n_features, uniform_importance, dtype=float)
        return importance, f"{family}:fallback_uniform"

    # Convert to numpy array
    if isinstance(raw_importance, pd.Series):
        importance = raw_importance.values
    else:
        importance = np.asarray(raw_importance, dtype=float)

    # Flatten if needed
    importance = importance.flatten()

    # Clean NaN / inf
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad / truncate to match feature count
    if importance.size < n_features:
        importance = np.pad(importance, (0, n_features - importance.size), constant_values=0.0)
    elif importance.size > n_features:
        importance = importance[:n_features]

    # Fallback if truly no signal (all zeros)
    if not np.any(importance > 0):
        importance = np.full(n_features, uniform_importance, dtype=float)
        return importance, f"{family}:fallback_uniform_no_signal"

    return importance, None


class DummyModel:
    """Dummy model for selection methods that don't train actual models."""

    def __init__(self, importance, fallback_reason=None):
        self.importance = importance
        self._fallback_reason = fallback_reason

    def get_feature_importance(self):
        return self.importance


@register_trainer('mutual_information')
def train_mutual_information(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Compute mutual information importance scores.

    Mutual information measures the dependency between each feature and the target.
    Higher values indicate stronger relationships.

    Args:
        model_config: Configuration (not used)
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments (ignored)

    Returns:
        TrainerResult with dummy model containing importance scores
    """
    try:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X

        # Mutual information doesn't support NaN
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        if is_classification:
            importance_values = mutual_info_classif(
                X_dense, y, random_state=model_seed, discrete_features='auto'
            )
        else:
            importance_values = mutual_info_regression(
                X_dense, y, random_state=model_seed, discrete_features='auto'
            )

        model = DummyModel(importance_values)
        train_score = 0.0  # No model to score

        # Create importance Series
        importance = pd.Series(importance_values, index=feature_names_dense)

        logger.debug(f"    mutual_information: computed for {len(feature_names_dense)} features")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Mutual Information failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('univariate_selection')
def train_univariate_selection(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Compute univariate selection scores (f_classif/f_regression).

    Uses F-statistics to measure the relationship between each feature and target.

    Args:
        model_config: Configuration (not used)
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed (not used - univariate is deterministic)
        **kwargs: Additional arguments (ignored)

    Returns:
        TrainerResult with dummy model containing importance scores
    """
    try:
        from sklearn.feature_selection import f_regression, f_classif
        from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X

        # Univariate selection doesn't support NaN
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Suppress division by zero warnings (expected for zero-variance features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if is_classification:
                scores, pvalues = f_classif(X_dense, y)
            else:
                scores, pvalues = f_regression(X_dense, y)

        # Handle NaN/inf in scores
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Use absolute values for ranking (negative correlations still useful)
        abs_scores = np.abs(scores)

        # Normalize scores
        max_score = np.max(abs_scores)
        if max_score > 0:
            importance_values = abs_scores / max_score
        else:
            importance_values = abs_scores

        # Log if we had negative scores
        if np.any(scores < 0):
            n_negative = np.sum(scores < 0)
            logger.debug(
                f"    univariate_selection: {n_negative}/{len(scores)} features "
                f"had negative F-statistics, using absolute values"
            )

        model = DummyModel(importance_values)
        train_score = 0.0

        logger.debug(f"    univariate_selection: computed for {len(feature_names_dense)} features")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Univariate Selection failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


# Complex selection methods extraction status
_RFE_FULLY_EXTRACTED = True  # RFE is now fully extracted
_BORUTA_FULLY_EXTRACTED = True  # Boruta is now fully extracted
_STABILITY_FULLY_EXTRACTED = True  # Stability Selection is now fully extracted


@register_trainer('rfe')
def train_rfe(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Compute Recursive Feature Elimination rankings.

    RFE recursively removes features and builds a model with remaining features,
    ranking features by when they were eliminated.

    Args:
        model_config: Configuration dict with:
            - n_features_to_select: Number of features to select (default: 20% of features)
            - step: Number of features to remove per iteration (default: 5)
            - estimator_n_estimators: Trees in RandomForest estimator (default: 100)
            - estimator_max_depth: Max depth of trees (default: 10)
            - estimator_n_jobs: Jobs for RandomForest (default: 1)
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments (target_column, symbol used for seed generation)

    Returns:
        TrainerResult with DummyModel containing importance scores
    """
    try:
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Load config from SST with fallbacks
        try:
            from CONFIG.config_loader import get_cfg
            rfe_cfg = get_cfg(
                "preprocessing.multi_model_feature_selection.rfe",
                default={},
                config_name="preprocessing_config"
            )
        except Exception:
            rfe_cfg = {}

        # Extract config values
        default_n_features = max(1, int(0.2 * len(feature_names)))
        n_features_to_select = min(
            model_config.get('n_features_to_select', rfe_cfg.get('n_features_to_select', default_n_features)),
            len(feature_names)
        )
        step = model_config.get('step', rfe_cfg.get('step', 5))
        estimator_n_estimators = model_config.get(
            'estimator_n_estimators',
            rfe_cfg.get('estimator_n_estimators', 100)
        )
        estimator_max_depth = model_config.get(
            'estimator_max_depth',
            rfe_cfg.get('estimator_max_depth', 10)
        )
        estimator_n_jobs = model_config.get(
            'estimator_n_jobs',
            rfe_cfg.get('estimator_n_jobs', 1)
        )

        # Create estimator based on task type
        if is_classification:
            estimator = RandomForestClassifier(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=model_seed,
                n_jobs=estimator_n_jobs
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=estimator_n_estimators,
                max_depth=estimator_max_depth,
                random_state=model_seed,
                n_jobs=estimator_n_jobs
            )

        # Create and fit RFE selector
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)

        # Use threading utilities if available
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('RandomForest', total_threads=default_threads())
            estimator.set_params(n_jobs=plan['OMP'])
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                selector.fit(X, y)
        except ImportError:
            selector.fit(X, y)

        # Convert ranking to importance (lower rank = more important)
        ranking = selector.ranking_

        if ranking is None or len(ranking) == 0:
            # Fallback: uniform importance
            importance_values = np.ones(len(feature_names)) * 1e-6
        else:
            # Convert ranking to importance: 1/rank
            importance_values = 1.0 / (ranking + 1e-6)
            # Normalize
            max_importance = np.max(importance_values)
            if max_importance > 0:
                importance_values = importance_values / max_importance

        model = DummyModel(importance_values)
        train_score = 0.0  # No model to score for selection methods

        logger.debug(f"    rfe: selected {n_features_to_select} features, "
                    f"step={step}, estimator_n_estimators={estimator_n_estimators}")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"RFE failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('boruta')
def train_boruta(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Run Boruta feature selection algorithm.

    Boruta is a wrapper method that uses shadow features to determine
    feature importance relative to random noise. It identifies all-relevant
    features, not just the best subset.

    Args:
        model_config: Configuration dict with:
            - enabled: Whether Boruta is enabled (default: True)
            - n_estimators: Trees in ExtraTrees estimator (default: 300)
            - max_depth: Max depth of trees (default: 6)
            - max_iter: Maximum iterations (default: 50)
            - perc: Percentile threshold for shadow comparison (default: 95)
            - n_jobs: Parallel jobs (default: 1)
            - max_features_threshold: Max features before skipping (default: 200)
            - max_samples_threshold: Max samples before subsampling (default: 20000)
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments (target_column, symbol for seed)

    Returns:
        TrainerResult with DummyModel containing Boruta importance scores
    """
    import math
    import time

    try:
        # Check if Boruta is enabled
        try:
            from CONFIG.config_loader import get_cfg
            boruta_cfg = get_cfg(
                "preprocessing.multi_model_feature_selection.boruta",
                default={},
                config_name="preprocessing_config"
            )
        except Exception:
            boruta_cfg = {}

        boruta_enabled = model_config.get('enabled', boruta_cfg.get('enabled', True))
        if not boruta_enabled:
            logger.info("    boruta: SKIPPED - disabled in config")
            return TrainerResult(
                model=DummyModel(np.zeros(len(feature_names))),
                train_score=0.0,
                error="boruta_disabled"
            )

        # Check dataset size thresholds
        max_features_threshold = boruta_cfg.get('max_features_threshold', 200)
        max_samples_threshold = boruta_cfg.get('max_samples_threshold', 20000)
        n_features = len(feature_names) if feature_names else X.shape[1]
        n_samples = len(y) if y is not None else X.shape[0]

        if n_features > max_features_threshold:
            logger.info(f"    boruta: SKIPPED - too many features ({n_features} > {max_features_threshold})")
            return TrainerResult(
                model=DummyModel(np.zeros(len(feature_names))),
                train_score=0.0,
                error=f"boruta_skipped_features_{n_features}"
            )

        # Import Boruta
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
        except ImportError as e:
            logger.error(f"Boruta not available: {e}")
            return TrainerResult(model=None, train_score=0.0, error=f"import_error: {e}")

        # Make data sklearn-safe (impute NaNs)
        X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Subsample large datasets
        subsample_cfg = boruta_cfg.get('subsample_large_datasets', {})
        subsample_enabled = subsample_cfg.get('enabled', True)
        subsample_threshold = subsample_cfg.get('threshold', 10000)
        subsample_max_samples = subsample_cfg.get('max_samples', 10000)

        if subsample_enabled and n_samples > subsample_threshold:
            from sklearn.model_selection import train_test_split
            try:
                if is_classification:
                    X_dense, _, y, _ = train_test_split(
                        X_dense, y,
                        train_size=min(subsample_max_samples, n_samples),
                        stratify=y,
                        random_state=model_seed
                    )
                else:
                    indices = np.random.RandomState(model_seed).choice(
                        n_samples, size=min(subsample_max_samples, n_samples), replace=False
                    )
                    X_dense = X_dense[indices]
                    y = y[indices]
                logger.info(f"    boruta: Subsampled {n_samples} → {len(y)} samples")
            except Exception as e:
                logger.debug(f"    boruta: Subsampling failed: {e}, using full dataset")

        # Extract config values
        boruta_n_estimators = model_config.get('n_estimators', boruta_cfg.get('n_estimators', 300))
        boruta_max_depth = model_config.get('max_depth', boruta_cfg.get('max_depth', 6))
        boruta_max_iter = model_config.get('max_iter', boruta_cfg.get('max_iter', 50))
        boruta_n_jobs = model_config.get('n_jobs', boruta_cfg.get('n_jobs', 1))
        boruta_verbose = model_config.get('verbose', boruta_cfg.get('verbose', 0))
        boruta_perc = model_config.get('perc', 95)

        # Adaptive max_iter based on dataset size
        adaptive_cfg = boruta_cfg.get('adaptive_max_iter', {})
        if adaptive_cfg.get('enabled', True):
            current_n_samples = len(y)
            small_threshold = adaptive_cfg.get('small_dataset_threshold', 5000)
            if current_n_samples < small_threshold:
                boruta_max_iter = adaptive_cfg.get('small_dataset_max_iter', 30)
            elif current_n_samples < adaptive_cfg.get('medium_dataset_threshold', 20000):
                boruta_max_iter = adaptive_cfg.get('medium_dataset_max_iter', 50)
            else:
                boruta_max_iter = adaptive_cfg.get('large_dataset_max_iter', 75)

        # Create ExtraTrees estimator
        from TRAINING.common.utils.config_cleaner import clean_config_for_estimator

        if is_classification:
            et_config = {
                'n_estimators': boruta_n_estimators,
                'max_depth': boruta_max_depth,
                'n_jobs': boruta_n_jobs,
                'class_weight': 'balanced_subsample' if task_type == TaskType.BINARY else 'balanced'
            }
            et_config_clean = clean_config_for_estimator(
                ExtraTreesClassifier, et_config,
                extra_kwargs={'random_state': model_seed},
                family_name='boruta_et'
            )
            base_estimator = ExtraTreesClassifier(**et_config_clean, random_state=model_seed)
        else:
            et_config = {
                'n_estimators': boruta_n_estimators,
                'max_depth': boruta_max_depth,
                'n_jobs': boruta_n_jobs
            }
            et_config_clean = clean_config_for_estimator(
                ExtraTreesRegressor, et_config,
                extra_kwargs={'random_state': model_seed},
                family_name='boruta_et'
            )
            base_estimator = ExtraTreesRegressor(**et_config_clean, random_state=model_seed)

        # Create and fit Boruta
        boruta = BorutaPy(
            base_estimator,
            n_estimators='auto',
            verbose=boruta_verbose,
            random_state=model_seed,
            max_iter=boruta_max_iter,
            perc=boruta_perc
        )

        fit_start = time.time()

        # Use threading utilities if available
        try:
            from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
            plan = plan_for_family('RandomForest', total_threads=default_threads())
            with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                boruta.fit(X_dense, y)
        except ImportError:
            boruta.fit(X_dense, y)

        fit_elapsed = time.time() - fit_start
        n_iterations = getattr(boruta, 'n_iter_', boruta_max_iter)
        logger.info(f"    boruta: Fit completed in {fit_elapsed:.1f}s ({n_iterations} iterations)")

        # Extract results
        ranking = boruta.ranking_
        selected = boruta.support_
        tentative_mask = (ranking == 2)

        n_confirmed = selected.sum()
        n_tentative = tentative_mask.sum()
        n_rejected = (ranking > 2).sum()
        logger.info(f"    boruta: {n_confirmed} confirmed, {n_rejected} rejected, {n_tentative} tentative")

        # Convert to importance vector
        importance_values = _boruta_to_importance(
            support=selected,
            support_weak=tentative_mask,
            ranking=ranking,
            n_features=X_dense.shape[1]
        )

        # Handle no-signal case
        fallback_reason = None
        if importance_values is None:
            logger.debug("    boruta: No stable features identified, using uniform fallback")
            importance_values, fallback_reason = _normalize_importance_local(
                raw_importance=None,
                n_features=X_dense.shape[1],
                family="boruta",
                feature_names=feature_names_dense
            )

        # Create model with Boruta metadata
        class BorutaDummyModel(DummyModel):
            def __init__(self, importance, fallback_reason, selected, tentative_mask, ranking):
                super().__init__(importance, fallback_reason)
                self._selected = selected
                self._tentative_mask = tentative_mask
                self._ranking = ranking

            def get_boruta_labels(self):
                return {
                    'confirmed': self._selected,
                    'tentative': self._tentative_mask,
                    'rejected': self._ranking > 2 if self._ranking is not None else None,
                    'ranking': self._ranking
                }

        model = BorutaDummyModel(importance_values, fallback_reason, selected, tentative_mask, ranking)
        train_score = math.nan  # Boruta is not a predictive model

        return TrainerResult(model=model, train_score=train_score)

    except ImportError as e:
        logger.error(f"Boruta import failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=f"import_error: {e}")
    except Exception as e:
        logger.error(f"Boruta failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))


@register_trainer('stability_selection')
def train_stability_selection(
    model_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_seed: int,
    **kwargs
) -> TrainerResult:
    """Run stability selection for feature importance.

    Stability selection uses bootstrap resampling to identify features that
    are consistently selected across subsamples using L1-regularized models.

    Args:
        model_config: Configuration dict with:
            - n_bootstrap: Number of bootstrap iterations (default: 50)
            - n_splits: Number of CV splits (default: 3)
            - Cs: Number of C values for LogisticRegressionCV (default: 10)
            - max_iter: Max iterations for fitting (default: 1000)
            - n_jobs: Parallel jobs (default: 1)
        X: Feature matrix
        y: Target array
        feature_names: Feature names
        model_seed: Random seed for reproducibility
        **kwargs: Additional arguments:
            - target_column: Target column name (for purge calculation)
            - data_interval_minutes: Data interval in minutes (for purge)

    Returns:
        TrainerResult with DummyModel containing stability scores
    """
    try:
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        # Extract kwargs needed for purge calculation
        target_column = kwargs.get('target_column')
        data_interval_minutes = kwargs.get('data_interval_minutes')

        # Determine task type
        task_type = detect_task_type(y)
        is_classification = task_type in (TaskType.BINARY, TaskType.MULTICLASS)

        # Set up purged cross-validation to prevent temporal leakage
        try:
            from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
            from TRAINING.ranking.utils.purge import get_purge_overlap_bars

            leakage_config = _load_leakage_config()
            target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
            purge_overlap = get_purge_overlap_bars(target_horizon_minutes, data_interval_minutes)

            n_splits = model_config.get('n_splits', 3)
            purged_cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_overlap=purge_overlap)
        except ImportError as e:
            logger.warning(f"Could not set up purged CV: {e}, using default CV")
            purged_cv = 3  # Fallback to simple 3-fold CV

        # Extract config values
        n_bootstrap = model_config.get('n_bootstrap', 50)
        stability_cs = model_config.get('Cs', 10)
        stability_max_iter = model_config.get('max_iter', 1000)
        stability_n_jobs = model_config.get('n_jobs', 1)

        # Initialize stability scores
        stability_scores = np.zeros(X.shape[1])

        # Create seeded RNG for deterministic bootstrap sampling
        bootstrap_rng = np.random.RandomState(model_seed)

        # Bootstrap iterations
        successful_iterations = 0
        for _ in range(n_bootstrap):
            indices = bootstrap_rng.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]

            try:
                # Clean config to prevent double seed argument
                from TRAINING.common.utils.config_cleaner import clean_config_for_estimator

                if is_classification:
                    lr_config = {
                        'Cs': stability_cs,
                        'cv': purged_cv,
                        'max_iter': stability_max_iter,
                        'n_jobs': stability_n_jobs
                    }
                    lr_config_clean = clean_config_for_estimator(
                        LogisticRegressionCV, lr_config,
                        extra_kwargs={'random_state': model_seed},
                        family_name='stability_selection'
                    )
                    model = LogisticRegressionCV(**lr_config_clean, random_state=model_seed)
                else:
                    lasso_config = {
                        'cv': purged_cv,
                        'max_iter': stability_max_iter,
                        'n_jobs': stability_n_jobs
                    }
                    lasso_config_clean = clean_config_for_estimator(
                        LassoCV, lasso_config,
                        extra_kwargs={'random_state': model_seed},
                        family_name='stability_selection'
                    )
                    model = LassoCV(**lasso_config_clean, random_state=model_seed)

                # Use threading utilities if available
                try:
                    from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
                    plan = plan_for_family('Lasso', total_threads=default_threads())
                    if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
                        model.set_params(n_jobs=plan['OMP'])
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        model.fit(X_boot, y_boot)
                except ImportError:
                    model.fit(X_boot, y_boot)

                # Extract coefficients and count non-zero features
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                stability_scores += (np.abs(coef) > 1e-6).astype(int)
                successful_iterations += 1

            except Exception as e:
                logger.debug(f"    stability_selection: Bootstrap iteration failed: {e}")
                continue

        if successful_iterations == 0:
            logger.warning("    stability_selection: All bootstrap iterations failed")
            return TrainerResult(model=None, train_score=0.0, error="All bootstrap iterations failed")

        # Normalize to 0-1 (fraction of times selected)
        raw_importance = stability_scores / successful_iterations

        # Use normalize_importance for edge case handling
        importance_values, fallback_reason = _normalize_importance_local(
            raw_importance=raw_importance,
            n_features=X.shape[1],
            family="stability_selection",
            feature_names=feature_names
        )

        if fallback_reason:
            logger.debug(f"    stability_selection: {fallback_reason}")

        model = DummyModel(importance_values, fallback_reason=fallback_reason)
        train_score = 0.0  # No single model to score

        logger.debug(f"    stability_selection: {successful_iterations}/{n_bootstrap} successful iterations")

        return TrainerResult(model=model, train_score=train_score)

    except Exception as e:
        logger.error(f"Stability selection failed: {e}")
        return TrainerResult(model=None, train_score=0.0, error=str(e))
