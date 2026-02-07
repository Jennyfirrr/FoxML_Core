# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Comparison and drift detection for reproducibility tracking.

Provides functions for:
- Computing drift between current and previous runs
- Sample-adjusted statistical comparisons
- Finding comparable previous runs from index

Note: The full implementation is currently in the parent reproducibility_tracker.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .types import DriftCategory, DriftMetrics, ComparisonStatus, ComparisonResult

logger = logging.getLogger(__name__)


# Classification threshold constants
Z_SCORE_STABLE = 1.0
Z_SCORE_DRIFTING = 2.0


def classify_by_z_score(z_score: Optional[float]) -> str:
    """Classify drift status based on z-score.

    Args:
        z_score: Z-score of the difference between runs

    Returns:
        Classification string: 'STABLE', 'DRIFTING', or 'DIVERGED'
    """
    if z_score is None:
        return 'UNKNOWN'
    if z_score < Z_SCORE_STABLE:
        return 'STABLE'
    elif z_score < Z_SCORE_DRIFTING:
        return 'DRIFTING'
    else:
        return 'DIVERGED'


def compute_sample_adjusted_z_score(
    prev_value: float,
    curr_value: float,
    prev_n: int,
    curr_n: int,
    prev_std: Optional[float] = None,
    curr_std: Optional[float] = None,
) -> Optional[float]:
    """Compute sample-adjusted z-score for comparing two runs.

    For AUC metrics, estimates variance as: AUC * (1 - AUC) / N

    Args:
        prev_value: Previous run metric value
        curr_value: Current run metric value
        prev_n: Previous run sample size
        curr_n: Current run sample size
        prev_std: Previous run standard deviation (optional, for fallback)
        curr_std: Current run standard deviation (optional, for fallback)

    Returns:
        Z-score or None if variance cannot be estimated
    """
    # Sample-adjusted variance estimation for AUC
    if prev_value > 0 and prev_value < 1:
        var_prev = prev_value * (1 - prev_value) / prev_n
    elif prev_std and prev_std > 0:
        var_prev = (prev_std ** 2) / prev_n
    else:
        var_prev = None

    if curr_value > 0 and curr_value < 1:
        var_curr = curr_value * (1 - curr_value) / curr_n
    elif curr_std and curr_std > 0:
        var_curr = (curr_std ** 2) / curr_n
    else:
        var_curr = None

    if var_prev is None or var_curr is None:
        return None

    delta = abs(curr_value - prev_value)
    sigma = math.sqrt(var_prev + var_curr)

    if sigma > 0:
        return delta / sigma
    return None


def is_n_ratio_comparable(
    prev_n: int,
    curr_n: int,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """Check if two runs have comparable sample sizes.

    Args:
        prev_n: Previous run sample size
        curr_n: Current run sample size
        threshold: Minimum ratio to be considered comparable

    Returns:
        Tuple of (is_comparable, n_ratio)
    """
    if max(prev_n, curr_n) == 0:
        return False, 0.0
    n_ratio = min(prev_n, curr_n) / max(prev_n, curr_n)
    return n_ratio >= threshold, n_ratio


def compare_within_cohort(
    prev_run: Dict[str, Any],
    curr_run: Dict[str, Any],
    thresholds: Dict[str, Dict[str, float]],
    metric_type: str = 'roc_auc'
) -> Tuple[str, float, float, Optional[float], Dict[str, Any]]:
    """Compare runs within the same cohort using sample-adjusted statistics.

    This is a delegating wrapper that calls the method on ReproducibilityTracker.

    Args:
        prev_run: Previous run data dict
        curr_run: Current run data dict
        thresholds: Comparison thresholds config
        metric_type: Metric type to compare

    Returns:
        Tuple of (classification, abs_diff, rel_diff, z_score, stats_dict)
    """
    from ..reproducibility_tracker import ReproducibilityTracker

    # Create a tracker instance to use the method
    tracker = ReproducibilityTracker.__new__(ReproducibilityTracker)
    tracker.thresholds = thresholds

    return tracker._compare_within_cohort(prev_run, curr_run, metric_type)


def get_last_comparable_run(
    repro_base_dir: Path,
    stage: str,
    target: str,
    view: Optional[str] = None,
    symbol: Optional[str] = None,
    model_family: Optional[str] = None,
    cohort_id: Optional[str] = None,
    current_N: Optional[int] = None,
    n_ratio_threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    """Find the last comparable run from index.parquet.

    This is a delegating wrapper that calls the method on ReproducibilityTracker.

    Args:
        repro_base_dir: Base directory for reproducibility files
        stage: Pipeline stage
        target: Target/item name
        view: Route type (CROSS_SECTIONAL/INDIVIDUAL)
        symbol: Symbol name (for INDIVIDUAL mode)
        model_family: Model family (for TRAINING)
        cohort_id: Cohort ID (if already computed)
        current_N: Current n_effective (for N ratio check)
        n_ratio_threshold: Override default N ratio threshold

    Returns:
        Previous run metrics dict or None if no comparable run found
    """
    from ..reproducibility_tracker import ReproducibilityTracker

    # Create a tracker instance with the provided directory
    tracker = ReproducibilityTracker.__new__(ReproducibilityTracker)
    tracker._repro_base_dir = repro_base_dir
    tracker.n_ratio_threshold = n_ratio_threshold

    return tracker.get_last_comparable_run(
        stage=stage,
        target=target,
        view=view,
        symbol=symbol,
        model_family=model_family,
        cohort_id=cohort_id,
        current_N=current_N,
        n_ratio_threshold=n_ratio_threshold
    )


__all__ = [
    'Z_SCORE_STABLE',
    'Z_SCORE_DRIFTING',
    'classify_by_z_score',
    'compute_sample_adjusted_z_score',
    'is_n_ratio_comparable',
    'compare_within_cohort',
    'get_last_comparable_run',
]
