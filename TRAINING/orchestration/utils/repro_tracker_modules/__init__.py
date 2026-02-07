# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Reproducibility Tracking Module

Tracks and compares run results across pipeline stages to verify reproducible behavior.
Supports target ranking, feature selection, and other pipeline stages.

Submodules:
    - types: Enums and data types (DriftCategory, ComparisonStatus, etc.)
    - cohort: Cohort directory management and metadata
    - comparison: Run comparison and drift computation (planned)
    - index: Index management and matching (planned)
    - persistence: Save/load operations for run data (planned)

Usage:
    from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker

    tracker = ReproducibilityTracker(output_dir=Path("results"))
    tracker.log_comparison(
        stage="target_ranking",
        target="y_will_swing_low_15m_0.05",
        metrics={"auc": 0.751, "composite_score": 0.764}
    )
"""

# Import types from submodules (new modular structure)
from .types import (
    DriftCategory,
    ComparisonStatus,
    DriftMetrics,
    CohortMetadata,
    ComparisonResult,
    SAMPLE_SIZE_BINS,
    get_sample_size_bin,
)

# Import cohort functions from submodules
from .cohort import (
    compute_cohort_id,
    get_cohort_metadata,
)

# Import comparison functions from submodules
from .comparison import (
    Z_SCORE_STABLE,
    Z_SCORE_DRIFTING,
    classify_by_z_score,
    compute_sample_adjusted_z_score,
    is_n_ratio_comparable,
    compare_within_cohort,
    get_last_comparable_run,
)

# Import persistence functions from submodules
from .persistence import (
    load_previous_run,
    save_run,
    read_log_file,
    get_run_history,
)

# Import main class and utility functions from parent file
# These will be migrated to submodules incrementally
from ..reproducibility_tracker import (
    ReproducibilityTracker,
    _normalize_view_for_comparison,
    _normalize_stage_for_comparison,
    _write_atomic_json_with_lock,
    _construct_comparison_group_key_from_dict,
    _extract_horizon_minutes_sst,
)

__all__ = [
    # Types
    'DriftCategory',
    'ComparisonStatus',
    'DriftMetrics',
    'CohortMetadata',
    'ComparisonResult',
    'SAMPLE_SIZE_BINS',
    'get_sample_size_bin',
    # Cohort functions
    'compute_cohort_id',
    'get_cohort_metadata',
    # Comparison functions
    'Z_SCORE_STABLE',
    'Z_SCORE_DRIFTING',
    'classify_by_z_score',
    'compute_sample_adjusted_z_score',
    'is_n_ratio_comparable',
    'compare_within_cohort',
    'get_last_comparable_run',
    # Persistence functions
    'load_previous_run',
    'save_run',
    'read_log_file',
    'get_run_history',
    # Main class and utilities (from parent)
    'ReproducibilityTracker',
    '_normalize_view_for_comparison',
    '_normalize_stage_for_comparison',
    '_write_atomic_json_with_lock',
    '_construct_comparison_group_key_from_dict',
    '_extract_horizon_minutes_sst',
]
