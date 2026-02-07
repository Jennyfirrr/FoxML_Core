# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-Sectional Target Construction

This package provides functions for computing cross-sectionally normalized
targets suitable for ranking-based training objectives.

Target Types:
    - **cs_percentile**: Percentile rank [0, 1], recommended for ranking losses
    - **cs_zscore**: Robust z-score, preserves magnitude information
    - **vol_scaled**: Volatility-adjusted cross-sectional returns

Usage:
    from TRAINING.common.targets import (
        compute_cs_target,
        compute_cs_percentile_target,
        compute_cs_zscore_target,
        compute_vol_scaled_cs_target,
        CrossSectionalTargetType,
        validate_no_future_leakage,
        validate_cs_target_quality,
    )

    # Recommended: use unified interface
    y = compute_cs_target(df, target_type="cs_percentile", return_col="fwd_ret_5m")

    # Or use specific functions directly
    y = compute_cs_percentile_target(df, return_col="fwd_ret_5m")

    # Validate targets
    validate_no_future_leakage(df, target_col="cs_target")

See:
    - .claude/plans/cross-sectional-ranking-objective.md (master plan)
    - .claude/plans/cs-ranking-phase1-targets.md (this phase)
"""

from TRAINING.common.targets.cross_sectional import (
    CrossSectionalTargetType,
    compute_cs_percentile_target,
    compute_cs_target,
    compute_cs_zscore_target,
    compute_vol_scaled_cs_target,
)
from TRAINING.common.targets.validators import (
    validate_cs_target_quality,
    validate_no_future_leakage,
    validate_vol_column_no_leakage,
)

__all__ = [
    # Target types
    "CrossSectionalTargetType",
    # Target computation functions
    "compute_cs_target",
    "compute_cs_percentile_target",
    "compute_cs_zscore_target",
    "compute_vol_scaled_cs_target",
    # Validators
    "validate_no_future_leakage",
    "validate_cs_target_quality",
    "validate_vol_column_no_leakage",
]
