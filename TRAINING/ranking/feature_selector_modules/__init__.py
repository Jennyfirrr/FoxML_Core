# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Selection Module

Multi-model consensus feature selection for ML pipeline.

Submodules:
    - config: Configuration hashing and loading
    - core: Main selection functions

Usage:
    from TRAINING.ranking.feature_selector import select_features_for_target

    features, importance_df = select_features_for_target(
        target_column="y_will_swing_low_15m_0.05",
        symbols=["AAPL", "MSFT"],
        data_dir=Path("data"),
        top_n=50
    )

Note: The full implementation is currently in the parent feature_selector.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

# Import config utilities
from .config import (
    compute_feature_selection_config_hash,
    load_multi_model_config,
)

# Import core functions
from .core import (
    select_features_for_target,
    rank_features_multi_model,
)

# Re-export from parent for backward compatibility
from TRAINING.ranking.feature_selector import FeatureImportanceResult

__all__ = [
    # Config utilities
    'compute_feature_selection_config_hash',
    'load_multi_model_config',
    # Core functions
    'select_features_for_target',
    'rank_features_multi_model',
    # Data types
    'FeatureImportanceResult',
]
