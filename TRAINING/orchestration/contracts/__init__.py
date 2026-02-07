# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Stage boundary contracts for pipeline data flow.

This module provides typed data contracts that ensure consistent
interfaces between pipeline stages.

Phase 7: Stage Boundaries
- FeatureSelectionView: Enum for feature selection views
- FeatureSelectionResult: Typed result from feature selection
- filter_target_features_at_boundary: Stage boundary filtering

Phase 8: API Design
- FeatureSelectionRequest: Request object for select_features_for_target()
- RankingRequest: Request object for rank_targets()
"""

from .feature_selection import (
    FeatureSelectionView,
    FeatureSelectionResult,
    filter_target_features_at_boundary,
)

from .requests import (
    FeatureSelectionRequest,
    RankingRequest,
)

__all__ = [
    # Phase 7: Stage Boundaries
    "FeatureSelectionView",
    "FeatureSelectionResult",
    "filter_target_features_at_boundary",
    # Phase 8: API Design
    "FeatureSelectionRequest",
    "RankingRequest",
]
