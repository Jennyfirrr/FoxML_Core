# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Selection stage boundary contract.

This module provides the typed data contract for feature selection output,
ensuring consistent interfaces regardless of routing view (CROSS_SECTIONAL,
SYMBOL_SPECIFIC, BOTH, or BLOCKED).

SB-001 to SB-004: Normalized return types for feature selection
SB-008, SB-009: Consistent empty states and documented structures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FeatureSelectionView(str, Enum):
    """
    Feature selection view types.

    Extends the base View enum to include BOTH and BLOCKED states
    that are specific to feature selection results.
    """
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
    BOTH = "BOTH"
    BLOCKED = "BLOCKED"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_routing_view(cls, view: str) -> "FeatureSelectionView":
        """
        Convert routing view string to FeatureSelectionView.

        Args:
            view: Routing view string ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "BOTH")

        Returns:
            Corresponding FeatureSelectionView
        """
        normalized = view.upper().replace("-", "_").replace(" ", "_")
        if normalized in ("CROSS_SECTIONAL", "CS"):
            return cls.CROSS_SECTIONAL
        elif normalized in ("SYMBOL_SPECIFIC", "SS", "INDIVIDUAL", "LOSO"):
            return cls.SYMBOL_SPECIFIC
        elif normalized == "BOTH":
            return cls.BOTH
        else:
            raise ValueError(f"Unknown routing view: {view}")


@dataclass
class FeatureSelectionResult:
    """
    SST data contract for feature selection stage output.

    Provides consistent interface regardless of routing view.
    This replaces the inconsistent return types:
    - CROSS_SECTIONAL: List[str]
    - SYMBOL_SPECIFIC: Dict[str, List[str]]
    - BOTH: Dict with 'cross_sectional', 'symbol_specific', 'route' keys
    - BLOCKED: []

    All views now return FeatureSelectionResult with typed fields.

    Attributes:
        target: Target column name
        view: The feature selection view used
        cross_sectional_features: Features for cross-sectional view (empty if SS only)
        symbol_specific_features: Features by symbol (empty dict if CS only)
        blocked_reason: Reason for blocking (only set if view == BLOCKED)
        feature_count: Total count of selected features (computed)
    """
    target: str
    view: FeatureSelectionView

    # CROSS_SECTIONAL features (empty list if SYMBOL_SPECIFIC only)
    cross_sectional_features: List[str] = field(default_factory=list)

    # SYMBOL_SPECIFIC features (empty dict if CROSS_SECTIONAL only)
    symbol_specific_features: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    blocked_reason: Optional[str] = None
    feature_count: int = 0

    def __post_init__(self):
        """Compute feature_count after initialization."""
        cs_count = len(self.cross_sectional_features)
        ss_count = sum(len(v) for v in self.symbol_specific_features.values())
        self.feature_count = cs_count + ss_count

    def is_empty(self) -> bool:
        """Check if no features were selected."""
        return self.feature_count == 0

    def is_blocked(self) -> bool:
        """Check if this result represents a blocked target."""
        return self.view == FeatureSelectionView.BLOCKED

    def get_all_features(self) -> List[str]:
        """Get all unique features across all views (sorted for determinism)."""
        features = set(self.cross_sectional_features)
        for symbol_features in self.symbol_specific_features.values():
            features.update(symbol_features)
        return sorted(features)  # DETERMINISM: Return sorted list, not set

    def to_legacy_format(self) -> Any:
        """
        Convert to legacy format for backward compatibility.

        Returns:
            Legacy format:
            - BLOCKED: []
            - CROSS_SECTIONAL: List[str]
            - SYMBOL_SPECIFIC: Dict[str, List[str]]
            - BOTH: {'cross_sectional': List[str], 'symbol_specific': Dict[str, List[str]], 'route': 'BOTH'}
        """
        if self.view == FeatureSelectionView.BLOCKED:
            return []
        elif self.view == FeatureSelectionView.CROSS_SECTIONAL:
            return self.cross_sectional_features
        elif self.view == FeatureSelectionView.SYMBOL_SPECIFIC:
            return self.symbol_specific_features
        else:  # BOTH
            return {
                'cross_sectional': self.cross_sectional_features,
                'symbol_specific': self.symbol_specific_features,
                'route': 'BOTH'
            }

    @classmethod
    def from_legacy_format(
        cls,
        target: str,
        data: Any,
        view_hint: Optional[FeatureSelectionView] = None
    ) -> "FeatureSelectionResult":
        """
        Parse from legacy format with view detection.

        Args:
            target: Target column name
            data: Legacy format data
            view_hint: Optional hint for ambiguous cases

        Returns:
            FeatureSelectionResult with detected view type
        """
        if data is None or (isinstance(data, list) and len(data) == 0):
            return cls(
                target=target,
                view=FeatureSelectionView.BLOCKED,
                blocked_reason="No features selected"
            )
        elif isinstance(data, list):
            return cls(
                target=target,
                view=FeatureSelectionView.CROSS_SECTIONAL,
                cross_sectional_features=data
            )
        elif isinstance(data, dict):
            if 'cross_sectional' in data and 'symbol_specific' in data:
                return cls(
                    target=target,
                    view=FeatureSelectionView.BOTH,
                    cross_sectional_features=data.get('cross_sectional', []),
                    symbol_specific_features=data.get('symbol_specific', {})
                )
            else:
                # Dict without 'cross_sectional' key = SYMBOL_SPECIFIC
                return cls(
                    target=target,
                    view=FeatureSelectionView.SYMBOL_SPECIFIC,
                    symbol_specific_features=data
                )
        else:
            raise ValueError(f"Unknown format for {target}: {type(data)}")

    def validate(self) -> List[str]:
        """
        Validate result and return list of issues.

        Returns:
            List of validation issue strings (empty if valid)
        """
        issues = []

        if self.view == FeatureSelectionView.BLOCKED and not self.blocked_reason:
            issues.append("BLOCKED view should have blocked_reason")

        if self.view == FeatureSelectionView.CROSS_SECTIONAL:
            if self.symbol_specific_features:
                issues.append("CROSS_SECTIONAL should not have symbol_specific_features")

        if self.view == FeatureSelectionView.SYMBOL_SPECIFIC:
            if self.cross_sectional_features:
                issues.append("SYMBOL_SPECIFIC should not have cross_sectional_features")

        if self.view == FeatureSelectionView.BOTH:
            if not self.cross_sectional_features and not self.symbol_specific_features:
                issues.append("BOTH view should have features in at least one field")

        return issues

    def __repr__(self) -> str:
        return (
            f"FeatureSelectionResult(target={self.target!r}, view={self.view.value}, "
            f"feature_count={self.feature_count})"
        )


def filter_target_features_at_boundary(
    target_features: Dict[str, FeatureSelectionResult],
    allowed_targets: Optional[Set[str]] = None,
    skip_blocked: bool = True
) -> Dict[str, FeatureSelectionResult]:
    """
    Filter target features at stage boundary.

    Called BETWEEN stages, not mid-stage. This ensures filtering
    happens at well-defined boundaries with consistent behavior.

    SB-005: Extract post-stage filtering to boundary function

    Args:
        target_features: Dict mapping target names to FeatureSelectionResult
        allowed_targets: Optional set of targets to keep (None = keep all)
        skip_blocked: Whether to skip BLOCKED results

    Returns:
        Filtered dict of target features
    """
    filtered = {}

    # Determine targets to iterate (sorted for determinism)
    targets_to_check = sorted(allowed_targets) if allowed_targets else sorted(target_features.keys())

    for target in targets_to_check:
        if target not in target_features:
            continue

        result = target_features[target]

        # Handle legacy format by converting to typed result
        if not isinstance(result, FeatureSelectionResult):
            result = FeatureSelectionResult.from_legacy_format(target, result)

        if skip_blocked and result.is_blocked():
            logger.debug(f"Skipping {target} (BLOCKED)")
            continue

        filtered[target] = result

    return filtered
