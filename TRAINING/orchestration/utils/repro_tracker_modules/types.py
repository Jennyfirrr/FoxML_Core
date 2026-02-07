# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Type definitions for reproducibility tracking.

Provides enums and data classes used across the reproducibility tracker modules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class DriftCategory(Enum):
    """Categories for drift classification between runs."""
    STABLE = "stable"
    MINOR_DRIFT = "minor_drift"
    MAJOR_DRIFT = "major_drift"
    DEGRADED = "degraded"
    IMPROVED = "improved"
    NEW = "new"


class ComparisonStatus(Enum):
    """Status of a comparison between runs."""
    MATCHED = "matched"  # Found matching previous run
    NO_MATCH = "no_match"  # No matching run found
    FIRST_RUN = "first_run"  # First run for this cohort
    ERROR = "error"  # Error during comparison


@dataclass
class DriftMetrics:
    """Drift metrics between current and previous run.

    Attributes:
        category: Drift classification
        auc_delta: Change in AUC score
        composite_delta: Change in composite score
        z_score: Standardized score relative to historical distribution
        is_significant: Whether drift exceeds threshold
        details: Additional drift details
    """
    category: DriftCategory
    auc_delta: Optional[float] = None
    composite_delta: Optional[float] = None
    z_score: Optional[float] = None
    is_significant: bool = False
    details: Optional[Dict[str, Any]] = None


@dataclass
class CohortMetadata:
    """Metadata for a reproducibility cohort.

    Attributes:
        cohort_id: Unique identifier for the cohort
        view: Processing view (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
        stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, etc.)
        universe_sig: Universe signature for data scoping
        model_family: Model family name (if applicable)
        target: Target column name
        symbol: Symbol name (for SYMBOL_SPECIFIC view)
        config_fingerprint: Configuration fingerprint
    """
    cohort_id: str
    view: str
    stage: str
    universe_sig: Optional[str] = None
    model_family: Optional[str] = None
    target: Optional[str] = None
    symbol: Optional[str] = None
    config_fingerprint: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing current run with previous run.

    Attributes:
        status: Comparison status
        previous_run_id: ID of the matched previous run
        drift: Drift metrics (if comparison was made)
        cohort_metadata: Cohort metadata
        error_message: Error message (if status is ERROR)
    """
    status: ComparisonStatus
    previous_run_id: Optional[str] = None
    drift: Optional[DriftMetrics] = None
    cohort_metadata: Optional[CohortMetadata] = None
    error_message: Optional[str] = None


# Sample size bins for threshold selection
SAMPLE_SIZE_BINS = {
    "tiny": {"min": 0, "max": 100, "label": "<100"},
    "small": {"min": 100, "max": 1000, "label": "100-1K"},
    "medium": {"min": 1000, "max": 10000, "label": "1K-10K"},
    "large": {"min": 10000, "max": 100000, "label": "10K-100K"},
    "huge": {"min": 100000, "max": float('inf'), "label": ">100K"},
}


def get_sample_size_bin(n_samples: int) -> str:
    """Get the sample size bin for a given sample count.

    Args:
        n_samples: Number of samples

    Returns:
        Bin name (tiny, small, medium, large, huge)
    """
    for bin_name, bin_config in SAMPLE_SIZE_BINS.items():
        if bin_config["min"] <= n_samples < bin_config["max"]:
            return bin_name
    return "huge"  # Default to huge if not found
