# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Reproducibility Tracker Mixins.

Provides modular mixins for the ReproducibilityTracker class:
- IndexManagerMixin: Global index.parquet management
- CohortManagerMixin: Cohort ID computation and directory management
- ComparisonEngineMixin: Run comparison and drift computation
- LoggingAPIMixin: Main public logging API (log_comparison, log_run)

Usage:
    from TRAINING.orchestration.utils.repro_tracker_mixins import (
        IndexManagerMixin,
        CohortManagerMixin,
        ComparisonEngineMixin,
        LoggingAPIMixin,
    )

    class ReproducibilityTracker(
        IndexManagerMixin,
        CohortManagerMixin,
        ComparisonEngineMixin,
        LoggingAPIMixin,
        ...
    ):
        ...
"""

from .index_manager import IndexManagerMixin
from .cohort_manager import CohortManagerMixin
from .comparison_engine import ComparisonEngineMixin
from .logging_api import LoggingAPIMixin

__all__ = [
    "IndexManagerMixin",
    "CohortManagerMixin",
    "ComparisonEngineMixin",
    "LoggingAPIMixin",
]
