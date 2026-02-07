# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cohort management for reproducibility tracking.

Provides functions for:
- Computing cohort IDs from metadata
- Creating and managing cohort directories
- Extracting cohort metadata from additional_data

Note: The full implementation is currently in the parent reproducibility_tracker.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .types import CohortMetadata

logger = logging.getLogger(__name__)


def compute_cohort_id(
    view: str,
    stage: str,
    universe_sig: Optional[str] = None,
    model_family: Optional[str] = None,
    target: Optional[str] = None,
    symbol: Optional[str] = None,
    config_fingerprint: Optional[str] = None,
) -> str:
    """Compute a cohort ID from metadata.

    The cohort ID uniquely identifies a group of comparable runs.
    It's computed from the relevant metadata fields that define
    what makes two runs comparable.

    Args:
        view: Processing view (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
        stage: Pipeline stage
        universe_sig: Universe signature
        model_family: Model family name
        target: Target column name
        symbol: Symbol name
        config_fingerprint: Configuration fingerprint

    Returns:
        Cohort ID string (hex hash)
    """
    # Import from parent module
    from ..reproducibility_tracker import ReproducibilityTracker

    # Create a tracker instance to use the method
    # This is a workaround until full extraction
    tracker = ReproducibilityTracker.__new__(ReproducibilityTracker)
    tracker._cohort_config_keys = [
        'view', 'stage', 'universe_sig', 'model_family',
        'target', 'symbol', 'config_fingerprint'
    ]

    # Build data dict
    data = {
        'view': view,
        'stage': stage,
        'universe_sig': universe_sig,
        'model_family': model_family,
        'target': target,
        'symbol': symbol,
        'config_fingerprint': config_fingerprint,
    }

    return tracker._compute_cohort_id(data)


def get_cohort_metadata(
    additional_data: Dict[str, Any],
    stage: str,
    view: Optional[str] = None,
) -> CohortMetadata:
    """Extract cohort metadata from additional_data dict.

    Args:
        additional_data: Dictionary with run metadata
        stage: Pipeline stage
        view: Override view (uses additional_data['view'] if not provided)

    Returns:
        CohortMetadata with extracted fields
    """
    return CohortMetadata(
        cohort_id="",  # Will be computed
        view=view or additional_data.get('view', 'CROSS_SECTIONAL'),
        stage=stage,
        universe_sig=additional_data.get('universe_sig'),
        model_family=additional_data.get('model_family'),
        target=additional_data.get('target'),
        symbol=additional_data.get('symbol'),
        config_fingerprint=additional_data.get('config_fingerprint'),
    )


__all__ = [
    'compute_cohort_id',
    'get_cohort_metadata',
]
