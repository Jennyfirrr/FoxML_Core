# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Column Exclusion Utilities - Single Source of Truth

This module provides the canonical definition of non-feature column patterns
(target columns, metadata columns) that should be excluded from feature lists.

SST Principle: This is the single source of truth for "what counts as a non-feature column".
Both leakage filtering and registry coverage calculation use these patterns to ensure
consistent behavior and avoid circular imports.
"""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# SST constant - single definition of non-feature patterns
# These patterns identify columns that are definitely NOT features:
# - Target columns (y_*, fwd_ret_*, barrier_*, etc.)
# - Metadata columns (ts, timestamp, symbol, etc.)
# - Forward-looking columns (next_*, future_*)
HARD_CODED_NON_FEATURE_PATTERNS = {
    'prefix_patterns': ['p_', 'y_', 'fwd_ret_', 'ret_zscore_', 'tth_', 'mfe_', 'mdd_', 'barrier_', 'next_', 'future_'],
    'exact_patterns': ['ts', 'timestamp', 'symbol', 'date', 'time']
}


def exclude_non_feature_columns(
    columns: List[str],
    reason: str = "non-feature-exclusion"
) -> Tuple[List[str], List[str]]:
    """
    Exclude non-feature columns (targets, metadata) from a column list.
    
    This function applies the canonical non-feature patterns to filter out columns
    that should never be counted as features (target columns, metadata, etc.).
    
    Args:
        columns: List of column names to filter
        reason: Label for logging (optional, for diagnostics)
    
    Returns:
        Tuple of (kept_columns, excluded_columns)
        - kept_columns: Columns that passed the filter (may still not be in registry)
        - excluded_columns: Columns that matched non-feature patterns
    
    Note:
        This does NOT verify that kept columns are valid features - that's handled
        by the registry check. This only filters out columns that are definitely
        NOT features (targets, metadata).
    """
    excluded = []
    patterns = HARD_CODED_NON_FEATURE_PATTERNS
    
    # Apply prefix patterns
    for prefix in patterns.get('prefix_patterns', []):
        for col in columns:
            if col not in excluded and col.startswith(prefix):
                excluded.append(col)
    
    # Apply exact patterns
    exact_set = set(patterns.get('exact_patterns', []))
    for col in columns:
        if col not in excluded and col in exact_set:
            excluded.append(col)
    
    # Apply regex patterns (if any)
    for pattern in patterns.get('regex_patterns', []):
        try:
            regex = re.compile(pattern)
            for col in columns:
                if col not in excluded and regex.match(col):
                    excluded.append(col)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}' in {reason}: {e}")
    
    # Apply keyword patterns (substring match, case-insensitive) - if any
    for keyword in patterns.get('keyword_patterns', []):
        keyword_lower = keyword.lower()
        for col in columns:
            if col not in excluded and keyword_lower in col.lower():
                excluded.append(col)
    
    # Return kept columns and excluded columns
    kept = [col for col in columns if col not in excluded]
    return kept, excluded
