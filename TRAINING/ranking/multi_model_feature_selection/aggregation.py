# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Aggregation module for multi-model feature selection.

This module provides functions for aggregating feature importance scores
across multiple model families and symbols.

Note: The main implementation is currently in the parent multi_model_feature_selection.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .types import ImportanceResult

logger = logging.getLogger(__name__)


def aggregate_multi_model_importance(
    all_results: Optional[List[ImportanceResult]] = None,
    model_families_config: Optional[Dict[str, Dict[str, Any]]] = None,
    aggregation_config: Optional[Dict[str, Any]] = None,
    top_n: Optional[int] = None,
    all_family_statuses: Optional[List[Dict[str, Any]]] = None,
    # Legacy parameters (for backward compatibility)
    results: Optional[List[ImportanceResult]] = None,
    model_weights: Optional[Dict[str, float]] = None,
    aggregation_method: str = "weighted_rank",
    top_k: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """
    Aggregate importance scores across multiple model families.

    This function supports two calling conventions:

    1. Full pipeline mode (feature_selector.py):
       aggregate_multi_model_importance(
           all_results=results,
           model_families_config=config,
           aggregation_config=agg_config,
           top_n=100,
           all_family_statuses=statuses
       )
       Returns: Tuple[pd.DataFrame, List[str]]

    2. Simple mode (legacy):
       aggregate_multi_model_importance(
           results=results,
           model_weights=weights,
           aggregation_method="weighted_rank",
           top_k=50
       )
       Returns: pd.DataFrame

    Args:
        all_results: List of ImportanceResult (full pipeline mode)
        model_families_config: Model family configuration dict
        aggregation_config: Aggregation configuration dict
        top_n: Number of top features to select
        all_family_statuses: Optional status info for excluded families
        results: List of ImportanceResult (legacy mode)
        model_weights: Optional weights per model family (legacy)
        aggregation_method: Method for combining scores (legacy)
        top_k: Optional limit for top features (legacy)

    Returns:
        DataFrame or Tuple[DataFrame, List[str]] depending on mode
    """
    # Import from parent module (the .py file, not the package)
    # Use importlib to get the actual file, not the package
    import importlib.util
    from pathlib import Path

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.aggregate_multi_model_importance

    # Determine which mode we're in
    if all_results is not None and model_families_config is not None:
        # Full pipeline mode
        return _impl(
            all_results=all_results,
            model_families_config=model_families_config,
            aggregation_config=aggregation_config or {},
            top_n=top_n,
            all_family_statuses=all_family_statuses,
        )
    elif results is not None:
        # Legacy simple mode - build minimal config and call full impl
        # Note: This path may need adjustment based on actual usage
        logger.warning("Using legacy aggregate_multi_model_importance API - consider updating to full API")
        return _impl(
            all_results=results,
            model_families_config={},
            aggregation_config={"method": aggregation_method},
            top_n=top_k,
            all_family_statuses=None,
        )
    else:
        raise ValueError("Must provide either all_results or results parameter")


def compute_target_confidence(
    aggregated_importance: pd.DataFrame,
    results: List[ImportanceResult],
    min_models_agree: int = 3,
) -> Dict[str, Any]:
    """
    Compute confidence metrics for target predictability.

    Based on how consistently features are ranked across model families,
    this provides a confidence score for whether the target is predictable.

    Args:
        aggregated_importance: Aggregated importance DataFrame
        results: Original ImportanceResult list
        min_models_agree: Minimum models that must agree for "high" confidence

    Returns:
        Dict with confidence_score, agreement_metrics, and recommendation
    """
    # Import from parent module file (not the package) using importlib
    import importlib.util
    from pathlib import Path

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.compute_target_confidence

    return _impl(
        aggregated_importance=aggregated_importance,
        results=results,
        min_models_agree=min_models_agree,
    )


__all__ = [
    'aggregate_multi_model_importance',
    'compute_target_confidence',
]
