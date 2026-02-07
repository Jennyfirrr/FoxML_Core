# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Persistence module for multi-model feature selection.

This module provides functions for saving and loading feature selection
results, including importance scores, metadata, and reproducibility tracking.

Note: The main implementation is currently in the parent multi_model_feature_selection.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .types import ImportanceResult

logger = logging.getLogger(__name__)


def save_multi_model_results(
    summary_df: pd.DataFrame,
    selected_features: List[str],
    all_results: List[ImportanceResult],
    output_dir: Path,
    metadata: Dict[str, Any],
    universe_sig: Optional[str] = None,
) -> None:
    """
    Save multi-model feature selection results.

    Target-first structure (with OutputLayout when universe_sig provided):
    targets/<target>/reproducibility/{view}/universe={universe_sig}/feature_importances/
      {model}_importances.csv
      feature_importance_multi_model.csv
      feature_importance_with_boruta_debug.csv
      model_agreement_matrix.csv
    targets/<target>/reproducibility/{view}/universe={universe_sig}/selected_features.txt

    Falls back to legacy structure without universe scoping when universe_sig not provided.

    Args:
        summary_df: DataFrame with feature importance summary
        selected_features: List of selected feature names
        all_results: List of ImportanceResult from all models
        output_dir: Directory to save results
        metadata: Metadata dict (must include 'target' or 'target_column', and 'view')
        universe_sig: Optional universe signature for canonical paths
    """
    # Import from parent module file (not the package) using importlib
    # This avoids the package/module naming collision
    import importlib.util

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.save_multi_model_results

    return _impl(
        summary_df=summary_df,
        selected_features=selected_features,
        all_results=all_results,
        output_dir=output_dir,
        metadata=metadata,
        universe_sig=universe_sig,
    )


def load_previous_model_results(
    output_dir: Optional[Path],
    symbol: str,
    target_column: str,
    model_family: str,
) -> Optional[Dict[str, Any]]:
    """
    Load previous model results for reproducibility comparison.

    Args:
        output_dir: Output directory with previous results
        symbol: Symbol name
        target_column: Target column name
        model_family: Model family name

    Returns:
        Dict with previous results or None if not found
    """
    # Import from parent module file (not the package) using importlib
    import importlib.util

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.load_previous_model_results

    return _impl(
        output_dir=output_dir,
        symbol=symbol,
        target_column=target_column,
        model_family=model_family,
    )


def save_model_metadata(
    output_dir: Optional[Path],
    symbol: str,
    target_column: str,
    model_family: str,
    score: float,
    importance: pd.Series,
    reproducibility: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save metadata for a single model training run.

    Args:
        output_dir: Output directory
        symbol: Symbol name
        target_column: Target column name
        model_family: Model family name
        score: Training score
        importance: Feature importance Series
        reproducibility: Optional reproducibility metrics
    """
    # Import from parent module file (not the package) using importlib
    import importlib.util

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.save_model_metadata

    return _impl(
        output_dir=output_dir,
        symbol=symbol,
        target_column=target_column,
        model_family=model_family,
        score=score,
        importance=importance,
        reproducibility=reproducibility,
    )


__all__ = [
    'save_multi_model_results',
    'load_previous_model_results',
    'save_model_metadata',
]
