# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Core feature selection functions.

Provides the main entry points for feature selection:
- select_features_for_target: Main selection function
- rank_features_multi_model: Legacy wrapper

These are delegating wrappers that call the original implementation
in feature_selector.py for backward compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# SST: Import View enum for type hints
from TRAINING.orchestration.utils.scope_resolution import View

logger = logging.getLogger(__name__)


def select_features_for_target(
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    model_families_config: Optional[Dict[str, Dict[str, Any]]] = None,
    multi_model_config: Optional[Dict[str, Any]] = None,
    max_samples_per_symbol: Optional[int] = None,
    top_n: Optional[int] = None,
    output_dir: Optional[Path] = None,
    feature_selection_config: Optional[Any] = None,
    explicit_interval: Optional[int | str] = None,
    experiment_config: Optional[Any] = None,
    view: str | View = View.CROSS_SECTIONAL,
    symbol: Optional[str] = None,
    force_refresh: bool = False,
    universe_sig: Optional[str] = None,
    run_identity: Optional[Any] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top features for a target using multi-model consensus.

    This function processes all symbols, aggregates feature importance across
    model families, and returns the top N features. All leakage-free behavior
    is preserved (PurgedTimeSeriesSplit, leakage filtering, etc.).

    Args:
        target_column: Target column name
        symbols: List of symbols to process
        data_dir: Directory containing symbol data
        model_families_config: Optional model families config (overrides multi_model_config) [LEGACY]
        multi_model_config: Optional multi-model config dict [LEGACY]
        max_samples_per_symbol: Maximum samples per symbol
        top_n: Number of top features to return
        output_dir: Optional output directory for results
        feature_selection_config: Optional FeatureSelectionConfig object [NEW - preferred]
        explicit_interval: Optional explicit interval (e.g., "5m" or 5)
        experiment_config: Optional ExperimentConfig (for data.bar_interval)
        view: View enum or "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Required for SYMBOL_SPECIFIC view
        force_refresh: If True, bypass cache and re-run Phase 2
        universe_sig: Universe signature from SST (resolved_data_config)
        run_identity: Finalized RunIdentity for hash-based storage

    Returns:
        Tuple of (selected_feature_names, importance_dataframe)
    """
    # Import from parent module
    from TRAINING.ranking.feature_selector import (
        select_features_for_target as _select_features_for_target
    )

    return _select_features_for_target(
        target_column=target_column,
        symbols=symbols,
        data_dir=data_dir,
        model_families_config=model_families_config,
        multi_model_config=multi_model_config,
        max_samples_per_symbol=max_samples_per_symbol,
        top_n=top_n,
        output_dir=output_dir,
        feature_selection_config=feature_selection_config,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config,
        view=view,
        symbol=symbol,
        force_refresh=force_refresh,
        universe_sig=universe_sig,
        run_identity=run_identity,
    )


def rank_features_multi_model(
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    multi_model_config: Optional[Dict[str, Any]] = None,
    max_samples_per_symbol: Optional[int] = None,
    top_n: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Legacy wrapper for backward compatibility.

    Use select_features_for_target instead for new code.

    Args:
        target_column: Target column name
        symbols: List of symbols to process
        data_dir: Directory containing symbol data
        multi_model_config: Optional multi-model config dict
        max_samples_per_symbol: Maximum samples per symbol
        top_n: Number of top features to return
        output_dir: Optional output directory for results

    Returns:
        Tuple of (selected_feature_names, importance_dataframe)
    """
    # Import from parent module
    from TRAINING.ranking.feature_selector import (
        rank_features_multi_model as _rank_features_multi_model
    )

    return _rank_features_multi_model(
        target_column=target_column,
        symbols=symbols,
        data_dir=data_dir,
        multi_model_config=multi_model_config,
        max_samples_per_symbol=max_samples_per_symbol,
        top_n=top_n,
        output_dir=output_dir,
    )


__all__ = [
    'select_features_for_target',
    'rank_features_multi_model',
]
