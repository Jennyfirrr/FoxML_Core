# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Symbol processing module for multi-model feature selection.

This module provides symbol-level processing for the multi-model feature
selection pipeline. Each symbol is processed independently with multiple
model families to extract feature importance.

Note: The main implementation is currently in the parent multi_model_feature_selection.py
file. This module provides the interface and will be fully extracted in a future phase.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .types import ImportanceResult

logger = logging.getLogger(__name__)


# Delegate to main implementation
# This provides a clean import path while the full extraction is pending
def process_single_symbol(
    symbol: str,
    data_path: Path,
    target_column: str,
    model_families_config: Dict[str, Dict[str, Any]],
    max_samples: int = None,
    explicit_interval: Optional[Union[int, str]] = None,
    experiment_config: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    selected_features: Optional[List[str]] = None,
    run_identity: Optional[Any] = None,
) -> Tuple[List[ImportanceResult], List[Dict[str, Any]]]:
    """
    Process a single symbol with multiple model families.

    This function:
    1. Loads and validates the symbol's data
    2. Applies leakage filtering based on target horizon
    3. Trains each enabled model family
    4. Extracts and aggregates importance scores
    5. Saves stability snapshots for reproducibility tracking

    Args:
        symbol: Symbol name (e.g., "AAPL")
        data_path: Path to the symbol's data file
        target_column: Target column for prediction (e.g., "fwd_ret_5m")
        model_families_config: Configuration for each model family
        max_samples: Maximum samples per symbol (default from config)
        explicit_interval: Data interval override (e.g., "5m", 5)
        experiment_config: Optional experiment configuration
        output_dir: Output directory for snapshots
        selected_features: Optional pre-selected feature list from shared harness
        run_identity: Optional RunIdentity for snapshot storage

    Returns:
        Tuple of:
        - List of ImportanceResult (one per model family)
        - List of family status dicts (for tracking skipped/failed families)
    """
    # Import from parent module file (not the package) using importlib
    import importlib.util

    parent_file = Path(__file__).parent.parent / "multi_model_feature_selection.py"
    spec = importlib.util.spec_from_file_location("multi_model_feature_selection_file", parent_file)
    parent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_module)
    _impl = parent_module.process_single_symbol

    return _impl(
        symbol=symbol,
        data_path=data_path,
        target_column=target_column,
        model_families_config=model_families_config,
        max_samples=max_samples,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config,
        output_dir=output_dir,
        selected_features=selected_features,
        run_identity=run_identity,
    )


__all__ = ['process_single_symbol']
