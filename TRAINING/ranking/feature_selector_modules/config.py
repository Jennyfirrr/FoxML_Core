# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Configuration utilities for feature selection.

Provides functions for:
- Computing configuration hashes for caching
- Loading multi-model configuration

These functions are extracted from feature_selector.py for modularity.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# DETERMINISM_CRITICAL: Feature selection order must be deterministic
from TRAINING.common.utils.determinism_ordering import sorted_items

logger = logging.getLogger(__name__)


def compute_feature_selection_config_hash(
    target_column: str,
    symbols: list[str],
    model_families_config: Dict[str, Dict[str, Any]],
    view: str,
    symbol: Optional[str],
    max_samples_per_symbol: Optional[int],
    min_cs: Optional[int],
    max_cs_samples: Optional[int],
    explicit_interval: Optional[int | str],
    aggregation_config: Dict[str, Any],
    top_n: Optional[int],
    lookback_minutes: Optional[float] = None,
    window_minutes: Optional[float] = None,
) -> str:
    """
    Compute a deterministic hash of feature selection configuration.

    This hash is used to determine if cached results can be reused,
    bypassing Phase 2 (symbol processing) when configs are identical.

    Uses centralized config hashing utility for consistency.

    Args:
        target_column: Target column name
        symbols: List of symbols to process (sorted for hash stability)
        model_families_config: Model families configuration
        view: View type (CROSS_SECTIONAL/SYMBOL_SPECIFIC/LOSO)
        symbol: Symbol for SYMBOL_SPECIFIC view
        max_samples_per_symbol: Maximum samples per symbol
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum cross-sectional samples
        explicit_interval: Explicit interval
        aggregation_config: Aggregation configuration
        top_n: Number of top features to return
        lookback_minutes: Lookback window in minutes (interval-dependent)
        window_minutes: Feature window in minutes (interval-dependent)

    Returns:
        Hex digest of config hash (truncated to 16 characters for backward compatibility)
    """
    from TRAINING.common.utils.config_hashing import compute_config_hash

    # Build config dict for hashing (only include parameters that affect results)
    config_dict = {
        'target': target_column,
        'symbols': sorted(symbols),  # Sort for stability
        'view': view,
        'symbol': symbol,  # For SYMBOL_SPECIFIC view
        'max_samples_per_symbol': max_samples_per_symbol,
        'min_cs': min_cs,
        'max_cs_samples': max_cs_samples,
        'explicit_interval': explicit_interval,
        'lookback_minutes': lookback_minutes,
        'window_minutes': window_minutes,
        'top_n': top_n,
        'aggregation': aggregation_config,
        # Include enabled model families and their configs (only enabled ones)
        'model_families': {
            name: cfg for name, cfg in sorted_items(model_families_config)
            if isinstance(cfg, dict) and cfg.get('enabled', False)
        }
    }

    # Compute hash using centralized utility (SHA256, truncated to 16 chars for backward compatibility)
    config_hash = compute_config_hash(config_dict)[:16]

    return config_hash


def load_multi_model_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load multi-model feature selection configuration.

    Delegates to the original implementation in multi_model_feature_selection.

    Args:
        config_path: Optional path to config file (default: CONFIG/multi_model_feature_selection.yaml)

    Returns:
        Configuration dictionary
    """
    from TRAINING.ranking.multi_model_feature_selection import load_multi_model_config as _load_multi_model_config
    return _load_multi_model_config(config_path)


__all__ = [
    'compute_feature_selection_config_hash',
    'load_multi_model_config',
]
