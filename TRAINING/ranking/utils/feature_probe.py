# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Single-symbol feature probing for column projection optimization.

This module provides fast feature importance probing using a single representative
symbol. Combined with preflight leakage filtering, it enables efficient column
projection that reduces memory usage by ~67% beyond lazy loading alone.

Architecture:
    1. Preflight: Schema-only leakage filter → ~300 safe columns
    2. Probe: Single-symbol importance filter → ~100 important columns
    3. Full load: All symbols with only ~100 columns

Key Insight (Cross-Sectional Representativeness):
    For cross-sectional models with standardized feature engineering:
    - Same feature formula across all symbols (RSI_14 computed identically)
    - Same target definition across all symbols
    - Same market microstructure (liquid US equities)

    One symbol IS representative for feature importance ranking.

Example:
    ```python
    from TRAINING.ranking.utils.feature_probe import probe_features_for_target
    from TRAINING.data.loading.unified_loader import UnifiedDataLoader

    loader = UnifiedDataLoader(data_dir="/data/prices", interval="5m")

    # Preflight: ~300 safe columns
    preflight_features = preflight_filter_features(...)["fwd_ret_60m"]

    # Probe: ~100 important columns
    probed_features, importances = probe_features_for_target(
        loader=loader,
        symbols=["AAPL", "GOOGL", "MSFT"],
        target="fwd_ret_60m",
        preflight_features=preflight_features,
        top_n=100,
    )

    # Full load: Only ~100 columns across all symbols
    mtf_data = loader.load_for_target(symbols, target, probed_features)
    ```

SST Compliance:
    - Deterministic: sorted(symbols)[0] for probe symbol, sorted feature output
    - Config access: Uses get_cfg() for probe settings
    - Error handling: Returns preflight_features on failure (fail-open for this optimization)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from CONFIG.config_loader import get_cfg
from TRAINING.common.determinism import BASE_SEED
from TRAINING.data.loading.unified_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


def probe_features_for_target(
    loader: UnifiedDataLoader,
    symbols: List[str],
    target: str,
    preflight_features: List[str],
    top_n: Optional[int] = None,
    probe_rows: Optional[int] = None,
    importance_threshold: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Load single symbol, run quick importance, return top N features.

    Uses LightGBM feature importance on a single representative symbol to
    identify the most predictive features. This enables column projection
    that significantly reduces memory usage when loading all symbols.

    Args:
        loader: UnifiedDataLoader instance (already configured with data_dir and interval)
        symbols: Full symbol list (first alphabetically used for probe)
        target: Target column name (e.g., "fwd_ret_60m")
        preflight_features: Features from preflight filter (~300)
        top_n: Maximum features to keep (default from config: 100)
        probe_rows: Rows to load for probe (default from config: 10000)
        importance_threshold: Minimum cumulative importance to keep (default from config)
        seed: Random seed for LightGBM (default from determinism system)

    Returns:
        Tuple of:
            - top_features_list: Sorted list of top N important features
            - importance_dict: Dictionary mapping feature name -> importance score

    Note:
        On failure, returns (preflight_features, {}) to allow pipeline to continue
        with all preflight features (fail-open for this optimization).

    Example:
        ```python
        probed, importances = probe_features_for_target(
            loader=loader,
            symbols=["AAPL", "GOOGL", "MSFT"],
            target="fwd_ret_60m",
            preflight_features=["close", "volume", "rsi_14", ...],
            top_n=100,
        )
        print(f"Probed {len(probed)} features, top: {probed[:5]}")
        ```
    """
    # Load config defaults
    if top_n is None:
        top_n = int(get_cfg("intelligent_training.lazy_loading.probe_top_n", default=100))
    if probe_rows is None:
        probe_rows = int(
            get_cfg("intelligent_training.lazy_loading.probe_rows", default=10000)
        )
    if importance_threshold is None:
        importance_threshold = float(
            get_cfg(
                "intelligent_training.lazy_loading.probe_importance_threshold",
                default=0.0001,
            )
        )
    if seed is None:
        seed = BASE_SEED if BASE_SEED is not None else 42

    # Validate inputs
    if not symbols:
        logger.warning("No symbols provided for probe, returning preflight features")
        return sorted(preflight_features), {}

    if not preflight_features:
        logger.warning("No preflight features provided for probe")
        return [], {}

    # Skip probing if we already have few features
    if len(preflight_features) <= top_n:
        logger.info(
            f"🔬 Probe skipped: {len(preflight_features)} preflight features "
            f"≤ {top_n} top_n threshold"
        )
        return sorted(preflight_features), {}

    # DETERMINISTIC: Always use first symbol alphabetically
    probe_symbol = sorted(symbols)[0]

    logger.info(
        f"🔬 Probing features for '{target}' using symbol={probe_symbol}, "
        f"preflight={len(preflight_features)}, rows={probe_rows}"
    )

    try:
        # Load single symbol with preflight features
        probe_data = loader.load_for_target(
            symbols=[probe_symbol],
            target=target,
            features=preflight_features,
            max_rows_per_symbol=probe_rows,
        )

        if probe_symbol not in probe_data:
            logger.warning(
                f"Probe symbol '{probe_symbol}' not in loaded data, "
                f"returning preflight features"
            )
            return sorted(preflight_features), {}

        df = probe_data[probe_symbol]

        # Validate target column exists
        if target not in df.columns:
            logger.warning(
                f"Target '{target}' not in probe data columns, "
                f"returning preflight features"
            )
            return sorted(preflight_features), {}

        # Get available features (intersection of preflight and actual columns)
        available_features = sorted([f for f in preflight_features if f in df.columns])

        if len(available_features) < 5:
            logger.warning(
                f"Only {len(available_features)} features available in probe data, "
                f"returning preflight features"
            )
            return sorted(preflight_features), {}

        # Prepare X, y for quick pruner
        X = df[available_features].values
        y = df[target].values

        # Drop rows with NaN in target
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < 100:
            logger.warning(
                f"Only {len(y)} valid samples for probe, returning preflight features"
            )
            return sorted(preflight_features), {}

        logger.info(f"   Probe data: {X.shape[0]} samples × {X.shape[1]} features")

        # Run existing quick_importance_prune
        # Note: quick_importance_prune uses `min_features` to keep at least N features
        from TRAINING.ranking.utils.feature_pruning import quick_importance_prune

        _, pruned_names, stats = quick_importance_prune(
            X,
            y,
            available_features,
            min_features=top_n,  # Use top_n as min_features
            cumulative_threshold=importance_threshold,
            seed=seed,
        )

        # Extract importance dict
        importance_dict = stats.get("full_importance_dict", {})

        # DETERMINISTIC: Sort output features
        pruned_names = sorted(pruned_names)

        logger.info(
            f"✅ Probe complete: {len(available_features)} → {len(pruned_names)} features "
            f"(reduced by {100 * (1 - len(pruned_names) / len(available_features)):.1f}%)"
        )

        # Log top features
        if stats.get("top_10_features"):
            top_features_str = ", ".join(stats["top_10_features"][:5])
            logger.info(f"   Top features: {top_features_str}")

        return pruned_names, importance_dict

    except Exception as e:
        logger.warning(
            f"Feature probe failed for '{target}': {e}. "
            f"Falling back to preflight features."
        )
        return sorted(preflight_features), {}


def probe_all_targets(
    loader: UnifiedDataLoader,
    symbols: List[str],
    target_features: Dict[str, List[str]],
    top_n: Optional[int] = None,
    probe_rows: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Probe features for multiple targets at once.

    Convenience function that runs probe_features_for_target for each target
    in the target_features dict (output from preflight_filter_features).

    Args:
        loader: UnifiedDataLoader instance
        symbols: Full symbol list
        target_features: Dict mapping target -> preflight features
        top_n: Maximum features to keep per target
        probe_rows: Rows to load for each probe

    Returns:
        Dict mapping target -> probed features (sorted)

    Example:
        ```python
        # Run preflight
        target_features = preflight_filter_features(...)

        # Run probe for all targets
        probed_features = probe_all_targets(loader, symbols, target_features)

        for target, features in probed_features.items():
            print(f"{target}: {len(features)} features")
        ```
    """
    probed: Dict[str, List[str]] = {}

    # DETERMINISTIC: Sort targets for consistent processing order
    for target in sorted(target_features.keys()):
        preflight_features = target_features[target]
        features, _ = probe_features_for_target(
            loader=loader,
            symbols=symbols,
            target=target,
            preflight_features=preflight_features,
            top_n=top_n,
            probe_rows=probe_rows,
        )
        probed[target] = features

    return probed
