# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Pre-flight leakage detection: filter features BEFORE loading data.

This module provides fast, schema-only leakage filtering that runs before
data loading. This enables:

1. Fail-fast: Detect leakage issues in <5 seconds (vs 5+ minutes after loading)
2. Memory efficiency: Know which columns to load before loading any data
3. Per-target feature lists: Build target -> features map for column projection

Example:
    ```python
    from TRAINING.ranking.utils.preflight_leakage import preflight_filter_features

    # Run pre-flight check (no data loading)
    target_features = preflight_filter_features(
        data_dir="/data/prices",
        symbols=["AAPL", "GOOGL"],
        targets=["fwd_ret_60m", "fwd_ret_120m"],
        interval_minutes=5
    )

    # Now load data with column projection
    for target, features in target_features.items():
        mtf_data = loader.load_for_target(symbols, target, features)
        # ... train model ...
    ```

SST Compliance:
    - Deterministic: sorted columns, sorted targets
    - Config access: Uses get_cfg() for thresholds
    - Error handling: Uses LeakageError for leakage failures
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from TRAINING.common.exceptions import LeakageError
from TRAINING.data.loading.unified_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


def preflight_filter_features(
    data_dir: Path,
    symbols: List[str],
    targets: List[str],
    interval_minutes: int = 5,
    use_registry: bool = True,
    for_ranking: bool = False,
    min_features_required: int = 5,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Run leakage filtering on schema only (no data loading).

    This is the core pre-flight check that enables lazy data loading.
    It reads only parquet metadata (~1ms/file) and runs the full leakage
    filtering pipeline on column names.

    Args:
        data_dir: Data directory containing parquet files
        symbols: List of symbols to read schema from
        targets: List of target column names to filter for
        interval_minutes: Data interval in minutes (for registry filtering)
        use_registry: Whether to use feature registry for structural validation
        for_ranking: If True, use permissive rules (for ranking stage)
                    If False, use strict rules (for training stage)
        min_features_required: Minimum features per target (raises if below)
        verbose: Whether to log detailed information

    Returns:
        Dictionary mapping target -> sorted list of allowed feature columns

    Raises:
        LeakageError: If any target has fewer than min_features_required features
        ValueError: If no parquet files found for symbols

    Example:
        ```python
        target_features = preflight_filter_features(
            data_dir=Path("/data/prices"),
            symbols=["AAPL", "GOOGL", "MSFT"],
            targets=["fwd_ret_60m", "fwd_ret_120m", "y_will_peak_15m_0.8"],
            interval_minutes=5,
            use_registry=True,
            for_ranking=False  # Strict mode for training
        )

        # Result:
        # {
        #     "fwd_ret_60m": ["close", "volume", "rsi_14", ...],
        #     "fwd_ret_120m": ["close", "volume", "rsi_14", ...],
        #     "y_will_peak_15m_0.8": ["close", "volume", "rsi_14", ...],
        # }
        ```
    """
    import time

    start_time = time.time()

    # Step 1: Read schemas from parquet metadata (~1ms per file)
    loader = UnifiedDataLoader(data_dir=data_dir, interval=f"{interval_minutes}m")
    schemas = loader.read_schema(symbols)

    if not schemas:
        raise ValueError(
            f"No parquet files found in {data_dir} for symbols {symbols}. "
            f"Check that data directory and interval are correct."
        )

    schema_time = time.time() - start_time
    if verbose:
        logger.info(
            f"📋 Pre-flight: Read schema for {len(schemas)}/{len(symbols)} symbols "
            f"in {schema_time*1000:.1f}ms"
        )

    # Step 2: Get common columns across all symbols (for cross-sectional)
    common_columns = loader.get_common_columns(list(schemas.keys()))

    if not common_columns:
        raise ValueError(
            f"No common columns found across symbols {list(schemas.keys())}. "
            f"Check that parquet files have consistent schemas."
        )

    if verbose:
        logger.info(f"   Common columns: {len(common_columns)} across {len(schemas)} symbols")

    # Step 3: Filter features for each target (metadata-only)
    from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target

    target_features: Dict[str, List[str]] = {}
    failed_targets: List[tuple] = []

    # DETERMINISTIC: Sort targets for consistent processing order
    for target in sorted(targets):
        try:
            allowed = filter_features_for_target(
                all_columns=common_columns,
                target_column=target,
                verbose=False,  # Reduce noise, log summary at end
                use_registry=use_registry,
                data_interval_minutes=interval_minutes,
                for_ranking=for_ranking,
            )

            # Validate minimum features
            if len(allowed) < min_features_required:
                failed_targets.append((target, len(allowed)))
            else:
                target_features[target] = allowed

        except Exception as e:
            logger.error(f"Pre-flight filter failed for target '{target}': {e}")
            failed_targets.append((target, 0))

    # Step 4: Report results
    filter_time = time.time() - start_time - schema_time
    total_time = time.time() - start_time

    if verbose:
        avg_features = (
            sum(len(f) for f in target_features.values()) / len(target_features)
            if target_features
            else 0
        )
        logger.info(
            f"✅ Pre-flight passed: {len(target_features)}/{len(targets)} targets, "
            f"avg {avg_features:.0f} features/target "
            f"(schema: {schema_time*1000:.1f}ms, filter: {filter_time*1000:.1f}ms, "
            f"total: {total_time*1000:.1f}ms)"
        )

    # Step 5: Fail if any targets have insufficient features
    if failed_targets:
        failed_str = ", ".join(
            f"'{t}' ({n} features)" for t, n in failed_targets
        )
        raise LeakageError(
            f"Pre-flight check failed: {len(failed_targets)} target(s) have fewer than "
            f"{min_features_required} allowed features: {failed_str}. "
            f"Check feature registry and excluded_features.yaml. "
            f"Mode: {'ranking (permissive)' if for_ranking else 'training (strict)'}."
        )

    return target_features


def preflight_check_target(
    data_dir: Path,
    symbols: List[str],
    target: str,
    interval_minutes: int = 5,
    use_registry: bool = True,
    for_ranking: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Run pre-flight check for a single target.

    Convenience wrapper for preflight_filter_features when checking one target.

    Args:
        data_dir: Data directory
        symbols: List of symbols
        target: Target column name
        interval_minutes: Data interval in minutes
        use_registry: Whether to use feature registry
        for_ranking: Whether this is for ranking (permissive) or training (strict)
        verbose: Whether to log details

    Returns:
        Sorted list of allowed feature columns

    Raises:
        LeakageError: If target has zero allowed features
    """
    result = preflight_filter_features(
        data_dir=data_dir,
        symbols=symbols,
        targets=[target],
        interval_minutes=interval_minutes,
        use_registry=use_registry,
        for_ranking=for_ranking,
        min_features_required=1,  # Allow single feature for single-target check
        verbose=verbose,
    )
    return result.get(target, [])


def get_preflight_summary(
    target_features: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Generate summary statistics from pre-flight results.

    Args:
        target_features: Result from preflight_filter_features()

    Returns:
        Dictionary with summary statistics:
        - n_targets: Number of targets
        - n_features_avg: Average features per target
        - n_features_min: Minimum features for any target
        - n_features_max: Maximum features for any target
        - common_features: Features present in ALL targets
        - target_feature_counts: Dict of target -> feature count
    """
    if not target_features:
        return {
            "n_targets": 0,
            "n_features_avg": 0,
            "n_features_min": 0,
            "n_features_max": 0,
            "common_features": [],
            "target_feature_counts": {},
        }

    # Count features per target
    counts = {t: len(f) for t, f in target_features.items()}

    # Find features common to ALL targets
    common = None
    for features in target_features.values():
        feature_set = set(features)
        if common is None:
            common = feature_set
        else:
            common &= feature_set

    return {
        "n_targets": len(target_features),
        "n_features_avg": sum(counts.values()) / len(counts),
        "n_features_min": min(counts.values()),
        "n_features_max": max(counts.values()),
        "common_features": sorted(common) if common else [],
        "target_feature_counts": counts,
    }
