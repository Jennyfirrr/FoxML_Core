# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cross-sectional data loading utilities for ranking scripts.

This module provides functions to load and prepare cross-sectional data
that matches the training pipeline's data structure.
"""


import numpy as np
import pandas as pd
import polars as pl

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View

# Polars-native conversion helpers (memory-efficient)
from TRAINING.common.utils.polars_to_numpy import (
    polars_to_sklearn_dense,
    polars_extract_column_as_numpy,
    polars_cross_sectional_filter,
    polars_cross_sectional_sample,
    polars_valid_mask,
)
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
import warnings

from TRAINING.common.utils.fingerprinting import _compute_feature_fingerprint

logger = logging.getLogger(__name__)


def _log_feature_set(
    stage: str,
    feature_names: List[str],
    previous_names: Optional[List[str]] = None,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Log feature set with fingerprint and delta tracking.
    
    Args:
        stage: Stage name (e.g., "SAFE_CANDIDATES", "AFTER_DROP_ALL_NAN")
        feature_names: Current feature names
        previous_names: Previous feature names (for delta computation)
        logger_instance: Logger to use (defaults to module logger)
    """
    if logger_instance is None:
        logger_instance = logger
    
    n_features = len(feature_names)
    set_fingerprint, order_fingerprint = _compute_feature_fingerprint(feature_names, set_invariant=True)
    fingerprint = set_fingerprint  # Use set-invariant for logging (backward compatibility)
    
    # Check for duplicates
    unique_names = set(feature_names)
    has_duplicates = len(unique_names) != n_features
    if has_duplicates:
        duplicates = [name for name in unique_names if feature_names.count(name) > 1]
        logger_instance.error(
            f"  🚨 FEATURESET [{stage}]: {n_features} features, fingerprint={fingerprint}, "
            f"DUPLICATES DETECTED: {duplicates}"
        )
        return
    
    # Compute delta if previous set provided
    if previous_names is not None:
        prev_set = set(previous_names)
        curr_set = set(feature_names)
        added = sorted(curr_set - prev_set)
        removed = sorted(prev_set - curr_set)
        
        # Check for order changes (if sets are equal but order differs)
        order_changed = False
        if not added and not removed and len(previous_names) == len(feature_names):
            prev_order_fp, _ = _compute_feature_fingerprint(previous_names, set_invariant=False)
            _, curr_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=False)
            if prev_order_fp != curr_order_fp:
                order_changed = True
        
        if added or removed:
            delta_str = f", added={len(added)}, removed={len(removed)}"
            if added and len(added) <= 5:
                delta_str += f" (added: {added})"
            elif added:
                delta_str += f" (added: {added[:3]}... +{len(added)-3} more)"
            if removed and len(removed) <= 5:
                delta_str += f" (removed: {removed})"
            elif removed:
                delta_str += f" (removed: {removed[:3]}... +{len(removed)-3} more)"
        elif order_changed:
            delta_str = " (order changed)"
        else:
            delta_str = " (no changes)"
    else:
        delta_str = ""
    
    logger_instance.info(
        f"  📊 FEATURESET [{stage}]: n={n_features}, fingerprint={fingerprint}{delta_str}"
    )


def _prepare_ranking_data_polars_native(
    combined_pl: pl.DataFrame,
    target_column: str,
    time_col: Optional[str],
    feature_names: Optional[List[str]],
    mtf_data: Dict[str, pd.DataFrame],
    effective_min_cs: int,
    max_cs_samples: Optional[int],
    view: Any,
    effective_requested_view: Optional[str],
    view_reason: str,
    view_policy: str,
    data_scope: str,
    n_symbols_available: int,
    min_cs: int,
    loader_contract: Optional[Dict],
    universe_sig: str,
    loaded_symbols_list: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Polars-native data preparation for ranking (memory-efficient, no Pandas intermediate).

    This function processes ranking data entirely in Polars, extracting numpy arrays
    directly without the Pandas conversion that causes memory spikes.

    Args:
        combined_pl: Polars DataFrame with combined symbol data
        target_column: Target column name
        time_col: Timestamp column name (or None)
        feature_names: List of feature names (auto-discovered if None)
        mtf_data: Original mtf_data dict (for feature discovery reference)
        effective_min_cs: Effective minimum cross-sectional size
        max_cs_samples: Maximum samples per timestamp
        view: View setting
        effective_requested_view: Requested view
        view_reason: Reason for view selection
        view_policy: View policy
        data_scope: Data scope (PANEL or SINGLE_SYMBOL)
        n_symbols_available: Number of symbols available
        min_cs: Minimum cross-sectional size
        loader_contract: Loader contract dict
        universe_sig: Universe signature
        loaded_symbols_list: List of loaded symbols

    Returns:
        Same tuple as prepare_cross_sectional_data_for_ranking
    """
    import warnings

    # Apply cross-sectional filtering and sampling in Polars
    if time_col is not None:
        # Filter by min_cs
        combined_pl = polars_cross_sectional_filter(combined_pl, time_col, effective_min_cs)

        if len(combined_pl) == 0:
            logger.warning(f"No data after min_cs filter - all timestamps have < {effective_min_cs} symbols")
            return (None,) * 6

        # Apply cross-sectional sampling
        if max_cs_samples:
            combined_pl = polars_cross_sectional_sample(
                combined_pl, time_col, max_cs_samples, symbol_col="symbol"
            )
            logger.info(f"After max_cs_samples={max_cs_samples} filter: {combined_pl.shape}")

        # Sort by timestamp
        combined_pl = combined_pl.sort(time_col)
    else:
        # No time column - panel data requires timestamps for purging
        logger.error("CRITICAL: No time column found in panel data. Time-based purging is REQUIRED.")
        return (None,) * 6

    # Auto-discover features if not provided
    if feature_names is None:
        # Use first symbol in sorted order for consistent feature discovery
        sample_symbol = sorted(mtf_data.keys())[0] if mtf_data else None
        if sample_symbol:
            sample_cols = mtf_data[sample_symbol].columns.tolist()
            feature_names = sorted([col for col in sample_cols
                            if not any(col.startswith(prefix) for prefix in
                                     ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                      'tth_', 'p_', 'barrier_', 'hit_'])
                            and col not in [time_col, target_column, 'symbol']])
        else:
            feature_names = []
        logger.info(f"Auto-discovered {len(feature_names)} features")

    # Extract target
    if target_column not in combined_pl.columns:
        logger.error(f"Target '{target_column}' not in combined data")
        return (None,) * 6

    y = polars_extract_column_as_numpy(combined_pl, target_column, dtype=np.float32, replace_inf=True)

    # Extract features using Polars-native conversion
    # CRITICAL: For ranking, we do NOT impute - imputation happens inside CV loop
    # So we use imputation_strategy that preserves NaN (not actually imputing)
    available_features = [f for f in feature_names if f in combined_pl.columns]
    missing_features = [f for f in feature_names if f not in combined_pl.columns]

    if missing_features:
        logger.debug(f"Dropped {len(missing_features)} features not in DataFrame: {missing_features[:10]}...")

    if not available_features:
        logger.error("❌ No features remaining after filtering - cannot train models")
        return (None,) * 6

    # Log feature sets
    _log_feature_set("SAFE_CANDIDATES", available_features)

    # Select and cast features
    cast_exprs = []
    for col in sorted(available_features):
        cast_exprs.append(
            pl.col(col).cast(pl.Float32, strict=False).alias(col)
        )

    feature_pl = combined_pl.select(cast_exprs)

    # Replace inf/-inf with null
    replace_inf_exprs = []
    for col in sorted(available_features):
        replace_inf_exprs.append(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    feature_pl = feature_pl.select(replace_inf_exprs)

    # Check for all-null columns and drop them
    null_counts = feature_pl.select([
        pl.col(c).is_null().all().alias(c) for c in sorted(available_features)
    ]).row(0)

    all_null_cols = [
        sorted(available_features)[i] for i, is_all_null in enumerate(null_counts) if is_all_null
    ]

    if all_null_cols:
        logger.debug(f"Dropped {len(all_null_cols)} all-NaN feature columns")
        available_features = [f for f in available_features if f not in set(all_null_cols)]
        feature_pl = feature_pl.select(sorted(available_features))

    _log_feature_set("AFTER_DROP_ALL_NAN", available_features)

    if len(available_features) == 0:
        logger.error("❌ No features remaining after filtering - cannot train models")
        return (None,) * 6

    if len(available_features) < 5:
        logger.warning(
            f"⚠️  Very few features ({len(available_features)}) remaining after filtering. "
            f"Model training may fail or produce poor results."
        )

    # Extract to numpy (NO imputation - ranking preserves NaN for CV-safe imputation)
    X = feature_pl.to_numpy()
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    if X.shape[0] == 0:
        logger.error("Feature matrix is empty - no data to process")
        return (None,) * 6

    # Compute validity mask
    target_valid = ~np.isnan(y) & np.isfinite(y)

    # Feature validity: allow up to 50% NaN per row
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])
    feature_valid = feature_nan_ratio <= 0.5

    valid_mask = target_valid & feature_valid

    if not valid_mask.any():
        logger.error(f"No valid data after cleaning:")
        logger.error(f"  Target: {len(y)} total, {target_valid.sum()} valid ({np.isnan(y).sum()} NaN)")
        logger.error(f"  Features: {X.shape[0]} rows, {X.shape[1]} cols, {feature_valid.sum()} valid rows")
        return (None,) * 6

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = polars_extract_column_as_numpy(combined_pl, "symbol", dtype=None, replace_inf=False)[valid_mask]
    time_vals = polars_extract_column_as_numpy(combined_pl, time_col, dtype=None, replace_inf=False)[valid_mask] if time_col else None

    # Ensure time_vals is sorted
    if time_vals is not None and len(time_vals) > 1:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            sort_idx = np.argsort(time_vals)
            X_clean = X_clean[sort_idx]
            y_clean = y_clean[sort_idx]
            symbols_clean = symbols_clean[sort_idx]
            time_vals = time_vals[sort_idx]
            logger.debug(f"  Re-sorted data by timestamp (should be rare)")

    logger.info(f"✅ Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features (Polars-native)")
    logger.info(f"   Removed {len(X) - len(X_clean)} rows due to cleaning")
    logger.info(f"   ⚠️  Note: NaN values preserved for CV-safe imputation (no leakage)")

    # Build resolved_config
    resolved_config = {
        'view': view,
        'requested_view': effective_requested_view,
        'view_reason': view_reason,
        'view_policy': view_policy,
        'resolved_data_mode': view,
        'data_scope': data_scope,
        'n_symbols_loaded': n_symbols_available,
        'min_cs_required': min_cs,
        'effective_min_cs': effective_min_cs,
        'loader_contract': loader_contract,
        'universe_sig': universe_sig,
        'loaded_symbols': loaded_symbols_list,
        'requested_symbols': loader_contract.get('requested_symbols') if loader_contract else None,
        'conversion_method': 'polars_native',
    }

    return X_clean, y_clean, available_features, symbols_clean, time_vals, resolved_config


def load_mtf_data_for_ranking(
    data_dir: Path,
    symbols: List[str],
    max_rows_per_symbol: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load MTF data for multiple symbols (matches training pipeline structure).

    Uses UnifiedDataLoader internally for consistent behavior across all stages.
    Supports optional column projection for memory efficiency.

    Args:
        data_dir: Directory containing symbol data
        symbols: List of symbols to load
        max_rows_per_symbol: Optional limit on rows per symbol (most recent rows)
                            Default: None (load all). For ranking, use 10000-50000 for speed.
        columns: Optional list of columns to load. If None, loads all columns.
                 For column projection, specify target + features + metadata.

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    # Use UnifiedDataLoader for consistent loading across all stages
    try:
        from TRAINING.data.loading.unified_loader import UnifiedDataLoader

        # Detect interval from data_dir path (e.g., "interval=5m")
        interval = "5m"  # Default
        data_dir_str = str(data_dir)
        if "interval=" in data_dir_str:
            # Extract interval from path like ".../interval=5m/..."
            import re
            match = re.search(r"interval=(\d+[mhd]?)", data_dir_str)
            if match:
                interval = match.group(1)

        loader = UnifiedDataLoader(data_dir=data_dir, interval=interval)

        mtf_data = loader.load_data(
            symbols=symbols,
            columns=columns,  # None = load all (backward compatible)
            max_rows_per_symbol=max_rows_per_symbol,
        )

        if mtf_data:
            logger.info(f"Loaded {len(mtf_data)}/{len(symbols)} symbols via UnifiedDataLoader")

        return mtf_data

    except ImportError:
        # Fallback to legacy loading if UnifiedDataLoader not available
        logger.warning("UnifiedDataLoader not available, using legacy loading")
        return _load_mtf_data_for_ranking_legacy(
            data_dir, symbols, max_rows_per_symbol
        )


def _load_mtf_data_for_ranking_legacy(
    data_dir: Path,
    symbols: List[str],
    max_rows_per_symbol: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Legacy implementation of load_mtf_data_for_ranking (fallback)."""
    mtf_data = {}
    dropped_symbols = []  # Track dropped symbols with reasons

    for symbol in symbols:
        # Try different possible file locations (matching training pipeline)
        possible_paths = [
            data_dir / f"symbol={symbol}" / f"{symbol}.parquet",  # New structure
            data_dir / f"{symbol}.parquet",  # Direct file
            data_dir / f"{symbol}_mtf.parquet",  # Legacy format
        ]

        symbol_file = None
        for path in possible_paths:
            if path.exists():
                symbol_file = path
                break

        if symbol_file and symbol_file.exists():
            try:
                df = pd.read_parquet(symbol_file)

                # Check for empty DataFrame
                if df.empty:
                    dropped_symbols.append({
                        'symbol': symbol,
                        'reason': 'empty_dataframe',
                        'details': 'File exists but contains no rows'
                    })
                    logger.warning(f"Dropping {symbol}: empty DataFrame")
                    continue

                # Apply row limit if specified (most recent rows)
                if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                    df = df.tail(max_rows_per_symbol)
                    logger.debug(f"Limited {symbol} to {max_rows_per_symbol} most recent rows")

                mtf_data[symbol] = df
                logger.debug(f"Loaded {symbol}: {df.shape}")
            except Exception as e:
                dropped_symbols.append({
                    'symbol': symbol,
                    'reason': 'load_error',
                    'details': str(e)
                })
                logger.error(f"Error loading {symbol}: {e}")
        else:
            dropped_symbols.append({
                'symbol': symbol,
                'reason': 'file_not_found',
                'details': f'Tried: {possible_paths}'
            })
            logger.warning(f"File not found for {symbol}. Tried: {possible_paths}")
    
    # Log loader contract (requested vs loaded)
    n_requested = len(symbols)
    n_loaded = len(mtf_data)
    loaded_symbols = list(mtf_data.keys())
    
    logger.info(f"📦 Loader contract: requested={n_requested} symbols → loaded={n_loaded} symbols")
    logger.info(f"   Loaded symbols: {loaded_symbols}")
    
    if dropped_symbols:
        logger.warning(f"   Dropped {len(dropped_symbols)} symbols:")
        for drop_info in dropped_symbols:
            logger.warning(f"     - {drop_info['symbol']}: {drop_info['reason']} ({drop_info['details']})")
    
    # Store loader contract in mtf_data metadata (for later use)
    if mtf_data:
        # Attach metadata as a special key (will be filtered out during processing)
        mtf_data['__loader_contract__'] = {
            'requested_symbols': symbols,
            'loaded_symbols': loaded_symbols,
            'n_requested': n_requested,
            'n_loaded': n_loaded,
            'dropped_symbols': dropped_symbols
        }
    
    return mtf_data


def prepare_cross_sectional_data_for_ranking(
    mtf_data: Dict[str, pd.DataFrame],
    target_column: str,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    feature_time_meta_map: Optional[Dict[str, Any]] = None,  # NEW: Optional map of feature_name -> FeatureTimeMeta
    base_interval_minutes: Optional[float] = None,  # NEW: Base training grid interval (for alignment)
    allow_single_symbol: bool = False,  # NEW: Allow single symbol for SYMBOL_SPECIFIC view
    requested_view: Optional[str] = None,  # SST: View requested by caller/config
    output_dir: Optional[Path] = None  # NEW: Output directory for persisting view (SST)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Prepare cross-sectional data for ranking (simplified version of training pipeline).
    
    Args:
        mtf_data: Dictionary of symbol -> DataFrame
        target_column: Target column name
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp (default: 1000)
        feature_names: Optional list of feature names (auto-discovered if None)
    
    Returns:
        Tuple of (X, y, feature_names, symbols, time_vals) or (None,)*5 on error
    """
    if not mtf_data:
        logger.error("No data provided")
        return (None,) * 6
    
    # Default max_cs_samples to match training pipeline
    if max_cs_samples is None:
        max_cs_samples = 1000
        logger.info(f"Using default max_cs_samples={max_cs_samples}")
    
    logger.info(f"🎯 Building cross-sectional data for target: {target_column}")
    
    # Extract loader contract if available (and remove from mtf_data)
    loader_contract = mtf_data.pop('__loader_contract__', None)
    
    # CRITICAL: Always log symbol load report (requested vs loaded vs dropped)
    # This prevents confusion and makes regressions obvious
    n_symbols_available = len(mtf_data)
    loaded_symbols_list = list(mtf_data.keys())
    
    if loader_contract:
        n_requested = loader_contract['n_requested']
        requested_symbols_list = loader_contract.get('requested_symbols', [])
        dropped_symbols = loader_contract.get('dropped_symbols', [])
        
        # Build structured symbol load report
        logger.info(f"📦 Symbol load report:")
        logger.info(f"   Requested: {n_requested} symbols {requested_symbols_list}")
        logger.info(f"   Loaded: {n_symbols_available} symbols {loaded_symbols_list}")
        
        if dropped_symbols:
            dropped_dict = {d['symbol']: d.get('reason', 'unknown') for d in dropped_symbols}
            logger.warning(f"   Dropped: {len(dropped_symbols)} symbols {dropped_dict}")
        else:
            logger.info(f"   Dropped: 0 symbols")
    else:
        # Fallback: if loader contract not available, still log what we have
        logger.info(f"📦 Symbol load report:")
        logger.info(f"   Requested: unknown (loader contract not available)")
        logger.info(f"   Loaded: {n_symbols_available} symbols {loaded_symbols_list}")
        logger.warning(f"   Dropped: unknown (loader contract not available)")
        # Create minimal loader contract for error messages
        loader_contract = {
            'requested_symbols': [],
            'n_requested': None,
            'loaded_symbols': loaded_symbols_list,
            'n_loaded': n_symbols_available,
            'dropped_symbols': []
        }
    
    # CRITICAL: Enforce minimum symbols BEFORE building cross-sectional data
    # Cross-sectional ranking with too few symbols should hard-stop, not degrade silently
    # 
    # IMPORTANT DISTINCTION:
    #   - This check enforces: "N symbols loaded overall" (global availability)
    #   - Later, per-timestamp filtering enforces: "effective cross-sectional width per timestamp"
    #   - The hard-stop prevents "only 1-2 symbols total", while per-timestamp sampling enforces cross-sectional width
    # 
    # NOTE: For LOSO view:
    #   - This function is called with mtf_data containing (N-1) training symbols (validation symbol excluded)
    #   - The check is applied to the training set size (N-1), which is correct
    #   - We need at least MIN_SYMBOLS symbols loaded in the training set (global availability)
    #   - Per-timestamp filtering (below) ensures each timestamp has >= effective_min_cs symbols present
    #   - The validation symbol is loaded separately with min_cs=1 (see evaluate_target_predictability)
    # 
    # For CROSS_SECTIONAL view:
    #   - This function is called with mtf_data containing all N symbols
    #   - The check ensures we have at least MIN_SYMBOLS symbols loaded overall (global availability)
    #   - Per-timestamp filtering (below) ensures each timestamp has >= effective_min_cs symbols present
    
    # Minimum symbols required for meaningful cross-sectional analysis
    # Hard minimum: 3 symbols (below this, it's not truly cross-sectional)
    # Recommended: 10+ symbols for robust cross-sectional ranking
    # Exception: allow_single_symbol=True for SYMBOL_SPECIFIC view (intentional single-symbol time series)
    # Load from config (SST: config first, fallback to hardcoded default)
    try:
        from CONFIG.config_loader import get_cfg
        MIN_SYMBOLS_REQUIRED = int(get_cfg(
            "thresholds.min_symbols_required",
            default=3,
            config_name="feature_selection_config"
        ))
        RECOMMENDED_SYMBOLS = int(get_cfg(
            "thresholds.recommended_symbols",
            default=10,
            config_name="feature_selection_config"
        ))
    except Exception:
        # Fallback if config system unavailable (defensive boundary)
        MIN_SYMBOLS_REQUIRED = 3
        RECOMMENDED_SYMBOLS = 10
    
    # Skip symbol count check for SYMBOL_SPECIFIC view (intentional single-symbol)
    if not allow_single_symbol:
        if n_symbols_available < MIN_SYMBOLS_REQUIRED:
            error_msg = (
                f"CROSS_SECTIONAL mode requires >= {MIN_SYMBOLS_REQUIRED} symbols, but only {n_symbols_available} loaded. "
                f"Loaded symbols: {loaded_symbols_list}. "
                f"This would silently degrade into single-symbol time series masquerading as cross-sectional ranking. "
                f"Use SYMBOL_SPECIFIC mode for single-symbol ranking, or ensure sufficient symbols are available."
            )
            # Always include dropped symbols info if available
            if loader_contract and loader_contract.get('dropped_symbols'):
                dropped_dict = {d['symbol']: d.get('reason', 'unknown') for d in loader_contract['dropped_symbols']}
                error_msg += f" Dropped symbols: {dropped_dict}"
            elif loader_contract and loader_contract.get('requested_symbols'):
                # If we have requested list, show what was requested but not loaded
                requested_set = set(loader_contract['requested_symbols'])
                loaded_set = set(loaded_symbols_list)
                missing = requested_set - loaded_set
                if missing:
                    error_msg += f" Missing from requested list: {sorted(missing)}"
            
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Warn if using fewer than recommended symbols (but allow it)
        if n_symbols_available < RECOMMENDED_SYMBOLS:
            logger.warning(
                f"⚠️  CROSS_SECTIONAL mode with {n_symbols_available} symbols (recommended: >= {RECOMMENDED_SYMBOLS}). "
                f"Cross-sectional ranking may be less robust with fewer symbols. "
                f"Consider using SYMBOL_SPECIFIC mode for more reliable per-symbol ranking."
            )
    else:
        # SYMBOL_SPECIFIC view: single symbol is expected and valid
        if n_symbols_available == 1:
            logger.debug(f"SYMBOL_SPECIFIC view: processing single symbol {loaded_symbols_list[0]}")
        elif n_symbols_available > 1:
            logger.warning(
                f"SYMBOL_SPECIFIC view expected 1 symbol, but {n_symbols_available} loaded: {loaded_symbols_list}. "
                f"Proceeding with all loaded symbols."
            )
    
    # Compute effective_min_cs (should equal min_cs now, but keep for consistency)
    effective_min_cs = min(min_cs, n_symbols_available)
    min_cs_reason = "requested"  # Always requested now (we hard-stop if insufficient)
    
    # Compute universe signature for this symbol set
    from TRAINING.orchestration.utils.run_context import (
        compute_universe_signature, get_view_for_universe, save_run_context, validate_view_contract
    )
    universe_sig = compute_universe_signature(loaded_symbols_list)
    symbols_sample = loaded_symbols_list[:3] if len(loaded_symbols_list) > 3 else loaded_symbols_list
    logger.info(f"🔑 Universe: sig={universe_sig} n_symbols={n_symbols_available} sample={symbols_sample}")
    
    # SST: Use requested_view if provided
    effective_requested_view = requested_view
    
    # Load existing view for THIS universe only (not global)
    existing_entry = None
    if output_dir is not None:
        try:
            existing_entry = get_view_for_universe(output_dir, universe_sig)
            if existing_entry:
                # SST: Read view from entry
                cached_view = existing_entry.get('view')
                logger.debug(f"Found cached view for universe={universe_sig}: {cached_view}")
        except Exception as e:
            logger.debug(f"Could not load existing run context: {e}")
    
    # Load view policy from config (with backward compat for view_policy)
    view_policy = "auto"  # Default
    auto_flip_min_symbols = RECOMMENDED_SYMBOLS  # Default
    try:
        from CONFIG.config_loader import get_cfg
        # New config keys (view_policy, requested_view) with fallback to old keys (view_policy, requested_view)
        view_policy = get_cfg("training_config.routing.view_policy", 
                              default=get_cfg("training_config.routing.view_policy", default="auto"))
        auto_flip_min_symbols = get_cfg("training_config.routing.auto_flip_min_symbols", default=RECOMMENDED_SYMBOLS)
        if effective_requested_view is None:
            effective_requested_view = get_cfg("training_config.routing.requested_view",
                                     default=get_cfg("training_config.routing.requested_view", default=None))
    except Exception as e:
        logger.debug(f"Could not load view policy from config: {e}, using defaults")
    
    # Determine resolved view based on policy
    # Only reuse if we have a cached entry for THIS universe AND it doesn't conflict with requested_view
    should_reuse_cache = False
    if existing_entry:
        cached_view = existing_entry.get('view')
        original_reason = existing_entry.get("original_reason", "N/A")
        
        # SST: Check if requested_view conflicts with cached view
        # Normalize both to View enum for comparison (SST pattern)
        cached_view_enum = None
        if cached_view:
            try:
                cached_view_enum = View.from_string(cached_view) if isinstance(cached_view, str) else cached_view
            except (ValueError, AttributeError):
                # Fallback: treat as string if View.from_string fails
                cached_view_enum = cached_view
        
        requested_view_enum = None
        if effective_requested_view:
            try:
                requested_view_enum = View.from_string(effective_requested_view) if isinstance(effective_requested_view, str) else effective_requested_view
            except (ValueError, AttributeError):
                # Fallback: treat as string if View.from_string fails
                requested_view_enum = effective_requested_view
        
        # CRITICAL: Validate cached view against symbol count FIRST (before requested_view check)
        # This prevents invalid cached views from being reused regardless of requested_view
        # This is especially important when subsequent targets have requested_view=None (auto mode)
        is_cache_valid = True
        if cached_view_enum == View.SYMBOL_SPECIFIC and n_symbols_available > 1:
            logger.warning(
                f"⚠️  Cached view=SYMBOL_SPECIFIC incompatible with n_symbols={n_symbols_available}. "
                f"SYMBOL_SPECIFIC requires n_symbols=1. Rejecting cache entry."
            )
            is_cache_valid = False
            should_reuse_cache = False
            existing_entry = None
        
        # Only reuse cache if:
        # 1. Cached view is valid (passed validation above), AND
        # 2. requested_view is None (auto mode) OR requested_view matches cached view
        if is_cache_valid and (requested_view_enum is None or cached_view_enum == requested_view_enum):
            # Safe to reuse: cache is valid and compatible with requested_view
            should_reuse_cache = True
            view = cached_view
            view_reason = f"reusing cached view for universe={universe_sig} (originally: {original_reason})"
        elif is_cache_valid:
            # Cache is valid but requested_view conflicts - resolve fresh
            logger.warning(
                f"⚠️  View cache conflict: cached view={cached_view} for universe={universe_sig} "
                f"conflicts with requested_view={effective_requested_view}. Resolving fresh."
            )
            should_reuse_cache = False
            # Don't set view here - fall through to auto/force logic
    
    if not should_reuse_cache:
        # Validate requested_view is compatible with n_symbols BEFORE resolution
        # SYMBOL_SPECIFIC view requires n_symbols=1 (single symbol only)
        if effective_requested_view == View.SYMBOL_SPECIFIC.value and n_symbols_available > 1:
            logger.warning(
                f"⚠️  Invalid requested_view=SYMBOL_SPECIFIC for multi-symbol run (n_symbols={n_symbols_available}). "
                f"SYMBOL_SPECIFIC view requires n_symbols=1. Resolving to CROSS_SECTIONAL."
            )
            effective_requested_view = None  # Clear invalid request, let auto logic resolve to CROSS_SECTIONAL
        
        if view_policy == "force":
            # Force view: use requested_view exactly (no auto-flip)
            if effective_requested_view is None:
                logger.warning("view_policy=force but requested_view not set, defaulting to CROSS_SECTIONAL")
                effective_requested_view = View.CROSS_SECTIONAL.value
            # CRITICAL: Validate forced view matches symbol count
            if effective_requested_view == View.SYMBOL_SPECIFIC.value and n_symbols_available > 1:
                raise ValueError(
                    f"view_policy=force with requested_view=SYMBOL_SPECIFIC is invalid for multi-symbol run "
                    f"(n_symbols={n_symbols_available}). SYMBOL_SPECIFIC requires n_symbols=1."
                )
            view = effective_requested_view
            view_reason = f"view_policy=force, requested_view={effective_requested_view}"
        else:
            # Auto mode: resolve fresh based on panel size
            if n_symbols_available == 1:
                if effective_requested_view and effective_requested_view != "SINGLE_SYMBOL_TS":
                    view = effective_requested_view
                    view_reason = f"n_symbols=1, using requested_view={effective_requested_view} (per-symbol loop)"
                else:
                    view = "SINGLE_SYMBOL_TS"
                    view_reason = "n_symbols=1"
            elif n_symbols_available < auto_flip_min_symbols:
                # Small panel: recommend SYMBOL_SPECIFIC, BUT only if single symbol
                # For multi-symbol runs, must use CROSS_SECTIONAL (SYMBOL_SPECIFIC requires n_symbols=1)
                # Note: effective_requested_view was already validated above, so if it was SYMBOL_SPECIFIC, it's now None
                if n_symbols_available == 1:
                    # Single symbol: can use SYMBOL_SPECIFIC
                    view = View.SYMBOL_SPECIFIC
                    view_reason = f"n_symbols={n_symbols_available} (small panel, < {auto_flip_min_symbols} recommended)"
                else:
                    # Multi-symbol small panel: must use CROSS_SECTIONAL (SYMBOL_SPECIFIC requires n_symbols=1)
                    view = View.CROSS_SECTIONAL
                    view_reason = f"n_symbols={n_symbols_available} (small panel, < {auto_flip_min_symbols}, but multi-symbol so CROSS_SECTIONAL)"
            else:
                # Large panel: always CROSS_SECTIONAL
                view = View.CROSS_SECTIONAL
                view_reason = f"n_symbols={n_symbols_available} (full panel, >= {auto_flip_min_symbols})"
    
    # Validate view contract (only if we resolved a new view, not cached)
    if not should_reuse_cache:
        try:
            validate_view_contract(view, effective_requested_view, view_policy)
        except ValueError as e:
            logger.error(f"View contract validation failed: {e}")
            raise
    
    # Set data_scope based on current n_symbols_available (can vary per-symbol, non-immutable)
    if n_symbols_available == 1:
        data_scope = "SINGLE_SYMBOL"
    else:
        data_scope = "PANEL"
    
    # Persist view and data_scope to run context (SST)
    # view is immutable PER UNIVERSE, data_scope can be updated
    if output_dir is not None:
        try:
            # For cached entries that were reused, pass the original_reason (not the "reusing..." message)
            # If we resolved fresh (conflict or no cache), use the new view_reason
            if should_reuse_cache and existing_entry:
                save_view_reason = existing_entry.get("original_reason", view_reason)
            else:
                save_view_reason = view_reason
            save_run_context(
                output_dir=output_dir,
                view=view,
                requested_view=effective_requested_view,
                view_reason=save_view_reason,
                n_symbols=n_symbols_available,
                data_scope=data_scope,
                universe_signature=universe_sig,
                symbols=loaded_symbols_list
            )
        except Exception as e:
            logger.warning(f"Could not save run context: {e}")
    
    # Log requested vs effective (single authoritative line)
    # Normalize view to enum for comparison
    view_enum = View.from_string(view) if isinstance(view, str) else view
    data_type_label = "Cross-sectional sampling" if view_enum == View.CROSS_SECTIONAL else "Panel data sampling"
    logger.info(
        f"📊 {data_type_label}: "
        f"requested_min_cs={min_cs} → effective_min_cs={effective_min_cs} "
        f"(reason={min_cs_reason}, n_symbols={n_symbols_available}), "
        f"max_cs_samples={max_cs_samples}"
    )
    logger.info(f"📋 View resolution: requested_view={effective_requested_view or 'N/A'}, view={view} (reason: {view_reason})")
    
    # NEW: Multi-interval alignment support
    # If feature_time_meta_map and base_interval_minutes are provided, apply alignment
    use_alignment = (
        feature_time_meta_map is not None 
        and base_interval_minutes is not None 
        and len(feature_time_meta_map) > 0
    )
    
    # CRITICAL: Schema harmonization BEFORE pd.concat() (matches training pipeline)
    # This prevents union explosion when symbols have different schemas (versioned features, etc.)
    # For ranking, use intersection mode by default (common features only) to ensure consistency
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    align_mode = os.environ.get("CS_ALIGN_MODE", "intersect").lower()  # Default to intersect for ranking
    
    if align_cols and mtf_data:
        # DETERMINISTIC: Sort symbols before picking first DataFrame to ensure consistent column order
        first_symbol = sorted(mtf_data.keys())[0]
        first_df = mtf_data[first_symbol]
        
        if align_mode == "intersect":
            # Intersection mode: use only common columns across all symbols
            shared = None
            # DETERMINISTIC: Sort symbols before iteration
            for _sym in sorted(mtf_data.keys()):
                _df = mtf_data[_sym]
                cols = set(_df.columns)
                shared = cols if shared is None else (shared & cols)
            ordered_schema = [c for c in first_df.columns if c in (shared or set())]
            
            # Apply to all symbols
            # DETERMINISTIC: Sort symbols before iteration
            for sym in sorted(mtf_data.keys()):
                df = mtf_data[sym]
                if list(df.columns) != ordered_schema:
                    mtf_data[sym] = df.loc[:, ordered_schema]
            
            logger.info(f"🔧 Harmonized schema (intersect) with {len(ordered_schema)} columns")
        else:
            # Union mode: include all columns seen across symbols; fill missing as NaN
            union = []
            seen = set()
            # Start with first df order for determinism
            for c in first_df.columns:
                union.append(c)
                seen.add(c)
            # DETERMINISTIC: Sort symbols before iteration to ensure consistent column discovery order
            for _sym in sorted(mtf_data.keys()):
                _df = mtf_data[_sym]
                for c in _df.columns:
                    if c not in seen:
                        union.append(c)
                        seen.add(c)
            
            # Apply to all symbols
            # DETERMINISTIC: Sort symbols before iteration
            for sym in sorted(mtf_data.keys()):
                df = mtf_data[sym]
                if list(df.columns) != union:
                    mtf_data[sym] = df.reindex(columns=union)
            
            logger.info(f"🔧 Harmonized schema (union) with {len(union)} columns")
    
    # Combine all symbol data using streaming concat (memory efficient for large universes)
    # MEMORY OPTIMIZATION: streaming_concat converts to Polars lazy frames incrementally,
    # releasing each DataFrame after conversion, then collects with streaming mode
    from TRAINING.data.loading import streaming_concat
    import gc

    # Check if any symbols have the target column
    has_target = any(
        target_column in mtf_data[sym].columns
        for sym in mtf_data.keys()
        if mtf_data[sym] is not None
    )
    if not has_target:
        logger.error(f"Target '{target_column}' not found in any symbol")
        return (None,) * 6

    # Convert to streaming lazy frame (memory efficient)
    # This handles: sorted symbol order, symbol column, float32 casting, memory release
    combined_lf = streaming_concat(
        mtf_data,
        symbol_column="symbol",
        target_column=target_column,
        # use_float32 defaults to config: intelligent_training.lazy_loading.use_float32
        release_after_convert=True,
    )

    # Collect with streaming mode (processes in memory-efficient chunks)
    # MEMORY OPTIMIZATION: Split collect() and to_pandas() to enable logging and efficient conversion
    from TRAINING.common.memory import log_memory_phase, log_memory_delta

    mem_baseline = log_memory_phase("before_collect")
    combined_pl = combined_lf.collect(streaming=True)
    del combined_lf
    gc.collect()

    log_memory_delta("after_collect", mem_baseline)
    logger.info(f"Polars DataFrame shape: {combined_pl.shape}")

    # Check config for Polars-native conversion (skip Pandas intermediate when possible)
    # DISABLED by default - the current implementation holds intermediate DataFrames
    # longer than the Pandas path, causing higher memory usage. Needs redesign.
    use_polars_direct = False  # Default to DISABLED
    try:
        from CONFIG.config_loader import get_cfg
        use_polars_direct = get_cfg(
            "memory.polars_conversion.polars_direct_conversion",
            default=False, config_name="memory"
        )
    except Exception:
        pass

    # Normalize time column name (check in Polars frame)
    time_col = "timestamp" if "timestamp" in combined_pl.columns else ("ts" if "ts" in combined_pl.columns else None)

    # =========================================================================
    # POLARS-NATIVE PATH (memory-efficient: skips Pandas intermediate)
    # Only used when alignment is NOT needed and polars_direct_conversion is true
    # =========================================================================
    if use_polars_direct and not use_alignment:
        logger.info("📊 Using Polars-native path (no alignment needed, memory-efficient)")

        try:
            result = _prepare_ranking_data_polars_native(
                combined_pl=combined_pl,
                target_column=target_column,
                time_col=time_col,
                feature_names=feature_names,
                mtf_data=mtf_data,
                effective_min_cs=effective_min_cs,
                max_cs_samples=max_cs_samples,
                view=view,
                effective_requested_view=effective_requested_view,
                view_reason=view_reason,
                view_policy=view_policy,
                data_scope=data_scope,
                n_symbols_available=n_symbols_available,
                min_cs=min_cs,
                loader_contract=loader_contract,
                universe_sig=universe_sig,
                loaded_symbols_list=loaded_symbols_list,
            )
            del combined_pl
            gc.collect()
            log_memory_delta("after_polars_native_ranking", mem_baseline)
            return result
        except Exception as e:
            logger.warning(f"⚠️  Polars-native path failed: {e}. Falling back to Pandas path.")
            # Fall through to Pandas path

    # =========================================================================
    # PANDAS PATH (required when alignment is needed or as fallback)
    # =========================================================================
    # Convert to pandas for downstream processing
    # NOTE: We avoid use_pyarrow_extension_array=True because downstream code
    # expects numpy-backed arrays (e.g., .std(), .mean() methods)
    mem_before_pandas = log_memory_phase("before_to_pandas")
    combined_df = combined_pl.to_pandas()

    # Release Polars DataFrame IMMEDIATELY to free memory
    del combined_pl
    gc.collect()

    log_memory_delta("after_to_pandas", mem_before_pandas)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    if use_alignment and time_col is not None:
        # Import alignment function
        from TRAINING.ranking.utils.feature_alignment import align_features_asof
        
        # Separate features by whether they need alignment
        features_need_alignment = []
        features_no_alignment = []
        
        # Auto-discover features if not provided
        if feature_names is None:
            # DETERMINISTIC: Sort symbols before picking sample to ensure consistent feature discovery
            # Use first symbol in sorted order, not arbitrary dict iteration order
            sample_symbol = sorted(mtf_data.keys())[0] if mtf_data else None
            if sample_symbol:
                sample_cols = mtf_data[sample_symbol].columns.tolist()
                feature_names = sorted([col for col in sample_cols 
                                if not any(col.startswith(prefix) for prefix in 
                                         ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                          'tth_', 'p_', 'barrier_', 'hit_'])
                                and col not in [time_col, target_column, 'symbol']])
            else:
                feature_names = []
        
        for feat_name in feature_names:
            if feat_name in feature_time_meta_map:
                meta = feature_time_meta_map[feat_name]
                native_interval = meta.native_interval_minutes or base_interval_minutes
                # Need alignment if different interval OR has embargo OR has publish_offset
                if (native_interval != base_interval_minutes or 
                    meta.embargo_minutes != 0.0 or 
                    meta.publish_offset_minutes != 0.0):
                    features_need_alignment.append(feat_name)
                else:
                    features_no_alignment.append(feat_name)
            else:
                features_no_alignment.append(feat_name)
                logger.debug(f"Feature {feat_name} not in feature_time_meta_map - using standard merge")
        
        if features_need_alignment:
            logger.info(
                f"🔧 Multi-interval alignment: {len(features_need_alignment)} features need alignment "
                f"(native intervals differ or have embargo/publish_offset), {len(features_no_alignment)} use standard merge"
            )
            
            # Extract feature DataFrames for alignment (from original mtf_data)
            feature_dfs = {}
            for feat_name in features_need_alignment:
                # Combine feature across all symbols
                feat_data = []
                # DETERMINISTIC: Sort symbols before iteration to ensure consistent feature alignment order
                for symbol in sorted(mtf_data.keys()):
                    df = mtf_data[symbol]
                    if feat_name in df.columns and time_col in df.columns:
                        feat_df = df[[time_col, feat_name]].copy()
                        feat_df['symbol'] = symbol
                        feat_data.append(feat_df)
                if feat_data:
                    feature_dfs[feat_name] = pd.concat(feat_data, ignore_index=True)
                else:
                    logger.warning(f"Feature {feat_name} marked for alignment but not found in any symbol - skipping")
            
            if feature_dfs:
                # Use existing combined_df as base (not full cross product)
                # This is the base grid: all (symbol, timestamp) pairs that exist in the data
                base_df = combined_df[['symbol', time_col]].drop_duplicates().copy()
                
                # Align features onto existing base dataframe
                aligned_features_df = align_features_asof(
                    base_df,
                    feature_dfs,
                    {k: v for k, v in feature_time_meta_map.items() if k in features_need_alignment},
                    base_interval_minutes,
                    timestamp_column=time_col
                )
                
                # Merge aligned features back into combined_df
                # Drop aligned features from combined_df first (they'll be replaced by aligned versions)
                cols_to_drop = [f for f in features_need_alignment if f in combined_df.columns]
                if cols_to_drop:
                    combined_df = combined_df.drop(columns=cols_to_drop)
                
                # Merge aligned features
                aligned_cols = [f for f in features_need_alignment if f in aligned_features_df.columns]
                if aligned_cols:
                    combined_df = combined_df.merge(
                        aligned_features_df[['symbol', time_col] + aligned_cols],
                        on=['symbol', time_col],
                        how='left'
                    )
                
                # Log alignment stats
                for feat_name in aligned_cols:
                    null_rate = combined_df[feat_name].isna().mean()
                    if null_rate > 0:
                        logger.debug(f"  Aligned {feat_name}: null_rate={null_rate:.1%} (from embargo/staleness)")
            else:
                logger.warning("No features found for alignment - falling back to standard merge")
                use_alignment = False
        else:
            use_alignment = False
    
    # CRITICAL: Sort by timestamp IMMEDIATELY after combining
    # This ensures data is always sorted and prevents warnings later
    if time_col is not None:
        combined_df = combined_df.sort_values(time_col).reset_index(drop=True)
        logger.debug(f"Sorted combined data by {time_col}")
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    # But be lenient: if we have fewer symbols than min_cs, use what we have
    if time_col is not None:
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        combined_df = combined_df[cs >= effective_min_cs]
        logger.debug(f"After effective_min_cs={effective_min_cs} filter: {combined_df.shape}")
        
        if len(combined_df) == 0:
            logger.warning(f"No data after min_cs filter - all timestamps have < {effective_min_cs} symbols")
            return (None,) * 6
        
        # Apply cross-sectional sampling per timestamp
        # CRITICAL: Shuffle symbols within each timestamp to avoid bias
        # If data is sorted alphabetically, we'd always sample AAPL, AMZN, etc. and miss ZZZ
        if max_cs_samples:
            # Add random shuffle column per timestamp group
            # CRITICAL FIX: Use deterministic timestamp-based seeding instead of hash()
            # hash() output changes every Python restart (salted for security), breaking reproducibility
            # Using timestamp integer ensures same shuffle for same timestamp across all runs
            def _get_deterministic_shuffle(group):
                """Generate deterministic shuffle key based on timestamp"""
                timestamp = group.name  # The timestamp value for this group
                # Convert timestamp to integer seed (works for pd.Timestamp, datetime, or numeric)
                if isinstance(timestamp, pd.Timestamp):
                    seed = int(timestamp.timestamp()) % (2**31)  # Use timestamp as seed
                elif isinstance(timestamp, (int, float)):
                    seed = int(timestamp) % (2**31)
                else:
                    # Fallback: use string hash but with fixed seed
                    seed = hash(str(timestamp)) % (2**31)
                return np.random.RandomState(seed).permutation(len(group))
            
            combined_df["_shuffle_key"] = combined_df.groupby(time_col)["symbol"].transform(_get_deterministic_shuffle)
            # Count timestamps that hit the cap before filtering
            timestamp_counts = combined_df.groupby(time_col).size()
            cap_hit_count = (timestamp_counts > max_cs_samples).sum()
            total_timestamps = len(timestamp_counts)
            
            combined_df = (combined_df
                           .sort_values([time_col, "_shuffle_key"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns=["_shuffle_key"]))
            
            # INFO: Show shape + cap hit info (readable)
            if cap_hit_count > 0:
                logger.info(f"After max_cs_samples={max_cs_samples} filter: {combined_df.shape} "
                          f"(cap_hit: {cap_hit_count}/{total_timestamps} timestamps)")
            else:
                logger.info(f"After max_cs_samples={max_cs_samples} filter: {combined_df.shape} "
                          f"(cap_hit: 0/{total_timestamps} timestamps)")
            # DEBUG: Full timestamp-level detail
            if cap_hit_count > 0:
                logger.debug(f"max_cs_samples cap details: {cap_hit_count} timestamps exceeded limit "
                           f"(sample: {list(timestamp_counts[timestamp_counts > max_cs_samples].head(5).index)})")
            # Data is already sorted by [time_col, _shuffle_key], so it's sorted by time_col
    else:
        # CRITICAL: Panel data REQUIRES timestamps for time-based purging
        # Without timestamps, row-count purging causes catastrophic leakage (1 bar = N rows, not 1 row)
        logger.error("CRITICAL: No time column found in panel data. Time-based purging is REQUIRED.")
        logger.error("  Panel data structure: multiple symbols per timestamp means row-count purging is invalid.")
        logger.error("  Example: With 50 symbols, 1 bar = 50 rows. Purging 17 rows = ~20 seconds, not 60 minutes!")
        return (None,) * 6
    
    # Auto-discover features if not provided
    if feature_names is None:
        # DETERMINISTIC: Sort columns before filtering to ensure consistent feature discovery order
        # This prevents non-determinism if pd.concat() column order varies
        feature_names = sorted([col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                                  'tth_', 'p_', 'barrier_', 'hit_'])
                        and col not in (['symbol', time_col, target_column] if time_col else ['symbol', target_column])])
        logger.info(f"Auto-discovered {len(feature_names)} features")
    
    # Extract target
    if target_column not in combined_df.columns:
        logger.error(f"Target '{target_column}' not in combined data")
        return (None,) * 6
    
    y = combined_df[target_column].values
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan).values
    
    # Extract features
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric and handle infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Track feature counts for resolved_config
    features_safe = len(feature_names)  # Features before dropping NaN
    _log_feature_set("SAFE_CANDIDATES", feature_names)
    
    # Drop columns that are entirely NaN
    before_cols = feature_df.shape[1]
    feature_names_before_nan = feature_names.copy()
    feature_df = feature_df.dropna(axis=1, how='all')
    features_dropped_nan = before_cols - feature_df.shape[1]
    if features_dropped_nan:
        logger.debug(f"Dropped {features_dropped_nan} all-NaN feature columns")
        # Update feature_names to match
        dropped_nan_names = [f for f in feature_names if f not in feature_df.columns]
        feature_names = [f for f in feature_names if f in feature_df.columns]
        if dropped_nan_names:
            logger.debug(f"  Dropped all-NaN columns: {dropped_nan_names[:10]}{'...' if len(dropped_nan_names) > 10 else ''}")
    _log_feature_set("AFTER_DROP_ALL_NAN", feature_names, previous_names=feature_names_before_nan)
    
    features_final = len(feature_names)  # Features after all filtering
    
    # Ensure only numeric dtypes
    feature_names_before_numeric = feature_names.copy()
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        dropped_non_numeric = [c for c in feature_df.columns if c not in numeric_cols]
        feature_df = feature_df[numeric_cols]
        feature_names = [f for f in feature_names if f in numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns: {dropped_non_numeric[:10]}{'...' if len(dropped_non_numeric) > 10 else ''}")
    # CRITICAL: Reindex DataFrame columns to match feature_names order (prevent order drift)
    # After cleaning, DataFrame columns might be reordered - enforce authoritative order
    # The feature_names list IS the authoritative order - DataFrame columns must match it
    if isinstance(feature_df, pd.DataFrame):
        # Reindex columns to match feature_names order exactly
        # This prevents "(order changed)" warnings and ensures deterministic column alignment
        feature_df = feature_df.loc[:, [f for f in feature_names if f in feature_df.columns]]
    
    _log_feature_set("AFTER_CLEANING", feature_names, previous_names=feature_names_before_numeric)
    
    # Check if we have any features left
    if len(feature_names) == 0:
        logger.error("❌ No features remaining after filtering - cannot train models")
        return (None,) * 6
    
    # Warn if very few features (may cause training issues)
    if len(feature_names) < 5:
        logger.warning(
            f"⚠️  Very few features ({len(feature_names)}) remaining after filtering. "
            f"Model training may fail or produce poor results. "
            f"Consider relaxing feature filtering rules or adding more features to the registry."
        )
    
    # Build feature matrix
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Check if we have any data
    if X.shape[0] == 0:
        logger.error("Feature matrix is empty - no data to process")
        return (None,) * 6
    
    # Clean data: remove rows with invalid target or too many NaN features
    target_valid = ~np.isnan(y) & np.isfinite(y)
    
    # Compute feature NaN ratio safely (suppress warning for empty slices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])  # All invalid if empty
            logger.warning("Feature matrix has zero columns - all features invalid")
    
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    valid_mask = target_valid & feature_valid
    
    if not valid_mask.any():
        logger.error(f"No valid data after cleaning:")
        logger.error(f"  Target: {len(y)} total, {target_valid.sum()} valid ({np.isnan(y).sum()} NaN, {np.sum(~np.isfinite(y))} inf)")
        logger.error(f"  Features: {X.shape[0]} rows, {X.shape[1]} cols, {feature_valid.sum()} valid rows")
        if X.shape[0] > 0:
            logger.error(f"  Feature NaN: {np.isnan(X).sum()} total, mean per row: {feature_nan_ratio.mean():.2%}")
        return (None,) * 6
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # CRITICAL FIX: Do NOT impute here - this causes data leakage!
    # Imputation must happen INSIDE the CV loop (fit on train, transform test)
    # The imputation will be handled by sklearn Pipeline in train_and_evaluate_models
    # We only remove rows with >50% NaN features, but keep NaN values for proper CV imputation
    
    # CRITICAL: Ensure time_vals is sorted (required for PurgedTimeSeriesSplit)
    # Data should already be sorted from earlier steps, but verify and fix if needed
    if time_vals is not None and len(time_vals) > 1:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            # Sort silently - data should be pre-sorted, but handle edge cases
            sort_idx = np.argsort(time_vals)
            X_clean = X_clean[sort_idx]
            y_clean = y_clean[sort_idx]
            symbols_clean = symbols_clean[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.debug(f"  Re-sorted data by timestamp (should be rare)")
    
    logger.info(f"✅ Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"   Removed {len(X) - len(X_clean)} rows due to cleaning")
    logger.info(f"   ⚠️  Note: NaN values preserved for CV-safe imputation (no leakage)")
    
    # Store resolved view and loader contract in a metadata dict (for telemetry)
    # This will be extracted by callers and passed to reproducibility tracker
    resolved_config = {
        # SST canonical fields
        'view': view,
        'requested_view': effective_requested_view,
        'view_reason': view_reason,
        'view_policy': view_policy,
        # DEPRECATED aliases (for backward compat)
        'resolved_data_mode': view,  # DEPRECATED: Use view
        'view': view,  # DEPRECATED: Use view
        'requested_view': effective_requested_view,  # DEPRECATED: Use requested_view
        'view_reason': view_reason,  # DEPRECATED: Use view_reason
        'view_policy': view_policy,  # DEPRECATED: Use view_policy
        # Other fields
        'data_scope': data_scope,  # Data scope (PANEL or SINGLE_SYMBOL) - can vary per-symbol
        'n_symbols_loaded': n_symbols_available,
        'min_cs_required': min_cs,
        'effective_min_cs': effective_min_cs,
        'loader_contract': loader_contract,
        # SST for scope partitioning - universe_sig is born here from loaded (not requested) symbols
        'universe_sig': universe_sig,
        'loaded_symbols': loaded_symbols_list,
        'requested_symbols': loader_contract.get('requested_symbols') if loader_contract else None,
    }
    
    # Attach to mtf_data for callers to extract (non-intrusive)
    # Callers can access via: resolved_config = mtf_data.get('__resolved_config__')
    # But we don't have mtf_data in return, so we'll need to pass this separately
    # For now, log it and callers can extract from logs or we'll add it to return later
    
    return X_clean, y_clean, feature_names, symbols_clean, time_vals, resolved_config

