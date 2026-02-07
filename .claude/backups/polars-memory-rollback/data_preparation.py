# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

# ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# CRITICAL: Set LD_LIBRARY_PATH for conda CUDA libraries BEFORE any imports
# This must happen before TensorFlow tries to load CUDA libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []
    if conda_lib not in current_ld_path:
        new_paths.append(conda_lib)
    if conda_targets_lib not in current_ld_path:
        new_paths.append(conda_targets_lib)
    if new_paths:
        updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
        os.environ["LD_LIBRARY_PATH"] = updated_ld_path

# Show TensorFlow warnings so user knows if GPU isn't working
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# project root: TRAINING/training_strategies/*.py -> parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Propagate to spawned processes (spawned interpreter reads PYTHONPATH at startup)
os.environ.setdefault("PYTHONPATH", str(_PROJECT_ROOT))

# Set up all paths using centralized utilities
# Note: setup_all_paths already adds CONFIG to sys.path
from TRAINING.common.utils.path_setup import setup_all_paths
_PROJECT_ROOT, _TRAINING_ROOT, _CONFIG_DIR = setup_all_paths(_PROJECT_ROOT)

# Import config loader (CONFIG is already in sys.path from setup_all_paths)
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    # Only log at debug level to avoid misleading warnings
    logging.getLogger(__name__).debug("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
from TRAINING.common.utils.process_cleanup import setup_loky_cleanup_from_config
setup_loky_cleanup_from_config()

# DETERMINISM: Bootstrap reproducibility BEFORE any ML libraries
import TRAINING.common.repro_bootstrap  # noqa: F401 - MUST be first ML import

"""
Enhanced Training Script with Multiple Strategies - Full Original Functionality

Replicates ALL functionality from train_mtf_cross_sectional_gpu.py but with:
- Modular architecture
- 3 training strategies (single-task, multi-task, cascade)
- All 20 model families from original script
- GPU acceleration
- Memory management
- Batch processing
- Cross-sectional training
- Target discovery
- Data validation
"""

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Paths are already set up above

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES

# Standard library imports
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl

# Import USE_POLARS - defined in utils.py, but we need it here
# Use environment variable directly to avoid circular import
import os
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"

# Import target router utilities from TRAINING root
from TRAINING.orchestration.routing.target_router import route_target
# safe_target_extraction may be in utils or defined elsewhere
try:
    from TRAINING.training_strategies.utils import safe_target_extraction
except ImportError:
    # Fallback definition
    def safe_target_extraction(df, target):
        if target in df.columns:
            return df[target], target
        # Try common variations
        for col in df.columns:
            if col.endswith(target) or target in col:
                return df[col], col
        raise ValueError(f"Target {target} not found in dataframe")

# Polars-native conversion helpers (memory-efficient)
from TRAINING.common.utils.polars_to_numpy import (
    polars_to_sklearn_dense,
    polars_extract_column_as_numpy,
    polars_valid_mask,
)

# Setup logger
logger = logging.getLogger(__name__)

"""Data preparation functions for training strategies."""

def _resolve_training_registry_overlay(
    output_dir: Optional[Path],
    experiment_config: Optional[Any],
    target: str,
    current_bar_minutes: Optional[float]
) -> Optional['RegistryOverlayResolution']:
    """
    Resolve registry overlay for training stage (SST helper - no duplication).
    
    Uses same resolver as feature selection for consistency.
    
    Args:
        output_dir: Optional output directory (for run root resolution)
        experiment_config: Optional experiment config (for config override)
        target: Target column name
        current_bar_minutes: Current bar interval in minutes
    
    Returns:
        RegistryOverlayResolution if resolved, None otherwise
    """
    if not output_dir:
        return None
    
    try:
        from TRAINING.orchestration.utils.target_first_paths import run_root
        from TRAINING.ranking.utils.registry_overlay_resolver import (
            resolve_registry_overlay_dir_for_feature_selection,
            RegistryOverlayResolution
        )
        
        run_output_root = run_root(output_dir)
        overlay_resolution = resolve_registry_overlay_dir_for_feature_selection(
            run_output_root=run_output_root,
            experiment_config=experiment_config,
            target_column=target,
            current_bar_minutes=current_bar_minutes
        )
        
        if overlay_resolution.overlay_kind == "patch":
            logger.debug(
                f"📋 Training: Using registry patch for {target}: {overlay_resolution.patch_file.name} "
                f"(signature: {overlay_resolution.overlay_signature[:16] if overlay_resolution.overlay_signature else 'none'}...)"
            )
        elif overlay_resolution.overlay_kind == "config":
            logger.debug(f"Training: Using config registry overlay for {target}: {overlay_resolution.overlay_dir}")
        
        return overlay_resolution
    except Exception as e:
        logger.debug(f"Could not resolve registry overlay for {target} (training): {e}")
        return None


def prepare_training_data_cross_sectional(mtf_data: Dict[str, pd.DataFrame], 
                                       target: str, 
                                       feature_names: List[str] = None,
                                       min_cs: int = 10,
                                       max_cs_samples: int = None,
                                       routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None,
                                       *,  # Keyword-only separator
                                       output_dir: Optional[Path] = None,
                                       experiment_config: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Prepare cross-sectional training data with polars optimization for memory efficiency."""
    
    logger.info(f"🎯 Building cross-sectional training data for target: {target}")
    if max_cs_samples is None:
        # Load from config if available, otherwise use default
        if _CONFIG_AVAILABLE:
            max_cs_samples = get_cfg("pipeline.data_limits.max_cross_sectional_samples", default=None, config_name="pipeline_config")
            # If max_cross_sectional_samples is None, fall back to max_cs_samples (which defaults to 1000)
            if max_cs_samples is None:
                max_cs_samples = get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config")
            # Convert to int if not None
            if max_cs_samples is not None:
                max_cs_samples = int(max_cs_samples)
        else:
            max_cs_samples = 1000  # Fallback if config system unavailable (defensive boundary)
        logger.info(f"📊 Using default aggressive sampling: max {max_cs_samples} samples per timestamp")
    else:
        logger.info(f"📊 Cross-sectional sampling: max {max_cs_samples} samples per timestamp")
    
    # Detect interval first (needed for overlay resolution)
    detected_interval = None
    if mtf_data:
        from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
        # DETERMINISTIC: Use lexicographically first symbol to ensure same dataframe across runs
        first_symbol = min(mtf_data.keys())
        first_df = mtf_data[first_symbol]
        detected_interval = detect_interval_from_dataframe(first_df, timestamp_column='ts', default=5)
        if detected_interval <= 0:
            detected_interval = 5
    
    # Resolve overlay ONCE (SST - single source of truth)
    overlay_resolution = None
    registry_overlay_dir = None
    if output_dir:
        overlay_resolution = _resolve_training_registry_overlay(
            output_dir=output_dir,
            experiment_config=experiment_config,
            target=target,
            current_bar_minutes=detected_interval
        )
        registry_overlay_dir = overlay_resolution.overlay_dir if overlay_resolution else None
    
    # Call backend with resolved overlay_dir (not output_dir/experiment_config)
    if USE_POLARS:
        result = _prepare_training_data_polars(
            mtf_data, target, feature_names, min_cs, max_cs_samples, routing_decisions,
            registry_overlay_dir=registry_overlay_dir
        )
    else:
        result = _prepare_training_data_pandas(
            mtf_data, target, feature_names, min_cs, max_cs_samples, routing_decisions,
            registry_overlay_dir=registry_overlay_dir
        )
    
    # Inject metadata ONCE (clean unpack/repack - no tuple mutation)
    if isinstance(result, tuple) and len(result) >= 8:
        X, y, feature_names_out, symbols, indices, feat_cols, time_vals, routing_meta = result
        
        # Include registry overlay metadata (for reproducibility)
        if overlay_resolution:
            # DETERMINISTIC: Use resolved paths for consistent string representation
            registry_overlay_dir_str = None
            patch_file_str = None
            if overlay_resolution.overlay_dir:
                registry_overlay_dir_str = str(overlay_resolution.overlay_dir.resolve())
            if overlay_resolution.patch_file:
                patch_file_str = str(overlay_resolution.patch_file.resolve())
            
            routing_meta['registry_overlay'] = {
                'overlay_applied': True,
                'overlay_kind': overlay_resolution.overlay_kind,
                'registry_overlay_dir': registry_overlay_dir_str,
                'patch_file': patch_file_str,
                'overlay_signature': overlay_resolution.overlay_signature,
            }
        else:
            routing_meta['registry_overlay'] = {
                'overlay_applied': False,
                'overlay_kind': 'none',
                'registry_overlay_dir': None,
                'patch_file': None,
                'overlay_signature': None,
            }
        
        return X, y, feature_names_out, symbols, indices, feat_cols, time_vals, routing_meta
    
    return result

def _prepare_training_data_polars(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None,
                                 routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None,
                                 *,  # Keyword-only separator
                                 registry_overlay_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Polars-based data preparation for memory efficiency with cross-sectional sampling."""
    
    # Initialize feature auditor for tracking feature drops
    from TRAINING.ranking.utils.feature_audit import FeatureAuditor
    auditor = FeatureAuditor(target=target)
    
    logger.info(f"🎯 Building cross-sectional training data (polars, memory-efficient) for target: {target}")
    
    # Harmonize schema across symbols to avoid width mismatches on concat
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    align_mode = os.environ.get("CS_ALIGN_MODE", "union").lower()
    ordered_schema = None
    if align_cols and mtf_data:
        # DETERMINISTIC: Use lexicographically first symbol to ensure same dataframe across runs
        first_symbol = min(mtf_data.keys())
        first_df = mtf_data[first_symbol]
        if align_mode == "intersect":
            shared = None
            # DETERMINISM: Use sorted keys for deterministic schema computation
            for _sym in sorted(mtf_data.keys()):
                _df = mtf_data[_sym]
                cols = list(_df.columns)
                shared = set(cols) if shared is None else (shared & set(cols))
            ordered_schema = [c for c in first_df.columns if c in (shared or set())]
            logger.info(f"🔧 [polars] Harmonized schema (intersect) with {len(ordered_schema)} columns")
        else:
            # union
            union = []
            seen = set()
            for c in first_df.columns:
                union.append(c); seen.add(c)
            # DETERMINISM: Use sorted keys for deterministic schema computation
            for _sym in sorted(mtf_data.keys()):
                _df = mtf_data[_sym]
                for c in _df.columns:
                    if c not in seen:
                        union.append(c); seen.add(c)
            ordered_schema = union
            logger.info(f"🔧 [polars] Harmonized schema (union) with {len(ordered_schema)} columns")

    # Apply schema harmonization to mtf_data in place (if needed)
    # MEMORY OPTIMIZATION: streaming_concat handles float32 casting and incremental memory release
    if ordered_schema is not None:
        for symbol in sorted(mtf_data.keys()):
            df = mtf_data[symbol]
            if df is None:
                continue
            if align_mode == "intersect":
                mtf_data[symbol] = df[ordered_schema]
            else:
                mtf_data[symbol] = df.reindex(columns=ordered_schema)

    # Convert to streaming lazy frame using DRY helper
    # This handles: sorted symbol order, symbol column, float32 casting, memory release
    from TRAINING.data.loading import streaming_concat
    from TRAINING.common.memory import log_memory_phase, log_memory_delta
    import gc

    combined_lf = streaming_concat(
        mtf_data,
        symbol_column="symbol",
        target_column=target,
        # use_float32 defaults to config: intelligent_training.lazy_loading.use_float32
        release_after_convert=True,
    )

    # Collect with streaming mode (memory efficient for large universes)
    # MEMORY LOGGING: Track memory at phase boundaries to identify spikes
    mem_baseline = log_memory_phase("before_polars_collect")
    combined_pl = combined_lf.collect(streaming=True)
    del combined_lf
    gc.collect()
    log_memory_delta("after_polars_collect", mem_baseline)
    logger.info(f"Combined data shape (polars): {combined_pl.shape}")
    
    # Auto-discover features if not provided, then validate with registry
    if feature_names is None:
        all_cols = combined_pl.columns
        feature_names = [col for col in all_cols 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', 'timestamp', 'ts']]
    
    # Record requested features
    auditor.record_requested(feature_names)
    logger.info(f"📊 Feature audit [{target}]: {len(feature_names)} features requested")
    
    # Validate features with registry (if enabled)
    if feature_names:
        try:
            from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
            from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
            
            # Detect data interval for horizon conversion
            # DETERMINISTIC: Use lexicographically first symbol to ensure same dataframe across runs
            first_symbol = min(mtf_data.keys())
            first_df = mtf_data[first_symbol]
            detected_interval = detect_interval_from_dataframe(first_df, timestamp_column='ts', default=5)
            # Ensure interval is valid (> 0)
            if detected_interval <= 0:
                detected_interval = 5
                logger.warning(f"  Invalid detected interval, using default: 5m")
            
            # Filter features using registry
            all_columns = list(combined_pl.columns)
            validated_features = filter_features_for_target(
                all_columns,
                target,
                verbose=True,  # Enable verbose to see what's being filtered
                use_registry=True,  # Enable registry validation
                data_interval_minutes=detected_interval,
                registry_overlay_dir=registry_overlay_dir  # Use parameter passed from wrapper
            )
            
            # Record registry filtering
            auditor.record_registry_allowed(validated_features, all_columns)
            logger.info(f"📊 Feature audit [{target}]: {len(validated_features)} features allowed by registry (from {len(all_columns)} total columns)")
            
            # Keep only features that are both in feature_names and validated
            feature_names = [f for f in feature_names if f in validated_features]
            
            # FIX: Ensure basic OHLCV features are always included (they're always safe - current bar only)
            ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
            for ohlcv_feat in ohlcv_features:
                if ohlcv_feat in all_columns and ohlcv_feat not in feature_names:
                    feature_names.append(ohlcv_feat)
                    logger.debug(f"  Added missing OHLCV feature: {ohlcv_feat}")
            
            if len(feature_names) < len([f for f in feature_names if f in validated_features]):
                logger.info(f"  Feature registry: Validated {len(feature_names)} features for target {target}")
        except Exception as e:
            logger.warning(f"  Feature registry validation failed: {e}. Using provided features as-is.")
    
    # Normalize time column name
    ts_name = "timestamp" if "timestamp" in combined_pl.columns else ("ts" if "ts" in combined_pl.columns else None)
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    if ts_name:
        combined_pl = combined_pl.filter(
            pl.len().over(ts_name) >= min_cs
        )
    
    # Apply cross-sectional sampling if specified
    if max_cs_samples and ts_name:
        logger.info(f"📊 Applying cross-sectional sampling: max {max_cs_samples} samples per timestamp")
        
        # Use deterministic per-timestamp sampling with simple approach
        combined_pl = (
            combined_pl
            .sort([ts_name])
            .group_by(ts_name, maintain_order=True)
            .head(max_cs_samples)
        )
        
        logger.info(f"Cross-sectional sampling applied")
    
    # Extract target and features using polars
    try:
        # Get target column - use Polars-native extraction (no Pandas intermediate)
        y = polars_extract_column_as_numpy(combined_pl, target, dtype=np.float32, replace_inf=True)
        
        # CRITICAL FIX: Track feature pipeline stages separately
        # Stage 1: requested → allowed (registry filtering - expected)
        # Stage 2: allowed → present (schema mismatch - NOT expected, indicates bug)
        # Stage 3: present → used (dtype/nan filtering - may be expected)
        
        # Get registry-allowed count (before schema check)
        registry_allowed_count = len(feature_names)  # feature_names is already registry-filtered at this point
        requested_count = len(auditor.requested_features) if hasattr(auditor, 'requested_features') and auditor.requested_features else registry_allowed_count
        
        # Check which allowed features are actually present in Polars frame
        available_features = [f for f in feature_names if f in combined_pl.columns]
        missing_in_polars = [c for c in feature_names if c not in combined_pl.columns]
        
        # CRITICAL FIX (Pitfall B): Diagnose missing allowed features with close matches
        if missing_in_polars:
            logger.error(f"🚨 CRITICAL [{target}]: {len(missing_in_polars)} registry-allowed features missing from polars frame")
            logger.error(f"  Missing features: {missing_in_polars[:20]}{'...' if len(missing_in_polars) > 20 else ''}")
            
            # Find close matches for missing features (helps diagnose name mismatches)
            from difflib import get_close_matches
            all_polars_cols = list(combined_pl.columns)
            close_matches = {}
            for missing_feat in missing_in_polars[:10]:  # Limit to first 10 for performance
                matches = get_close_matches(missing_feat, all_polars_cols, n=3, cutoff=0.6)
                if matches:
                    close_matches[missing_feat] = matches
            if close_matches:
                logger.error(f"  Close matches found in polars columns:")
                for missing, matches in close_matches.items():
                    logger.error(f"    '{missing}' → {matches}")
            
            if auditor:
                for feat in missing_in_polars:
                    auditor.record_drop(feat, "missing_from_polars", f"Feature not in polars frame columns")
        
        # Record features present in Polars
        auditor.record_present_in_polars(combined_pl, feature_names)
        present_count = len(auditor.present_in_polars) if hasattr(auditor, 'present_in_polars') else len(available_features)
        logger.info(
            f"📊 Feature audit [{target}]: "
            f"requested={requested_count} → allowed={registry_allowed_count} → present={present_count} "
            f"(allowed→present drop: {len(missing_in_polars)})"
        )
        
        # Build feature_cols with only existing columns
        feature_cols = [target] + available_features + ['symbol'] + ([ts_name] if ts_name and ts_name in combined_pl.columns else [])
        
        # CRITICAL FIX (Pitfall A): Check threshold on allowed → present (not requested → present)
        # This prevents false positives when registry intentionally prunes features
        if registry_allowed_count > 0 and present_count < registry_allowed_count * 0.5:
            error_msg = (
                f"🚨 CRITICAL [{target}]: Feature schema mismatch detected! "
                f"Registry allowed {registry_allowed_count} features, but only {present_count} exist in polars frame "
                f"(ratio={present_count/registry_allowed_count:.1%}). "
                f"This indicates a schema breach or feature name mismatch. "
                f"Missing allowed features: {missing_in_polars[:20]}{'...' if len(missing_in_polars) > 20 else ''}"
            )
            if close_matches:
                error_msg += f"\n  Close matches found (possible name mismatches): {close_matches}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        data_pl = combined_pl.select(feature_cols)

        # Release combined_pl after selecting columns (we only need data_pl now)
        del combined_pl
        gc.collect()

        # Check config for Polars-native conversion
        # DISABLED by default - the current implementation holds intermediate DataFrames
        # longer than the Pandas path, causing higher memory usage. Needs redesign.
        use_polars_direct = False  # Default to DISABLED
        fallback_to_pandas = True
        if _CONFIG_AVAILABLE:
            try:
                use_polars_direct = get_cfg(
                    "memory.polars_conversion.polars_direct_conversion",
                    default=False, config_name="memory"
                )
                fallback_to_pandas = get_cfg(
                    "memory.polars_conversion.fallback_to_pandas",
                    default=True, config_name="memory"
                )
            except Exception:
                pass  # Use defaults

        logger.info(f"Extracted target {target} from polars data")

        # =========================================================================
        # POLARS-NATIVE PATH (memory-efficient: skips Pandas intermediate)
        # =========================================================================
        if use_polars_direct:
            try:
                mem_before_native = log_memory_phase("before_polars_native_convert")
                result = _process_combined_data_polars_native(
                    data_pl, target, y, available_features,
                    auditor=auditor, routing_decisions=routing_decisions
                )
                del data_pl
                gc.collect()
                log_memory_delta("after_polars_native_convert", mem_before_native)

                if result[0] is not None:
                    logger.info(f"✅ Polars-native conversion succeeded for {target}")
                    # Continue to audit report writing below
                else:
                    raise ValueError("Polars-native conversion returned None")

            except Exception as e:
                if fallback_to_pandas:
                    logger.warning(f"⚠️  Polars-native conversion failed for {target}: {e}. Falling back to Pandas path.")
                    # Fall through to Pandas path
                    use_polars_direct = False
                else:
                    logger.error(f"❌ Polars-native conversion failed for {target}: {e}")
                    return (None,)*8

        # =========================================================================
        # PANDAS PATH (fallback or when polars_direct_conversion=false)
        # =========================================================================
        if not use_polars_direct:
            # Convert to pandas for sklearn compatibility
            # NOTE: We avoid use_pyarrow_extension_array=True because downstream code
            # expects numpy-backed arrays (e.g., .std(), .mean() methods)
            mem_before_pandas = log_memory_phase("before_to_pandas_train")
            combined_df = data_pl.to_pandas()

            # Release Polars DataFrame IMMEDIATELY
            del data_pl
            gc.collect()
            log_memory_delta("after_to_pandas_train", mem_before_pandas)

            logger.info(f"🔍 Debug [{target}]: After polars→pandas conversion: combined_df shape={combined_df.shape}, "
                       f"feature_names count={len(feature_names)}, "
                       f"features in df={len([f for f in feature_names if f in combined_df.columns])}")

            # Continue with pandas-based processing (pass auditor for tracking and routing_decisions)
            result = _process_combined_data_pandas(combined_df, target, feature_names, auditor=auditor, routing_decisions=routing_decisions)

    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Write audit report if auditor was used
    if auditor and len(auditor.drop_records) > 0:
        try:
            # Try to get output directory from environment or use default
            import os
            output_dir = os.getenv("TRAINING_OUTPUT_DIR", "output")
            audit_dir = Path(output_dir) / "artifacts" / "feature_audits"
            audit_dir.mkdir(parents=True, exist_ok=True)
            auditor.write_report(audit_dir)
        except Exception as e:
            logger.warning(f"Failed to write feature audit report: {e}")
    
    return result

def _prepare_training_data_pandas(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None,
                                 routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None,
                                 *,  # Keyword-only separator
                                 registry_overlay_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Pandas-based data preparation (fallback)."""

    # Combine all symbol data using streaming concat (memory efficient for large universes)
    # MEMORY OPTIMIZATION: streaming_concat converts to Polars lazy frames incrementally,
    # releasing each DataFrame after conversion, then collects with streaming mode
    from TRAINING.data.loading import streaming_concat
    from TRAINING.common.memory import log_memory_phase, log_memory_delta
    import gc

    combined_lf = streaming_concat(
        mtf_data,
        symbol_column="symbol",
        target_column=target,
        # use_float32 defaults to config: intelligent_training.lazy_loading.use_float32
        release_after_convert=True,
    )

    # Collect with streaming mode and convert to pandas
    # MEMORY LOGGING: Split operations to track where spike occurs
    mem_baseline = log_memory_phase("before_collect_pandas_path")
    combined_pl = combined_lf.collect(streaming=True)
    del combined_lf
    gc.collect()
    log_memory_delta("after_collect_pandas_path", mem_baseline)
    logger.info(f"Polars DataFrame shape: {combined_pl.shape}")

    # Convert to pandas - avoid pyarrow extension arrays for numpy compatibility
    mem_before_pandas = log_memory_phase("before_to_pandas_fallback")
    combined_df = combined_pl.to_pandas()
    del combined_pl
    gc.collect()
    log_memory_delta("after_to_pandas_fallback", mem_before_pandas)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Normalize time column name
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    
    # Enforce min_cs and apply sampling
    if time_col is not None:
        # enforce min_cs
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        combined_df = combined_df[cs >= min_cs]
        # per-timestamp deterministic sampling
        if max_cs_samples:
            combined_df["_rn"] = combined_df.groupby(time_col).cumcount()
            combined_df = (combined_df
                           .sort_values([time_col, "_rn"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns="_rn"))
    
    # Auto-discover features, then validate with registry
    if feature_names is None:
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', time_col]]
    
    # Validate features with registry (if enabled)
    if feature_names:
        try:
            from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
            from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
            
            # Detect data interval for horizon conversion
            detected_interval = detect_interval_from_dataframe(combined_df, timestamp_column=time_col or 'ts', default=5)
            # Ensure interval is valid (> 0)
            if detected_interval <= 0:
                detected_interval = 5
                logger.warning(f"  Invalid detected interval, using default: 5m")
            
            # Filter features using registry
            all_columns = combined_df.columns.tolist()
            validated_features = filter_features_for_target(
                all_columns,
                target,
                verbose=True,  # Enable verbose to see what's being filtered
                use_registry=True,  # Enable registry validation
                data_interval_minutes=detected_interval,
                registry_overlay_dir=registry_overlay_dir  # Use parameter passed from wrapper
            )
            
            # Keep only features that are both in feature_names and validated
            feature_names = [f for f in feature_names if f in validated_features]
            
            # FIX: Ensure basic OHLCV features are always included (they're always safe - current bar only)
            ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
            for ohlcv_feat in ohlcv_features:
                if ohlcv_feat in all_columns and ohlcv_feat not in feature_names:
                    feature_names.append(ohlcv_feat)
                    logger.debug(f"  Added missing OHLCV feature: {ohlcv_feat}")
            
            if len(feature_names) > 0:
                logger.info(f"  Feature registry: Validated {len(feature_names)} features for target {target}")
        except Exception as e:
            logger.warning(f"  Feature registry validation failed: {e}. Using provided features as-is.")
    
    return _process_combined_data_pandas(combined_df, target, feature_names, auditor=None, routing_decisions=routing_decisions)

def _process_combined_data_polars_native(
    data_pl: pl.DataFrame,
    target: str,
    y: np.ndarray,
    feature_names: List[str],
    auditor=None,
    routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """
    Process combined data using Polars-native conversion (no Pandas intermediate).

    This is the memory-efficient path that eliminates the Polars→Pandas conversion spike.

    Args:
        data_pl: Polars DataFrame with target, features, symbol, and optionally timestamp
        target: Target column name
        y: Pre-extracted target array (already converted to numpy)
        feature_names: List of feature column names
        auditor: Optional FeatureAuditor for tracking
        routing_decisions: Optional routing decisions dict

    Returns:
        Same tuple as _process_combined_data_pandas for compatibility
    """
    from sklearn.impute import SimpleImputer

    # Route target to get task specification
    route_info = route_target(target)
    spec = route_info['spec']
    logger.info(f"[Router] Target {target} → {spec.task} task (objective={spec.objective})")

    # Enhance routing_meta with routing_decisions context if available
    route_decision_info = None
    if routing_decisions and target in routing_decisions:
        route_decision_info = routing_decisions[target]
        logger.info(f"[Routing] Using routing decision for {target}: route={route_decision_info.get('route', 'UNKNOWN')}")

    # CRITICAL FIX: feature_names is empty check
    if not feature_names:
        logger.error(f"❌ CRITICAL [{target}]: feature_names is empty or None! Cannot proceed.")
        return (None,)*8

    # Record features kept for training
    if auditor:
        auditor.record_kept_for_training(None, feature_names)
        logger.info(f"📊 Feature audit [{target}]: {len(feature_names)} features kept for training")

    # Extract time column name
    ts_name = "timestamp" if "timestamp" in data_pl.columns else ("ts" if "ts" in data_pl.columns else None)

    # Load imputation strategy from config
    if _CONFIG_AVAILABLE:
        try:
            imputation_strategy = get_cfg("preprocessing.imputation.strategy", default="median", config_name="preprocessing_config")
        except Exception:
            imputation_strategy = "median"
    else:
        imputation_strategy = "median"

    # Use Polars-native conversion (MEMORY EFFICIENT: no Pandas intermediate)
    try:
        X, final_feature_cols, impute_values = polars_to_sklearn_dense(
            data_pl, feature_names, imputation_strategy=imputation_strategy, return_impute_values=True
        )
    except ValueError as e:
        logger.error(f"❌ CRITICAL [{target}]: Polars-native conversion failed: {e}")
        return (None,)*8

    # Record features used in final X matrix
    if auditor:
        auditor.record_used_in_X(final_feature_cols, X)
        logger.info(f"📊 Feature audit [{target}]: {len(final_feature_cols)} features used in final X matrix")

    # Extract symbols (Polars native)
    symbols = polars_extract_column_as_numpy(data_pl, "symbol", dtype=None, replace_inf=False)

    # Extract time values if available
    time_vals = None
    if ts_name and ts_name in data_pl.columns:
        time_vals = data_pl.get_column(ts_name).to_numpy()

    # Compute validity mask using Polars (target validity + feature NaN ratio)
    # Note: y is already extracted, so we check for NaN/inf directly
    target_valid = ~np.isnan(y) & np.isfinite(y)

    # Feature NaN ratio (use pre-imputation state - check X directly)
    # Since X is already imputed by polars_to_sklearn_dense, we need to compute this differently
    # Actually, we can just check if all imputation went OK
    # For now, use the simpler approach: X should have no NaN after imputation
    feature_valid = ~np.isnan(X).any(axis=1) if X.shape[1] > 0 else np.ones(len(X), dtype=bool)

    valid_mask = target_valid & feature_valid

    if not valid_mask.any():
        logger.error(f"❌ [{target}]: No valid data after cleaning")
        logger.error(f"❌ [{target}]: Target stats - total={len(y)}, valid={target_valid.sum()}, "
                    f"NaN={np.isnan(y).sum()}, inf={np.sum(~np.isfinite(y))}")
        logger.error(f"❌ [{target}]: Feature stats - rows={X.shape[0]}, cols={X.shape[1]}, "
                    f"valid_rows={feature_valid.sum()}")
        return (None,)*8

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = symbols[valid_mask]
    time_vals_clean = time_vals[valid_mask] if time_vals is not None else None

    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"Removed {len(X) - len(X_clean)} rows due to cleaning")

    # Apply routing-based label preparation
    y_prepared, sample_weights, group_sizes, routing_meta = route_info['prepare_fn'](y_clean, time_vals_clean)

    # Store routing metadata for trainer
    routing_meta['target'] = target
    routing_meta['spec'] = spec
    routing_meta['sample_weights'] = sample_weights
    routing_meta['group_sizes'] = group_sizes
    routing_meta['conversion_method'] = 'polars_native'  # Track which path was used

    # Enhance routing_meta with routing_decisions context if available
    if route_decision_info:
        routing_meta['route'] = route_decision_info.get('route', 'CROSS_SECTIONAL')
        routing_meta['route_reason'] = route_decision_info.get('reason')
        for key in ['auc', 'symbol_auc_mean', 'symbol_auc_median', 'frac_symbols_good', 'winner_symbols']:
            if key in route_decision_info:
                routing_meta[key] = route_decision_info[key]

    logger.info(f"[Routing] Prepared {spec.task} task: y_shape={y_prepared.shape}, has_weights={sample_weights is not None}, has_groups={group_sizes is not None}")

    return X_clean, y_prepared, feature_names, symbols_clean, np.arange(len(X_clean)), final_feature_cols, time_vals_clean, routing_meta


def _process_combined_data_pandas(combined_df: pd.DataFrame, target: str, feature_names: List[str], auditor=None, routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Process combined data using pandas."""
    
    # Route target to get task specification
    # Use routing_decisions if available, otherwise fallback to route_target inference
    route_info = route_target(target)
    spec = route_info['spec']
    logger.info(f"[Router] Target {target} → {spec.task} task (objective={spec.objective})")
    
    # Enhance routing_meta with routing_decisions context if available
    route_decision_info = None
    if routing_decisions and target in routing_decisions:
        route_decision_info = routing_decisions[target]
        logger.info(f"[Routing] Using routing decision for {target}: route={route_decision_info.get('route', 'UNKNOWN')}")
    
    # Extract target using safe extraction
    try:
        target_series, actual_col = safe_target_extraction(combined_df, target)
        # Sanitize target: replace inf/-inf with NaN
        target_series = target_series.replace([np.inf, -np.inf], np.nan)
        y = target_series.values
        logger.info(f"Extracted target {target} from column {actual_col}")
    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Extract feature matrix - handle non-numeric columns
    # CRITICAL: Check if feature_names is empty or None
    if not feature_names:
        logger.error(f"❌ CRITICAL [{target}]: feature_names is empty or None! Cannot proceed.")
        return (None,)*8
    
    # Check which features actually exist in combined_df
    existing_features = [f for f in feature_names if f in combined_df.columns]
    missing_cols = [f for f in feature_names if f not in combined_df.columns]
    
    if not existing_features:
        logger.error(f"❌ CRITICAL [{target}]: NONE of the {len(feature_names)} selected features exist in combined_df!")
        logger.error(f"❌ [{target}]: Selected features: {feature_names[:20]}")
        logger.error(f"❌ [{target}]: Sample of combined_df columns: {list(combined_df.columns)[:20]}")
        return (None,)*8
    
    if missing_cols:
        logger.warning(f"🔍 Debug [{target}]: {len(missing_cols)} selected features missing from combined_df: {missing_cols[:10]}")
        logger.warning(f"🔍 Debug [{target}]: Using {len(existing_features)} existing features instead of {len(feature_names)}")
        feature_names = existing_features  # Use only existing features
    
    feature_df = combined_df[feature_names].copy()
    
    # Record features kept for training (before coercion)
    if auditor:
        auditor.record_kept_for_training(feature_df, feature_names)
        logger.info(f"📊 Feature audit [{target}]: {len(auditor.kept_for_training)} features kept for training (before coercion)")
    
    # DIAGNOSTIC: Log initial state before coercion
    logger.info(f"🔍 Debug [{target}]: Initial feature_df shape={feature_df.shape}, "
               f"feature_names count={len(feature_names)}, "
               f"columns in df={len([c for c in feature_names if c in feature_df.columns])}")
    
    # DIAGNOSTIC: Check NaN ratios BEFORE coercion
    if len(feature_df.columns) > 0:
        pre_coerce_nan_ratios = feature_df.isna().mean()
        all_nan_before = pre_coerce_nan_ratios[pre_coerce_nan_ratios == 1.0]
        if len(all_nan_before) > 0:
            logger.warning(f"🔍 Debug [{target}]: {len(all_nan_before)} features are ALL NaN BEFORE coercion: {list(all_nan_before.index)[:10]}")
    
    # Convert to numeric, coerce errors to NaN, and sanitize infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # DIAGNOSTIC: Check NaN ratios AFTER coercion
    if len(feature_df.columns) > 0:
        post_coerce_nan_ratios = feature_df.isna().mean()
        all_nan_after = post_coerce_nan_ratios[post_coerce_nan_ratios == 1.0]
        if len(all_nan_after) > 0:
            logger.error(f"🔍 Debug [{target}]: {len(all_nan_after)} features became ALL NaN AFTER coercion: {list(all_nan_after.index)[:10]}")
            # Log sample of what these columns look like in raw data
            sample_cols = list(all_nan_after.index)[:5]
            for col in sample_cols:
                if col in combined_df.columns:
                    raw_sample = combined_df[col].head(10)
                    raw_dtype = combined_df[col].dtype
                    logger.error(f"🔍 Debug [{target}]: Column '{col}' (dtype={raw_dtype}) raw sample: {raw_sample.tolist()}")
    
    # Drop columns that are entirely NaN after coercion
    before_cols = feature_df.shape[1]
    dropped_cols = feature_df.columns[feature_df.isna().all()].tolist()
    feature_df = feature_df.dropna(axis=1, how='all')
    dropped_all_nan = before_cols - feature_df.shape[1]
    if dropped_all_nan:
        logger.warning(f"🔧 Dropped {dropped_all_nan} all-NaN feature columns after coercion")
        if auditor:
            auditor.record_dropped_all_nan(dropped_cols, combined_df)
        
        # CRITICAL: If ALL features were dropped, this is fatal
        if feature_df.shape[1] == 0:
            logger.error(f"❌ CRITICAL [{target}]: ALL {before_cols} selected features became all-NaN after coercion!")
            logger.error(f"❌ [{target}]: Selected features: {feature_names[:20]}...")
            logger.error(f"❌ [{target}]: This indicates a mismatch between feature_names and actual data columns/dtypes")
            
            # Write debug file
            try:
                import os
                debug_dir = Path("debug_feature_coercion")
                debug_dir.mkdir(exist_ok=True)
                from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                debug_path = debug_dir / f"all_nan_features_{normalize_target_name(target)}.npz"
                np.savez_compressed(
                    debug_path,
                    feature_names=np.array(feature_names, dtype=object),
                    combined_df_columns=np.array(combined_df.columns.tolist(), dtype=object),
                    missing_cols=np.array(missing_cols, dtype=object) if missing_cols else np.array([], dtype=object),
                )
                logger.error(f"❌ [{target}]: Wrote debug file to {debug_path}")
            except Exception as e:
                logger.error(f"❌ [{target}]: Failed to write debug file: {e}")
            
            return (None,)*8
    
    # Ensure only numeric dtypes remain (guard against objects/arrays)
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        dropped_non_numeric = [c for c in feature_df.columns if c not in numeric_cols]
        feature_df = feature_df[numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns")
        if auditor:
            auditor.record_dropped_non_numeric(dropped_non_numeric, combined_df)
    
    # Build float32 matrix safely
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Record features used in final X matrix
    if auditor:
        final_feature_names = feature_df.columns.tolist()
        auditor.record_used_in_X(final_feature_names, X)
        logger.info(f"📊 Feature audit [{target}]: {len(final_feature_names)} features used in final X matrix")
    
    # CRITICAL: Guard against empty feature matrix
    if X.shape[1] == 0:
        logger.error(f"❌ CRITICAL [{target}]: Feature matrix X has 0 columns after coercion and filtering!")
        logger.error(f"❌ [{target}]: Cannot proceed with training - no usable features")
        return (None,)*8
    
    # DIAGNOSTIC: Log X shape and feature stats
    logger.info(f"🔍 Debug [{target}]: X shape={X.shape}, y shape={y.shape}, "
               f"X NaN count={np.isnan(X).sum()}, y NaN count={np.isnan(y).sum()}")
    
    # Clean data - be more lenient with NaN values
    target_valid = ~np.isnan(y)
    
    # Compute feature NaN ratio safely (handle empty X case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])  # All invalid if empty
            logger.error(f"❌ [{target}]: X has zero columns or rows - cannot compute feature_nan_ratio")
            return (None,)*8
    
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    # Treat inf in target as invalid as well
    y_is_finite = np.isfinite(y)
    valid_mask = target_valid & feature_valid & y_is_finite
    
    if not valid_mask.any():
        logger.error(f"❌ [{target}]: No valid data after cleaning")
        logger.error(f"❌ [{target}]: Target stats - total={len(y)}, valid={target_valid.sum()}, "
                    f"NaN={np.isnan(y).sum()}, inf={np.sum(~np.isfinite(y))}")
        logger.error(f"❌ [{target}]: Feature stats - rows={X.shape[0]}, cols={X.shape[1]}, "
                    f"valid_rows={feature_valid.sum()}, mean_NaN_ratio={feature_nan_ratio.mean():.2%}")
        return (None,)*8
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    
    # Get final feature column names (after all filtering/dropping)
    # This should match the columns in X_clean (which comes from feature_df after filtering)
    final_feature_cols = list(feature_df.columns)  # Actual columns after filtering/dropping
    
    # Fill remaining NaN values with median (load strategy from config if available)
    from sklearn.impute import SimpleImputer
    if _CONFIG_AVAILABLE:
        try:
            imputation_strategy = get_cfg("preprocessing.imputation.strategy", default="median", config_name="preprocessing_config")
        except Exception:
            imputation_strategy = "median"
    else:
        imputation_strategy = "median"
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_clean = imputer.fit_transform(X_clean)
    
    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"Removed {len(X) - len(X_clean)} rows due to cleaning")
    
    # Determine time column and extract time values
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # Apply routing-based label preparation
    y_prepared, sample_weights, group_sizes, routing_meta = route_info['prepare_fn'](y_clean, time_vals)
    
    # Store routing metadata for trainer
    routing_meta['target'] = target
    routing_meta['spec'] = spec
    routing_meta['sample_weights'] = sample_weights
    routing_meta['group_sizes'] = group_sizes
    
    # Enhance routing_meta with routing_decisions context if available
    if route_decision_info:
        routing_meta['route'] = route_decision_info.get('route', 'CROSS_SECTIONAL')
        routing_meta['route_reason'] = route_decision_info.get('reason')
        # Include other routing decision metadata
        for key in ['auc', 'symbol_auc_mean', 'symbol_auc_median', 'frac_symbols_good', 'winner_symbols']:
            if key in route_decision_info:
                routing_meta[key] = route_decision_info[key]
    
    logger.info(f"[Routing] Prepared {spec.task} task: y_shape={y_prepared.shape}, has_weights={sample_weights is not None}, has_groups={group_sizes is not None}")
    
    # Return with prepared labels instead of raw labels
    # Note: We return routing_meta in the imputer slot (slot 7) for now - trainer can extract it
    # feat_cols should be the actual column names after filtering (numeric_cols), not the input feature_names
    final_feature_cols = list(feature_df.columns)  # Actual columns after filtering/dropping
    return X_clean, y_prepared, feature_names, symbols_clean, np.arange(len(X_clean)), final_feature_cols, time_vals, routing_meta

