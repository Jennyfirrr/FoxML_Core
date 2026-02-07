# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Core Target Predictability Ranking Logic

This module contains the evaluate_target_predictability function and related helpers
for evaluating target predictability across symbols.

Extracted from model_evaluation.py as part of Phase 1 modular decomposition.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np

# SST: Import View and Stage enums for consistent view/stage handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import sorted_unique
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config, get_experiment_config_path, load_experiment_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.config_helpers import get_importance_top_fraction as _get_importance_top_fraction

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.common.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)



# Import dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.composite_score import calculate_composite_score
from TRAINING.ranking.predictability.data_loading import load_sample_data, prepare_features_and_target, get_model_config
from TRAINING.ranking.predictability.leakage_detection import detect_leakage, _save_feature_importances, _log_suspicious_features, find_near_copy_features, _detect_leaking_features
from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import detect_and_fix_leakage, LeakageArtifacts


# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import compute_suspicion_score as _compute_suspicion_score


# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.reporting import log_canonical_summary as _log_canonical_summary

# Import safety gate from modular components
from TRAINING.ranking.predictability.model_evaluation.safety import enforce_final_safety_gate as _enforce_final_safety_gate

# Import threading utilities for smart thread management
try:
    from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
    _THREADING_UTILITIES_AVAILABLE = True
except ImportError:
    _THREADING_UTILITIES_AVAILABLE = False

# Initialize determinism system and get BASE_SEED
try:
    from TRAINING.common.determinism import init_determinism_from_config
    BASE_SEED = init_determinism_from_config()
except ImportError:
    # Fallback: load from config directly if determinism module not available
    try:
        from CONFIG.config_loader import get_cfg
        BASE_SEED = int(get_cfg("pipeline.determinism.base_seed", default=42, config_name="pipeline_config"))
    except Exception:
        BASE_SEED = 42  # Final fallback matches pipeline.yaml default

# NOTE: _enforce_final_safety_gate is now imported from the safety submodule (see import above)
# The function was extracted as part of Phase 1 modular decomposition.


# Import train_and_evaluate_models from sibling module
from TRAINING.ranking.predictability.model_evaluation.training import train_and_evaluate_models


def evaluate_target_predictability(
    target: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = None,
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (e.g., "5m")
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: Union[str, View] = View.CROSS_SECTIONAL,  # View enum or "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or "LOSO"
    symbol: Optional[str] = None,  # Required for SYMBOL_SPECIFIC and LOSO views
    scope_purpose: str = "FINAL",  # "FINAL" or "ROUTING_EVAL" - controls where artifacts are written
    run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object for authoritative signatures
    allow_current_run_overlay: bool = False,  # NEW: explicit flag from auto-fix rerun
    attempt_id: Optional[int] = None,  # NEW: Attempt identifier for rerun tracking (defaults to 0)
    coverage_breakdowns_dict: Optional[Dict[str, Any]] = None,  # NEW: Optional dict to store coverage_breakdown (CS-only)
    registry: Optional[Any] = None  # NEW: Pass registry instance (reuse same instance)
) -> TargetPredictabilityScore:
    """Evaluate predictability of a single target across symbols"""
    
    # SST: Extract universe_sig from run_identity for consistent artifact scoping
    # This ensures all artifacts within this stage use the same universe signature
    early_universe_sig = getattr(run_identity, 'dataset_signature', None) if run_identity else None
    # Fallback: compute from symbols if run_identity doesn't have dataset_signature
    if early_universe_sig is None and symbols:
        try:
            from TRAINING.orchestration.utils.run_context import compute_universe_signature
            early_universe_sig = compute_universe_signature(symbols)
            logger.debug(f"Computed early_universe_sig={early_universe_sig[:8]}... from symbols (run_identity.dataset_signature not available)")
        except Exception as e:
            logger.debug(f"Could not compute universe_sig from symbols: {e}")
    
    # Ensure numpy is available (imported at module level, but ensure it's accessible)
    import numpy as np  # Use global import from top of file
    
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
    else:
        log_cfg = _DummyLoggingConfig()
    
    # ============================================================================
    # CONFIG TRACE: Data loading limits (with provenance)
    # ============================================================================
    import os
    config_provenance = {}
    
    # Load default max_rows_per_symbol from config if not provided
    if max_rows_per_symbol is None:
        # First check experiment config if available
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            config_provenance['max_rows_per_symbol'] = f"experiment_config.max_samples_per_symbol = {max_rows_per_symbol}"
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config")
        else:
            # Try reading from experiment config YAML directly
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    if _CONFIG_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        import yaml
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                    exp_data = exp_yaml.get('data', {})
                    if 'max_samples_per_symbol' in exp_data:
                        max_rows_per_symbol = exp_data['max_samples_per_symbol']
                        config_provenance['max_rows_per_symbol'] = f"experiment YAML data.max_samples_per_symbol = {max_rows_per_symbol}"
                        logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass
            
            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                if _CONFIG_AVAILABLE:
                    try:
                        max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                        config_provenance['max_rows_per_symbol'] = f"pipeline_config.pipeline.data_limits.default_max_rows_per_symbol_ranking = {max_rows_per_symbol} (default=50000)"
                    except Exception:
                        max_rows_per_symbol = 50000
                        config_provenance['max_rows_per_symbol'] = f"hardcoded default = 50000"
                else:
                    max_rows_per_symbol = 50000
                    config_provenance['max_rows_per_symbol'] = f"hardcoded default = 50000 (config unavailable)"
    else:
        config_provenance['max_rows_per_symbol'] = f"passed as parameter = {max_rows_per_symbol}"
    
    # Trace max_cs_samples
    if max_cs_samples is None:
        # First check experiment config YAML
        if experiment_config:
            try:
                exp_name = experiment_config.name
                if _CONFIG_AVAILABLE:
                    exp_yaml = load_experiment_config(exp_name)
                else:
                    import yaml
                    exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                    if exp_file.exists():
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                    else:
                        exp_yaml = {}
                    exp_data = exp_yaml.get('data', {})
                    if 'max_cs_samples' in exp_data:
                        max_cs_samples = exp_data['max_cs_samples']
                        config_provenance['max_cs_samples'] = f"experiment YAML data.max_cs_samples = {max_cs_samples}"
                        logger.debug(f"Using max_cs_samples={max_cs_samples} from experiment config YAML")
            except Exception:
                pass
        
        # Fallback to pipeline config
        if max_cs_samples is None:
            if _CONFIG_AVAILABLE:
                try:
                    max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
                    config_provenance['max_cs_samples'] = f"pipeline_config.pipeline.data_limits.max_cs_samples = {max_cs_samples} (default=1000)"
                except Exception:
                    max_cs_samples = 1000
                    config_provenance['max_cs_samples'] = f"hardcoded default = 1000"
            else:
                max_cs_samples = 1000
                config_provenance['max_cs_samples'] = f"hardcoded default = 1000 (config unavailable)"
    else:
        config_provenance['max_cs_samples'] = f"passed as parameter = {max_cs_samples}"
    
    # Log config trace
    logger.info("=" * 80)
    logger.info("📋 CONFIG TRACE: Data Loading Limits (with provenance)")
    logger.info("=" * 80)
    logger.info(f"   Working directory: {os.getcwd()}")
    logger.info(f"   Experiment config: {experiment_config.name if experiment_config else 'None'}")
    logger.info("")
    logger.info("   🔍 Resolved values:")
    logger.info(f"      max_rows_per_symbol: {max_rows_per_symbol}")
    logger.info(f"         Source: {config_provenance.get('max_rows_per_symbol', 'unknown')}")
    logger.info(f"      max_cs_samples: {max_cs_samples}")
    logger.info(f"         Source: {config_provenance.get('max_cs_samples', 'unknown')}")
    logger.info(f"      min_cs: {min_cs}")
    logger.info("=" * 80)
    logger.info("")
    
    # Convert dict config to TargetConfig if needed
    if isinstance(target_config, dict):
        target_column = target_config['target_column']
        display_name = target_config.get('display_name', target)
        # Infer task type from column name (will be refined with actual data)
        task_type = TaskType.from_target_column(target_column)
        target_config_obj = TargetConfig(
            name=target,
            target_column=target_column,
            task_type=task_type,
            display_name=display_name,
            **{k: v for k, v in target_config.items() 
               if k not in ['target_column', 'display_name']}
        )
    else:
        target_config_obj = target_config
        target_column = target_config_obj.target_column
        display_name = target_config_obj.display_name or target
    # Normalize view to enum for validation
    view_enum = View.from_string(view) if isinstance(view, str) else view
    
    # Validate view and symbol parameters
    if view_enum == View.SYMBOL_SPECIFIC and symbol is None:
        raise ValueError(f"symbol parameter required for View.SYMBOL_SPECIFIC view")
    # LOSO is not a View enum value, check as string
    if isinstance(view, str) and view == "LOSO" and symbol is None:
        raise ValueError(f"symbol parameter required for LOSO view")
    if view_enum == View.CROSS_SECTIONAL and symbol is not None:
        # CRITICAL FIX: Only auto-detect SYMBOL_SPECIFIC if this is actually a single-symbol run
        # Check symbols parameter (list) to determine if single-symbol
        is_single_symbol = (symbols and len(symbols) == 1) if symbols else False
        if is_single_symbol:
            logger.info(f"Auto-detecting SYMBOL_SPECIFIC view (symbol={symbol} provided with CROSS_SECTIONAL, single-symbol run)")
            view = View.SYMBOL_SPECIFIC
            view_enum = View.SYMBOL_SPECIFIC
        else:
            # Multi-symbol CROSS_SECTIONAL run - clear symbol to prevent incorrect SYMBOL_SPECIFIC detection
            logger.debug(f"CROSS_SECTIONAL run with {len(symbols) if symbols else 'unknown'} symbols - keeping CROSS_SECTIONAL view, ignoring symbol parameter")
            symbol = None
    
    # Load view from run context (SST) if available
    # For per-symbol loops, use cached view as requested_view to prevent view contract violations
    view_from_context = None
    requested_view_from_context = view  # Default to view parameter (use auto-detected view if changed above)
    try:
        from TRAINING.orchestration.utils.run_context import load_run_context
        if output_dir:
            context = load_run_context(output_dir)
            if context:
                view_from_context = context.get("view")
                # CRITICAL FIX: If view was auto-detected to SYMBOL_SPECIFIC above, use it as requested_view
                # This ensures the data preparation function receives the correct view instead of CROSS_SECTIONAL from context
                if view_enum == View.SYMBOL_SPECIFIC and symbol is not None:
                    # Use auto-detected view instead of context (prevents single-symbol runs from using CROSS_SECTIONAL)
                    requested_view_from_context = View.SYMBOL_SPECIFIC.value
                    logger.debug(f"Using auto-detected SYMBOL_SPECIFIC view as requested_view (symbol={symbol})")
                elif view_enum == View.SYMBOL_SPECIFIC and view_from_context:
                    # For per-symbol loops (SYMBOL_SPECIFIC with single symbol), use cached view as requested_view
                    # This prevents the resolver from trying to change view to SINGLE_SYMBOL_TS
                    requested_view_from_context = view_from_context
                else:
                    requested_view_from_context = context.get("requested_view") or view
    except Exception as e:
        logger.debug(f"Could not load view from run context: {e}")
    
    # Load data based on view
    # Note: Header log moved to after data prep to show resolved mode
    from TRAINING.ranking.utils.cross_sectional_data import load_mtf_data_for_ranking, prepare_cross_sectional_data_for_ranking
    from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
    from TRAINING.ranking.utils.target_conditional_exclusions import (
        generate_target_exclusion_list,
        load_target_exclusion_list
    )
    
    # For SYMBOL_SPECIFIC and LOSO, filter symbols
    symbols_to_load = symbols
    if view_enum == View.SYMBOL_SPECIFIC:
        symbols_to_load = [symbol]
    elif view == "LOSO":
        # LOSO: train on all symbols except symbol, validate on symbol
        symbols_to_load = [s for s in symbols if s != symbol]
        validation_symbol = symbol
    else:
        validation_symbol = None
    
    # ========================================================================
    # LAZY LOADING: Preflight + Probe for large universes (memory optimization)
    # ========================================================================
    # For large universes (>100 symbols), use column projection to reduce memory
    columns_to_load = None  # None = load all (backward compatible)

    logger.info(f"🔍 [LAZY_DEBUG] Checking lazy loading config for {len(symbols_to_load)} symbols...")

    try:
        from CONFIG.config_loader import get_cfg

        # First try experiment config (has precedence)
        lazy_enabled = False
        probe_enabled = True
        probe_top_n = 100
        probe_rows = 10000

        if experiment_config:
            # Check experiment config directly (higher precedence)
            exp_name = experiment_config.name if hasattr(experiment_config, 'name') else str(experiment_config)
            try:
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    import yaml
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    lazy_cfg = exp_yaml.get('intelligent_training', {}).get('lazy_loading', {})
                    if lazy_cfg:
                        lazy_enabled = lazy_cfg.get('enabled', False)
                        probe_enabled = lazy_cfg.get('probe_features', True)
                        probe_top_n = int(lazy_cfg.get('probe_top_n', 100))
                        probe_rows = int(lazy_cfg.get('probe_rows', 10000))
                        logger.info(f"🔍 [LAZY_DEBUG] Loaded from experiment config: {exp_name}")
            except Exception as e:
                logger.debug(f"Could not load experiment config: {e}")

        # Fallback to base config if not set by experiment
        if not lazy_enabled:
            lazy_enabled = get_cfg("intelligent_training.lazy_loading.enabled", default=False)
            probe_enabled = get_cfg("intelligent_training.lazy_loading.probe_features", default=True)
            probe_top_n = int(get_cfg("intelligent_training.lazy_loading.probe_top_n", default=100))
            probe_rows = int(get_cfg("intelligent_training.lazy_loading.probe_rows", default=10000))

        logger.info(f"🔍 [LAZY_DEBUG] Config loaded: lazy_enabled={lazy_enabled}, probe_enabled={probe_enabled}, "
                    f"probe_top_n={probe_top_n}, symbols={len(symbols_to_load)}")

        # Only use lazy loading for large universes (>50 symbols)
        use_lazy = lazy_enabled and len(symbols_to_load) > 50

        if not use_lazy:
            logger.info(f"🔍 [LAZY_DEBUG] Lazy loading SKIPPED: lazy_enabled={lazy_enabled}, symbols={len(symbols_to_load)} (threshold=50)")

        if use_lazy:
            logger.info(f"🎯 Lazy loading enabled for {len(symbols_to_load)} symbols")

            # Step 1: Preflight filter (schema-only leakage filtering)
            try:
                from TRAINING.ranking.utils.preflight_leakage import preflight_filter_features
                from TRAINING.data.loading.unified_loader import UnifiedDataLoader

                # Get interval from data_dir path
                interval = "5m"
                interval_minutes = 5
                data_dir_str = str(data_dir)
                if "interval=" in data_dir_str:
                    import re
                    match = re.search(r"interval=(\d+)([mhd]?)", data_dir_str)
                    if match:
                        interval = match.group(1) + (match.group(2) or 'm')
                        interval_minutes = int(match.group(1))

                loader = UnifiedDataLoader(data_dir=data_dir, interval=interval)

                # Preflight uses data_dir and symbols to read schema internally
                # DETERMINISM: Sort symbols before sampling to ensure consistent preflight results
                preflight_result = preflight_filter_features(
                    data_dir=data_dir,
                    symbols=sorted(symbols_to_load)[:20],  # Sample for schema (sorted for determinism)
                    targets=[target_column],
                    interval_minutes=interval_minutes,
                    for_ranking=True,
                    verbose=False,
                )
                preflight_features = preflight_result.get(target_column, [])

                if preflight_features:
                    logger.info(f"   📋 Preflight: schema → {len(preflight_features)} safe columns")

                    # Step 2: Probe (single-symbol importance filtering)
                    if probe_enabled and len(preflight_features) > probe_top_n:
                        from TRAINING.ranking.utils.feature_probe import probe_features_for_target

                        probed_features, _ = probe_features_for_target(
                            loader=loader,
                            symbols=symbols_to_load,
                            target=target_column,
                            preflight_features=preflight_features,
                            top_n=probe_top_n,
                            probe_rows=probe_rows,
                        )
                        logger.info(f"   🔬 Probe: {len(preflight_features)} → {len(probed_features)} important columns")
                        columns_to_load = probed_features
                    else:
                        columns_to_load = preflight_features

                    # Always include target and metadata columns
                    metadata_cols = ['ts', 'symbol', 'date', 'time', target_column]
                    for col in metadata_cols:
                        if col not in columns_to_load:
                            columns_to_load.append(col)
                    columns_to_load = sorted(set(columns_to_load))

            except Exception as e:
                logger.warning(f"⚠️ Lazy loading setup failed: {e}. Falling back to full load.")
                columns_to_load = None

    except Exception as e:
        logger.debug(f"Lazy loading config not available: {e}")

    logger.info(f"Loading data for {len(symbols_to_load)} symbol(s) (max {max_rows_per_symbol} rows per symbol)...")
    if columns_to_load:
        logger.info(f"   📊 Column projection: loading {len(columns_to_load)} columns (vs all)")
    if view == "LOSO":
        logger.info(f"  LOSO: Training on {len(symbols_to_load)} symbols, validating on {validation_symbol}")
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols_to_load, max_rows_per_symbol=max_rows_per_symbol, columns=columns_to_load)
    
    if not mtf_data:
        logger.error(f"No data loaded for any symbols")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Apply leakage filtering to feature list BEFORE preparing data (with registry validation)
    # Get all columns from first symbol to determine available features
    sample_df = next(iter(mtf_data.values()))
    all_columns = sample_df.columns.tolist()
    
    # DIAGNOSTIC: Log all_columns count to trace feature set sizes
    logger.info(
        f"[FEATURE_DIAG] After data load: len(all_columns)={len(all_columns)}, "
        f"symbol={list(mtf_data.keys())[0] if mtf_data else 'N/A'}, view={view}"
    )

    # TARGET-CONDITIONAL EXCLUSIONS: Generate per-target exclusion list
    # This implements "Target-Conditional Feature Selection" - tailoring features to target physics
    target_conditional_exclusions = []
    exclusion_metadata = {}
    target_exclusion_dir = None
    
    if output_dir:
        # Determine base output directory (RESULTS/{run}/)
        # output_dir might be: RESULTS/{run}/target_rankings/ or RESULTS/{run}/
        if output_dir.name == "target_rankings":
            base_output_dir = output_dir.parent
        else:
            base_output_dir = output_dir
        
        # Save feature exclusions to target-first structure scoped by view
        # (targets/<target>/reproducibility/<VIEW>/[symbol=...]/feature_exclusions/)
        from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
        target_clean = normalize_target_name(target)
        
        from TRAINING.orchestration.utils.target_first_paths import (
            ensure_scoped_artifact_dir, ensure_target_structure
        )
        ensure_target_structure(base_output_dir, target_clean)
        
        # CRITICAL FIX: Ensure symbol is available for SYMBOL_SPECIFIC view
        symbol_for_exclusion = symbol if ('symbol' in locals() and symbol) else None
        if view_enum == View.SYMBOL_SPECIFIC and symbol_for_exclusion is None:
            # Try to get from symbols_to_load if available
            if 'symbols_to_load' in locals() and symbols_to_load and len(symbols_to_load) == 1:
                symbol_for_exclusion = symbols_to_load[0]
                logger.debug(f"Derived symbol={symbol_for_exclusion} from symbols_to_load for feature_exclusions path")
        
        # CRITICAL: Use early_universe_sig for initial directory creation (for loading existing exclusions)
        # Will be updated to canonical universe_sig_for_writes before saving new exclusions
        target_exclusion_dir = ensure_scoped_artifact_dir(
            base_output_dir, target_clean, "feature_exclusions",
            view=view, symbol=symbol_for_exclusion, universe_sig=early_universe_sig,
            stage=Stage.TARGET_RANKING,  # Explicit stage for proper scoping
            attempt_id=attempt_id if attempt_id is not None else 0  # Per-attempt artifacts
        )
        
        # Try to load existing exclusion list first (check target-first structure)
        existing_exclusions = load_target_exclusion_list(target, target_exclusion_dir)
        if existing_exclusions is None:
            # Fallback to old target-first location (unscoped by view)
            from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
            legacy_target_dir = get_target_reproducibility_dir(base_output_dir, target_clean) / "feature_exclusions"
            existing_exclusions = load_target_exclusion_list(target, legacy_target_dir)
        if existing_exclusions is None:
            # Fallback to legacy REPRODUCIBILITY location
            legacy_repro_base = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING"
            legacy_exclusion_dir = legacy_repro_base / view / target_clean / "feature_exclusions"
            existing_exclusions = load_target_exclusion_list(target, legacy_exclusion_dir)
        if existing_exclusions is not None:
            target_conditional_exclusions = existing_exclusions
            logger.info(
                f"📋 Loaded existing target-conditional exclusions for {target}: "
                f"{len(target_conditional_exclusions)} features "
                f"(from {target_exclusion_dir})"
            )
        else:
            # Generate new exclusion list
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
            except Exception:
                registry = None
            
            # Detect interval for lookback calculation
            from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
            temp_interval = detect_interval_from_dataframe(sample_df, explicit_interval=explicit_interval)
            
            target_conditional_exclusions, exclusion_metadata = generate_target_exclusion_list(
                target=target,
                all_features=all_columns,
                interval_minutes=temp_interval,
                output_dir=target_exclusion_dir,
                registry=registry
            )
            
            # Also save to legacy location for backward compatibility
            if target_conditional_exclusions:
                try:
                    import shutil
                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                    safe_target = normalize_target_name(target)
                    exclusion_file = target_exclusion_dir / f"{safe_target}_exclusions.yaml"
                    legacy_exclusion_file = legacy_exclusion_dir / f"{safe_target}_exclusions.yaml"
                    if exclusion_file.exists():
                        shutil.copy2(exclusion_file, legacy_exclusion_file)
                        logger.debug(f"Saved exclusion file to legacy location: {legacy_exclusion_file}")
                except Exception as e:
                    logger.debug(f"Failed to copy exclusion file to legacy location: {e}")
            
            if target_conditional_exclusions:
                logger.info(
                    f"📋 Generated target-conditional exclusions for {target}: "
                    f"{len(target_conditional_exclusions)} features excluded "
                    f"(horizon={exclusion_metadata.get('target_horizon_minutes', 'unknown')}m, "
                    f"semantics={exclusion_metadata.get('target_semantics', {})})"
                )
    else:
        # No output_dir - skip target-conditional exclusions (backward compatibility)
        logger.debug("No output_dir provided - skipping target-conditional exclusions")

    # Detect data interval for horizon conversion (use explicit_interval if provided)
    from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
    detected_interval = detect_interval_from_dataframe(
        sample_df,
        timestamp_column='ts', 
        default=5,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config
    )
    
    # Extract target horizon for error messages
    from TRAINING.ranking.utils.leakage_filtering import _load_leakage_config, _extract_horizon
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    target_horizon_bars = None
    if target_horizon_minutes is not None and detected_interval > 0:
        # CRITICAL: Use centralized conversion helper (not //)
        from TRAINING.common.utils.horizon_conversion import horizon_minutes_to_bars
        target_horizon_bars = horizon_minutes_to_bars(
            horizon_minutes=float(target_horizon_minutes),
            interval_minutes=float(detected_interval)
        )
    
    # Use target-aware filtering with registry validation
    # Apply target-conditional exclusions BEFORE global filtering
    # This ensures target-specific rules are applied first
    columns_after_target_exclusions = [c for c in all_columns if c not in target_conditional_exclusions]
    
    if target_conditional_exclusions:
        logger.info(
            f"  🎯 Target-conditional exclusions: Removed {len(target_conditional_exclusions)} features "
            f"({len(columns_after_target_exclusions)} remaining before global filtering)"
        )
    
    # Determine registry_overlay_dir (explicit discovery only)
    registry_overlay_dir = None
    
    # ALLOWED: Derive from output_dir ONLY if allow_current_run_overlay=True
    # (explicit parameter from auto-fix rerun callsite, not a heuristic)
    if allow_current_run_overlay and output_dir is not None and target_column:
        patch_dir = output_dir / "registry_patches"
        if patch_dir.exists():
            from TRAINING.common.registry_patch_naming import find_patch_file
            if find_patch_file(patch_dir, target_column):
                registry_overlay_dir = patch_dir
                logger.debug(f"Using patches from current run for {target_column} (auto-fix rerun)")
    
    # PREFERRED: Explicit config override
    if experiment_config and hasattr(experiment_config, 'registry_overlay_dir'):
        registry_overlay_dir = experiment_config.registry_overlay_dir
    
    # Apply global filtering (registry, patterns, etc.)
    safe_columns = filter_features_for_target(
        columns_after_target_exclusions,  # Use pre-filtered columns
        target_column,
        verbose=True,
        use_registry=True,  # Enable registry validation
        data_interval_minutes=detected_interval,
        for_ranking=True,  # Use permissive rules for ranking (allow basic OHLCV/TA)
        dropped_tracker=dropped_tracker if 'dropped_tracker' in locals() else None,  # Pass tracker for sanitizer tracking
        registry_overlay_dir=registry_overlay_dir  # Explicit or None
    )
    
    # DIAGNOSTIC: Log safe_columns count before coverage calculation
    logger.info(
        f"[FEATURE_DIAG] After filtering: len(safe_columns)={len(safe_columns)}, "
        f"len(columns_after_target_exclusions)={len(columns_after_target_exclusions)}, "
        f"target={target_column}"
    )
    
    excluded_count = len(all_columns) - len(safe_columns) - 1  # -1 for target itself
    features_safe = len(safe_columns)
    logger.debug(f"Filtered out {excluded_count} potentially leaking features (kept {features_safe} safe features)")
    
    # Diagnostic: Count columns by suspicious prefix groups
    from collections import Counter
    def count_by_prefix(columns: List[str], prefixes: List[str]) -> Dict[str, int]:
        counts = Counter()
        for col in columns:
            for prefix in prefixes:
                if col.startswith(prefix):
                    counts[prefix] += 1
                    break
        return dict(counts)
    
    suspicious_prefixes = ['y_', 'fwd_ret_', 'p_', 'barrier_', 'ts', 'timestamp', 'symbol']
    safe_counts = count_by_prefix(safe_columns, suspicious_prefixes)
    all_counts = count_by_prefix(all_columns, suspicious_prefixes)
    
    logger.info(
        f"Feature filtering summary: all_columns={len(all_columns)}, safe_columns={len(safe_columns)}. "
        f"Suspicious prefixes in safe_columns: {safe_counts}. "
        f"Suspicious prefixes in all_columns: {all_counts}"
    )
    
    # If safe_columns still contains target columns, log warning
    if any(safe_counts.get(p, 0) > 0 for p in ['y_', 'fwd_ret_']):
        logger.warning(
            f"⚠️  Target columns still present in safe_columns after filtering! "
            f"Counts: {safe_counts}. This will cause low coverage."
        )
    
    # NEW: Track early filter drops (schema/pattern/registry filtering) - set-based comparison
    if 'dropped_tracker' in locals() and dropped_tracker is not None and 'all_columns_before_filter' in locals():
        early_filtered = sorted(list(set(all_columns_before_filter) - set(safe_columns)))
        if early_filtered:
            dropped_tracker.add_early_filter_summary(
                filter_name="schema_pattern_registry",
                dropped_count=len(early_filtered),
                top_samples=early_filtered[:10],
                rule_hits=None  # Could be enhanced to track which rules hit
            )
    
    # CRITICAL: Check if we have enough features to train
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_REQUIRED = int(ranking_cfg.get('min_features_required', 2))
        except Exception:
            MIN_FEATURES_REQUIRED = 2
    else:
        MIN_FEATURES_REQUIRED = 2
    
    if len(safe_columns) < MIN_FEATURES_REQUIRED:
        # Always log both minutes and bars for clarity
        if target_horizon_minutes is not None and target_horizon_bars is not None:
            horizon_info = f"horizon_minutes={target_horizon_minutes:.1f}m, horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
        elif target_horizon_bars is not None:
            horizon_info = f"horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
        else:
            horizon_info = "this horizon"
        logger.error(
            f"❌ INSUFFICIENT FEATURES: Only {len(safe_columns)} features remain after filtering "
            f"(minimum required: {MIN_FEATURES_REQUIRED}). "
            f"This target may not be predictable with current feature set. "
            f"Consider:\n"
            f"  1. Adding more features to CONFIG/feature_registry.yaml with allowed_horizons including {horizon_info}\n"
            f"  2. Relaxing feature registry rules for short-horizon targets\n"
            f"  3. Checking if excluded_features.yaml is too restrictive\n"
            f"  4. Skipping this target and focusing on targets with longer horizons"
        )
        # Return -999.0 to indicate this target should be skipped (same as degenerate targets)
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=target_config_obj.task_type,
            auc=-999.0,  # Flag for filtering (same as degenerate targets)
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Track feature counts (will be updated after data preparation)
    features_dropped_nan = 0
    features_final = features_safe
    
    # NEW: Track NaN drops - capture BEFORE data prep (set-based comparison)
    feature_names_before_data_prep = safe_columns.copy() if 'safe_columns' in locals() else []
    
    # CRITICAL: Initialize resolved_config early to avoid "referenced before assignment" errors
    # We need it for SYMBOL_SPECIFIC view data preparation
    resolved_config = None
    from TRAINING.ranking.utils.resolved_config import create_resolved_config
    
    # Get n_symbols_available from mtf_data (needed for resolved_config creation)
    n_symbols_available = len(mtf_data) if mtf_data is not None else 0
    
    # Create baseline resolved_config early (WITH feature lookback computation)
    # This is needed for SYMBOL_SPECIFIC view data preparation
    selected_features = safe_columns.copy() if safe_columns else []
    resolved_config = create_resolved_config(
        requested_min_cs=min_cs if view_enum != View.SYMBOL_SPECIFIC else 1,
        n_symbols_available=n_symbols_available,
        max_cs_samples=max_cs_samples,
        interval_minutes=detected_interval,
        horizon_minutes=target_horizon_minutes,
        feature_names=selected_features,
        experiment_config=experiment_config
    )
    
    # Prepare data based on view
    if view_enum == View.SYMBOL_SPECIFIC:
        # For symbol-specific, prepare single-symbol time series data
        # Use same function but with single symbol (min_cs=1 effectively)
        # allow_single_symbol=True bypasses the minimum symbol count check
        X, y, feature_names, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=1, max_cs_samples=max_cs_samples, feature_names=safe_columns,
            feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config else None,
            base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None,
            allow_single_symbol=True,  # Allow single symbol for SYMBOL_SPECIFIC view
            requested_view=requested_view_from_context,
            output_dir=output_dir
        )
        # Verify we only have one symbol
        unique_symbols = set(symbols_array) if symbols_array is not None else set()
        if len(unique_symbols) > 1:
            logger.warning(f"SYMBOL_SPECIFIC view expected 1 symbol, got {len(unique_symbols)}: {unique_symbols}")
    elif view == "LOSO":
        # LOSO: prepare training data (all symbols except validation symbol)
        X_train, y_train, feature_names_train, symbols_array_train, time_vals_train, resolved_data_config_train = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns,
            requested_view=requested_view_from_context,
            output_dir=output_dir
        )
        # Load validation symbol data separately
        validation_mtf_data = load_mtf_data_for_ranking(data_dir, [validation_symbol], max_rows_per_symbol=max_rows_per_symbol)
        X_val, y_val, feature_names_val, symbols_array_val, time_vals_val, resolved_data_config_val = prepare_cross_sectional_data_for_ranking(
            validation_mtf_data, target_column, min_cs=1, max_cs_samples=None, feature_names=safe_columns,
            allow_single_symbol=True,  # LOSO validation symbol is intentionally single-symbol
            requested_view=requested_view_from_context,
            output_dir=output_dir
        )
        # For LOSO, we'll use a special CV that trains on X_train and validates on X_val
        # For now, combine them and use a custom splitter (will be implemented in train_and_evaluate_models)
        # TODO: Implement LOSO-specific CV splitter
        logger.warning("LOSO view: Using combined data for now (LOSO-specific CV splitter not yet implemented)")
        X = X_train  # Will be handled by LOSO-specific logic
        y = y_train
        feature_names = feature_names_train
        symbols_array = symbols_array_train
        time_vals = time_vals_train
    else:
        # CROSS_SECTIONAL: standard pooled data
        X, y, feature_names, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns,
            requested_view=requested_view_from_context,
            output_dir=output_dir
        )
    
    # ========================================================================
    # SST: Use resolve_write_scope for canonical scope resolution
    # ========================================================================
    # Canonical SST-derived scope resolution with:
    # - Asymmetric rule: blocks SS→CS promotion (min_cs=1 bug)
    # - Symbol derivation: auto-derive from SST symbols[0] when unambiguous
    # - Strict mode: raises on any scope ambiguity
    from TRAINING.orchestration.utils.scope_resolution import resolve_write_scope
    
    # Check if strict mode is enabled
    strict_scope = False
    try:
        from CONFIG.config_loader import load_config
        cfg = load_config()
        strict_scope = getattr(getattr(getattr(cfg, 'safety', None), 'output_layout', None), 'strict_scope_partitioning', False)
    except Exception:
        pass
    
    # Resolve write scope using SST helper (replaces manual .get() calls)
    view_for_writes = view_from_context
    symbol_for_writes = None
    universe_sig_for_writes = None
    try:
        view_for_writes, symbol_for_writes, universe_sig_for_writes = resolve_write_scope(
            resolved_data_config=resolved_data_config,
            caller_view=view_from_context,
            caller_symbol=symbol if ('symbol' in locals() and symbol) else None,  # Pass symbol if available for SYMBOL_SPECIFIC view
            strict=strict_scope
        )
    except Exception as e:
        logger.debug(f"resolve_write_scope failed: {e}, using caller-provided values")
        # Fallback to manual extraction if resolve_write_scope fails
        if resolved_data_config:
            view_from_resolved = resolved_data_config.get("view") or view_from_context
            # CRITICAL FIX: Validate resolved view matches symbol count (same validation as resolve_write_scope)
            sst_symbols = resolved_data_config.get('symbols', [])
            if view_from_resolved == View.SYMBOL_SPECIFIC.value and len(sst_symbols) > 1:
                logger.warning(
                    f"⚠️  Fallback: Invalid resolved view=SYMBOL_SPECIFIC for multi-symbol run (n_symbols={len(sst_symbols)}). "
                    f"SYMBOL_SPECIFIC requires n_symbols=1. Using caller_view={view_from_context} instead."
                )
                view_for_writes = view_from_context
            else:
                view_for_writes = view_from_resolved
            universe_sig_for_writes = resolved_data_config.get("universe_sig")
            # ROOT CAUSE DEBUG: Log universe_sig_for_writes value
            if universe_sig_for_writes:
                logger.debug(f"ROOT CAUSE DEBUG: universe_sig_for_writes={universe_sig_for_writes[:8]}... (from resolved_data_config)")
            else:
                logger.warning(f"ROOT CAUSE DEBUG: universe_sig_for_writes is None (resolved_data_config.get('universe_sig') returned None)")
    
    # CRITICAL: Update early_universe_sig and train_universe_sig to use canonical universe_sig_for_writes if available
    # This ensures ALL artifacts (feature_exclusions, featureset_artifacts, feature_importances) use the same universe signature
    if universe_sig_for_writes:
        if universe_sig_for_writes != early_universe_sig:
            logger.debug(f"Updating early_universe_sig from {early_universe_sig[:8] if early_universe_sig else 'None'}... to canonical {universe_sig_for_writes[:8]}...")
            early_universe_sig = universe_sig_for_writes
        if 'train_universe_sig' in locals() and universe_sig_for_writes != train_universe_sig:
            logger.debug(f"Updating train_universe_sig from {train_universe_sig[:8] if train_universe_sig else 'None'}... to canonical {universe_sig_for_writes[:8]}...")
            train_universe_sig = universe_sig_for_writes
        
        # CRITICAL: If we generated new exclusions, move them to canonical batch_ directory
        # This ensures feature_exclusions are in the same batch_ directory as other artifacts
        if 'target_exclusion_dir' in locals() and target_exclusion_dir and output_dir:
            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name, ensure_scoped_artifact_dir
            safe_target = normalize_target_name(target)
            old_exclusion_file = target_exclusion_dir / f"{safe_target}_exclusions.yaml"
            if old_exclusion_file.exists():
                # Re-create directory with canonical universe_sig
                canonical_exclusion_dir = ensure_scoped_artifact_dir(
                    base_output_dir, target_clean, "feature_exclusions",
                    view=view, symbol=symbol_for_exclusion, universe_sig=universe_sig_for_writes,
                    stage=Stage.TARGET_RANKING,
                    attempt_id=attempt_id if attempt_id is not None else 0
                )
                # Move exclusion file to canonical directory if different
                if canonical_exclusion_dir != target_exclusion_dir:
                    import shutil
                    canonical_exclusion_file = canonical_exclusion_dir / f"{safe_target}_exclusions.yaml"
                    if not canonical_exclusion_file.exists():
                        shutil.move(str(old_exclusion_file), str(canonical_exclusion_file))
                        logger.debug(f"Moved exclusion file to canonical batch_ directory: {canonical_exclusion_file}")
                    target_exclusion_dir = canonical_exclusion_dir
    
    # Update header log after data prep to show resolved view
    view_final = view_for_writes
    requested_view_final = requested_view_from_context
    if resolved_data_config:
        requested_view_final = resolved_data_config.get("requested_view") or requested_view_final
    
    # Log resolved scope if different from caller view
    if view_for_writes != view_from_context:
        logger.warning(
            f"SST OVERRIDE: Using view={view_for_writes} instead of "
            f"caller view={view_from_context} for downstream writes"
        )
    if universe_sig_for_writes:
        logger.debug(f"SST universe_sig={universe_sig_for_writes[:8]}... for writes")
    
    # ========================================================================
    # END PATCH 0
    # ========================================================================
    
    # Header log showing ACTUAL write view (not just requested view)
    # Use view_for_writes if set (from SST override), otherwise fall back to view_final
    effective_view_for_log = view_for_writes if 'view_for_writes' in dir() and view_for_writes else view_final
    symbol_display = f" (symbol={symbol})" if symbol else ""
    
    # Show both requested and effective if they differ (e.g., SS->CS promotion blocked)
    if effective_view_for_log and effective_view_for_log != requested_view_final:
        view_display = f"{effective_view_for_log}{symbol_display} (requested={requested_view_final})"
    else:
        view_display = f"{effective_view_for_log or requested_view_final or 'N/A'}{symbol_display}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} ({target_column}) - {view_display}")
    logger.info(f"{'='*60}")
    
    # Update feature counts after data preparation
    if feature_names is not None:
        features_final = len(feature_names)
        features_dropped_nan = features_safe - features_final
    
    # Store cohort metadata context for later use in reproducibility tracking
    # These will be used to extract cohort metadata at the end of the function
    # ROOT CAUSE FIX: Extract universe_sig more robustly for cohort_context
    universe_sig_for_context = None
    try:
        universe_sig_for_context = universe_sig_for_writes
    except NameError:
        pass
    if not universe_sig_for_context and resolved_data_config:
        universe_sig_for_context = resolved_data_config.get('universe_sig')
    cohort_context = {
        'X': X,
        'y': y,  # Label vector for data fingerprint
        'time_vals': time_vals,
        'symbols_array': symbols_array,
        'mtf_data': mtf_data,
        'symbols': symbols,
        'min_cs': min_cs,
        'max_cs_samples': max_cs_samples,
        # SST: Use resolved universe_sig from resolve_write_scope (canonical)
        'universe_sig': universe_sig_for_context
    }
    
    if X is None or y is None:
        logger.error(f"Failed to prepare cross-sectional data for {target_column}")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # NOTE: resolved_config was already initialized earlier (before data preparation)
    # This section updates it with post-pruning feature information
    # Get n_symbols_available from mtf_data (if not already set)
    if 'n_symbols_available' not in locals():
        n_symbols_available = len(mtf_data) if mtf_data is not None else 0
    
    # Update resolved_config with post-pruning feature information
    # CRITICAL FIX: Compute feature lookback early to ensure purge is large enough
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    selected_features = feature_names.copy() if feature_names else []
    
    # Update config (WITH feature lookback computation for auto-adjustment)
    # The auto-fix logic in create_resolved_config will increase purge if feature_lookback > purge
    resolved_config = create_resolved_config(
        requested_min_cs=min_cs if view_enum != View.SYMBOL_SPECIFIC else 1,
        n_symbols_available=n_symbols_available,
        max_cs_samples=max_cs_samples,
        interval_minutes=detected_interval,
        horizon_minutes=target_horizon_minutes,
        feature_lookback_max_minutes=None,  # Will be computed from feature_names
        purge_buffer_bars=5,  # Default from config
        default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
        features_safe=features_safe,
        features_dropped_nan=features_dropped_nan,
        features_final=len(selected_features),
        view=view_for_writes if 'view_for_writes' in locals() else view,  # Use SST-resolved view
        symbol=symbol_for_writes if 'symbol_for_writes' in locals() else symbol,  # Use SST-resolved symbol
        feature_names=selected_features,  # Pass feature names for lookback computation
        recompute_lookback=True,  # CRITICAL: Compute feature lookback to auto-adjust purge
        experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
    )
    
    if log_cfg.cv_detail:
        logger.info(f"  ✅ Baseline resolved config (pre-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    logger.info(f"Cross-sectional data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Symbols: {len(set(symbols_array))} unique symbols")
    
    # Infer task type from data (needed for leak scan)
    y_sample = pd.Series(y).dropna()
    task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
    
    # PRE-TRAINING LEAK SCAN: Detect and remove near-copy features before model training
    logger.info("🔍 Pre-training leak scan: Checking for near-copy features...")
    feature_names_before_leak_scan = feature_names.copy()
    
    # Check for duplicate column names before leak scan
    if len(feature_names) != len(set(feature_names)):
        # DETERMINISM: Use sorted_unique for deterministic iteration order
        duplicates = [name for name in sorted_unique(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES DETECTED before leak scan: {duplicates}")
        raise ValueError(f"Duplicate feature names detected: {duplicates}")
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    leaky_features = find_near_copy_features(X_df, y_series, task_type)
    
    if leaky_features:
        logger.error(
            f"  ❌ CRITICAL: Found {len(leaky_features)} leaky features that are near-copies of target: {leaky_features}"
        )
        logger.error(
            f"  Removing leaky features and continuing with {X.shape[1] - len(leaky_features)} features..."
        )
        
        # Remove leaky features
        leaky_indices = [i for i, name in enumerate(feature_names) if name in leaky_features]
        X = np.delete(X, leaky_indices, axis=1)
        feature_names = [name for name in feature_names if name not in leaky_features]
        
        # CRITICAL: Reindex X columns to match feature_names order (prevent order drift)
        # After leak removal, ensure X columns match feature_names order exactly
        # This prevents "(order changed)" warnings and ensures deterministic column alignment
        if X.shape[1] != len(feature_names):
            logger.warning(
                f"  ⚠️ Column count mismatch after leak removal: X.shape[1]={X.shape[1]}, "
                f"len(feature_names)={len(feature_names)}. This should not happen."
            )
        # Note: For numpy arrays, column order is implicit via feature_names list
        # The feature_names list IS the authoritative order - X columns must match it
        
        logger.info(f"  After leak removal: {X.shape[1]} features remaining")
        from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
        
        # If we removed too many features, mark as insufficient
        # Load from config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                ranking_cfg = leakage_cfg.get('ranking', {})
                MIN_FEATURES_AFTER_LEAK_REMOVAL = int(ranking_cfg.get('min_features_after_leak_removal', 2))
            except Exception:
                MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        else:
            MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        
        if X.shape[1] < MIN_FEATURES_AFTER_LEAK_REMOVAL:
            logger.error(
                f"  ❌ Too few features remaining after leak removal ({X.shape[1]}). "
                f"Marking target as LEAKAGE_DETECTED."
            )
            return TargetPredictabilityScore(
                target=target,
                target_column=target_column,
                task_type=task_type,
                auc=-999.0,
                std_score=0.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={},
                composite_score=0.0,
                leakage_flag="LEAKAGE_DETECTED"
            )
    else:
        logger.info("  ✅ No obvious leaky features detected in pre-training scan")
        from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
    
    # CRITICAL: Early exit if too few features (before wasting time training models)
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_FOR_MODEL = int(ranking_cfg.get('min_features_for_model', 3))
        except Exception:
            MIN_FEATURES_FOR_MODEL = 3
    else:
        MIN_FEATURES_FOR_MODEL = 3
    
    if X.shape[1] < MIN_FEATURES_FOR_MODEL:
        logger.warning(
            f"Too few features ({X.shape[1]}) after filtering (minimum: {MIN_FEATURES_FOR_MODEL}); "
            f"marking target as degenerate and skipping model training."
        )
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if we get further
            auc=-999.0,  # Flag for filtering
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Task type already inferred above for leak scan
    
    # Validate target
    is_valid, error_msg = validate_target(y, task_type=task_type)
    if not is_valid:
        logger.warning(f"Skipping: {error_msg}")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=task_type,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Check if target is degenerate
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.warning(f"Skipping: Target has only {len(unique_vals)} unique value(s)")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=task_type,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # For classification, check if classes are too imbalanced for CV
    if len(unique_vals) <= 10:  # Likely classification
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.warning(f"Skipping: Smallest class has only {min_class_count} sample(s) (too few for CV)")
            return TargetPredictabilityScore(
                target=target,
                target_column=target_column,
                task_type=task_type,
                auc=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
    
    # CRITICAL: Recompute resolved_config AFTER pruning (if pruning happened)
    # This ensures feature_lookback_max is computed from actual pruned features
    # If pruning didn't happen or failed, we keep the baseline config (already assigned above)
    # Note: Pruning happens inside train_and_evaluate_models, so we need to handle it there
    # For now, we'll recompute here if feature_names changed (indicating pruning happened externally)
    # The actual post-prune recomputation happens in train_and_evaluate_models
    
    # Log baseline config summary
    if log_cfg.cv_detail:
        resolved_config.log_summary(logger)

    # Log active leakage policy (CRITICAL for audit)
    policy = "strict"  # Default
    try:
        from CONFIG.config_loader import get_cfg
        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
    except Exception:
        pass
    # Add dev_mode indicator
    dev_mode_indicator = ""
    try:
        from CONFIG.dev_mode import get_dev_mode
        if get_dev_mode():
            dev_mode_indicator = " [DEV_MODE]"
    except Exception:
        pass
    logger.info(f"🔒 Leakage policy: {policy} (strict=hard-stop, drop_features=auto-drop, warn=log-only){dev_mode_indicator}")
    
    # FINAL GATEKEEPER: Enforce safety at the last possible moment
    # This runs AFTER all loading/merging/sanitization is done
    # It physically drops features that violate the policy cap from the dataframe
    # This is the "worry-free" auto-corrector that handles race conditions
    from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
    pre_gatekeeper_fp, _ = _compute_feature_fingerprint(feature_names, set_invariant=True)
    _log_feature_set("PRE_GATEKEEPER", feature_names, previous_names=None, logger_instance=logger)
    
    # Load and parse config once
    from TRAINING.ranking.utils.leakage_budget import load_lookback_budget_spec, compute_policy_cap_minutes
    
    spec, warnings = load_lookback_budget_spec("safety_config")
    for warning in warnings:
        logger.warning(f"Config validation: {warning}")
    
    # Compute policy cap
    policy_cap_result = compute_policy_cap_minutes(spec, target_horizon_minutes, detected_interval)
    
    # Log diagnostics
    logger.info(
        f"🛡️ Gatekeeper policy cap: {policy_cap_result.cap_minutes:.1f}m "
        f"(source: {policy_cap_result.source})"
    )
    if policy_cap_result.diagnostics.get("horizon_missing"):
        logger.warning(f"⚠️ Horizon missing → using min_minutes={spec.min_minutes:.1f}m as fallback")
    if policy_cap_result.diagnostics.get("clamped"):
        clamp_info = policy_cap_result.diagnostics["clamped"]
        logger.info(
            f"📊 Policy cap clamped: {clamp_info['original']:.1f}m → {clamp_info['clamped_to']:.1f}m "
            f"(max_minutes={clamp_info['max_minutes']:.1f}m)"
        )
    
    # Call gatekeeper with policy cap
    X, feature_names, gate_report = _enforce_final_safety_gate(
        X=X,
        feature_names=feature_names,
        policy_cap_minutes=policy_cap_result.cap_minutes,  # Explicit, no None
        interval_minutes=detected_interval,
        feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config else None,
        base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None,
        logger=logger,
        dropped_tracker=dropped_tracker if 'dropped_tracker' in locals() else None
    )
    
    # Update resolved_config with gate_report
    if resolved_config and gate_report.get("enforced_feature_set"):
        resolved_config._gatekeeper_enforced = gate_report["enforced_feature_set"]
    
    # CRITICAL: Log POST_GATEKEEPER stage explicitly
    post_gatekeeper_fp, post_gatekeeper_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
    _log_feature_set("POST_GATEKEEPER", feature_names, previous_names=None, logger_instance=logger)
    
    # CRITICAL: Hard-fail check: POST_GATEKEEPER must have ZERO unknowns in strict mode
    # This is the contract: post-enforcement stages should never see unknowns
    if resolved_config and hasattr(resolved_config, '_gatekeeper_enforced'):
        enforced_gatekeeper = resolved_config._gatekeeper_enforced
        
        # PHASE 1: Persist FeatureSet artifact for debugging
        if output_dir is not None:
            try:
                from TRAINING.ranking.utils.feature_set_artifact import create_artifact_from_enforced
                artifact = create_artifact_from_enforced(
                    enforced_gatekeeper,
                    stage="POST_GATEKEEPER",
                    removal_reasons={}
                )
                # Save to target-first structure (targets/<target>/reproducibility/featureset_artifacts/)
                if target_column:
                    # Find base run directory using SST helper
                    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                    base_output_dir = get_run_root(output_dir)
                    
                    if base_output_dir.exists():
                        try:
                            from TRAINING.orchestration.utils.target_first_paths import (
                                ensure_scoped_artifact_dir, ensure_target_structure
                            )
                            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                            target_clean = normalize_target_name(target_column)
                            ensure_target_structure(base_output_dir, target_clean)
                            
                            # CRITICAL FIX: Ensure symbol is available for SYMBOL_SPECIFIC view
                            symbol_for_artifact = symbol if ('symbol' in locals() and symbol) else None
                            view_enum_for_artifact = View.from_string(view) if isinstance(view, str) else view
                            if view_enum_for_artifact == View.SYMBOL_SPECIFIC and symbol_for_artifact is None:
                                # Try to get from symbols_array if available
                                if 'symbols_array' in locals() and symbols_array is not None and len(symbols_array) > 0:
                                    unique_symbols = set(symbols_array)
                                    if len(unique_symbols) == 1:
                                        symbol_for_artifact = list(unique_symbols)[0]
                                        logger.debug(f"Derived symbol={symbol_for_artifact} from symbols_array for POST_GATEKEEPER artifact path")
                            
                            # CRITICAL: Use universe_sig_for_writes (canonical) if available, fallback to early_universe_sig
                            # This ensures featureset_artifacts uses the same universe_sig as cohort metadata
                            universe_sig_for_artifact = universe_sig_for_writes if 'universe_sig_for_writes' in locals() and universe_sig_for_writes else early_universe_sig
                            
                            target_artifact_dir = ensure_scoped_artifact_dir(
                                base_output_dir, target_clean, "featureset_artifacts",
                                view=view, symbol=symbol_for_artifact, universe_sig=universe_sig_for_artifact,
                                stage=Stage.TARGET_RANKING,  # Explicit stage for proper scoping
                                attempt_id=attempt_id if attempt_id is not None else 0  # Per-attempt artifacts
                            )
                            artifact.save(target_artifact_dir)
                            logger.debug(f"Saved POST_GATEKEEPER artifact to view-scoped location: {target_artifact_dir}")
                        except Exception as e2:
                            logger.debug(f"Failed to save POST_GATEKEEPER artifact to target-first location: {e2}")
                
                # Target-first structure only - no legacy writes
            except Exception as e:
                logger.debug(f"  ⚠️  Failed to persist POST_GATEKEEPER artifact: {e}")
        if len(enforced_gatekeeper.unknown) > 0:
            policy = "strict"
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
            except Exception:
                pass
            
            if policy == "strict":
                error_msg = (
                    f"🚨 POST_GATEKEEPER CONTRACT VIOLATION: {len(enforced_gatekeeper.unknown)} features have unknown lookback (inf). "
                    f"In strict mode, post-enforcement stages must have ZERO unknowns. "
                    f"Gatekeeper should have quarantined these. "
                    f"Sample: {enforced_gatekeeper.unknown[:10]}"
                )
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
            else:
                logger.warning(
                    f"⚠️ POST_GATEKEEPER: {len(enforced_gatekeeper.unknown)} features have unknown lookback (inf). "
                    f"Policy={policy} allows this, but this is unexpected after enforcement."
                )
        
        # CRITICAL: Boundary assertion - validate feature_names matches gatekeeper EnforcedFeatureSet
        from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
        try:
            assert_featureset_hash(
                label="POST_GATEKEEPER",
                expected=enforced_gatekeeper,
                actual_features=feature_names,
                logger_instance=logger,
                allow_reorder=False  # Strict order check
            )
        except RuntimeError as e:
            # Log but don't fail - this is a validation check
            logger.error(f"POST_GATEKEEPER assertion failed: {e}")
            # Fix it: use enforced.features (the truth)
            feature_names = enforced_gatekeeper.features.copy()
            logger.info(f"Fixed: Updated feature_names to match gatekeeper_enforced.features")
    
    # NOTE: MODEL_TRAIN_INPUT fingerprint will be computed in train_and_evaluate_models AFTER pruning
    # Pruning happens inside train_and_evaluate_models, so we can't set it here
    
    # CRITICAL: Recompute resolved_config.feature_lookback_max AFTER Final Gatekeeper
    # The audit system uses this value, so it must reflect the ACTUAL features that will be trained
    # (not the original features before the gatekeeper dropped problematic ones)
    if feature_names and len(feature_names) > 0:
        from TRAINING.ranking.utils.leakage_budget import compute_budget
        from TRAINING.ranking.utils.resolved_config import compute_feature_lookback_max
        
        # Get registry for lookback calculation
        registry = None
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception:
            pass
        
        # Load lookback_budget_minutes cap for consistency with gatekeeper
        lookback_budget_cap_for_budget = None
        budget_cap_provenance_post_gatekeeper = None
        try:
            from CONFIG.config_loader import get_cfg, get_config_path
            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            config_path = get_config_path("safety_config")
            budget_cap_provenance_post_gatekeeper = f"safety_config.yaml:{config_path} → safety.leakage_detection.lookback_budget_minutes = {budget_cap_raw} (default='auto')"
            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                lookback_budget_cap_for_budget = float(budget_cap_raw)
        except Exception as e:
            budget_cap_provenance_post_gatekeeper = f"config lookup failed: {e}"
        
        # Log config trace for budget compute
        logger.info(f"📋 CONFIG TRACE (POST_GATEKEEPER budget): {budget_cap_provenance_post_gatekeeper}")
        logger.info(f"   → max_lookback_cap_minutes passed to compute_budget: {lookback_budget_cap_for_budget}")
        
        # Compute budget from FINAL feature set (post gatekeeper)
        # NOTE: MODEL_TRAIN_INPUT fingerprint will be computed later in train_and_evaluate_models AFTER pruning
        # For now, validate against post_gatekeeper fingerprint
        budget, computed_fp, computed_order_fp = compute_budget(
            feature_names,
            detected_interval,
            resolved_config.horizon_minutes if resolved_config else 60.0,
            registry=registry,
            max_lookback_cap_minutes=lookback_budget_cap_for_budget,  # Pass cap for consistency
            expected_fingerprint=post_gatekeeper_fp,
            stage="POST_GATEKEEPER"
        )
        
        # SANITY CHECK: Verify POST_GATEKEEPER max_lookback_minutes respects the cap
        # CRITICAL: Use the canonical map that was already computed (don't recompute)
        # The budget was computed using canonical_lookback_map, so we can use that same map
        if lookback_budget_cap_for_budget is not None:
            # Get canonical map from the budget computation (it was passed in)
            # We need to recompute it here since we don't have a reference, but we'll use the same logic
            # Actually, better: use compute_feature_lookback_max which builds the canonical map correctly
            from TRAINING.ranking.utils.leakage_budget import compute_feature_lookback_max
            lookback_result = compute_feature_lookback_max(
                feature_names,
                detected_interval,
                max_lookback_cap_minutes=lookback_budget_cap_for_budget,
                registry=registry,
                expected_fingerprint=post_gatekeeper_fp,
                stage="POST_GATEKEEPER_sanity_check"
            )
            # CRITICAL: Use the EXACT SAME oracle as final enforcement
            # This is the single source of truth - if it disagrees, we have split-brain
            actual_max_from_features = lookback_result.max_minutes if lookback_result.max_minutes is not None else 0.0
            budget_max = budget.max_feature_lookback_minutes
            
            # CRITICAL: Hard-fail on mismatch (split-brain detection)
            # Both should use the same canonical map, so they MUST agree
            if abs(actual_max_from_features - budget_max) > 1.0:
                # This is a real bug - different code paths are computing different lookbacks
                logger.error(
                    f"🚨 SPLIT-BRAIN DETECTED (POST_GATEKEEPER): "
                    f"budget.max={budget_max:.1f}m vs actual_max_from_features={actual_max_from_features:.1f}m. "
                    f"This indicates different code paths are computing different lookbacks. "
                    f"Both should use the same canonical map from compute_feature_lookback_max()."
                )
                # In strict mode, this is a hard-stop
                policy = "strict"
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                except Exception:
                    pass
                
                if policy == "strict":
                    raise RuntimeError(
                        f"🚨 SPLIT-BRAIN DETECTED (POST_GATEKEEPER): "
                        f"budget.max={budget_max:.1f}m vs actual_max_from_features={actual_max_from_features:.1f}m. "
                        f"This indicates different code paths are computing different lookbacks. "
                        f"Training blocked until this is fixed."
                    )
            
            # Use actual max from features for the sanity check (the truth)
            if actual_max_from_features > lookback_budget_cap_for_budget:
                # CRITICAL: In strict mode, this is a hard-stop (gatekeeper should have caught this)
                policy = "strict"
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                except Exception:
                    pass
                
                error_msg = (
                    f"🚨 POST_GATEKEEPER sanity check FAILED: actual_max_from_features={actual_max_from_features:.1f}m > cap={lookback_budget_cap_for_budget:.1f}m. "
                    f"Gatekeeper should have dropped features exceeding cap."
                )
                
                if policy == "strict":
                    raise RuntimeError(error_msg + " (policy: strict - training blocked)")
                else:
                    logger.error(error_msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
            else:
                logger.info(
                    f"✅ POST_GATEKEEPER sanity check PASSED: actual_max_from_features={actual_max_from_features:.1f}m <= cap={lookback_budget_cap_for_budget:.1f}m"
                )
        else:
            logger.debug(f"📊 POST_GATEKEEPER max_lookback: {budget.max_feature_lookback_minutes:.1f}m (no cap set)")
        
        # Validate fingerprint consistency (invariant check)
        if computed_fp != post_gatekeeper_fp:
            logger.error(
                f"🚨 FINGERPRINT MISMATCH (POST_GATEKEEPER): compute_budget={computed_fp} != "
                f"post_gatekeeper={post_gatekeeper_fp}. "
                f"This indicates lookback computed on different feature set than enforcement."
            )
        
        # Store for validation in train_and_evaluate_models
        gatekeeper_output_fingerprint = post_gatekeeper_fp
        
        # Update resolved_config with the new lookback (from features that actually remain)
        resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
        if log_cfg.cv_detail:
            logger.info(
                f"📊 Updated feature_lookback_max after Final Gatekeeper: {budget.max_feature_lookback_minutes:.1f}m "
                f"(from {len(feature_names)} remaining features, fingerprint={computed_fp}, stage=POST_GATEKEEPER)"
            )
        
            # CRITICAL: Enforce leakage policy (strict/drop_features/warn)
            # Design: purge covers feature lookback, embargo covers target horizon
            # Validate TWO separate constraints (not a single combined requirement)
            if resolved_config.purge_minutes is not None:
                purge_minutes = resolved_config.purge_minutes
                embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                
                # Load policy and over_budget_action from config
                policy = "strict"  # Default: strict
                over_budget_action = "drop"  # Default: drop (for gatekeeper behavior)
                buffer_minutes = 5.0  # Default buffer
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                    over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
                    buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                except Exception:
                    pass
                
                # Constraint 1: purge must cover feature lookback
                purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                purge_violation = purge_minutes < purge_required
                
                # Constraint 2: embargo must cover target horizon
                # Guard: horizon_minutes may be None (e.g., for some target types)
                if budget.horizon_minutes is not None:
                    embargo_required = budget.horizon_minutes + buffer_minutes
                    embargo_violation = embargo_minutes < embargo_required
                else:
                    # If horizon is None, skip embargo validation (not applicable)
                    embargo_violation = False
                    embargo_required = None
                
                if purge_violation or embargo_violation:
                    # Build detailed violation message
                    violations = []
                    if purge_violation:
                        violations.append(
                            f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                            f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                        )
                    if embargo_violation:
                        violations.append(
                            f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                            f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                        )
                    
                    msg = f"🚨 LEAKAGE VIOLATION: {'; '.join(violations)}"
                
                if policy == "strict":
                    # Hard-stop: raise exception
                    raise RuntimeError(msg + " (policy: strict - training blocked)")
                elif policy == "drop_features":
                    # Drop features that cause violation, recompute budget
                    logger.warning(msg + " (policy: drop_features - dropping violating features)")
                    # Find features with lookback > (purge - buffer)
                    # Note: purge covers lookback, not lookback+horizon
                    max_allowed_lookback = purge_minutes - buffer_minutes
                    violating_features = []
                    for feat_name in feature_names:
                        spec_lookback = None
                        if registry is not None:
                            try:
                                metadata = registry.get_feature_metadata(feat_name)
                                lag_bars = metadata.get('lag_bars')
                                if lag_bars is not None and lag_bars >= 0:
                                    spec_lookback = float(lag_bars * detected_interval)
                            except Exception:
                                pass
                        
                        from TRAINING.ranking.utils.leakage_budget import infer_lookback_minutes
                        lookback = infer_lookback_minutes(
                            feat_name,
                            detected_interval,
                            spec_lookback_minutes=spec_lookback,
                            registry=registry
                        )
                        
                        if lookback > max_allowed_lookback:
                            violating_features.append(feat_name)
                    
                    # Drop violating features
                    if violating_features:
                        logger.warning(f"   Dropping {len(violating_features)} features with lookback > {max_allowed_lookback:.1f}m")
                        logger.info(f"   Policy: drop_features (auto-drop violating features)")
                        logger.info(f"   Drop list ({len(violating_features)} features): {', '.join(violating_features[:10])}")
                        if len(violating_features) > 10:
                            logger.info(f"   ... and {len(violating_features) - 10} more")
                        keep_indices = [i for i, name in enumerate(feature_names) if name not in violating_features]
                        X = X[:, keep_indices]
                        feature_names = [name for i, name in enumerate(feature_names) if i in keep_indices]
                        
                        # Recompute budget on remaining features
                        from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
                        after_drop_fp, after_drop_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
                        budget, budget_fp, budget_order_fp = compute_budget(
                            feature_names,
                            detected_interval,
                            resolved_config.horizon_minutes if resolved_config else 60.0,
                            registry=registry,
                            expected_fingerprint=after_drop_fp,
                            stage="after_policy_drop"
                        )
                        
                        # Validate fingerprint
                        if budget_fp != after_drop_fp:
                            logger.error(
                                f"🚨 FINGERPRINT MISMATCH (after_drop): budget={budget_fp} != expected={after_drop_fp}"
                            )
                        resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                        
                        # Verify violation is resolved (check both constraints)
                        buffer_minutes = 5.0
                        purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                        embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else resolved_config.purge_minutes
                        
                        # Guard: horizon_minutes may be None (e.g., for some target types)
                        if budget.horizon_minutes is not None:
                            embargo_required = budget.horizon_minutes + buffer_minutes
                            embargo_violation = embargo_minutes < embargo_required
                        else:
                            # If horizon is None, skip embargo validation (not applicable)
                            embargo_violation = False
                            embargo_required = None
                        
                        if resolved_config.purge_minutes < purge_required or embargo_violation:
                            violations = []
                            if resolved_config.purge_minutes < purge_required:
                                violations.append(f"purge ({resolved_config.purge_minutes:.1f}m) < {purge_required:.1f}m")
                            if embargo_violation:
                                violations.append(f"embargo ({embargo_minutes:.1f}m) < {embargo_required:.1f}m")
                            raise RuntimeError(
                                f"🚨 LEAKAGE VIOLATION PERSISTS after dropping features: {'; '.join(violations)}"
                            )
                        logger.info(
                            f"   ✅ Violation resolved: "
                            f"purge ({resolved_config.purge_minutes:.1f}m) >= {purge_required:.1f}m, "
                            f"embargo ({embargo_minutes:.1f}m) >= {embargo_required:.1f}m"
                        )
                else:  # policy == "warn"
                    # Log warning but continue (NOT recommended)
                    logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
    
    if X.shape[1] == 0:
        logger.error("❌ FINAL GATEKEEPER: All features were dropped! Cannot train models.")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=task_type,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )

    # Train and evaluate on cross-sectional data (single evaluation, not per-symbol)
    all_model_scores = []
    all_importances = []
    all_suspicious_features = {}
    fold_timestamps = None  # Initialize fold_timestamps for later use
    
    try:
        # Use detected_interval from outer scope (already computed above)
        # No need to recompute here
        
        # CRITICAL: Validate fingerprint consistency before training
        # The feature_names passed to train_and_evaluate_models should match post_gatekeeper fingerprint
        # NOTE: Pruning happens INSIDE train_and_evaluate_models, so MODEL_TRAIN_INPUT will be POST_PRUNE
        if 'gatekeeper_output_fingerprint' in locals():
            from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
            current_fp, current_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
            if current_fp != gatekeeper_output_fingerprint:
                logger.warning(
                    f"⚠️  FINGERPRINT CHANGE (pre_training): POST_GATEKEEPER={gatekeeper_output_fingerprint} -> "
                    f"pre_training={current_fp}. "
                    f"This is expected if features were modified between gatekeeper and train_and_evaluate_models."
                )
        
        # NEW: Initialize dropped features tracker for telemetry (EARLY - before any filtering)
        # This must happen BEFORE filter_features_for_target so sanitizer can track drops
        from TRAINING.ranking.utils.dropped_features_tracker import DroppedFeaturesTracker
        dropped_tracker = DroppedFeaturesTracker()
        
        # NEW: Track early filter drops (schema/pattern/registry) - capture before filter_features_for_target
        all_columns_before_filter = columns_after_target_exclusions.copy() if 'columns_after_target_exclusions' in locals() else []
        
        # Compute partial identity for quick_pruner (split_signature computed after folds created)
        partial_identity = None
        try:
            from TRAINING.common.utils.fingerprinting import (
                RunIdentity, compute_target_fingerprint, compute_routing_fingerprint
            )
            from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
            
            # Dataset signature
            dataset_payload = {
                "data_dir": str(data_dir),
                "symbols": sorted(symbols),
            }
            if max_rows_per_symbol is not None:
                dataset_payload["max_rows_per_symbol"] = max_rows_per_symbol
            dataset_signature = sha256_full(canonical_json(dataset_payload))
            
            # Target signature
            target_sig = compute_target_fingerprint(target=target_column)
            if target_sig and len(target_sig) == 16:
                target_sig = sha256_full(target_sig)
            
            # Routing signature
            view_for_id = view_for_writes if 'view_for_writes' in locals() else view
            symbol_for_id = symbol_for_writes if 'symbol_for_writes' in locals() else symbol
            routing_sig, routing_payload = compute_routing_fingerprint(
                view=view_for_id,
                symbol=symbol_for_id,
            )
            
            # Get seed - try multiple sources with fallback to default
            train_seed = None
            if experiment_config and hasattr(experiment_config, 'seed'):
                train_seed = experiment_config.seed
            if train_seed is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    train_seed = get_cfg("pipeline.determinism.base_seed", default=42)
                except Exception:
                    train_seed = 42  # FALLBACK_DEFAULT_OK
            
            # Compute hparams_signature for TARGET_RANKING evaluation models
            hparams_sig = ""
            try:
                from TRAINING.common.utils.fingerprinting import compute_hparams_fingerprint
                eval_params = {
                    "model_families": sorted(model_families) if model_families else [],
                }
                hparams_sig = compute_hparams_fingerprint(
                    model_family="target_ranking_eval",
                    params=eval_params,
                )
            except Exception:
                pass
            
            # FP-005: Create partial identity with None fallbacks (not empty strings)
            # Empty strings mask missing signatures; None is semantically correct
            partial_identity = RunIdentity(
                dataset_signature=dataset_signature,  # FP-005: None not empty string
                split_signature=None,  # Computed inside train_and_evaluate_models
                target_signature=target_sig,  # FP-005: None not empty string
                feature_signature=None,
                hparams_signature=hparams_sig,  # Hash of evaluation model families
                routing_signature=routing_sig,  # FP-005: None not empty string
                routing_payload=routing_payload,
                train_seed=train_seed,
                is_final=False,
            )
        except Exception as e:
            logger.debug(f"Failed to compute partial identity: {e}")
        
        result = train_and_evaluate_models(
            X, y, feature_names, task_type, model_families, multi_model_config,
            target_column=target_column,
            data_interval_minutes=detected_interval,  # Auto-detected or default
            time_vals=time_vals,  # Pass timestamps for fold tracking
            explicit_interval=explicit_interval,  # Pass explicit interval for consistency
            experiment_config=experiment_config,  # Pass experiment config
            output_dir=output_dir,  # Pass output directory for stability snapshots
            resolved_config=resolved_config,  # Pass resolved config with correct purge/embargo (post-pruning)
            dropped_tracker=dropped_tracker,  # Pass tracker for telemetry
            view=view_for_writes if 'view_for_writes' in locals() else view,  # Use SST-resolved view
            symbol=symbol_for_writes if 'symbol_for_writes' in locals() else symbol,  # Use SST-resolved symbol
            run_identity=partial_identity,  # Pass partial identity for quick_pruner
        )
        
        if result is None or len(result) != 7:
            logger.warning(f"train_and_evaluate_models returned unexpected value: {result}")
            return TargetPredictabilityScore(
                target=target,
                target_column=target_column,
                task_type=task_type,
                auc=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        model_metrics, primary_scores, importance, suspicious_features, feature_importances, fold_timestamps, _perfect_correlation_models = result
        
        # CRITICAL: Extract actual pruned feature count from feature_importances
        # feature_importances contains the features that were actually used (after pruning)
        actual_pruned_feature_count = 0
        if feature_importances:
            # Get feature count from first model's importances (all models use same features after pruning)
            first_model_importances = next(iter(feature_importances.values()))
            if isinstance(first_model_importances, dict):
                actual_pruned_feature_count = len(first_model_importances)
            elif isinstance(first_model_importances, (list, np.ndarray)):
                actual_pruned_feature_count = len(first_model_importances)
        # Fallback to len(feature_names) if we can't extract from importances
        if actual_pruned_feature_count == 0:
            actual_pruned_feature_count = len(feature_names) if feature_names else 0
        
        # NOTE: _perfect_correlation_models is now only for tracking/debugging.
        # High training accuracy alone is NOT a reliable leakage signal (especially for tree models),
        # so we no longer mark targets as LEAKAGE_DETECTED based on this.
        # Real leakage defense: schema filters + pre-training scan + time-purged CV.
        if _perfect_correlation_models:
            logger.debug(
                f"  Models with high training accuracy (may be overfitting): {_perfect_correlation_models}. "
                f"Check CV metrics to assess real predictive power."
            )
        
        # Save aggregated feature importances (respect view: CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
        # PATCH 4: Use SST-derived view/symbol/universe_sig for proper scoping
        # ROOT CAUSE DEBUG: Log condition check
        has_feature_importances = bool(feature_importances)
        has_output_dir = output_dir is not None
        logger.debug(f"ROOT CAUSE DEBUG: feature_importances check - has_feature_importances={has_feature_importances}, has_output_dir={has_output_dir}, feature_importances_keys={list(feature_importances.keys()) if feature_importances else 'None'}")
        if feature_importances and output_dir:
            # Use SST-derived values from Patch 0
            # Normalize view_for_importances - use enum value for consistency
            view_for_importances_raw = view_for_writes if 'view_for_writes' in locals() else (view if 'view' in locals() else View.CROSS_SECTIONAL)
            view_for_importances = View.from_string(view_for_importances_raw) if isinstance(view_for_importances_raw, str) else view_for_importances_raw
            symbol_for_importances = symbol_for_writes if 'symbol_for_writes' in locals() else (symbol if ('symbol' in locals() and symbol) else None)
            # FIX 1: Don't set universe_sig_for_importances here - it will be set later after universe_sig_for_writes is available
            # universe_sig_for_importances will be set when building _feature_importances_to_save (line 6942)
            
            # CRITICAL FIX: Auto-detect SYMBOL_SPECIFIC view if symbol is provided (same logic as line 5309-5313)
            # BUT ONLY if this is actually a single-symbol run (not multi-symbol CROSS_SECTIONAL)
            if view_for_importances == View.CROSS_SECTIONAL and symbol_for_importances is not None:
                # Validate this is actually a single-symbol run before auto-detecting
                is_single_symbol = False
                if 'resolved_data_config' in locals() and resolved_data_config:
                    sst_symbols = resolved_data_config.get('symbols', [])
                    is_single_symbol = len(sst_symbols) == 1
                elif 'symbols_array' in locals() and symbols_array is not None:
                    is_single_symbol = len(symbols_array) == 1
                
                if is_single_symbol:
                    logger.info(f"Auto-detecting SYMBOL_SPECIFIC view for feature importances (symbol={symbol_for_importances} provided with CROSS_SECTIONAL, single-symbol run)")
                    view_for_importances = View.SYMBOL_SPECIFIC
                else:
                    # Multi-symbol CROSS_SECTIONAL run - clear symbol to prevent SYMBOL_SPECIFIC detection
                    num_symbols = len(sst_symbols) if ('resolved_data_config' in locals() and resolved_data_config and resolved_data_config.get('symbols')) else (len(symbols_array) if ('symbols_array' in locals() and symbols_array is not None) else 'unknown')
                    logger.debug(f"CROSS_SECTIONAL run with {num_symbols} symbols - keeping CROSS_SECTIONAL view, clearing symbol for feature importances")
                    symbol_for_importances = None
            
            # CRITICAL FIX: If view is SYMBOL_SPECIFIC, ensure symbol is set
            if view_for_importances == View.SYMBOL_SPECIFIC and symbol_for_importances is None:
                # Fallback: try to get symbol from function parameter or resolved_data_config
                if 'symbol' in locals() and symbol:
                    symbol_for_importances = symbol
                    logger.debug(f"Using symbol={symbol_for_importances} from function parameter for SYMBOL_SPECIFIC view")
                elif 'resolved_data_config' in locals() and resolved_data_config:
                    sst_symbols = resolved_data_config.get('symbols', [])
                    if len(sst_symbols) == 1:
                        symbol_for_importances = sst_symbols[0]
                        logger.debug(f"Derived symbol={symbol_for_importances} from resolved_data_config for SYMBOL_SPECIFIC view")
                if symbol_for_importances is None:
                    logger.warning(f"SYMBOL_SPECIFIC view but symbol is None - feature importances may fail to save")
            
            # Compute RunIdentity for target ranking snapshots
            target_ranking_identity = None
            try:
                from TRAINING.common.utils.fingerprinting import (
                    RunIdentity, compute_target_fingerprint, compute_routing_fingerprint,
                    compute_split_fingerprint, compute_hparams_fingerprint,
                    compute_feature_fingerprint_from_specs,
                    canonicalize_timestamp, get_identity_mode
                )
                from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
                
                # Dataset signature (finance-safe: no path noise, includes date range)
                # Get identity mode from config
                identity_mode = get_identity_mode()
                # For identity computation, ALWAYS assume UTC for naive timestamps.
                # This is deterministic (same naive timestamp -> same UTC result) and
                # doesn't affect other strict mode behaviors (required fields, validation).
                # Naive timestamps are a DATA FORMAT issue, not a reproducibility concern.
                assume_utc = True
                
                # Extract date range - handle empty and failures properly
                data_start_utc = None
                data_end_utc = None
                timestamp_canon_failed = False
                empty_time_vals = False
                
                if time_vals is None or len(time_vals) == 0:
                    # Empty timestamps - different handling by mode
                    empty_time_vals = True
                    if identity_mode == "strict":
                        raise ValueError("Cannot compute dataset signature: time_vals is empty (strict mode)")
                    else:
                        logger.warning("time_vals is empty - dataset identity may be incomplete")
                else:
                    try:
                        # Epoch-safe min/max computation
                        # Note: numpy/pandas imports here are scoped to this block only
                        # They don't affect module-level pd which is imported at top of file
                        import numpy as _np_local
                        import pandas as _pd_local
                        arr = _np_local.asarray(time_vals)
                        
                        if _np_local.issubdtype(arr.dtype, _np_local.number):
                            # Epoch array: compute min/max numerically, then canonicalize with unit inference
                            ts_min_raw = int(arr.min())
                            ts_max_raw = int(arr.max())
                            data_start_utc = canonicalize_timestamp(ts_min_raw, assume_utc_for_naive=assume_utc)
                            data_end_utc = canonicalize_timestamp(ts_max_raw, assume_utc_for_naive=assume_utc)
                        else:
                            # Datetime-like: type-stable vectorized conversion
                            t = _pd_local.to_datetime(time_vals, utc=False, errors="raise")
                            ts_min, ts_max = t.min(), t.max()
                            data_start_utc = canonicalize_timestamp(ts_min, assume_utc_for_naive=assume_utc)
                            data_end_utc = canonicalize_timestamp(ts_max, assume_utc_for_naive=assume_utc)
                    except Exception as e:
                        if identity_mode == "strict":
                            raise ValueError(f"Cannot compute dataset signature: timestamp canonicalization failed: {e}")
                        else:
                            logger.warning(f"Timestamp canonicalization failed (relaxed mode): {e}")
                            timestamp_canon_failed = True
                
                # Build payload - finance-safe, no path noise
                dataset_payload = {
                    # Core identity
                    "symbols_digest": sha256_full(canonical_json(sorted(symbols))),
                    "start_ts_utc": data_start_utc,
                    "end_ts_utc": data_end_utc,
                    
                    # Row-shaping filters
                    "max_rows_per_symbol": max_rows_per_symbol,
                    
                    # Sampling method versioning
                    "sampling_method": "stable_seed_from:v1",
                    
                    # Row footprint
                    "n_rows_total": len(time_vals) if time_vals is not None else None,
                    
                    # Failure/empty markers (prevent false hash matches)
                    "timestamp_canon_failed": timestamp_canon_failed if timestamp_canon_failed else None,
                    "empty_time_vals": empty_time_vals if empty_time_vals else None,
                }
                # data_dir REMOVED - filesystem noise, not identity
                
                dataset_signature = sha256_full(canonical_json(dataset_payload))
                
                # Target signature
                target_signature = compute_target_fingerprint(target=target_column)
                if target_signature and len(target_signature) == 16:
                    target_signature = sha256_full(target_signature)
                
                # Routing signature
                routing_signature, routing_payload = compute_routing_fingerprint(
                    view=view_for_importances,
                    symbol=symbol_for_importances,
                )
                
                # Split signature from fold_timestamps
                split_signature = None
                if fold_timestamps:
                    fold_boundaries = []
                    for fold_info in fold_timestamps:
                        if isinstance(fold_info, dict):
                            start = fold_info.get('train_start')
                            end = fold_info.get('test_end')
                            if start and end:
                                fold_boundaries.append((start, end))
                    if fold_boundaries:
                        # Get purge/embargo from resolved_config if available
                        purge_min = resolved_config.purge_minutes if resolved_config and hasattr(resolved_config, 'purge_minutes') else 0.0
                        embargo_min = resolved_config.embargo_minutes if resolved_config and hasattr(resolved_config, 'embargo_minutes') else 0.0
                        split_signature = compute_split_fingerprint(
                            cv_method="purged_kfold",
                            n_folds=len(fold_boundaries),
                            purge_minutes=purge_min,
                            embargo_minutes=embargo_min,
                            fold_boundaries=fold_boundaries,
                            split_seed=None,
                        )
                
                # Get seed - SST fallback chain
                train_seed = None
                if experiment_config and hasattr(experiment_config, 'seed'):
                    train_seed = experiment_config.seed
                if train_seed is None:
                    try:
                        from CONFIG.config_loader import get_cfg
                        train_seed = get_cfg("pipeline.determinism.base_seed", default=42)
                    except Exception:
                        train_seed = 42  # FALLBACK_DEFAULT_OK
                
                # FIX: Create per-model identity dict to avoid strict_key collisions
                # Each model family gets its own identity based on its hparams_signature
                per_model_identities = {}  # Dict[str, RunIdentity]
                
                # Check if we have any feature importances to process
                if not feature_importances:
                    logger.warning("No feature_importances available - cannot compute local target_ranking_identity")
                else:
                    for model_family, model_importances in feature_importances.items():
                        if not model_importances:
                            continue
                        
                        try:
                            # Hparams signature for this family
                            family_config = {}
                            if multi_model_config and 'model_families' in multi_model_config:
                                family_config = multi_model_config['model_families'].get(model_family, {})
                            hparams_signature = compute_hparams_fingerprint(
                                model_family=model_family,
                                params=family_config.get('params', family_config),
                            )
                            
                            # Feature signature from features in this model's importances (registry-resolved)
                            from TRAINING.common.utils.fingerprinting import resolve_feature_specs_from_registry
                            feature_specs = resolve_feature_specs_from_registry(list(model_importances.keys()))
                            feature_signature = compute_feature_fingerprint_from_specs(feature_specs)
                            
                            # FP-005: Create partial and finalize for THIS model family
                            # Use None not empty strings for missing signatures
                            partial = RunIdentity(
                                dataset_signature=dataset_signature,  # FP-005: None not empty string
                                split_signature=split_signature,  # FP-005: None not empty string
                                target_signature=target_signature,  # FP-005: None not empty string
                                feature_signature=None,
                                hparams_signature=hparams_signature,  # FP-005: None not empty string
                                routing_signature=routing_signature,  # FP-005: None not empty string
                                routing_payload=routing_payload,
                                train_seed=train_seed,
                                is_final=False,
                            )
                            per_model_identities[model_family] = partial.finalize(feature_signature)
                        except Exception as model_e:
                            logger.debug(f"Failed to compute identity for {model_family}: {model_e}")
                        # NO break - continue loop to build identity for ALL models
            except Exception as e:
                logger.warning(f"Failed to compute RunIdentity for target ranking: {e}")  # Upgraded from debug
            
            # Compute registry_overlay_signature if patches are loaded (before finalization)
            if run_identity and not run_identity.is_final:
                from TRAINING.common.utils.fingerprinting import compute_registry_signature
                
                # Determine overlay and override directories
                persistent_override_dir = None
                persistent_unblock_dir = None
                if target_column:
                    repo_root = Path(__file__).resolve().parents[3]
                    persistent_dir = repo_root / "CONFIG" / "data" / "feature_registry_per_target"
                    if persistent_dir.exists():
                        persistent_override_dir = persistent_dir
                        persistent_unblock_dir = persistent_dir  # Same directory, different suffix
                
                # Get overlay fingerprint and registry version hash
                auto_overlay_effective_hash = None
                registry_version_hash = None
                allow_overwrite = None
                
                try:
                    from TRAINING.common.feature_registry import get_registry
                    registry = get_registry(target_column=target_column)
                    
                    # Get overlay fingerprint
                    overlay_fingerprint = registry.get_overlay_fingerprint() if hasattr(registry, 'get_overlay_fingerprint') else None
                    if overlay_fingerprint:
                        auto_overlay_effective_hash = overlay_fingerprint.get('effective_hash')
                    
                    # Get registry version hash (compute from registry config_path)
                    if hasattr(registry, 'config_path') and registry.config_path and registry.config_path.exists():
                        import hashlib
                        registry_version_hash = hashlib.sha256(registry.config_path.read_bytes()).hexdigest()
                    
                    # Get allow_overwrite policy (from config)
                    try:
                        from CONFIG.config_loader import get_cfg
                        allow_overwrite = get_cfg('registry_autopatch.allow_overwrite', default=False)
                    except Exception:
                        allow_overwrite = False
                except Exception as e:
                    logger.debug(f"Could not get overlay fingerprint/registry version for signature: {e}")
                
                # Compute signature from effective policy used during this run
                registry_signature = compute_registry_signature(
                    registry_overlay_dir=registry_overlay_dir,  # From current run if allow_current_run_overlay=True
                    persistent_override_dir=persistent_override_dir,
                    persistent_unblock_dir=persistent_unblock_dir,
                    target_column=target_column,
                    current_bar_minutes=detected_interval,
                    auto_overlay_effective_hash=auto_overlay_effective_hash,
                    registry_version_hash=registry_version_hash,
                    allow_overwrite=allow_overwrite
                )
                
                # Set on run_identity (before finalization)
                if registry_signature:
                    run_identity.registry_overlay_signature = registry_signature
                    logger.debug(f"Set registry_overlay_signature: {registry_signature[:16]}...")
            
            # FIX: Pass per-model identity dict if we have any, else fallback to shared identity
            if per_model_identities:
                identity_for_save = per_model_identities  # Dict[str, RunIdentity]
                logger.debug(f"Created per-model identities for {len(per_model_identities)} model families")
            else:
                identity_for_save = run_identity  # Fallback to shared identity
                if identity_for_save is None:
                    # Check strict mode and log appropriately
                    try:
                        from TRAINING.common.determinism import is_strict_mode
                        if is_strict_mode():
                            logger.error("STRICT MODE: No RunIdentity available for feature importance snapshot - reproducibility compromised")
                        else:
                            logger.warning("No RunIdentity available for feature importance snapshot (non-strict mode)")
                    except Exception:
                        logger.warning("No RunIdentity available for feature importance snapshot")
            
            # FIX 5: Feature importances will be saved after valid_for_ranking is determined (see below)
            # Store reference for later conditional save
            # FIX 1: Use universe_sig_for_writes directly (canonical) if available, otherwise None
            # This ensures we use the correct universe signature that was set after resolved_data_config
            # Remove unreliable locals() check - use variable directly with fallback to resolved_data_config
            universe_sig_for_save = None
            try:
                # Try to use universe_sig_for_writes if it exists
                universe_sig_for_save = universe_sig_for_writes
            except NameError:
                # Variable doesn't exist, try resolved_data_config
                pass
            # Fallback: extract directly from resolved_data_config if universe_sig_for_writes is None
            if not universe_sig_for_save and 'resolved_data_config' in locals() and resolved_data_config:
                universe_sig_for_save = resolved_data_config.get('universe_sig')
                if universe_sig_for_save:
                    logger.debug(f"ROOT CAUSE DEBUG: universe_sig_for_save={universe_sig_for_save[:8]}... (from resolved_data_config fallback)")
            # ROOT CAUSE DEBUG: Log final universe_sig_for_save value
            if not universe_sig_for_save:
                logger.warning(f"ROOT CAUSE DEBUG: universe_sig_for_save is None after all fallbacks - feature importances may not be saved")
            else:
                logger.debug(f"ROOT CAUSE DEBUG: universe_sig_for_save={universe_sig_for_save[:8]}... (final value)")
            _feature_importances_to_save = {
                'target_column': target_column,
                'symbol': symbol_for_importances,
                'importances': feature_importances,
                'output_dir': output_dir,
                'view': view_for_importances,
                'universe_sig': universe_sig_for_save,
                'run_identity': identity_for_save,
                'model_metrics': model_metrics,
                'attempt_id': attempt_id if attempt_id is not None else 0,
            }
        else:
            _feature_importances_to_save = None
        
        # Store suspicious features
        if suspicious_features:
            all_suspicious_features = suspicious_features
            # symbol_for_log should be symbol, not view - this looks like a bug, but preserve behavior
            symbol_for_log = symbol if ('symbol' in locals() and symbol) else None
            _log_suspicious_features(target_column, symbol_for_log, suspicious_features, output_dir=output_dir)
        
        # AUTO-FIX LEAKAGE: If leakage detected, automatically fix and re-run
        # Initialize autofix_info to None (will be set if auto-fixer runs)
        autofix_info = None
        
        # Extract leakage detection into reusable function (Phase 1: Extract and validate)
        # Build TargetContext if needed (will be created inside detect_and_fix_leakage if None)
        target_ctx_for_leakage = None
        try:
            from TRAINING.common.leakage_auto_fixer import TargetContext
            target_ctx_for_leakage = TargetContext.from_target(
                target=target,
                bar_minutes=detected_interval,  # SST-driven (from resolved_config or auto-detection)
                experiment_config=experiment_config
            )
            if target_ctx_for_leakage is None:
                logger.debug(f"Could not create TargetContext for {target}, will be created inside detect_and_fix_leakage if needed")
        except Exception as e:
            logger.debug(f"Could not create TargetContext for leakage detection: {e}, will be created inside detect_and_fix_leakage if needed")
        
        # Build LeakageArtifacts from available data
        leakage_artifacts = LeakageArtifacts(
            model_metrics=model_metrics,
            primary_scores=primary_scores,
            feature_importances=feature_importances,
            perfect_correlation_models=_perfect_correlation_models,
            X=X,
            y=y,
            feature_names=feature_names,
            safe_columns=safe_columns if 'safe_columns' in locals() else None
        )
        
        # Build io_context dict
        io_context = {
            'output_dir': output_dir,
            'experiment_config': experiment_config,
            'run_identity': run_identity,
            'target_column': target_column,
            'symbols_array': symbols_array if 'symbols_array' in locals() else None,
            'task_type': task_type,
            'detected_interval': detected_interval
        }
        
        # Call extracted leakage detection function
        should_rerun, autofix_info, detector_failed = detect_and_fix_leakage(
            target_ctx=target_ctx_for_leakage,
            artifacts=leakage_artifacts,
            io_context=io_context,
            autofix_enabled=True  # TARGET_RANKING always enables autofix (report-only mode is for FEATURE_SELECTION)
        )
        
        if detector_failed:
            logger.warning("⚠️  Leakage detection failed (non-fatal, continuing execution)")
        
        # Define should_auto_fix for reporting (based on whether autofix_info exists)
        # This replaces the old should_auto_fix variable that was in the extracted code
        should_auto_fix = autofix_info is not None
        
        # OLD CODE BELOW - REPLACED BY EXTRACTED FUNCTION
        # Keeping for reference during validation, will be removed after testing
        """
        # Load thresholds from config (with sensible defaults)
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                auto_fix_cfg = leakage_cfg.get('auto_fix_thresholds', {})
                cv_threshold = float(auto_fix_cfg.get('cv_score', 0.99))
                accuracy_threshold = float(auto_fix_cfg.get('training_accuracy', 0.999))
                r2_threshold = float(auto_fix_cfg.get('training_r2', 0.999))
                correlation_threshold = float(auto_fix_cfg.get('perfect_correlation', 0.999))
                auto_fix_enabled = leakage_cfg.get('auto_fix_enabled', True)
                auto_fix_min_confidence = float(leakage_cfg.get('auto_fix_min_confidence', 0.8))
                auto_fix_max_features = int(leakage_cfg.get('auto_fix_max_features_per_run', 20))
            except Exception as e:
                logger.debug(f"Failed to load leakage detection config: {e}, using defaults")
                cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
                accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
                r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
                correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
                auto_fix_min_confidence = 0.8  # FALLBACK_DEFAULT_OK
                auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        else:
            # FALLBACK_DEFAULT_OK: Fallback defaults (config not available)
            cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
            accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
            r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
            correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
            auto_fix_min_confidence = 0.8
            auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        
        # Check if auto-fixer is enabled
        if not auto_fix_enabled:
            logger.debug("Auto-fixer is disabled in config")
            should_auto_fix = False
        else:
            should_auto_fix = False
            
            # Check 1: Perfect CV scores (cross-validation)
            # CRITICAL: Use actual CV scores from model_scores (primary_scores), not model_metrics
            # model_metrics may contain training scores, but model_scores contains CV scores
            max_cv_score = None
            if primary_scores:
                # primary_scores contains CV scores from cross_val_score
                valid_cv_scores = [s for s in primary_scores.values() if s is not None and not np.isnan(s)]
                if valid_cv_scores:
                    max_cv_score = max(valid_cv_scores)
            
            # Fallback: try to extract from model_metrics if primary_scores unavailable
            # But be careful - model_metrics['accuracy'] etc. should now contain CV scores after our fix above
            if max_cv_score is None and model_metrics:
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        # Get CV score (should be CV after our fix, but double-check it's not training_accuracy)
                        cv_score_val = metrics.get('roc_auc') or metrics.get('r2') or metrics.get('accuracy')
                        # Exclude training scores explicitly
                        if cv_score_val is not None and not np.isnan(cv_score_val):
                            # Skip if this looks like a training score (training_accuracy exists and matches)
                            if 'training_accuracy' in metrics and abs(cv_score_val - metrics['training_accuracy']) < 0.001:
                                continue  # This is likely a training score, skip it
                            if max_cv_score is None or cv_score_val > max_cv_score:
                                max_cv_score = cv_score_val
            
            if max_cv_score is not None and max_cv_score >= cv_threshold:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect CV scores detected (max_cv={max_cv_score:.4f} >= {cv_threshold:.1%}) - enabling auto-fix mode")
            
            # Check 2: Perfect in-sample training accuracy with suspicion score gating
            # Use suspicion score to distinguish overfit noise from real leakage
            if not should_auto_fix and model_metrics:
                logger.debug(f"Checking model_metrics for perfect scores: {list(model_metrics.keys())}")
                
                # Compute suspicion score for each model with perfect train accuracy
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        logger.debug(f"  {model_name} metrics: {list(metrics.keys())}")
                        
                        # Get train and CV scores
                        train_acc = metrics.get('training_accuracy')
                        cv_acc = metrics.get('accuracy')  # CV accuracy
                        train_r2 = metrics.get('training_r2')
                        cv_r2 = metrics.get('r2')  # CV R²
                        
                        # Check classification
                        if train_acc is not None and train_acc >= accuracy_threshold:
                            logger.debug(f"    {model_name} training_accuracy: {train_acc:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_acc,
                                cv_score=cv_acc,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='classification'
                            )
                            
                            # Only auto-fix if suspicion score crosses threshold
                            suspicion_threshold = 0.5  # Load from config if available
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training accuracy in {model_name} "
                                            f"(train={train_acc:.1%}, cv={cv_acc_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                # Overfit noise - log once at INFO level
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_acc:.1%}, "
                                         f"cv={cv_acc_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_acc is not None and cv_acc >= accuracy_threshold:
                            # CV accuracy alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV accuracy detected in {model_name} "
                                        f"({cv_acc:.1%} >= {accuracy_threshold:.1%}) - enabling auto-fix mode")
                            break
                        
                        # Check regression
                        if train_r2 is not None and train_r2 >= r2_threshold:
                            logger.debug(f"    {model_name} training_r2 (correlation): {train_r2:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_r2,
                                cv_score=cv_r2,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='regression'
                            )
                            
                            suspicion_threshold = 0.5
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training correlation in {model_name} "
                                            f"(train={train_r2:.4f}, cv={cv_r2_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_r2:.4f}, "
                                         f"cv={cv_r2_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_r2 is not None and cv_r2 >= r2_threshold:
                            # CV R² alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV R² detected in {model_name} "
                                        f"({cv_r2:.4f} >= {r2_threshold:.4f}) - enabling auto-fix mode")
                            break
            
            # Check 3: Models that triggered perfect correlation warnings (fallback check)
            # Note: _perfect_correlation_models is populated inside train_and_evaluate_models,
            # but we check model_metrics above which covers the same cases, so this is just a safety check
            if not should_auto_fix and _perfect_correlation_models:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect correlation detected in models: {', '.join(_perfect_correlation_models)} (>= {correlation_threshold:.1%}) - enabling auto-fix mode")
        
        if should_auto_fix:
            try:
                from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer, TargetContext
                
                logger.info("🔧 Auto-fixing detected leaks...")
                logger.info(f"   Initializing LeakageAutoFixer (backups disabled)...")
                # Backups are disabled by default - no backup directory will be created
                fixer = LeakageAutoFixer(backup_configs=False, output_dir=output_dir)
                
                # Create TargetContext from SST (detected_interval is SST-driven bar_minutes)
                target_ctx = TargetContext.from_target(
                    target=target,
                    bar_minutes=detected_interval,  # SST-driven (from resolved_config or auto-detection)
                    experiment_config=experiment_config
                )
                
                if target_ctx is None:
                    logger.warning(f"Could not create TargetContext for {target}, skipping per-target registry updates")
                    target_ctx = None
                
                # Get run_id for evidence tracking
                run_id = None
                if run_identity:
                    run_id = getattr(run_identity, 'run_id', None) or str(run_identity)
                
                # Convert X to DataFrame if needed (auto-fixer expects DataFrame)
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = X
                
                # Convert y to Series if needed
                if not isinstance(y, pd.Series):
                    y_series = pd.Series(y)
                else:
                    y_series = y
                
                # Aggregate feature importances across all models
                aggregated_importance = {}
                if feature_importances:
                    # Sort model names for deterministic order (ensures reproducible aggregations)
                    for model_name in sorted(feature_importances.keys()):
                        importances = feature_importances[model_name]
                        if isinstance(importances, dict):
                            for feat, imp in importances.items():
                                if feat not in aggregated_importance:
                                    aggregated_importance[feat] = []
                                aggregated_importance[feat].append(imp)
                
                # Average importance across models (sort features for deterministic order)
                avg_importance = {feat: np.mean(imps) for feat, imps in sorted(aggregated_importance.items())} if aggregated_importance else {}
                
                # Get actual training accuracy from model_metrics (not CV scores)
                # This is critical - we detected perfect training accuracy, so pass that value
                actual_train_score = None
                if model_metrics:
                    for model_name, metrics in model_metrics.items():
                        if isinstance(metrics, dict):
                            # For classification, prefer training_accuracy (in-sample), fall back to CV accuracy
                            if 'training_accuracy' in metrics and metrics['training_accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['training_accuracy']
                                logger.debug(f"Using training accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'accuracy' in metrics and metrics['accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['accuracy']
                                logger.debug(f"Using CV accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            # For regression, prefer training_r2 (in-sample correlation), fall back to CV R²
                            elif 'training_r2' in metrics and metrics['training_r2'] >= r2_threshold:
                                actual_train_score = metrics['training_r2']
                                logger.debug(f"Using training correlation {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'r2' in metrics and metrics['r2'] >= r2_threshold:
                                actual_train_score = metrics['r2']
                                logger.debug(f"Using CV R² {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                
                # Fallback to CV score if no perfect training score found
                # CRITICAL: Use the same max_cv_score we computed above for consistency
                if actual_train_score is None:
                    if max_cv_score is not None:
                        actual_train_score = max_cv_score
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from model_metrics)")
                    else:
                        actual_train_score = max(primary_scores.values()) if primary_scores else None
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from primary_scores)")
                
                # Log what we're passing to auto-fixer (enhanced visibility)
                # CRITICAL: Clarify which feature set is being used for scanning vs training
                train_feature_set_size = len(feature_names)  # Features used for training (after pruning)
                scan_feature_set_size = len(safe_columns) if 'safe_columns' in locals() else len(feature_names)  # Features available for scanning
                scan_scope = "full_safe" if scan_feature_set_size > train_feature_set_size else "trained_only"
                
                train_score_str = f"{actual_train_score:.4f}" if actual_train_score is not None else "None"
                logger.info(f"🔧 Auto-fixer inputs: train_score={train_score_str}, "
                           f"train_feature_set_size={train_feature_set_size}, "
                           f"scan_feature_set_size={scan_feature_set_size}, "
                           f"scan_scope={scan_scope}, "
                           f"model_importance keys={len(avg_importance)}")
                if avg_importance:
                    top_5 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.debug(f"   Top 5 features by importance: {', '.join([f'{f}={imp:.4f}' for f, imp in top_5])}")
                else:
                    logger.warning(f"   ⚠️  No aggregated importance available! feature_importances keys: {list(feature_importances.keys()) if feature_importances else 'None'}")
                
                # Detect leaks
                detections = fixer.detect_leaking_features(
                    X=X_df, y=y_series, feature_names=feature_names,
                    target_column=target_column,
                    symbols=pd.Series(symbols_array) if symbols_array is not None else None,
                    task_type='classification' if task_type == TaskType.BINARY_CLASSIFICATION or task_type == TaskType.MULTICLASS_CLASSIFICATION else 'regression',
                    data_interval_minutes=detected_interval,
                    model_importance=avg_importance if avg_importance else None,
                    train_score=actual_train_score,
                    test_score=None  # CV scores are already validation scores
                )
                
                if detections:
                    logger.warning(f"🔧 Auto-detected {len(detections)} leaking features")
                    # Apply fixes (with high confidence threshold to avoid false positives)
                    updates, autofix_info = fixer.apply_fixes(
                        detections, 
                        min_confidence=auto_fix_min_confidence, 
                        max_features=auto_fix_max_features,
                        dry_run=False,
                        target=target,
                        target_ctx=target_ctx,  # NEW: explicit context for per-target patches
                        run_id=run_id  # NEW: for evidence tracking
                    )
                    if autofix_info.modified_configs:
                        logger.info(f"✅ Auto-fixed leaks. Configs updated.")
                        logger.info(f"   Updated: {len(updates.get('excluded_features_updates', {}).get('exact_patterns', []))} exact patterns, "
                                  f"{len(updates.get('excluded_features_updates', {}).get('prefix_patterns', []))} prefix patterns")
                        logger.info(f"   Rejected: {len(updates.get('feature_registry_updates', {}).get('rejected_features', []))} features in registry")
                    else:
                        logger.warning("⚠️  Auto-fix detected leaks but no configs were modified")
                        logger.warning("   This usually means all detections were below confidence threshold")
                        logger.warning(f"   Check logs above for confidence distribution details")
                    # Log backup info if available
                    if autofix_info.backup_files:
                        logger.info(f"📦 Backup created: {len(autofix_info.backup_files)} backup file(s)")
                else:
                    logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
                    # Still create backup even when no leaks detected (to preserve state history)
                    # This ensures we have a backup whenever auto-fix mode is triggered
                    # But only if backup_configs is enabled
                    backup_files = []
                    if fixer.backup_configs:
                        try:
                            backup_files = fixer._backup_configs(
                                target=target,
                                max_backups_per_target=None  # Use instance config
                            )
                            if backup_files:
                                logger.info(f"📦 Backup created (no leaks detected): {len(backup_files)} backup file(s)")
                        except Exception as backup_error:
                            logger.warning(f"Failed to create backup when no leaks detected: {backup_error}")
            except Exception as e:
                logger.warning(f"Auto-fix failed: {e}", exc_info=True)
        """
        # END OF OLD CODE - REPLACED BY EXTRACTED FUNCTION
        
        # Ensure primary_scores is a dict
        if primary_scores is None:
            logger.warning(f"primary_scores is None, skipping")
            return TargetPredictabilityScore(
                target=target,
                target_column=target_column,
                task_type=task_type,
                auc=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        if not isinstance(primary_scores, dict):
            logger.warning(f"primary_scores is not a dict (got {type(primary_scores)}), skipping")
            return TargetPredictabilityScore(
                target=target,
                target_column=target_column,
                task_type=task_type,
                auc=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        all_model_scores.append(primary_scores)
        all_importances.append(importance)
        
        scores_str = ", ".join([f"{k}={v:.3f}" for k, v in primary_scores.items()])
        logger.info(f"Scores: {scores_str}, importance={importance:.2f}")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.warning(f"Failed: {error_msg}")
        logger.error(f"Full traceback:\n{tb_str}")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=task_type,
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    if not all_model_scores:
        logger.warning(f"No successful evaluations for {target} (skipping)")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if target succeeds
            auc=-999.0,  # Flag for degenerate/failed targets
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Aggregate across models (skip NaN scores)
    # Note: With cross-sectional data, we only have one evaluation, not per-symbol
    all_scores_by_model = defaultdict(list)
    all_fold_scores = []  # Collect all fold scores across all models for distributional analysis
    for scores_dict in all_model_scores:
        # Defensive check: skip None or non-dict entries
        if scores_dict is None or not isinstance(scores_dict, dict):
            logger.warning(f"Skipping invalid scores_dict: {type(scores_dict)}")
            continue
        for model_name, score in scores_dict.items():
            if not (np.isnan(score) if isinstance(score, (float, np.floating)) else False):
                all_scores_by_model[model_name].append(score)
                # If score is from a single fold (not aggregated), add to fold_scores
                # Note: We'll collect actual fold scores separately if available
    
    # Calculate statistics (only from models that succeeded)
    model_means = {model: np.mean(scores) for model, scores in all_scores_by_model.items() if scores}
    if not model_means:
        logger.warning(f"No successful model evaluations for {target}")
        return TargetPredictabilityScore(
            target=target,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default
            auc=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            leakage_flag="OK",
            suspicious_features=None
        )
    
    auc = np.mean(list(model_means.values()))
    std_score = np.std(list(model_means.values())) if len(model_means) > 1 else 0.0
    mean_importance = np.mean(all_importances)
    consistency = 1.0 - (std_score / (abs(auc) + 1e-6))
    
    # Determine task type (already inferred from data above)
    final_task_type = task_type
    
    # === P0 CORRECTNESS: Compute new snapshot contract fields ===
    # FIX 4: n_cs_valid should be number of valid timestamps, not number of models
    # Use time_vals if available (deterministic - comes from data loading)
    # For CROSS_SECTIONAL: count unique timestamps (multiple symbols per timestamp)
    # For SYMBOL_SPECIFIC: count unique timestamps (one symbol, multiple timestamps)
    if 'time_vals' in locals() and time_vals is not None and len(time_vals) > 0:
        # Count unique timestamps (deterministic - np.unique returns sorted array)
        n_cs_valid = len(np.unique(time_vals[~np.isnan(time_vals)])) if np.any(~np.isnan(time_vals)) else len(time_vals)
        n_cs_total = n_cs_valid  # Total = valid (after filtering)
    else:
        # Fallback: use X length (deterministic - same data always gives same length)
        n_cs_valid = len(X) if 'X' in locals() and X is not None else 0
        n_cs_total = n_cs_valid
    
    # Track invalid reasons (why model evaluations failed)
    # Note: In current architecture, these are tracked per-model-family not per-timestamp
    invalid_reason_counts = {}
    invalid_count = n_cs_total - n_cs_valid
    if invalid_count > 0:
        invalid_reason_counts["model_evaluation_failed"] = invalid_count
    
    # === FIX: Extract IC from model_metrics for regression (not R²) ===
    # For regression, auc contains R² (can be negative), but we need IC for routing
    # model_metrics contains full metrics dict: {model_name: {'ic': float, 'r2': float, ...}}
    ic_mean = None
    if final_task_type == TaskType.REGRESSION:
        # Check if model_metrics is available (it's assigned earlier in the function)
        try:
            if model_metrics and isinstance(model_metrics, dict):
                ic_values = []
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict) and 'ic' in metrics:
                        ic_val = metrics.get('ic')
                        if ic_val is not None and not np.isnan(ic_val):
                            ic_values.append(ic_val)
                if ic_values:
                    ic_mean = np.mean(ic_values)
                    logger.debug(f"Extracted IC from model_metrics: {len(ic_values)} models, mean IC={ic_mean:.4f}")
                else:
                    logger.warning(f"No valid IC values found in model_metrics for regression target {target}")
                    # Fallback: use 0.0 (null baseline) if IC not available
                    ic_mean = 0.0
            else:
                logger.warning(f"model_metrics not available or invalid for regression target {target}, using 0.0 as fallback")
                ic_mean = 0.0
        except NameError:
            # model_metrics not in scope (shouldn't happen, but defensive)
            logger.warning(f"model_metrics not in scope for regression target {target}, using 0.0 as fallback")
            ic_mean = 0.0
    
    # Compute t-stat for skill normalization (universal signal-above-null metric)
    # Note: This will be recomputed in calculate_composite_score_tstat with proper guards,
    # but we compute it here for backward compatibility in snapshot
    primary_metric_tstat = None
    if n_cs_valid > 1 and std_score > 0:
        # Use centered primary_mean for t-stat computation
        # t-stat = mean / (std / sqrt(n)) = mean / se
        primary_se_temp = std_score / np.sqrt(n_cs_valid)
        if primary_se_temp > 0:
            # Use centered mean (will be set below)
            primary_metric_tstat = None  # Will be set after we compute centered mean
    elif n_cs_valid == 1:
        # With only one model, can't compute t-stat (no variance estimate)
        primary_metric_tstat = None
    
    # Classification-specific: compute centered AUC for proper aggregation
    auc_mean_raw = None
    auc_excess_mean = None
    if final_task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        # Check for failed target (auc = -999.0 is sentinel for failed evaluation)
        if auc < -900.0 or np.isnan(auc):
            # Failed target: don't compute excess, use null baseline
            auc_mean_raw = None
            auc_excess_mean = None
        else:
            auc_mean_raw = auc  # Raw 0-1 scale
            # Center around 0.5 (null baseline for random classifier)
            auc_excess_mean = auc - 0.5
    
    # === Phase 3.1: Use centered primary_metric_mean for composite score ===
    # Regression: Use IC (centered at 0, null baseline) - FIXED: extract from model_metrics, not R²
    # Classification: Use AUC-excess (centered at 0, null baseline)
    if final_task_type == TaskType.REGRESSION:
        if ic_mean is not None:
            primary_metric_mean_centered = ic_mean  # IC already centered at 0
        else:
            # Fallback: if IC not available, use 0.0 (null baseline)
            # This should not happen if model_metrics is properly populated
            logger.warning(f"IC not available for regression target {target}, using 0.0 as fallback")
            primary_metric_mean_centered = 0.0
    else:
        # Classification: use AUC-excess if available, otherwise 0.0 (null baseline)
        if auc_excess_mean is not None:
            primary_metric_mean_centered = auc_excess_mean  # AUC - 0.5, centered at 0
        else:
            # Failed target or invalid auc: use 0.0 (null baseline)
            primary_metric_mean_centered = 0.0
    
    # Compute primary_se (standard error) for SE-based stability
    if n_cs_valid > 1:
        primary_se = std_score / np.sqrt(n_cs_valid)
    elif n_cs_valid == 1:
        primary_se = std_score  # Conservative estimate with one observation
    else:
        primary_se = 0.0
    
    # Get metric name for logging
    if final_task_type == TaskType.REGRESSION:
        metric_name = "R²"
    elif final_task_type == TaskType.BINARY_CLASSIFICATION:
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        metric_name = "Accuracy"
    
    # Composite score using Phase 3.1 t-stat based normalization
    # This uses centered primary_mean, SE-based stability, and skill-gated composite
    try:
        from TRAINING.ranking.predictability.composite_score import calculate_composite_score_tstat
        
        # Load run_intent from config (SST style - same pattern as other config loading in this file)
        run_intent = None
        if _CONFIG_AVAILABLE:
            try:
                from CONFIG.config_loader import get_cfg
                # Try experiment config first (highest priority)
                if experiment_config and hasattr(experiment_config, 'run_intent'):
                    run_intent = experiment_config.run_intent
                else:
                    # Fallback to pipeline config
                    run_intent = get_cfg("pipeline.ranking.run_intent", default=None, config_name="pipeline_config")
            except Exception:
                pass  # Default to None (will default to "eval" in calculate_composite_score_tstat)
        
        # Compute registry_coverage_rate BEFORE composite score calculation (fixes timing issue)
        # This ensures eligibility gate has the value when it's checked
        # FIX ISSUE-015/007: Read telemetry for diagnostics only (do NOT set registry_coverage_rate)
        # Mismatch telemetry must never block canonical coverage computation for eligibility
        mismatch_registry_coverage_rate = None
        if 'mismatch_telemetry' in locals() and mismatch_telemetry:
            mismatch_registry_coverage_rate = mismatch_telemetry.get("registry_coverage_rate")
            # Guard formatting - only format if numeric to prevent crash
            if mismatch_registry_coverage_rate is not None and isinstance(mismatch_registry_coverage_rate, (int, float)):
                logger.debug(f"Previous run coverage: {mismatch_registry_coverage_rate:.2%} (diagnostic only, not used for eligibility)")
        
        # Always reset eligibility variables to force fresh computation
        # CRITICAL: Eligibility coverage must come from canonical computation, not telemetry
        registry_coverage_rate = None
        coverage_breakdown = None
        coverage_computation_error = False  # Track if computation failed (error) vs missing (genuine lack)
        
        # CRITICAL: registry is a function parameter - don't overwrite it!
        # The parameter already has default None, so no need to initialize here.
        # Removing the overwrite that was blocking the registry parameter from being used.
        
        # Initialize interval_for_filtering before try block (prevents UnboundLocalError in exception handler)
        # detected_interval is computed earlier in function (line 5582), so it's available in scope
        # Prefer detected_interval (already computed), else fallback to 5 (canonical default from detect_interval_from_dataframe)
        interval_for_filtering = detected_interval if (detected_interval and detected_interval > 0) else 5
        
        # CRITICAL: Avoid truthiness check on pandas/polars objects (if safe_columns is DataFrame, this raises ValueError)
        # Use explicit None check and len() safely
        n_features = None
        if safe_columns is not None:
            try:
                n_features = len(safe_columns)
            except Exception:
                n_features = None  # Handle case where len() fails (unlikely but defensive)
        
        # FIX ISSUE-015: Compute canonical coverage unconditionally when safe_columns exists
        # Do NOT gate on registry_coverage_rate - that variable should not control eligibility computation
        if safe_columns is not None and n_features is not None and n_features > 0:
            try:
                from TRAINING.ranking.utils.registry_coverage import compute_registry_coverage
                
                # Use passed registry if provided (ensures same instance)
                # CRITICAL: No hidden get_registry() fallback - explicit policy
                if registry is None:
                    try:
                        from TRAINING.common.determinism import is_strict_mode
                        strict_mode = is_strict_mode()
                    except Exception:
                        strict_mode = False
                    
                    if strict_mode:
                        # Strict mode: registry is required for coverage computation
                        # Include callsite context for debugging
                        target_name = target if 'target' in locals() else "unknown"
                        view_name = view if 'view' in locals() else "unknown"
                        raise ValueError(
                            f"registry is None in strict mode (stage=target_ranking, target={target_name}, view={view_name}). "
                            f"Coverage computation requires registry instance. "
                            f"Registry must be constructed in IntelligentTrainer.rank_targets_auto() and passed down. "
                            f"Set determinism mode to best_effort to skip coverage computation."
                        )
                    else:
                        # Best-effort mode: log and skip coverage/autopatch
                        logger.warning(
                            f"registry is None in best_effort mode. "
                            f"Skipping registry coverage computation and autopatch collection. "
                            f"Pass registry parameter for full functionality."
                        )
                        # Skip coverage computation entirely (don't call compute_registry_coverage)
                        coverage_breakdown = None
                        registry_coverage_rate = None
                        coverage_computation_error = False
                
                # Compute canonical coverage
                # Add instrumentation breadcrumbs to trace execution flow
                import time
                coverage_call_id = f"{time.monotonic_ns():x}"[-8:]  # Monotonic identifier (last 8 hex chars)
                try:
                    from TRAINING.common.determinism import is_strict_mode
                    strict_mode = is_strict_mode()
                except Exception:
                    strict_mode = False
                error_policy = "strict" if strict_mode else "best_effort"
                
                logger.debug(
                    "[coverage_call_id=%s] About to call compute_registry_coverage target=%s interval=%s strict_mode=%s",
                    coverage_call_id, target, interval_for_filtering, strict_mode
                )
                
                coverage_breakdown = compute_registry_coverage(
                    feature_names=safe_columns,
                    target=target_column if 'target_column' in locals() else target,
                    interval_minutes=interval_for_filtering,
                    horizon_minutes=None,  # Will be extracted from target name
                    registry=registry,
                    experiment_config=experiment_config if 'experiment_config' in locals() else None,
                    view=(view.value if hasattr(view, 'value') else view) if 'view' in locals() else None,
                    error_policy=error_policy
                )
                
                if coverage_breakdown:
                    cov_total_str = f"{coverage_breakdown.coverage_total:.4f}" if coverage_breakdown.coverage_total is not None else "N/A"
                    logger.info(
                        f"[coverage_call_id={coverage_call_id}] compute_registry_coverage returned: "
                        f"mode={coverage_breakdown.coverage_mode}, "
                        f"n_total={coverage_breakdown.n_total}, "
                        f"coverage_total={cov_total_str}, "
                        f"id={id(coverage_breakdown)}"
                    )
                else:
                    logger.info(
                        f"[coverage_call_id={coverage_call_id}] compute_registry_coverage returned: None"
                    )
                
                # Check for error_summary (best_effort mode error evidence)
                if coverage_breakdown and hasattr(coverage_breakdown, 'error_summary') and coverage_breakdown.error_summary:
                    logger.warning(
                        "[coverage_call_id=%s] Registry coverage computation failed (best_effort mode): %s: %s. "
                        "Check logs above for full traceback.",
                        coverage_call_id,
                        coverage_breakdown.error_summary.get('exception_type'),
                        coverage_breakdown.error_summary.get('exception_message')
                    )
                
                # Extract coverage rate based on mode
                # CRITICAL: Check if coverage_breakdown is None (can happen if compute_registry_coverage raises exception)
                if coverage_breakdown is None:
                    # This should not happen - compute_registry_coverage always returns CoverageBreakdown
                    # If we reach here, an exception occurred and coverage_breakdown was set to None in handler
                    # Only warn if not in legitimate modes (membership_only/unknown return valid CoverageBreakdown)
                    # CRITICAL: coverage_computation_error is guaranteed defined (initialized before try)
                    if not coverage_computation_error:
                        logger.warning(
                            f"Target {target}: coverage_breakdown is None after computation. "
                            f"This indicates an exception occurred (check logs above for REGISTRY_COVERAGE_ERROR)."
                        )
                    registry_coverage_rate = None
                elif coverage_breakdown.coverage_mode == "horizon_ok":
                    registry_coverage_rate = coverage_breakdown.coverage_total
                    # DIAGNOSTIC: Log what coverage rate is being used
                    logger.info(
                        f"[COVERAGE_DIAG] Using coverage_total for eligibility: {registry_coverage_rate:.4f} "
                        f"(from coverage_breakdown: n_total={coverage_breakdown.n_total}, "
                        f"n_in_registry_horizon_ok={coverage_breakdown.n_in_registry_horizon_ok})"
                    )
                elif coverage_breakdown.coverage_mode == "membership_only":
                    # Membership-only coverage must NOT feed eligibility gate in prod
                    # Set to None so gate treats it as missing
                    registry_coverage_rate = None
                    logger.warning(
                        f"Target {target}: coverage_mode='membership_only' (horizon conversion failed). "
                        f"Membership coverage={coverage_breakdown.coverage_in_registry:.2%}, "
                        f"but not using for eligibility gate (different metric)."
                    )
                else:  # unknown
                    registry_coverage_rate = None
                    logger.warning(
                        f"Target {target}: coverage_mode='unknown' (invalid interval/horizon). "
                        f"Cannot compute coverage for eligibility gate."
                    )
                
                # Store coverage_breakdown only for CS view (registry is global, SS is per-symbol)
                if coverage_breakdowns_dict is not None and coverage_breakdown is not None:
                    # Check if view is CROSS_SECTIONAL (enum or string)
                    is_cs_view = False
                    if hasattr(view, 'value'):
                        is_cs_view = (view.value == "CROSS_SECTIONAL" or view == View.CROSS_SECTIONAL)
                    elif isinstance(view, str):
                        is_cs_view = (view.upper() == "CROSS_SECTIONAL")
                    else:
                        is_cs_view = (view == View.CROSS_SECTIONAL)
                    
                    if is_cs_view:
                        # Use target as key (deterministic, CS-only)
                        coverage_breakdowns_dict[target] = coverage_breakdown
                
                # FIX ISSUE-002: Defensive formatting for coverage_total (None-safe)
                # Log coverage breakdown for diagnostics
                if coverage_breakdown is not None and coverage_breakdown.coverage_mode == "horizon_ok":
                    cov_str = f"{coverage_breakdown.coverage_total:.2%}" if coverage_breakdown.coverage_total is not None else "N/A"
                    logger.debug(
                        f"Canonical registry coverage for {target}: "
                        f"n_in_registry={coverage_breakdown.n_in_registry}/{coverage_breakdown.n_total}, "
                        f"n_horizon_ok={coverage_breakdown.n_in_registry_horizon_ok}, "
                        f"coverage_total={cov_str}, "
                        f"mode={coverage_breakdown.coverage_mode}"
                    )
                    # Log n_total for sanity check (if identical across targets, universe is wrong)
                    logger.debug(f"Coverage sanity check: n_total={coverage_breakdown.n_total} for target {target}")
            except Exception as e:
                # CRITICAL: Distinguish computation error from genuine missing coverage
                coverage_computation_error = True
                registry_coverage_rate = None
                coverage_breakdown = None
                
                # Log exception with context (logger.exception includes full traceback automatically)
                # CRITICAL: All variables are initialized, so no UnboundLocalError
                # Try to get registry path for context
                registry_path = None
                try:
                    from TRAINING.common.feature_registry import get_registry_path
                    registry_path = str(get_registry_path()) if get_registry_path() else None
                except Exception:
                    registry_path = None
                
                # Compute summary if coverage_breakdown was computed before error
                coverage_summary = None
                if 'coverage_breakdown' in locals() and coverage_breakdown is not None:
                    try:
                        from TRAINING.ranking.utils.registry_coverage import summarize_coverage_breakdown
                        coverage_summary = summarize_coverage_breakdown(coverage_breakdown)
                    except Exception:
                        pass
                
                logger.exception(
                    "[coverage_call_id=%s] REGISTRY_COVERAGE_ERROR: compute_registry_coverage failed "
                    "target=%s interval=%s view=%s symbol=%s n_features=%s registry_loaded=%s registry_path=%s. "
                    "Coverage summary: %s",
                    coverage_call_id if 'coverage_call_id' in locals() else 'unknown',
                    target_column if 'target_column' in locals() else target,
                    interval_for_filtering,
                    (view.value if hasattr(view, 'value') else view) if 'view' in locals() else None,
                    symbol if 'symbol' in locals() else None,
                    n_features,  # Use pre-computed value (safe)
                    registry is not None,  # Safe - initialized before try
                    registry_path,
                    coverage_summary
                )
                # Additional context for debugging
                if registry is None:
                    logger.debug("Registry was None - may indicate registry loading failure")
                if 'experiment_config' in locals() and experiment_config:
                    logger.debug(f"Experiment config available: {type(experiment_config).__name__}")
        
        # FIX ISSUE-015: Regression net - eligibility must not accept float-only coverage when canonical inputs exist
        # This invariant prevents someone from "helpfully" reintroducing registry_coverage_rate assignment later
        if safe_columns is not None and n_features is not None and n_features > 0 and coverage_breakdown is None:
            # Check if this is a computation error (from above) or genuine missing
            coverage_computation_error = 'coverage_computation_error' in locals() and coverage_computation_error
            
            if coverage_computation_error:
                # Already logged as REGISTRY_COVERAGE_ERROR above - don't log again
                # Will be handled by eligibility gate with REGISTRY_COVERAGE_ERROR reason
                pass
            else:
                # Genuine missing (computation succeeded but returned None/missing mode)
                logger.error(f"CRITICAL: safe_columns exists but coverage_breakdown is None. This violates SST - eligibility must use canonical coverage.")
                # Force compute as last resort (should not happen if code is correct)
                try:
                    from TRAINING.ranking.utils.registry_coverage import compute_registry_coverage
                    interval_for_filtering = detected_interval if ('detected_interval' in locals() and detected_interval and detected_interval > 0) else 5
                    # Use parameter if passed, otherwise try to load
                    if registry is None:
                        try:
                            from TRAINING.common.feature_registry import get_registry
                            registry = get_registry()
                        except Exception as e_registry:
                            logger.debug(f"Could not load registry for force compute: {e_registry}")
                            pass
                    
                    # Ensure all required variables are available
                    target_for_coverage = target_column if ('target_column' in locals() and target_column) else target
                    exp_config = experiment_config if ('experiment_config' in locals() and experiment_config) else None
                    view_for_coverage = (view.value if hasattr(view, 'value') else view) if ('view' in locals() and view) else None
                    
                    coverage_breakdown = compute_registry_coverage(
                        feature_names=safe_columns,
                        target=target_for_coverage,
                        interval_minutes=interval_for_filtering,
                        horizon_minutes=None,
                        registry=registry,
                        experiment_config=exp_config,
                        view=view_for_coverage,
                        error_policy=error_policy if 'error_policy' in locals() else "best_effort"
                    )
                    # CRITICAL: Check if coverage_breakdown is None (can happen if compute_registry_coverage raises exception)
                    if coverage_breakdown is not None:
                        cov_str = f"{coverage_breakdown.coverage_total:.2%}" if coverage_breakdown.coverage_total is not None else "N/A"
                        logger.info(f"✅ Force computed coverage breakdown: mode={coverage_breakdown.coverage_mode}, coverage={cov_str}")
                    else:
                        logger.warning(f"⚠️  Force compute returned None for target {target_for_coverage if 'target_for_coverage' in locals() else target}")
                        coverage_computation_error = True
                except Exception as e2:
                    # This is also a computation error - mark it
                    # logger.exception includes full traceback automatically
                    logger.exception(
                        f"REGISTRY_COVERAGE_ERROR: Force compute also failed for target={target_for_coverage if 'target_for_coverage' in locals() else target}. "
                        f"This indicates a persistent computation issue."
                    )
                    coverage_computation_error = True
        
        # Initialize eligibility variables before composite calculation (fail-closed defaults)
        # This prevents UnboundLocalError if calculate_composite_score_tstat fails
        valid_for_ranking = False  # Fail-closed default
        invalid_reasons = []
        effective_run_intent = "eval"  # Default run intent
        
        try:
            composite, composite_def, composite_ver, components, scoring_signature, eligibility = calculate_composite_score_tstat(
                primary_mean=primary_metric_mean_centered,
                primary_std=std_score,
                n_slices_valid=n_cs_valid,  # Now correctly represents timestamps
                n_slices_total=n_cs_total,
                task_type=final_task_type,
                scoring_config=None,  # Will load from metrics_schema.yaml (SST pattern)
                coverage_breakdown=coverage_breakdown,  # NEW: Canonical coverage breakdown (SST)
                registry_coverage_rate=registry_coverage_rate,  # DEPRECATED: Fallback for backward compat
                run_intent=run_intent,  # Pass from config
                view=(view_for_writes.value if hasattr(view_for_writes, 'value') else view_for_writes) if 'view_for_writes' in locals() else (
                    (view.value if hasattr(view, 'value') else view) if 'view' in locals() else "CROSS_SECTIONAL"
                ),  # FIX 4: Pass view for view-aware checks (convert enum to string)
                coverage_computation_error=coverage_computation_error,  # NEW: Pass error flag to distinguish error from missing
            )
            
            # Extract eligibility fields - avoid aliasing shared lists
            valid_for_ranking = eligibility.get("valid_for_ranking", False)
            invalid_reasons = list(eligibility.get("invalid_reasons", []))  # Copy to avoid mutating shared state
            warnings = list(eligibility.get("warnings", []))  # NEW: Extract warnings (non-blocking quality flags)
            effective_run_intent = eligibility.get("run_intent", "eval")
        except Exception as e:
            # Explicitly mark as invalid on exception (fail-closed)
            valid_for_ranking = False
            invalid_reasons = ["composite_score_exception"]  # New list, stable reason for determinism
            # Log exception details (logging is fine, determinism is about artifacts/ranking outputs)
            logger.exception(f"Phase 3.1 composite score calculation failed: {e}, falling back to legacy")
            # Set fallback values for composite calculation
            composite = 0.0
            composite_def = "legacy_fallback"
            composite_ver = "unknown"
            components = {}
            scoring_signature = None
            eligibility = {"valid_for_ranking": False, "invalid_reasons": ["composite_score_exception"], "warnings": [], "run_intent": "eval"}
            warnings = []  # Initialize warnings for exception path
        # Extract primary_se from components if available
        if components and "primary_se" in components:
            primary_se = components["primary_se"]
        # Log success for debugging
        if scoring_signature:
            logger.debug(f"Phase 3.2 composite score calculated successfully: version={composite_ver}, signature={scoring_signature[:16]}...")
            if not valid_for_ranking:
                logger.warning(f"Target {target} marked as invalid_for_ranking: {', '.join(invalid_reasons)}")
        else:
            logger.warning(f"Phase 3.2 composite score calculated but scoring_signature is None")
        
        # FIX 5: Save feature importances only if target is valid_for_ranking
        # FIX 1: Ensure universe_sig is available before saving (update if None - try multiple sources)
        if '_feature_importances_to_save' in locals() and _feature_importances_to_save:
            # Update universe_sig if it's None - try multiple sources
            if not _feature_importances_to_save.get('universe_sig'):
                # Try universe_sig_for_writes first
                try:
                    if universe_sig_for_writes:
                        _feature_importances_to_save['universe_sig'] = universe_sig_for_writes
                        logger.debug(f"Updated universe_sig from universe_sig_for_writes: {universe_sig_for_writes[:8]}...")
                except NameError:
                    pass
                # Fallback: try resolved_data_config
                if not _feature_importances_to_save.get('universe_sig') and 'resolved_data_config' in locals() and resolved_data_config:
                    universe_sig_from_config = resolved_data_config.get('universe_sig')
                    if universe_sig_from_config:
                        _feature_importances_to_save['universe_sig'] = universe_sig_from_config
                        logger.debug(f"Updated universe_sig from resolved_data_config: {universe_sig_from_config[:8]}...")
                # ROOT CAUSE DEBUG: Final fallback - extract from cohort_dir path if still None
                if not _feature_importances_to_save.get('universe_sig'):
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                        # Try to extract from any available cohort_dir path
                        if 'cohort_context' in locals() and cohort_context:
                            # cohort_context might have universe_sig
                            universe_sig_from_context = cohort_context.get('universe_sig')
                            if universe_sig_from_context:
                                _feature_importances_to_save['universe_sig'] = universe_sig_from_context
                                logger.debug(f"ROOT CAUSE DEBUG: Updated universe_sig from cohort_context: {universe_sig_from_context[:8]}...")
                        # Also try extracting from output_dir path if available
                        if not _feature_importances_to_save.get('universe_sig') and 'output_dir' in locals() and output_dir:
                            try:
                                path_info = parse_reproducibility_path(Path(output_dir))
                                universe_sig_from_path = path_info.get('universe_sig')
                                if universe_sig_from_path:
                                    _feature_importances_to_save['universe_sig'] = universe_sig_from_path
                                    logger.debug(f"ROOT CAUSE DEBUG: Updated universe_sig from output_dir path: {universe_sig_from_path[:8]}...")
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"ROOT CAUSE DEBUG: Failed to extract universe_sig from fallback sources: {e}")
        
        # ROOT CAUSE DEBUG: Log before save call
        if '_feature_importances_to_save' in locals() and _feature_importances_to_save:
            universe_sig_value = _feature_importances_to_save.get('universe_sig')
            logger.debug(f"ROOT CAUSE DEBUG: Before save - valid_for_ranking={valid_for_ranking}, universe_sig={universe_sig_value[:8] if universe_sig_value else 'None'}..., has_importances={bool(_feature_importances_to_save.get('importances'))}")
        
        # Always save feature importances (removed valid_for_ranking gate per user request)
        if '_feature_importances_to_save' in locals() and _feature_importances_to_save:
            if not _feature_importances_to_save.get('importances'):
                logger.warning(f"⚠️  Skipping feature importances save for {target}: no importances computed")
            elif not _feature_importances_to_save.get('universe_sig'):
                logger.error(f"❌ Skipping feature importances save for {target}: universe_sig is None (will cause SCOPE BUG error)")
            else:
                # Save feature importances regardless of valid_for_ranking status
                _save_feature_importances(
                    _feature_importances_to_save['target_column'],
                    _feature_importances_to_save['symbol'],
                    _feature_importances_to_save['importances'],
                    _feature_importances_to_save['output_dir'],
                    view=_feature_importances_to_save['view'],
                    universe_sig=_feature_importances_to_save['universe_sig'],
                    run_identity=_feature_importances_to_save['run_identity'],
                    model_metrics=_feature_importances_to_save['model_metrics'],
                    attempt_id=_feature_importances_to_save['attempt_id'],
                )
                if not valid_for_ranking:
                    logger.debug(f"Feature importances saved for {target} (target marked as invalid_for_ranking: {', '.join(invalid_reasons) if invalid_reasons else 'unknown'})")
    except Exception as e:
        import traceback
        # Use ERROR level so it's definitely visible - this is a critical failure
        logger.error(f"❌ Phase 3.1 composite score calculation failed: {e}, falling back to legacy")
        logger.error(f"Phase 3.1 failure traceback:\n{traceback.format_exc()}")
        # Fallback to legacy composite score
        composite, composite_def, composite_ver = calculate_composite_score(
            auc, std_score, mean_importance, len(all_scores_by_model), final_task_type
        )
        scoring_signature = None
        components = {}
        # Use computed primary_se from above (or 0.0 if not computed)
    
    # Detect potential leakage (use task-appropriate thresholds)
    # Pass X, y, time_vals, symbols for triage checks if available
    try:
        leakage_flag = detect_leakage(
            auc, composite, mean_importance, 
            target=target, model_scores=model_means, task_type=final_task_type,
            X=X, y=y, time_vals=time_vals, symbols=symbols_array if 'symbols_array' in locals() else None
        )
    except Exception as e:
        logger.exception(f"Leakage detection failed for {target}; marking as UNKNOWN")
        leakage_flag = "UNKNOWN"  # Safe default - allows pipeline to continue
    
    # Dominance Quarantine: Detect suspects based on dominant importance
    suspects = []
    runtime_quarantine_features = set()
    confirm_result = None
    post_quarantine_auc = None
    post_quarantine_leakage_flag = None
    
    try:
        from TRAINING.ranking.utils.dominance_quarantine import (
            DominanceConfig, detect_suspects, write_suspects_artifact_with_data,
            confirm_quarantine, persist_confirmed_quarantine
        )
        
        cfg = DominanceConfig.from_config()
        if cfg.enabled and all_feature_importances:
            # Convert importance dicts to percentages
            per_model_importance_pct = {}
            for model_name, imp_dict in all_feature_importances.items():
                if not imp_dict:
                    continue
                total = sum(abs(v) for v in imp_dict.values())
                if total > 0:
                    # Convert to percentages (0-100 scale)
                    imp_pct = {f: (abs(v) / total * 100.0) for f, v in imp_dict.items()}
                    per_model_importance_pct[model_name] = imp_pct
            
            if per_model_importance_pct:
                suspects = detect_suspects(per_model_importance_pct, cfg)
                
                if suspects and output_dir:
                    # Write suspects artifact
                    target_for_artifact = target_column if target_column else target
                    # Normalize resolved_view to enum
                    resolved_view_raw = view if 'view' in locals() else View.CROSS_SECTIONAL
                    resolved_view = View.from_string(resolved_view_raw) if isinstance(resolved_view_raw, str) else resolved_view_raw
                    symbol_for_artifact = symbol if 'symbol' in locals() else None
                    
                    try:
                        write_suspects_artifact_with_data(
                            output_dir=output_dir,
                            target=target_for_artifact,
                            view=resolved_view,
                            suspects=suspects,
                            symbol=symbol_for_artifact
                        )
                        logger.info(f"🔍 Dominance quarantine: Detected {len(suspects)} suspect feature(s): {[s.feature for s in suspects]}")
                    except Exception as e:
                        logger.warning(f"Failed to write suspects artifact: {e}")
                
                # Confirm pass: rerun with suspects removed if enabled
                if suspects and cfg.confirm_enabled and cfg.rerun_once:
                    suspect_features = {s.feature for s in suspects}
                    features_without_suspects = [f for f in feature_names if f not in suspect_features]
                    
                    if len(features_without_suspects) >= 2:  # Need at least 2 features
                        # Get indices of features to keep
                        feature_indices = [i for i, f in enumerate(feature_names) if f in features_without_suspects]
                        # Filter X to only include non-suspect features
                        if len(feature_indices) < len(feature_names):
                            X_filtered = X[:, feature_indices]
                        else:
                            X_filtered = X  # No suspects to remove (shouldn't happen, but safe)
                        
                        # Rerun importance producers with filtered features (same CV splitter, seeds, folds)
                        logger.info(f"🔄 Dominance confirm: Rerunning with {len(features_without_suspects)} features (removed {len(suspect_features)} suspects)")
                        
                        try:
                            # Reuse same CV splitter configuration
                            post_metrics, post_scores, post_mean_importance, _, _, _, _ = train_and_evaluate_models(
                                X=X_filtered,
                                y=y,
                                feature_names=features_without_suspects,
                                task_type=task_type,
                                model_families=model_families,
                                multi_model_config=multi_model_config,
                                target_column=target_column,
                                data_interval_minutes=data_interval_minutes,
                                time_vals=time_vals,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                output_dir=output_dir,
                                resolved_config=resolved_config,
                                dropped_tracker=dropped_tracker,
                                view=view_for_writes if 'view_for_writes' in locals() else view,  # Use SST-resolved view
                                symbol=symbol_for_writes if 'symbol_for_writes' in locals() else symbol,  # Use SST-resolved symbol
                                run_identity=partial_identity,  # Pass partial identity for quick_pruner
                            )
                            
                            # Compute post-quarantine mean score
                            if post_scores:
                                post_quarantine_auc = float(np.mean([s for s in post_scores.values() if s is not None and not np.isnan(s)]))
                            
                            # Evaluate confirm result
                            n_samples = len(y) if y is not None else 0
                            n_symbols = len(set(symbols_array)) if 'symbols_array' in locals() and symbols_array is not None else 1
                            
                            confirm_result = confirm_quarantine(
                                pre_auc=auc,
                                post_auc=post_quarantine_auc if post_quarantine_auc is not None else auc,
                                suspects=suspects,
                                n_samples=n_samples,
                                n_symbols=n_symbols,
                                cfg=cfg
                            )
                            
                            logger.info(
                                f"📊 Dominance confirm: confirmed={confirm_result.confirmed} reason={confirm_result.reason} "
                                f"drop_abs={confirm_result.drop_abs:.4f} drop_rel={confirm_result.drop_rel:.2%}"
                            )
                            
                            if confirm_result.confirmed and output_dir:
                                # Persist confirmed quarantine
                                try:
                                    persist_confirmed_quarantine(
                                        output_dir=output_dir,
                                        target=target_for_artifact,
                                        suspects=suspects,
                                        view=resolved_view,
                                        symbol=symbol_for_artifact
                                    )
                                    runtime_quarantine_features = suspect_features
                                    logger.info(f"✅ Dominance quarantine: Confirmed and quarantined {len(suspect_features)} feature(s)")
                                except Exception as e:
                                    logger.warning(f"Failed to persist confirmed quarantine: {e}")
                            
                            # Re-evaluate leakage on post-quarantine results
                            if post_quarantine_auc is not None:
                                post_composite, _, _ = calculate_composite_score(
                                    post_quarantine_auc,
                                    float(np.std([s for s in post_scores.values() if s is not None and not np.isnan(s)])) if post_scores else 0.0,
                                    post_mean_importance,
                                    len(post_scores),
                                    final_task_type
                                )
                                
                                try:
                                    post_quarantine_leakage_flag = detect_leakage(
                                        post_quarantine_auc,
                                        post_composite,
                                        post_mean_importance,
                                        target=target,
                                        model_scores=post_scores,
                                        task_type=final_task_type,
                                        X=X_filtered,
                                        y=y,
                                        time_vals=time_vals,
                                        symbols=symbols_array if 'symbols_array' in locals() else None
                                    )
                                except Exception as e:
                                    logger.exception(f"Post-quarantine leakage detection failed for {target}; marking as UNKNOWN")
                                    post_quarantine_leakage_flag = "UNKNOWN"  # Safe default - allows pipeline to continue
                        except Exception as e:
                            logger.warning(f"Dominance confirm pass failed: {e}")
    except Exception as e:
        logger.debug(f"Dominance quarantine detection failed: {e}")
    
    # Build detailed leakage flags for auto-rerun logic
    leakage_flags = {
        "perfect_train_acc": len(_perfect_correlation_models) > 0,  # Any model hit 100% training accuracy
        "high_auc": auc > 0.95 if final_task_type == TaskType.BINARY_CLASSIFICATION else False,
        "high_r2": auc > 0.80 if final_task_type == TaskType.REGRESSION else False,
        "suspicious_flag": leakage_flag != "OK"
    }
    
    # CRITICAL: Build LeakageAssessment to prevent contradictory reason strings
    from TRAINING.common.utils.leakage_assessment import LeakageAssessment
    
    # Determine CV suspicious flag (CV score too high suggests leakage, not just overfitting)
    cv_suspicious = False
    if primary_scores:
        valid_cv_scores = [s for s in primary_scores.values() if s is not None and not np.isnan(s)]
        if valid_cv_scores:
            max_cv_score = max(valid_cv_scores)
            # CV score >= 0.85 is suspicious (too good to be true)
            cv_suspicious = max_cv_score >= 0.85
    
    # Determine overfit_likely flag (perfect train but low CV = classic overfitting)
    overfit_likely = False
    if model_metrics:
        for model_name, metrics in model_metrics.items():
            if isinstance(metrics, dict):
                train_acc = metrics.get('training_accuracy')
                cv_acc = metrics.get('accuracy')
                train_r2 = metrics.get('training_r2')
                cv_r2 = metrics.get('r2')
                
                # Check if perfect train but low CV (classic overfitting)
                if train_acc is not None and train_acc >= 0.99:
                    if cv_acc is not None and cv_acc < 0.75:
                        overfit_likely = True
                        break
                if train_r2 is not None and train_r2 >= 0.99:
                    if cv_r2 is not None and cv_r2 < 0.50:
                        overfit_likely = True
                        break
    
    # Find models with AUC > 0.90
    auc_too_high_models = []
    if final_task_type == TaskType.BINARY_CLASSIFICATION and model_means:
        for model_name, score in model_means.items():
            if score is not None and not np.isnan(score) and score > 0.90:
                auc_too_high_models.append(model_name)
    
    # Extract CV metric name from scoring source (SST: same logic as train_and_evaluate_models())
    # Replicate the same task-type-based selection logic used in train_and_evaluate_models()
    cv_metric_name_for_reason = None
    if final_task_type == TaskType.REGRESSION:
        cv_metric_name_for_reason = "r2"
    elif final_task_type == TaskType.BINARY_CLASSIFICATION:
        cv_metric_name_for_reason = "roc_auc"
    else:  # MULTICLASS_CLASSIFICATION
        cv_metric_name_for_reason = "accuracy"
    
    # Extract CV metric value (finite only)
    cv_metric_value_for_reason = None
    if primary_scores:
        # primary_scores is Dict[str, float] - model_name -> CV score
        valid_cv_scores = [
            s for s in primary_scores.values() 
            if s is not None and np.isfinite(s)  # Use isfinite, not just isnan
        ]
        if valid_cv_scores:
            # All metrics (r2, roc_auc, accuracy) are "higher is better", so max() is correct
            cv_metric_value_for_reason = max(valid_cv_scores)
    
    # Get T-stat from composite score calculation (computed earlier, around line 7685)
    # The tstat is returned in components dict from calculate_composite_score_tstat() with key "skill_tstat"
    tstat_for_reason = None
    if components and isinstance(components, dict):
        tstat_raw = components.get("skill_tstat")
        if tstat_raw is not None and np.isfinite(tstat_raw):  # Use isfinite
            tstat_for_reason = tstat_raw
    
    # Build assessment with metrics
    assessment = LeakageAssessment(
        leak_scan_pass=not summary_leaky_features if 'summary_leaky_features' in locals() else True,
        cv_suspicious=cv_suspicious,
        overfit_likely=overfit_likely,
        auc_too_high_models=auc_too_high_models,
        cv_metric_name=cv_metric_name_for_reason,
        cv_metric_value=cv_metric_value_for_reason,
        primary_metric_tstat=tstat_for_reason
    )
    
    # Determine status: SUSPICIOUS targets should be excluded from rankings
    # High AUC/R² after auto-fix suggests structural leakage (target construction issue)
    if leakage_flag in ["SUSPICIOUS", "HIGH_SCORE"]:
        # If we have very high scores, this is likely structural leakage, not just feature leakage
        if final_task_type == TaskType.BINARY_CLASSIFICATION and auc > 0.95:
            final_status = "SUSPICIOUS_STRONG"
        elif final_task_type == TaskType.REGRESSION and auc > 0.80:
            final_status = "SUSPICIOUS_STRONG"
        else:
            final_status = "SUSPICIOUS"
    else:
        final_status = "OK"
    
    # Collect fold scores if available (from model evaluations)
    # Note: This is a simplified collection - actual per-fold scores would require
    # storing them during cross_val_score calls, which can be enhanced later
    aggregated_fold_scores = None
    if all_fold_scores and len(all_fold_scores) > 0:
        aggregated_fold_scores = [float(s) for s in all_fold_scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
        if len(aggregated_fold_scores) == 0:
            aggregated_fold_scores = None
    
    # Determine view for canonical metric naming
    # SST-resolved view_for_writes is preferred (handles auto-flip from CS to SS)
    result_view = view_for_writes if 'view_for_writes' in locals() and view_for_writes else (
        view if 'view' in locals() and view else View.CROSS_SECTIONAL.value
    )
    
    # === DUAL RANKING: Compute strict evaluation for mismatch telemetry ===
    # Screen evaluation (current): uses for_ranking=True (safe_family + registry)
    # Strict evaluation: uses for_ranking=False (registry-only, exact training universe)
    score_screen = composite if 'composite' in locals() else None
    score_strict = None
    strict_viability_flag = None
    rank_delta = None
    mismatch_telemetry = None
    
    # Only run strict evaluation if we have valid screen results and data
    if (score_screen is not None and score_screen > -999.0 and 
        'X' in locals() and X is not None and 'y' in locals() and y is not None and
        'safe_columns' in locals() and safe_columns):
        try:
            logger.info("  🔍 Running strict evaluation (registry-only features) for mismatch telemetry...")
            
            # Re-filter features with strict mode (registry-only)
            strict_columns = filter_features_for_target(
                columns_after_target_exclusions if 'columns_after_target_exclusions' in locals() else all_columns,
                target_column,
                verbose=False,  # Less verbose for second pass
                use_registry=True,
                data_interval_minutes=detected_interval,
                for_ranking=False,  # Strict: registry-only (exact training universe)
                dropped_tracker=None  # Don't track drops for strict pass
            )
            
            # Check if we have enough strict features
            if len(strict_columns) >= MIN_FEATURES_REQUIRED:
                # Find intersection of screen and strict features
                screen_feat_set = set(safe_columns)
                strict_feat_set = set(strict_columns)
                unknown_feature_count = len(screen_feat_set - strict_feat_set)
                
                # FIX ISSUE-012/013: Reuse canonical coverage_breakdown instead of recomputing
                # This ensures telemetry and eligibility use the same coverage computation and interval validation
                if 'coverage_breakdown' in locals() and coverage_breakdown is not None:
                    # Reuse the canonical coverage breakdown computed for eligibility
                    # Use coverage_total if horizon_ok, else coverage_in_registry (membership-only)
                    if coverage_breakdown.coverage_mode == "horizon_ok":
                        # FIX ISSUE-005: Rename to prevent accidental reuse with eligibility variable
                        telemetry_registry_coverage_rate = coverage_breakdown.coverage_total
                    else:
                        # For telemetry, use membership-only as fallback
                        # FIX ISSUE-005: Rename to prevent accidental reuse with eligibility variable
                        telemetry_registry_coverage_rate = coverage_breakdown.coverage_in_registry
                else:
                    # Fallback: recompute only if canonical breakdown not available
                    try:
                        from TRAINING.ranking.utils.registry_coverage import compute_registry_coverage
                        
                        # FIX ISSUE-013: Use same interval validation as eligibility (unified logic)
                        interval_for_telemetry = interval_for_filtering if 'interval_for_filtering' in locals() else (detected_interval if ('detected_interval' in locals() and detected_interval and detected_interval > 0) else 5)
                        
                        # Use parameter if passed, otherwise try to load
                        if registry is None:
                            try:
                                from TRAINING.common.feature_registry import get_registry
                                registry = get_registry()
                            except Exception:
                                pass
                        
                        # Compute canonical coverage for telemetry (only if not already computed)
                        telemetry_coverage_breakdown = compute_registry_coverage(
                            feature_names=safe_columns,
                            target=target_column if 'target_column' in locals() else target,
                            interval_minutes=interval_for_telemetry,
                            horizon_minutes=None,  # Will be extracted from target name
                            registry=registry,
                            experiment_config=experiment_config if 'experiment_config' in locals() else None,
                            view=(view.value if hasattr(view, 'value') else view) if 'view' in locals() else None
                        )
                        
                        # Use coverage_total if horizon_ok, else coverage_in_registry (membership-only)
                        if telemetry_coverage_breakdown.coverage_mode == "horizon_ok":
                            # FIX ISSUE-005: Rename to prevent accidental reuse with eligibility variable
                            telemetry_registry_coverage_rate = telemetry_coverage_breakdown.coverage_total
                        else:
                            # For telemetry, use membership-only as fallback
                            # FIX ISSUE-005: Rename to prevent accidental reuse with eligibility variable
                            telemetry_registry_coverage_rate = telemetry_coverage_breakdown.coverage_in_registry
                    except Exception as e:
                        # Fallback to simple computation if canonical function fails
                        logger.debug(f"Canonical coverage computation failed for telemetry: {e}, using simple computation")
                        # FIX ISSUE-018: Fix division by zero - check length, not truthiness (empty set is truthy)
                        # FIX ISSUE-005: Rename to prevent accidental reuse with eligibility variable
                        telemetry_registry_coverage_rate = len(strict_feat_set) / len(screen_feat_set) if len(screen_feat_set) > 0 else 0.0
                
                # Prepare data with strict features only
                # Find column indices for strict features in original data
                if 'mtf_data' in locals() and mtf_data is not None:
                    # Re-prepare data with strict features (simplified - reuse existing data prep logic)
                    # For now, we'll compute a simplified strict score using available features
                    # Full re-evaluation would require re-running the entire data preparation pipeline
                    logger.debug(f"  Strict features: {len(strict_columns)} (screen had {len(safe_columns)})")
                    logger.debug(f"  Unknown features in screen: {unknown_feature_count}")
                    
                    # For initial implementation, compute telemetry only
                    # TODO: Full implementation would re-run train_and_evaluate_models with strict features
                    # This requires re-preparing data, which is complex. For now, we compute telemetry only.
                    # score_strict will be computed in a future enhancement
                    # For now, use screen score as conservative estimate (strict score will be <= screen score)
                    score_strict = None  # Will be computed in full implementation
                    # strict_viability_flag will be computed after all targets are ranked (in target_ranker.py)
                    strict_viability_flag = None
                    
                    # Compute feature importance overlap (if available)
                    topk_overlap = None
                    if 'feature_importances' in locals() and feature_importances:
                        # Get top features from screen evaluation
                        try:
                            # Aggregate importances across models
                            aggregated_importance = {}
                            for model_name, imp_dict in feature_importances.items():
                                if isinstance(imp_dict, dict):
                                    for feat, imp_val in imp_dict.items():
                                        aggregated_importance[feat] = aggregated_importance.get(feat, 0.0) + abs(imp_val)
                            
                            # Get top-K features from screen
                            top_k = min(20, len(aggregated_importance))
                            screen_top_k = set(sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)[:top_k])
                            screen_top_k_features = {feat for feat, _ in screen_top_k}
                            
                            # Compute overlap with strict features
                            strict_top_k_features = screen_top_k_features & strict_feat_set
                            if screen_top_k_features:
                                topk_overlap = len(strict_top_k_features) / len(screen_top_k_features)
                        except Exception as e:
                            logger.debug(f"Failed to compute top-k overlap: {e}")
                    
                    # Build mismatch telemetry
                    mismatch_telemetry = {
                        "n_feats_screen": len(screen_feat_set),
                        "n_feats_strict": len(strict_feat_set),
                        "topk_overlap": topk_overlap if topk_overlap is not None else 0.0,
                        "unknown_feature_count": unknown_feature_count,
                        # FIX ISSUE-005: Use renamed variable to prevent accidental reuse
                        "registry_coverage_rate": telemetry_registry_coverage_rate if 'telemetry_registry_coverage_rate' in locals() else registry_coverage_rate
                    }
                    
                    # DETERMINISM: None-safe formatting for registry_coverage_rate
                    coverage_str = f"{registry_coverage_rate:.2%}" if registry_coverage_rate is not None else "N/A"
                    logger.info(
                        f"  📊 Mismatch telemetry: screen={len(screen_feat_set)} feats, "
                        f"strict={len(strict_feat_set)} feats, "
                        f"unknown={unknown_feature_count}, "
                        f"coverage={coverage_str}"
                    )
                else:
                    logger.debug("  Skipping strict evaluation: mtf_data not available")
            else:
                logger.debug(f"  Skipping strict evaluation: insufficient features ({len(strict_columns)} < {MIN_FEATURES_REQUIRED})")
                # FIX ISSUE-005: Use local variable name for telemetry to prevent collision
                telemetry_registry_coverage_rate = len(strict_columns) / len(safe_columns) if safe_columns else 0.0
                mismatch_telemetry = {
                    "n_feats_screen": len(safe_columns),
                    "n_feats_strict": len(strict_columns),
                    "topk_overlap": 0.0,
                    "unknown_feature_count": len(set(safe_columns) - set(strict_columns)),
                    "registry_coverage_rate": telemetry_registry_coverage_rate
                }
        except Exception as e:
            logger.warning(f"  Failed to compute strict evaluation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Store auto_fix_reason in result for routing traceability
    auto_fix_reason_str = assessment.auto_fix_reason() if 'assessment' in locals() else None
    
    result = TargetPredictabilityScore(
        target=target,
        target_column=target_column,
        task_type=final_task_type,
        auc=auc,
        std_score=std_score,
        mean_importance=mean_importance,
        consistency=consistency,
        n_models=len(all_scores_by_model),
        model_scores=model_means,
        composite_score=composite,
        composite_definition=composite_def,
        composite_version=composite_ver,
        leakage_flag=leakage_flag,
        suspicious_features=all_suspicious_features if all_suspicious_features else None,
        fold_timestamps=fold_timestamps,
        fold_scores=aggregated_fold_scores,
        leakage_flags=leakage_flags,
        autofix_info=autofix_info if 'autofix_info' in locals() else None,
        status=final_status,
        attempts=1,
        view=result_view,  # For canonical metric naming
        valid_for_ranking=valid_for_ranking,  # NEW
        invalid_reasons=invalid_reasons,  # NEW
        warnings=warnings if 'warnings' in locals() else [],  # NEW: Non-blocking quality flags
        run_intent=effective_run_intent,  # NEW
        # === Phase 3.1: Centered primary metric and SE-based stats ===
        primary_metric_mean=primary_metric_mean_centered,  # Centered: IC for regression, AUC-excess for classification
        primary_metric_std=std_score,  # Explicit (authoritative, same as std_score for now)
        primary_metric_tstat=components.get("skill_tstat") if components else primary_metric_tstat,  # From Phase 3.1 calculation
        n_cs_valid=n_cs_valid,  # Valid model evaluations (backward compat)
        n_cs_total=n_cs_total,  # Total model evaluations attempted (backward compat)
        invalid_reason_counts=invalid_reason_counts if invalid_reason_counts else None,
        auc_mean_raw=auc_mean_raw,  # Classification only: raw 0-1 AUC
        auc_excess_mean=auc_excess_mean,  # Classification only: centered AUC
        # Dual ranking fields (2026-01 filtering mismatch fix)
        score_screen=score_screen,
        score_strict=score_strict,
        strict_viability_flag=strict_viability_flag,
        rank_delta=rank_delta,  # Will be computed after ranking all targets
        mismatch_telemetry=mismatch_telemetry,
        auto_fix_reason=auto_fix_reason_str  # NEW: Auto-fix skip reason for routing traceability
    )
    
    # Add Phase 3.2 fields to result object (SST: load version from config)
    from TRAINING.ranking.predictability.metrics_schema import get_scoring_schema_version
    result.primary_se = primary_se
    result.scoring_signature = scoring_signature
    result.scoring_schema_version = get_scoring_schema_version()  # SST: Load from CONFIG/ranking/metrics_schema.yaml
    
    # Log canonical summary block (one block that can be screenshot for PR comments)
    # Use detected_interval from evaluate_target_predictability scope (defined at line ~2276)
    summary_interval = detected_interval if 'detected_interval' in locals() else None
    summary_horizon = target_horizon_minutes if 'target_horizon_minutes' in locals() else None
    summary_safe_features = len(safe_columns) if 'safe_columns' in locals() else 0
    summary_leaky_features = leaky_features if 'leaky_features' in locals() else []
    
    # Extract CV splitter info for logging
    splitter_name = None
    n_splits_val = None
    purge_minutes_val = None
    embargo_minutes_val = None
    max_lookback_val = None
    
    if 'resolved_config' in locals() and resolved_config:
        purge_minutes_val = resolved_config.purge_minutes
        embargo_minutes_val = resolved_config.embargo_minutes
        # CRITICAL: Use the FINAL lookback from resolved_config (should match POST_PRUNE recompute)
        # If there's a mismatch, the invariant check should have caught it
        max_lookback_val = resolved_config.feature_lookback_max_minutes
        splitter_name = "PurgedTimeSeriesSplit"  # Default for time-series CV
        n_splits_val = folds if 'folds' in locals() else None
        
        # SANITY CHECK: Verify resolved_config lookback matches what we computed at POST_PRUNE
        if 'computed_lookback' in locals() and computed_lookback is not None:
            if abs(max_lookback_val - computed_lookback) > 1.0:
                logger.error(
                    f"🚨 SUMMARY MISMATCH: resolved_config.feature_lookback_max_minutes={max_lookback_val:.1f}m "
                    f"but POST_PRUNE computed_lookback={computed_lookback:.1f}m. "
                    f"Using POST_PRUNE value for summary."
                )
                max_lookback_val = computed_lookback  # Use the correct value
        
        # Log lookback_budget_minutes cap status for auditability
        try:
            from CONFIG.config_loader import get_cfg
            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                logger.info(f"📊 lookback_budget_minutes cap: {float(budget_cap_raw):.1f}m (active)")
            else:
                logger.info(f"📊 lookback_budget_minutes cap: auto (no cap, using actual max)")
        except Exception:
            pass
    
    _log_canonical_summary(
        target=target,
        target_column=target_column,
        symbols=symbols,
        time_vals=time_vals,
        interval=summary_interval,
        horizon=summary_horizon,
        rows=len(X) if X is not None else 0,
        features_safe=summary_safe_features,
        features_pruned=actual_pruned_feature_count if 'actual_pruned_feature_count' in locals() else (len(feature_names) if feature_names else 0),
        leak_scan_verdict="PASS" if not summary_leaky_features else "FAIL",
        auto_fix_verdict="SKIPPED" if not should_auto_fix else ("RAN" if autofix_info and autofix_info.modified_configs else "NO_CHANGES"),
        auto_fix_reason=assessment.auto_fix_reason() if 'assessment' in locals() else None,
        cv_metric=f"{metric_name}={auc:.3f}±{std_score:.3f}",
        composite_score=composite,
        leakage_flag=leakage_flag,
        cohort_path=None,  # Will be set by reproducibility tracker
        splitter_name=splitter_name,
        purge_minutes=purge_minutes_val,
        embargo_minutes=embargo_minutes_val,
        max_feature_lookback_minutes=max_lookback_val,
        n_splits=n_splits_val
    )
    
    # Legacy summary line (backward compatibility)
    leakage_indicator = f" [{leakage_flag}]" if leakage_flag != "OK" else ""
    logger.debug(f"Legacy summary: {metric_name}={auc:.3f}±{std_score:.3f}, "
               f"importance={mean_importance:.2f}, composite={composite:.3f}{leakage_indicator}")
    
    # Store suspicious features in result for summary report
    result.suspicious_features = all_suspicious_features if all_suspicious_features else None
    
    # Top features are tracked in feature importances artifacts and metrics
    # Removed verbose logging for cleaner logs (all metrics are tracked elsewhere)
    
    # Track reproducibility: compare to previous target ranking run
    # This runs regardless of which entry point calls this function
    if output_dir and result.auc != -999.0:
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            
            # Use module-specific directory for reproducibility log
            # output_dir might be: output_dir_YYYYMMDD_HHMMSS/target_rankings/ or just output_dir_YYYYMMDD_HHMMSS
            # We want to store in target_rankings/ subdirectory for this module
            if output_dir.name == 'target_rankings':
                # Already in target_rankings subdirectory
                module_output_dir = output_dir
            elif (output_dir.parent / 'target_rankings').exists():
                # output_dir is parent, use target_rankings subdirectory
                module_output_dir = output_dir.parent / 'target_rankings'
            else:
                # Fallback: use output_dir directly (for standalone runs)
                module_output_dir = output_dir
            
            tracker = ReproducibilityTracker(
                output_dir=module_output_dir,
                search_previous_runs=True  # Search for previous runs in parent directories
            )
            
            # Automated audit-grade reproducibility tracking using RunContext
            try:
                from TRAINING.orchestration.utils.run_context import RunContext
                
                # Build RunContext from available data
                # Prefer symbols_array (from prepare_cross_sectional_data_for_ranking) over symbols list
                symbols_for_ctx = None
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_ctx = cohort_context.get('symbols_array')
                    if symbols_for_ctx is None:
                        symbols_for_ctx = cohort_context.get('symbols')
                elif 'symbols_array' in locals():
                    symbols_for_ctx = symbols_array
                elif 'symbols' in locals():
                    symbols_for_ctx = symbols
                
                # Use resolved_config values if available (single source of truth)
                # CRITICAL: Use feature_lookback_max_minutes from resolved_config (computed from FINAL feature set)
                if 'resolved_config' in locals() and resolved_config:
                    purge_minutes_val = resolved_config.purge_minutes
                    embargo_minutes_val = resolved_config.embargo_minutes
                    # Use actual computed lookback from final features (post gatekeeper + pruning)
                    feature_lookback_max = resolved_config.feature_lookback_max_minutes
                elif 'purge_minutes_val' not in locals() or purge_minutes_val is None:
                    # Fallback: compute from purge_time if available
                    if 'purge_time' in locals() and purge_time is not None:
                        try:
                            if hasattr(purge_time, 'total_seconds'):
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                embargo_minutes_val = purge_minutes_val  # Assume same
                        except Exception:
                            pass
                
                # Fallback: if resolved_config not available, try to compute from final feature_names
                if 'feature_lookback_max' not in locals() or feature_lookback_max is None:
                    # Try to compute from final feature_names if available
                    if 'feature_names' in locals() and feature_names and 'data_interval_minutes' in locals() and data_interval_minutes:
                        from TRAINING.ranking.utils.leakage_budget import compute_budget
                        try:
                            # Get horizon for budget calculation
                            horizon = target_horizon_minutes if 'target_horizon_minutes' in locals() else 60.0
                            budget, _, _ = compute_budget(feature_names, data_interval_minutes, horizon, stage="run_context_budget")
                            feature_lookback_max = budget.max_feature_lookback_minutes
                        except Exception:
                            # Fallback: conservative estimate
                            feature_lookback_max = None
                    else:
                        feature_lookback_max = None
                
                # Get seed from config for reproducibility
                try:
                    from CONFIG.config_loader import get_cfg
                    seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                except Exception:
                    seed_value = 42
                
                # Build RunContext
                # FIX: Set view in constructor for consistency (already has min_cs and max_cs_samples)
                # Use SST-resolved view_for_writes if available (handles auto-flip from CS to SS)
                view_for_ctx = view_for_writes if 'view_for_writes' in locals() else (view if 'view' in locals() else None)
                symbol_for_ctx = symbol_for_writes if 'symbol_for_writes' in locals() else (symbol if 'symbol' in locals() and symbol else None)
                # FIX: Get universe_sig from SST-resolved value or cohort_context
                universe_sig_for_ctx = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else (
                    cohort_context.get('universe_sig') if 'cohort_context' in locals() and cohort_context else None
                )
                # FIX 1: Extract horizon_minutes with fallback using SST function
                horizon_minutes_for_ctx = None
                if 'target_horizon_minutes' in locals() and target_horizon_minutes is not None:
                    horizon_minutes_for_ctx = target_horizon_minutes
                elif 'target_column' in locals() and target_column:
                    # Fallback: extract from target column name using SST function
                    try:
                        from TRAINING.common.utils.sst_contract import resolve_target_horizon_minutes
                        horizon_minutes_for_ctx = resolve_target_horizon_minutes(target_column)
                    except Exception:
                        # Last resort: try _extract_horizon
                        try:
                            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                            leakage_config = _load_leakage_config()
                            horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
                        except Exception:
                            pass
                
                ctx = RunContext(
                    X=cohort_context.get('X') if 'cohort_context' in locals() and cohort_context else None,
                    y=cohort_context.get('y') if 'cohort_context' in locals() and cohort_context else None,
                    feature_names=feature_names if 'feature_names' in locals() else None,
                    symbols=symbols_for_ctx,
                    time_vals=cohort_context.get('time_vals') if 'cohort_context' in locals() and cohort_context else None,
                    target_column=target_column,
                    target=target,
                    min_cs=cohort_context.get('min_cs') if 'cohort_context' in locals() and cohort_context else (min_cs if 'min_cs' in locals() else None),
                    max_cs_samples=cohort_context.get('max_cs_samples') if 'cohort_context' in locals() and cohort_context else (max_cs_samples if 'max_cs_samples' in locals() else None),
                    mtf_data=cohort_context.get('mtf_data') if 'cohort_context' in locals() and cohort_context else None,
                    cv_method="purged_kfold",
                    folds=folds if 'folds' in locals() else None,
                    horizon_minutes=horizon_minutes_for_ctx,
                    purge_minutes=purge_minutes_val,
                    fold_timestamps=fold_timestamps if 'fold_timestamps' in locals() else None,
                    feature_lookback_max_minutes=feature_lookback_max,
                    data_interval_minutes=data_interval_minutes if 'data_interval_minutes' in locals() else None,
                    stage=Stage.TARGET_RANKING,  # FIX: Use uppercase for consistency
                    output_dir=output_dir,
                    seed=seed_value,
                    view=view_for_ctx,  # FIX: Set view in constructor for diff telemetry
                    symbol=symbol_for_ctx,  # FIX: Set symbol in constructor for consistency
                    universe_sig=universe_sig_for_ctx  # FIX: Pass universe_sig for proper scope tracking
                )
                
                # Build metrics dict with regression features
                # FIX: Remove redundancy - use n_features_post_prune (more descriptive) and drop features_final
                n_features_final = len(feature_names) if 'feature_names' in locals() and feature_names else None
                
                # Build clean, grouped metrics dict (replaces flat, duplicate-heavy structure)
                from TRAINING.ranking.predictability.metrics_schema import build_clean_metrics_dict, compute_target_stats
                
                # Compute target stats (task-aware)
                target_stats = None
                if 'y' in locals() and y is not None:
                    try:
                        target_stats = compute_target_stats(result.task_type, y)
                        # Add n_effective if available (from cohort_metadata or computed)
                        n_effective_value = None
                        if 'cohort_metadata' in locals() and cohort_metadata:
                            n_effective_value = cohort_metadata.get('n_effective_cs') or cohort_metadata.get('n_effective')
                        if n_effective_value is None and 'n_effective' in locals():
                            n_effective_value = n_effective
                        if n_effective_value is not None:
                            target_stats['n_effective'] = int(n_effective_value)
                    except Exception as e:
                        logger.debug(f"Failed to compute target stats: {e}")
                
                # Get leakage info from result
                leakage_info = None
                result_dict = result.to_dict(filter_task_irrelevant=True)
                if 'leakage' in result_dict:
                    leakage_info = result_dict['leakage']
                
                # Get fold timestamps if available
                fold_timestamps = result.fold_timestamps if hasattr(result, 'fold_timestamps') and result.fold_timestamps else None
                
                # Build clean metrics using new structured format
                metrics_dict = build_clean_metrics_dict(
                    result=result,
                    target_stats=target_stats,
                    n_features_pre=features_safe if 'features_safe' in locals() else None,
                    n_features_post_prune=n_features_final,
                    features_safe=features_safe if 'features_safe' in locals() else None,
                    fold_timestamps=fold_timestamps,
                    leakage_info=leakage_info,
                )
                
                # Add metadata fields that MetricsWriter expects (these are added by MetricsWriter, but include for completeness)
                # Note: run_id, timestamp, reproducibility_mode, stage are added by MetricsWriter
                # We keep metric_name for backward compatibility in some contexts
                metrics_dict["metric_name"] = metric_name
                
                # Add view and symbol to RunContext if available (for dual-view target ranking)
                if 'view' in locals():
                    ctx.view = view
                if 'symbol' in locals() and symbol:
                    ctx.symbol = symbol
                
                # FIX: Aggregate prediction fingerprints from model_metrics for predictions_sha256
                # This must happen in the NEW log_run path (not just fallback)
                aggregated_prediction_fingerprint = None
                per_model_hashes = {}
                if 'model_metrics' in locals() and model_metrics:
                    import hashlib
                    pred_hashes = []
                    for model_name, metrics_entry in model_metrics.items():
                        if isinstance(metrics_entry, dict) and 'prediction_fingerprint' in metrics_entry:
                            pred_hash = metrics_entry['prediction_fingerprint'].get('prediction_hash', '')
                            if pred_hash:
                                pred_hashes.append(pred_hash)
                                per_model_hashes[model_name] = pred_hash
                    if pred_hashes:
                        combined = hashlib.sha256('|'.join(sorted(pred_hashes)).encode()).hexdigest()
                        aggregated_prediction_fingerprint = {'prediction_hash': combined}
                        logger.info(f"✅ Aggregated {len(pred_hashes)} prediction fingerprints for predictions_sha256")
                    else:
                        logger.warning(f"⚠️ No prediction_fingerprints found in model_metrics ({len(model_metrics)} models)")
                else:
                    logger.warning("⚠️ model_metrics not available for prediction fingerprint aggregation")
                
                # Use automated log_run API with prediction fingerprint
                # FIX: Pass partial_identity (computed with real data) for authoritative signatures
                # Pass attempt_id via additional_data_override (additional_data_with_cohort is built later in fallback path)
                additional_data_override = {}
                if attempt_id is not None:
                    additional_data_override['attempt_id'] = attempt_id
                
                audit_result = tracker.log_run(
                    ctx, metrics_dict,
                    additional_data_override=additional_data_override if additional_data_override else None,
                    prediction_fingerprint=aggregated_prediction_fingerprint,
                    run_identity=partial_identity,  # SST: Use locally-computed identity with real data
                )
                
                # Log audit report summary if available
                if audit_result.get("audit_report"):
                    audit_report = audit_result["audit_report"]
                    if audit_report.get("violations"):
                        logger.warning(f"🚨 Audit violations detected: {len(audit_report['violations'])}")
                        for violation in audit_report['violations']:
                            logger.warning(f"  - {violation['message']}")
                    if audit_report.get("warnings"):
                        logger.info(f"⚠️  Audit warnings: {len(audit_report['warnings'])}")
                        for warning in audit_report['warnings']:
                            logger.info(f"  - {warning['message']}")
                
                # Log trend summary if available (already logged by log_run, but include in result)
                if audit_result.get("trend_summary"):
                    trend = audit_result["trend_summary"]
                    # Trend summary is already logged by log_run, but we can add additional context here if needed
                    pass
                
            except ImportError:
                # Fallback to legacy API if RunContext not available
                logger.warning("RunContext not available, falling back to legacy reproducibility tracking")
                from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                
                # FIX: Get universe_sig for fallback path
                fallback_universe_sig = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else (
                    cohort_context.get('universe_sig') if 'cohort_context' in locals() and cohort_context else None
                )
                
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_extraction = cohort_context.get('symbols_array') or cohort_context.get('symbols')
                    cohort_metadata = extract_cohort_metadata(
                        X=cohort_context.get('X'),
                        symbols=symbols_for_extraction,
                        time_vals=cohort_context.get('time_vals'),
                        y=cohort_context.get('y'),
                        mtf_data=cohort_context.get('mtf_data'),
                        min_cs=cohort_context.get('min_cs'),
                        max_cs_samples=cohort_context.get('max_cs_samples'),
                        compute_data_fingerprint=True,
                        compute_per_symbol_stats=True,
                        universe_sig=fallback_universe_sig  # FIX: Pass universe_sig for proper scope tracking
                    )
                else:
                    cohort_metadata = extract_cohort_metadata(
                        symbols=symbols if 'symbols' in locals() else None,
                        mtf_data=mtf_data if 'mtf_data' in locals() else None,
                        min_cs=min_cs if 'min_cs' in locals() else None,
                        max_cs_samples=max_cs_samples if 'max_cs_samples' in locals() else None,
                        universe_sig=fallback_universe_sig  # FIX: Pass universe_sig for proper scope tracking
                    )
                
                cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
                metrics_with_cohort = {
                    "metric_name": metric_name,
                    "auc": result.auc,
                    "std_score": result.std_score,
                    "mean_importance": result.mean_importance,
                    "composite_score": result.composite_score,
                    # Regression features: feature counts
                    "n_features_pre": features_safe if 'features_safe' in locals() else None,
                    "n_features_post_prune": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    "features_safe": features_safe if 'features_safe' in locals() else None,
                    "features_final": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    **cohort_metrics
                }
                
                # Add task-aware target stats (replaces unconditional pos_rate)
                if 'y' in locals() and y is not None and 'result' in locals():
                    try:
                        from TRAINING.ranking.predictability.metrics_schema import compute_target_stats
                        target_stats = compute_target_stats(result.task_type, y)
                        metrics_with_cohort.update(target_stats)
                    except Exception as e:
                        logger.debug(f"Failed to compute target stats: {e}")
                
                # NOTE: NaN drops are now tracked immediately after data prep (above), not here
                
                # NEW: Add dropped features summary to additional_data for telemetry
                if 'dropped_tracker' in locals() and dropped_tracker is not None and not dropped_tracker.is_empty():
                    cohort_additional_data['dropped_features'] = dropped_tracker.get_summary()
                
                # Add resolved_data_config (mode, loader contract) to additional_data for telemetry
                if 'resolved_data_config' in locals() and resolved_data_config:
                    cohort_additional_data['view'] = resolved_data_config.get('view') or resolved_data_config.get('resolved_data_mode')
                    cohort_additional_data['view_reason'] = resolved_data_config.get('view_reason')
                    cohort_additional_data['loader_contract'] = resolved_data_config.get('loader_contract')
                
                additional_data_with_cohort = {
                    "n_models": result.n_models,
                    "leakage_flag": result.leakage_flag,
                    "task_type": result.task_type.name if hasattr(result.task_type, 'name') else str(result.task_type),
                    **cohort_additional_data
                }
                
                # Extract experiment_id from experiment_config.name (stable, no fallbacks)
                if experiment_config and hasattr(experiment_config, 'name') and experiment_config.name:
                    additional_data_with_cohort['experiment_id'] = experiment_config.name
                    logger.debug(f"Added experiment_id to additional_data: {experiment_config.name}")
                elif experiment_config and hasattr(experiment_config, 'name') and not experiment_config.name:
                    # name exists but is empty/None - log once per run
                    try:
                        from CONFIG.dev_mode import get_dev_mode
                        if get_dev_mode():
                            logger.info("experiment_id missing (experiment_config.name is empty); grouping disabled")
                        else:
                            logger.debug("experiment_id missing (experiment_config.name is empty); grouping disabled")
                    except Exception:
                        logger.debug("experiment_id missing (experiment_config.name is empty); grouping disabled")
                
                # Add attempt_id if provided (for rerun tracking)
                if attempt_id is not None:
                    additional_data_with_cohort['attempt_id'] = attempt_id
                
                # Add model_families for run recreation (already sorted at line 2073)
                if 'model_families' in locals() and model_families:
                    additional_data_with_cohort['model_families'] = model_families
                
                # Add data_dir for run recreation
                if 'data_dir' in locals() and data_dir:
                    additional_data_with_cohort['data_dir'] = str(data_dir)
                
                # Add view and symbol for dual-view target ranking
                if 'view' in locals():
                    additional_data_with_cohort['view'] = view
                if 'symbol' in locals() and symbol:
                    additional_data_with_cohort['symbol'] = symbol
                
                # Add CV details manually (legacy path)
                if 'target_horizon_minutes' in locals() and target_horizon_minutes is not None:
                    additional_data_with_cohort['horizon_minutes'] = target_horizon_minutes
                if 'purge_time' in locals() and purge_time is not None:
                    try:
                        if hasattr(purge_time, 'total_seconds'):
                            # Use purge_minutes_val if available (single source of truth)
                            if 'purge_minutes_val' in locals() and purge_minutes_val is not None:
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                            else:
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                    except Exception:
                        pass
                if 'folds' in locals() and folds is not None:
                    additional_data_with_cohort['folds'] = folds
                if 'fold_timestamps' in locals() and fold_timestamps:
                    additional_data_with_cohort['fold_timestamps'] = fold_timestamps
                if 'feature_names' in locals() and feature_names:
                    additional_data_with_cohort['feature_names'] = feature_names
                if 'data_interval_minutes' in locals() and data_interval_minutes is not None:
                    additional_data_with_cohort['data_interval_minutes'] = data_interval_minutes
                    # Use time-based value (1 day = 1440 minutes) - interval-agnostic
                    additional_data_with_cohort['feature_lookback_max_minutes'] = 1440.0
                
                # ========================================================================
                # PATCH 0: Use WriteScope for type-safe scope handling
                # ========================================================================
                # Stage is already imported globally at line 41, don't re-import
                from TRAINING.orchestration.utils.scope_resolution import (
                    WriteScope, ScopePurpose
                )
                
                # Determine purpose from caller (scope_purpose parameter)
                # ROUTING_EVAL: evaluating both modes to decide routing
                # FINAL: training after routing decision is made
                purpose = ScopePurpose.ROUTING_EVAL if scope_purpose == "ROUTING_EVAL" else ScopePurpose.FINAL
                
                # Create WriteScope from SST-derived values
                scope = None
                if universe_sig_for_writes:
                    try:
                        # Normalize view_for_writes to enum for comparison
                        view_for_writes_enum = View.from_string(view_for_writes) if isinstance(view_for_writes, str) else view_for_writes
                        if view_for_writes_enum == View.CROSS_SECTIONAL or symbol_for_writes is None:
                            scope = WriteScope.for_cross_sectional(
                                universe_sig=universe_sig_for_writes,
                                stage=Stage.TARGET_RANKING,
                                purpose=purpose
                            )
                        else:
                            scope = WriteScope.for_symbol_specific(
                                universe_sig=universe_sig_for_writes,
                                symbol=symbol_for_writes,
                                stage=Stage.TARGET_RANKING,
                                purpose=purpose
                            )
                    except ValueError as e:
                        logger.warning(f"Failed to create WriteScope: {e}")
                
                # Use WriteScope to populate additional_data correctly
                if scope:
                    scope.to_additional_data(additional_data_with_cohort)
                else:
                    # Legacy fallback (deprecated)
                    from TRAINING.orchestration.utils.scope_resolution import populate_additional_data
                    populate_additional_data(
                        additional_data_with_cohort,
                        view_for_writes=view_for_writes,
                        symbol_for_writes=symbol_for_writes,
                        universe_sig_for_writes=universe_sig_for_writes
                    )
                
                # Add seed for reproducibility tracking
                try:
                    from CONFIG.config_loader import get_cfg
                    seed = get_cfg("pipeline.determinism.base_seed", default=42)
                    additional_data_with_cohort['seed'] = seed
                except Exception:
                    # Fallback to default if config not available
                    additional_data_with_cohort['seed'] = 42
                
                # FIX: Aggregate prediction fingerprints from model_metrics for predictions_sha256
                # This enables determinism verification by tracking per-model prediction hashes
                aggregated_prediction_fingerprint = None
                per_model_hashes = {}  # For auditability
                try:
                    # DEBUG: Defensive check - ensure model_metrics is in scope and populated
                    # model_metrics is unpacked from train_and_evaluate_models result at line ~6419
                    if 'model_metrics' not in locals():
                        logger.warning("⚠️ model_metrics not in local scope - this is a bug!")
                        model_metrics = {}  # Defensive fallback
                    mm_count = len(model_metrics) if model_metrics else 0
                    logger.info(f"🔍 Aggregating prediction fingerprints from {mm_count} models in model_metrics: {list(model_metrics.keys()) if model_metrics else []}")
                    if model_metrics:
                        import hashlib
                        pred_hashes = []
                        models_without_fp = []
                        for model_name, metrics in model_metrics.items():
                            if isinstance(metrics, dict) and 'prediction_fingerprint' in metrics:
                                pred_hash = metrics['prediction_fingerprint'].get('prediction_hash', '')
                                if pred_hash:
                                    pred_hashes.append(pred_hash)
                                    per_model_hashes[model_name] = pred_hash  # Track per-model
                            else:
                                models_without_fp.append(model_name)
                        
                        if models_without_fp:
                            # DEBUG: Show what keys each model HAS if it's missing prediction_fingerprint
                            for m in models_without_fp:
                                m_metrics = model_metrics.get(m, {})
                                if isinstance(m_metrics, dict):
                                    logger.warning(f"🔍 Model {m} missing prediction_fingerprint, has keys: {list(m_metrics.keys())}")
                                else:
                                    logger.warning(f"🔍 Model {m} has non-dict metrics: {type(m_metrics)}")
                        
                        # DEBUG: Show models with fingerprints
                        logger.info(f"🔍 Models WITH prediction_fingerprint: {list(per_model_hashes.keys())}")
                        
                        if pred_hashes:
                            # Create combined hash from sorted prediction hashes
                            combined = hashlib.sha256('|'.join(sorted(pred_hashes)).encode()).hexdigest()
                            aggregated_prediction_fingerprint = {'prediction_hash': combined}
                            logger.info(f"✅ Aggregated {len(pred_hashes)} prediction fingerprints for predictions_sha256")
                            # Add per-model hashes to additional_data for auditability
                            if per_model_hashes:
                                additional_data_with_cohort['prediction_hashes'] = per_model_hashes
                        else:
                            logger.warning(f"⚠️ No prediction_fingerprints found in model_metrics ({len(model_metrics)} models). predictions_sha256 will be null.")
                    else:
                        logger.warning(f"⚠️ model_metrics is empty for prediction fingerprint aggregation. predictions_sha256 will be null.")
                except Exception as e:
                    logger.warning(f"Failed to aggregate prediction fingerprints: {e}")
                
                # FIX: Use partial_identity (computed with real data) instead of run_identity param
                tracker.log_comparison(
                    stage=scope.stage.value if scope else "TARGET_RANKING",  # FIX: Use uppercase
                    target=target,
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort,
                    view=scope.view.value if scope else view_for_writes,
                    symbol=scope.symbol if scope else symbol_for_writes,
                    run_identity=partial_identity,  # FIX: Use locally-computed identity with real data
                    prediction_fingerprint=aggregated_prediction_fingerprint,  # FIX: Pass aggregated predictions for predictions_sha256
                )
        except Exception as e:
            logger.warning(f"Reproducibility tracking failed for {target}: {e}")
            import traceback
            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
    
    return result

