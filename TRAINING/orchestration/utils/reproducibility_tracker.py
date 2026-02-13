# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Reproducibility Tracking Module

Tracks and compares run results across pipeline stages to verify reproducible behavior.
Supports target ranking, feature selection, and other pipeline stages.

Usage:
    from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
    
    tracker = ReproducibilityTracker(output_dir=Path("results"))
    tracker.log_comparison(
        stage="target_ranking",
        target="y_will_swing_low_15m_0.05",
        metrics={
            "auc": 0.751,
            "std_score": 0.029,
            "mean_importance": 0.23,
            "composite_score": 0.764,
            "metric_name": "ROC-AUC"
        }
    )
"""

import json
import logging
import hashlib
import traceback
import os
import sys
import platform
import socket
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
from enum import Enum
import math
import pandas as pd

# Import RunContext and AuditEnforcer for automated audit-grade tracking
try:
    from TRAINING.orchestration.utils.run_context import RunContext
    from TRAINING.common.utils.audit_enforcer import AuditEnforcer, AuditMode
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False
    RunContext = None

# SST: Import View and Stage enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM: Import deterministic filesystem helpers
from TRAINING.common.utils.determinism_ordering import iterdir_sorted

# Helper function to normalize view for comparisons (handles both enum and string)
def _normalize_view_for_comparison(view: Union[str, View, None]) -> str:
    """Normalize view to string for safe comparisons."""
    if view is None:
        return View.CROSS_SECTIONAL.value
    if isinstance(view, View):
        return view.value
    return str(view).upper()

# Helper function to normalize stage for comparisons (handles both enum and string)
def _normalize_stage_for_comparison(stage: Union[str, Stage, None]) -> str:
    """Normalize stage to string for safe comparisons."""
    if stage is None:
        return None
    if isinstance(stage, Stage):
        return stage.value
    return str(stage).upper()

# Import OutputLayout for view+universe scoped paths
try:
    from TRAINING.orchestration.utils.output_layout import (
        OutputLayout, 
        validate_cohort_metadata,
        _normalize_universe_sig,
        _normalize_view
    )
    _OUTPUT_LAYOUT_AVAILABLE = True
except ImportError:
    _OUTPUT_LAYOUT_AVAILABLE = False
    OutputLayout = None
    validate_cohort_metadata = None
    _normalize_universe_sig = None
    _normalize_view = None

# Import WriteScope for scope-safe writes
try:
    from TRAINING.orchestration.utils.scope_resolution import (
        WriteScope,
        ScopePurpose,
        View as ScopeView,
        Stage as ScopeStage
    )
    _WRITE_SCOPE_AVAILABLE = True
except ImportError:
    _WRITE_SCOPE_AVAILABLE = False
    WriteScope = None
    ScopePurpose = None
    ScopeView = None
    ScopeStage = None

# Use root logger to ensure messages are visible regardless of calling script's logger setup
logger = logging.getLogger(__name__)
# Ensure this logger propagates to root so messages are visible
logger.propagate = True

# Schema version for reproducibility files
REPRODUCIBILITY_SCHEMA_VERSION = 2  # v2: Tagged unions for ambiguous nulls

# Import from modular components
from TRAINING.orchestration.utils.reproducibility.utils import (
    collect_environment_info,
    compute_comparable_key,
    _get_main_logger,
    make_tagged_scalar,
    make_tagged_not_applicable,
    make_tagged_per_target_feature,
    make_tagged_auto,
    make_tagged_not_computed,
    make_tagged_omitted,
    extract_scalar_from_tagged,
    extract_embargo_minutes,
    extract_folds,
    Stage,
    RouteType,
    TargetRankingView,
    # SST accessor functions
    extract_n_effective,
    extract_universe_sig,
    extract_date_range,
    extract_pos_rate,
    extract_feature_counts,
    extract_target,
    extract_model_family,
    extract_run_id,
    extract_purge_minutes,
    extract_horizon_minutes,
)
from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json

# Import SST for comparison group key construction
from TRAINING.common.utils.fingerprinting import construct_comparison_group_key_from_dict

# Helper for inline usage
def _extract_horizon_minutes_sst(metadata, cv_details):
    return extract_horizon_minutes(metadata, cv_details)


# SST: Import canonical _sanitize_for_json from diff_telemetry (single source of truth)
from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json


def _write_atomic_json_with_lock(
    file_path: Path,
    data: Dict[str, Any],
    lock_timeout: float = 30.0
) -> None:
    """
    Write JSON file atomically with file locking to prevent race conditions.

    Uses fcntl.flock with LOCK_EX to ensure exclusive access during write.
    This prevents concurrent writes from multiple processes/threads.

    FIX: Keep file handle open for entire lock lifecycle to prevent race conditions.
    The previous implementation had a bug where it would try to release the lock
    by opening a new file handle after the original handle was closed.

    Args:
        file_path: Target file path
        data: Data to write (will be sanitized automatically)
        lock_timeout: Maximum time to wait for lock (seconds)

    Raises:
        IOError: If write fails or lock cannot be acquired
    """
    import time

    # Sanitize data before writing (convert Timestamps to ISO strings)
    sanitized_data = _sanitize_for_json(data)

    # Create lock file (same directory, .lock extension)
    lock_file = file_path.with_suffix('.lock')

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    lock_acquired = False
    lock_f = None

    try:
        # Keep file handle open for entire lock lifecycle
        lock_f = open(lock_file, 'w')

        # Non-blocking attempt first
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError:
            # Lock is held, wait for it with timeout
            elapsed = 0
            while elapsed < lock_timeout:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.1)  # Wait 100ms before retry
                    elapsed = time.time() - start_time

            if not lock_acquired:
                raise IOError(f"Could not acquire lock for {file_path} within {lock_timeout}s")

        # Lock acquired - perform write
        _write_atomic_json(file_path, sanitized_data)

    except Exception as e:
        raise IOError(f"Failed to write locked JSON to {file_path}: {e}") from e
    finally:
        # Always release lock and close file handle
        if lock_f is not None:
            if lock_acquired:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass  # Best effort unlock
            try:
                lock_f.close()
            except Exception:
                pass  # Best effort close


def _construct_comparison_group_key_from_dict(comparison_group: Dict[str, Any], stage: Union[str, Stage] = Stage.TRAINING) -> Optional[str]:
    """
    Construct comparison_group_key from comparison_group dict.
    
    DEPRECATED: Use construct_comparison_group_key_from_dict from fingerprinting.py directly.
    This wrapper exists for backward compatibility.
    
    Args:
        comparison_group: Dict of comparison group fields
        stage: Stage name for validation (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Returns:
        Comparison key string, or None if invalid
    """
    return construct_comparison_group_key_from_dict(comparison_group, mode="debug", stage=stage)


# All utility functions are now imported from reproducibility.utils (see imports above)


# extract_folds is already imported from reproducibility.utils above

# Import mixins for modular functionality
from TRAINING.orchestration.utils.repro_tracker_mixins import (
    IndexManagerMixin,
    CohortManagerMixin,
    ComparisonEngineMixin,
    LoggingAPIMixin,
)


class ReproducibilityTracker(IndexManagerMixin, CohortManagerMixin, ComparisonEngineMixin, LoggingAPIMixin):
    """
    Tracks run results and compares them to previous runs for reproducibility verification.
    
    Uses tolerance bands with STABLE/DRIFTING/DIVERGED classification instead of binary
    SAME/DIFFERENT. Only escalates to warnings for meaningful differences.
    """
    
    def __init__(
        self,
        output_dir: Path,
        log_file_name: str = "reproducibility_log.json",
        max_runs_per_item: int = 10,
        score_tolerance: float = 0.001,  # Legacy: kept for backward compat, but thresholds loaded from config
        importance_tolerance: float = 0.01,  # Legacy: kept for backward compat
        search_previous_runs: bool = False,  # If True, search parent directories for previous runs
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,  # Override config thresholds
        use_z_score: Optional[bool] = None,  # Override config use_z_score
        audit_mode: str = "warn"  # Audit enforcement mode: "off" | "warn" | "strict"
    ):
        """
        Initialize reproducibility tracker.
        
        Args:
            output_dir: Directory where reproducibility logs are stored (module-specific)
            log_file_name: Name of the JSON log file
            max_runs_per_item: Maximum number of runs to keep per item (prevents log bloat)
            score_tolerance: Legacy tolerance (kept for backward compat, but config thresholds used)
            importance_tolerance: Legacy tolerance (kept for backward compat, but config thresholds used)
            search_previous_runs: If True, search parent directories for previous runs from same module
            thresholds: Optional override for config thresholds (dict with 'roc_auc', 'composite', 'importance' keys)
            use_z_score: Optional override for config use_z_score setting
        """
        self.output_dir = Path(output_dir)
        # Store log file in module-specific directory: output_dir/reproducibility_log.json
        # This ensures each module (target_rankings, feature_selections, training_results) has its own log
        self.log_file = self.output_dir / log_file_name
        self.max_runs_per_item = max_runs_per_item
        self.search_previous_runs = search_previous_runs
        
        # Helper: Get base directory for REPRODUCIBILITY (should be at run level, not module level)
        # If output_dir is a module subdirectory, go up one level; otherwise use output_dir itself
        self._repro_base_dir = self._get_repro_base_dir()
        
        # Load thresholds from config using centralized utilities
        from TRAINING.orchestration.utils.reproducibility.config_loader import (
            load_thresholds,
            load_use_z_score,
            load_cohort_aware,
            load_n_ratio_threshold,
            load_cohort_config_keys
        )
        self.thresholds = load_thresholds(thresholds)
        self.use_z_score = load_use_z_score(use_z_score)
        
        # Load cohort-aware settings
        self.cohort_aware = load_cohort_aware()
        self.n_ratio_threshold = load_n_ratio_threshold()
        self.cohort_config_keys = load_cohort_config_keys()
        
        # Initialize audit enforcer
        audit_mode = self._load_audit_mode()
        if _AUDIT_AVAILABLE:
            self.audit_enforcer = AuditEnforcer(mode=audit_mode)
        else:
            self.audit_enforcer = None
            if audit_mode != "off":
                logger.warning("Audit enforcement not available (RunContext/AuditEnforcer not imported), disabling audit")
        
        # Initialize stats tracking
        # Stats file now goes to globals/ instead of REPRODUCIBILITY/
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        globals_dir = get_globals_dir(self._repro_base_dir)
        globals_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file = globals_dir / "stats.json"
        
        # Routing evaluation root: all ROUTING_EVAL purpose writes go here
        # This keeps evaluation artifacts separate from final artifacts
        self._routing_eval_root = self._repro_base_dir / "routing_evaluation"
        
        # Initialize metrics writer (if enabled)
        try:
            from TRAINING.common.utils.metrics import MetricsWriter, load_metrics_config
            metrics_config = load_metrics_config()
            if metrics_config.get("enabled", False):
                self.metrics = MetricsWriter(
                    output_dir=self._repro_base_dir,  # Base output dir (run level, not module-specific)
                    enabled=metrics_config.get("enabled", True),
                    baselines=metrics_config.get("baselines"),
                    drift=metrics_config.get("drift")
                )
                logger.info(f"✅ Metrics initialized and enabled (output_dir={self._repro_base_dir})")
            else:
                self.metrics = None
                logger.debug("Metrics is disabled in config")
        except Exception as e:
            logger.warning(f"⚠️  Metrics not available: {e}")
            import traceback
            logger.debug(f"Metrics initialization traceback: {traceback.format_exc()}")
            self.metrics = None
    
    def _get_repro_base_dir(self) -> Path:
        """
        Get the base directory for REPRODUCIBILITY structure.
        
        REPRODUCIBILITY should be at the run level, not the module level.
        If output_dir is a module subdirectory (target_rankings/, feature_selections/, training_results/),
        or inside REPRODUCIBILITY/{STAGE}/... structure, walk up to the run directory.
        
        Returns:
            Path to the run-level directory where REPRODUCIBILITY should be created
        """
        # Module subdirectories that indicate we need to go up one level
        module_subdirs = {"target_rankings", "feature_selections", "training_results"}
        
        # Walk up from output_dir to find the run-level directory
        current_dir = self.output_dir
        
        # If we're inside REPRODUCIBILITY/{STAGE}/... structure, walk up to run level
        # Check if we're in a REPRODUCIBILITY subdirectory
        if "REPRODUCIBILITY" in current_dir.parts:
            # Find the index of REPRODUCIBILITY in the path
            repro_idx = None
            for i, part in enumerate(current_dir.parts):
                if part == "REPRODUCIBILITY":
                    repro_idx = i
                    break
            
            if repro_idx is not None and repro_idx > 0:
                # Go up to the directory before REPRODUCIBILITY (run level)
                return Path(*current_dir.parts[:repro_idx])
        
        # If output_dir is a module subdirectory, go up to run level
        if current_dir.name in module_subdirs:
            return current_dir.parent
        
        # Otherwise, output_dir is already at run level
        return current_dir
    
    # Config loading methods now use centralized utilities
    def _load_thresholds(self, override: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
        """Load reproducibility thresholds from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_thresholds
        return load_thresholds(override)
    
    def _load_use_z_score(self, override: Optional[bool] = None) -> bool:
        """Load use_z_score setting from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_use_z_score
        return load_use_z_score(override)
    
    def _load_audit_mode(self) -> str:
        """Load audit mode from config. Defaults to 'off'."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_audit_mode
        return load_audit_mode()
    
    def _load_cohort_aware(self) -> bool:
        """Load cohort_aware setting from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_cohort_aware
        return load_cohort_aware()
    
    def _load_n_ratio_threshold(self) -> float:
        """Load n_ratio_threshold from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_n_ratio_threshold
        return load_n_ratio_threshold()
    
    def _load_cohort_config_keys(self) -> List[str]:
        """Load cohort_config_keys from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_cohort_config_keys
        return load_cohort_config_keys()
    
    @staticmethod
    def _compute_sample_size_bin(n_effective: int) -> Dict[str, Any]:
        """
        Compute sample size bin info (same logic as IntelligentTrainer._get_sample_size_bin).
        
        **Boundary Rules (CRITICAL - DO NOT CHANGE WITHOUT VERSIONING):**
        - Boundaries are EXCLUSIVE upper bounds: `bin_min <= n_effective < bin_max`
        - Example: `sample_25k-50k` means `25000 <= n_effective < 50000`
        
        **Binning Scheme Version:** `sample_bin_v1`
        
        Returns:
            Dict with keys: bin_name, bin_min, bin_max, binning_scheme_version
        """
        BINNING_SCHEME_VERSION = "sample_bin_v1"
        
        # Define bins with EXCLUSIVE upper bounds (bin_min <= N < bin_max)
        bins = [
            (0, 5000, "sample_0-5k"),
            (5000, 10000, "sample_5k-10k"),
            (10000, 25000, "sample_10k-25k"),
            (25000, 50000, "sample_25k-50k"),
            (50000, 100000, "sample_50k-100k"),
            (100000, 250000, "sample_100k-250k"),
            (250000, 500000, "sample_250k-500k"),
            (500000, 1000000, "sample_500k-1M"),
            (1000000, float('inf'), "sample_1M+")
        ]
        
        for bin_min, bin_max, bin_name in bins:
            if bin_min <= n_effective < bin_max:
                return {
                    "bin_name": bin_name,
                    "bin_min": bin_min,
                    "bin_max": bin_max if bin_max != float('inf') else None,
                    "binning_scheme_version": BINNING_SCHEME_VERSION
                }
        
        # Fallback (should never reach here)
        return {
            "bin_name": "sample_unknown",
            "bin_min": None,
            "bin_max": None,
            "binning_scheme_version": BINNING_SCHEME_VERSION
        }
    
    def _find_previous_log_files(self) -> List[Path]:
        """Find all previous reproducibility log files in parent directories (for same module)."""
        if not self.search_previous_runs:
            return []
        
        previous_logs = []
        try:
            current_dir = self.output_dir
            module_name = self.output_dir.name
            
            # Search up to 3 levels up for previous runs
            for _ in range(3):
                parent = current_dir.parent
                if not parent or parent == current_dir:
                    break
                
                # Look for timestamped directories (format: *_YYYYMMDD_HHMMSS or similar)
                if parent.exists():
                    try:
                        # Check if parent contains module subdirectories (target_rankings, feature_selections, etc.)
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for sibling_dir in iterdir_sorted(parent):
                            try:
                                if sibling_dir.is_dir() and sibling_dir != self.output_dir:
                                    # Check if this sibling has the same module subdirectory
                                    module_log = sibling_dir / module_name / self.log_file.name
                                    if module_log.exists():
                                        previous_logs.append(module_log)
                            except (PermissionError, OSError) as e:
                                logger.debug(f"Could not access sibling directory {sibling_dir}: {e}")
                                continue
                    except (PermissionError, OSError) as e:
                        logger.debug(f"Could not iterate parent directory {parent} for previous logs: {e}")
                        continue
                
                current_dir = parent
        except Exception as e:
            logger.warning(f"Error searching for previous log files: {e}")
            # Don't fail completely, just return empty list
        
        return previous_logs
    
    def load_previous_run(
        self,
        stage: str,
        target: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load the previous run's summary for a stage/item combination.
        
        Searches current log file first, then previous runs if search_previous_runs=True.
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            target: Name of the item (e.g., target name, symbol name)
        
        Returns:
            Dictionary with previous run results, or None if no previous run exists
        """
        key = f"{stage}:{target}"
        all_item_runs = []
        
        # First, try current log file
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    all_runs = json.load(f)
                item_runs = all_runs.get(key, [])
                if item_runs:
                    all_item_runs.extend(item_runs)
                    logger.debug(f"Found {len(item_runs)} run(s) in current log: {self.log_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Could not read current log file {self.log_file}: {e}")
        
        # Then, search previous runs if enabled
        if self.search_previous_runs:
            previous_logs = self._find_previous_log_files()
            for prev_log in previous_logs:
                try:
                    with open(prev_log, 'r') as f:
                        all_runs = json.load(f)
                    item_runs = all_runs.get(key, [])
                    if item_runs:
                        all_item_runs.extend(item_runs)
                        logger.debug(f"Found {len(item_runs)} run(s) in previous log: {prev_log}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"Could not read previous log file {prev_log}: {e}")
        
        if not all_item_runs:
            logger.debug(f"No previous runs found for {key}")
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        all_runs = json.load(f)
                        logger.debug(f"Available keys in current log: {list(all_runs.keys())[:10]}")
                except Exception as e:
                    # Best-effort: log file read failed, continue without previous run data
                    logger.debug(f"Could not read log file for debugging: {e}")
            return None
        
        # Sort by timestamp and return most recent
        all_item_runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        logger.debug(f"Found {len(all_item_runs)} total previous run(s) for {key}, using most recent")
        return all_item_runs[0]
    
    def save_run(
        self,
        stage: Union[str, Stage],
        target: str,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the current run's summary to the reproducibility log.
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            target: Name of the item (e.g., target name, symbol name)
            metrics: Dictionary of metrics to track (must include at least auc, std_score)
            additional_data: Optional additional data to store with the run
        """
        # Load existing runs
        all_runs = {}
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    all_runs = json.load(f)
            except (json.JSONDecodeError, IOError):
                all_runs = {}
        
        # Create key for this stage/item combination
        key = f"{stage}:{target}"
        
        # Initialize entry if needed
        if key not in all_runs:
            all_runs[key] = []
        
        # Create summary entry
        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
        from TRAINING.common.utils.determinism_ordering import sorted_items
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "target": target,
            "reproducibility_mode": "LEGACY",  # Track which mode was used
            **{k: float(v) if isinstance(v, (int, float)) else v
               for k, v in sorted_items(metrics)}
        }
        
        if additional_data:
            summary["additional_data"] = additional_data
        
        # Append to item's run history (keep last N runs)
        all_runs[key].append(summary)
        if len(all_runs[key]) > self.max_runs_per_item:
            all_runs[key] = all_runs[key][-self.max_runs_per_item:]
        
        # Save back to file
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(self.log_file, all_runs)
        except IOError as e:
            logger.warning(f"Could not save reproducibility log: {e}")
    
    def _classify_diff(
        self,
        prev_value: float,
        curr_value: float,
        prev_std: Optional[float],
        metric_type: str  # 'roc_auc', 'composite', or 'importance'
    ) -> Tuple[str, float, float, Optional[float]]:
        """
        Classify difference into STABLE/DRIFTING/DIVERGED tiers.
        
        Args:
            prev_value: Previous run value
            curr_value: Current run value
            prev_std: Previous run standard deviation (for z-score calculation)
            metric_type: Type of metric ('roc_auc', 'composite', 'importance')
        
        Returns:
            Tuple of (classification, abs_diff, rel_diff, z_score)
            classification: 'STABLE', 'DRIFTING', or 'DIVERGED'
        """
        diff = curr_value - prev_value
        abs_diff = abs(diff)
        
        # Calculate relative difference
        rel_diff = (abs_diff / max(abs(prev_value), 1e-8)) * 100 if prev_value != 0 else 0.0
        
        # Calculate z-score if std available and use_z_score enabled
        z_score = None
        if self.use_z_score and prev_std is not None and prev_std > 0:
            # Pooled std: use average of previous and current if available
            # For now, use previous std
            z_score = abs_diff / prev_std
        
        # Get thresholds for this metric type
        thresholds = self.thresholds.get(metric_type, self.thresholds.get('roc_auc'))
        abs_thr = thresholds.get('abs', 0.005)
        rel_thr = thresholds.get('rel', 0.02)
        z_thr = thresholds.get('z_score', 1.0)
        
        # Classification logic: require BOTH effect size AND statistical significance for DIVERGED
        # This prevents flagging tiny, statistically insignificant changes as DIVERGED
        # 
        # STABLE: small change AND not statistically significant
        # DRIFTING: moderate change OR borderline statistical significance
        # DIVERGED: large change AND statistically significant
        
        if z_score is not None:
            # Use z-score for statistical significance, abs/rel for effect size
            # Require BOTH big effect AND statistical significance for DIVERGED
            big_effect = abs_diff >= abs_thr or rel_diff >= rel_thr
            statistically_significant = z_score >= z_thr
            
            # For DIVERGED: need BOTH big effect AND statistical significance
            # Use stricter z_thr (2.0) for DIVERGED to require ~95% confidence
            div_thr = max(z_thr * 2.0, 2.0)  # At least 2.0 for DIVERGED
            is_diverged = big_effect and z_score >= div_thr
            
            if not big_effect and not statistically_significant:
                classification = 'STABLE'
            elif is_diverged:
                classification = 'DIVERGED'
            else:
                classification = 'DRIFTING'
        else:
            # Fallback to abs/rel thresholds (no z-score available)
            # Still require both abs AND rel for DIVERGED
            big_abs = abs_diff >= abs_thr
            big_rel = rel_diff >= rel_thr
            is_diverged = big_abs and big_rel
            
            if not big_abs and not big_rel:
                classification = 'STABLE'
            elif is_diverged:
                classification = 'DIVERGED'
            else:
                classification = 'DRIFTING'
        
        return classification, abs_diff, rel_diff, z_score
    
    def _extract_view(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Extract route type from additional_data (for feature_selection stage).
        
        **CRITICAL**: For FEATURE_SELECTION, map view to view:
        - view="CROSS_SECTIONAL" → view="CROSS_SECTIONAL"
        - view="SYMBOL_SPECIFIC" → view="SYMBOL_SPECIFIC"

        This ensures metrics is scoped correctly (features compared per-target, per-view, per-symbol).

        Returns:
            "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or None
        """
        if not additional_data:
            return None
        
        # Check explicit view
        view = additional_data.get('view')
        if view:
            return view.upper()
        
        # FIX: For FEATURE_SELECTION, map view to view (same as TARGET_RANKING)
        # This ensures proper scoping: features compared per-target, per-view, per-symbol
        view = additional_data.get('view')
        if view:
            # Normalize view to enum, then return string value
            try:
                view_enum = View.from_string(view)
                return view_enum.value
            except ValueError:
                # Handle legacy values
                if view.upper() == "CROSS_SECTIONAL":
                    return View.CROSS_SECTIONAL.value
                elif view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                    return View.SYMBOL_SPECIFIC.value
        
        # Infer from other fields (fallback)
        if additional_data.get('cross_sectional') or additional_data.get('is_cross_sectional'):
            return View.CROSS_SECTIONAL.value
        elif additional_data.get('symbol_specific') or additional_data.get('is_symbol_specific'):
            return View.SYMBOL_SPECIFIC.value
        
        # Default: assume CROSS_SECTIONAL if not specified
        return View.CROSS_SECTIONAL.value
    
    def _extract_symbol(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract symbol name from additional_data."""
        if not additional_data:
            return None
        return additional_data.get('symbol')
    
    def _extract_model_family(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract model family from additional_data."""
        if not additional_data:
            return None
        # Use SST accessor for model_family
        return extract_model_family(additional_data)
    
    def _save_to_cohort(
        self,
        stage: Union[str, Stage],
        target: str,
        cohort_id: str,
        cohort_metadata: Dict[str, Any],
        run_data: Dict[str, Any],
        view: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        trend_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint
    ) -> None:
        """
        Save run to cohort-specific directory with structured layout.
        
        Creates:
        - metadata.json: Full cohort metadata
        - metrics.json: Run metrics
        - drift.json: Comparison to previous run (if applicable)
        """
        # Get target-first cohort directory (no longer use legacy REPRODUCIBILITY structure)
        # We still call _get_cohort_dir for path calculation, but we'll use target_cohort_dir instead
        # This is just for logging/compatibility - actual writes go to target_cohort_dir
        legacy_cohort_dir = self._get_cohort_dir(stage, target, cohort_id, view, symbol, model_family)
        # Don't create legacy directory - we only use target-first structure now
        
        # Logging will happen after target_cohort_dir is created (below)
        main_logger = _get_main_logger()
        
        # Generate run_id - use deterministic derivation from RunIdentity (SST pattern)
        # Prefer run_identity if available, then extract from run_data/additional_data, then fallback
        run_id = None
        if run_identity is not None:
            try:
                from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                run_id = derive_run_id_from_identity(
                    run_identity=run_identity
                )
            except Exception as e:
                logger.debug(f"Failed to derive run_id from identity: {e}, trying extract_run_id")
        
        # Fallback to extract_run_id if identity derivation failed
        if not run_id:
            if not isinstance(run_data, dict):
                run_id = None
            else:
                # Pass additional_data as second parameter to allow multi-source extraction
                run_id = extract_run_id(run_data, additional_data)
        
        # Final fallback: use unstable run_id if no identity available
        if not run_id or not isinstance(run_id, str) or not run_id.strip():
            from TRAINING.orchestration.utils.manifest import derive_unstable_run_id, generate_run_instance_id
            run_id = derive_unstable_run_id(generate_run_instance_id())
            logger.debug("Using unstable run_id (no identity available)")
        
        # Now guaranteed to be a non-empty string - safe to call .replace()
        run_id_clean = run_id.replace(':', '-').replace('.', '-').replace('T', '_')
        
        # Normalize stage to enum, then to string for comparisons
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        stage_normalized = str(stage_enum)  # Stage enum's __str__ returns .value
        
        # Normalize view (accept both string and RouteType enum)
        # For TARGET_RANKING, use view from additional_data if available
        if stage_enum == Stage.TARGET_RANKING and not view:
            if additional_data and 'view' in additional_data:
                view = additional_data['view']  # Use view as view for TARGET_RANKING
        elif view and isinstance(view, RouteType):
            view = view.value
        
        # Extract symbols list from cohort_metadata, additional_data
        # Try multiple sources to get the actual symbol list
        symbols_list = None
        if additional_data and 'symbols' in additional_data:
            symbols_list = additional_data['symbols']
        elif cohort_metadata and 'symbols' in cohort_metadata:
            symbols_list = cohort_metadata['symbols']
        elif additional_data and 'symbol_list' in additional_data:
            symbols_list = additional_data['symbol_list']
        
        # For TARGET_RANKING with SYMBOL_SPECIFIC/LOSO view, use symbol from additional_data if available
        if stage_enum == Stage.TARGET_RANKING and not symbol:
            if additional_data and 'symbol' in additional_data:
                symbol = additional_data['symbol']  # Override symbol from additional_data
        
        # Normalize symbols: convert to list, remove duplicates, sort for stable diffs
        if symbols_list is not None:
            if isinstance(symbols_list, str):
                # Handle comma-separated string
                symbols_list = [s.strip() for s in symbols_list.split(',')]
            elif not isinstance(symbols_list, (list, tuple)):
                # Try to convert other iterables
                try:
                    symbols_list = list(symbols_list)
                except (TypeError, ValueError):
                    symbols_list = None
        
        # Clean and sort symbols (remove duplicates, sort for stable git diffs)
        # The extractor already provides sorted, deduplicated list, but be defensive
        if symbols_list:
            symbols_list = sorted(set(str(s).strip() for s in symbols_list if s))
            if not symbols_list:  # Empty after cleaning
                symbols_list = None
        
        # Build full metadata with schema version and explicit IDs
        # For TARGET_RANKING, FEATURE_SELECTION, and TRAINING, include view metadata
        # Schema v2: Use tagged unions for ambiguous nulls (omit non-applicable fields)
        
        # Extract view for stages that require it (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
        view_value = None
        if stage_enum in (Stage.TARGET_RANKING, Stage.FEATURE_SELECTION, Stage.TRAINING):
            # Try to get view from additional_data first
            if additional_data and 'view' in additional_data:
                view_value = additional_data['view'].upper() if isinstance(additional_data['view'], str) else additional_data['view']
            # Fallback: derive from view
            elif view:
                route_normalized = view.upper() if isinstance(view, str) else (view.value if hasattr(view, 'value') else str(view).upper() if view else None)
                # Normalize route_normalized to View enum
                route_enum = View.from_string(route_normalized) if isinstance(route_normalized, str) else route_normalized
                if route_enum == View.CROSS_SECTIONAL:
                    view_value = View.CROSS_SECTIONAL.value
                elif route_enum == View.SYMBOL_SPECIFIC:
                    view_value = View.SYMBOL_SPECIFIC.value
            # Default to CROSS_SECTIONAL if not found
            if not view_value:
                view_value = View.CROSS_SECTIONAL.value
        
        # ========================================================================
        # PR1 FIREWALL: OutputLayout validation for view+universe scoping
        # This is the choke point that catches all writes and validates scope
        # ========================================================================
        if _OUTPUT_LAYOUT_AVAILABLE and stage_enum in (Stage.TARGET_RANKING, Stage.FEATURE_SELECTION, Stage.TRAINING):
            # Extract and normalize metadata fields
            raw_view = cohort_metadata.get("view") or view_value
            normalized_view = _normalize_view({"view": raw_view}) if _normalize_view else None
            universe_sig = _normalize_universe_sig(cohort_metadata) if _normalize_universe_sig else None
            symbol_from_meta = symbol or cohort_metadata.get("symbol")
            # Use SST accessor for target
            target = extract_target(cohort_metadata) or target
            
            # Check strict mode config flag
            strict_mode = False
            try:
                from CONFIG.config_loader import load_config
                cfg = load_config()
                strict_mode = getattr(getattr(getattr(cfg, 'safety', None), 'output_layout', None), 'strict_scope_partitioning', False)
            except Exception as e:
                logger.debug(f"Could not load strict mode config: {e}, defaulting to non-strict")
            
            # ========================================================================
            # HARD INVARIANTS: These ALWAYS fire regardless of metadata completeness
            # Cohort prefix mismatches indicate upstream bugs that corrupt the output
            # ========================================================================
            
            # Invariant 1: cohort_id prefix must match view (ALWAYS enforced)
            if cohort_id and normalized_view:
                if _normalize_view_for_comparison(normalized_view) == View.CROSS_SECTIONAL.value and cohort_id.startswith("sy_"):
                    raise ValueError(
                        f"SCOPE VIOLATION: Cannot write sy_ cohort to CROSS_SECTIONAL view. "
                        f"cohort_id={cohort_id}, view={normalized_view}, stage={stage_normalized}, "
                        f"target={target}, symbol={symbol_from_meta}, universe_sig={universe_sig}. "
                        f"Check that view_for_writes comes from resolved_data_config['view']."
                    )
                if _normalize_view_for_comparison(normalized_view) == View.SYMBOL_SPECIFIC.value and cohort_id.startswith("cs_"):
                    raise ValueError(
                        f"SCOPE VIOLATION: Cannot write cs_ cohort to SYMBOL_SPECIFIC view. "
                        f"cohort_id={cohort_id}, view={normalized_view}, stage={stage_normalized}, "
                        f"target={target}, symbol={symbol_from_meta}, universe_sig={universe_sig}. "
                        f"This indicates a missing symbol in cohort computation."
                    )
            
            # Invariant 2: symbol presence must match view (ALWAYS enforced when view is known)
            if normalized_view:
                symbol_key_present = "symbol" in cohort_metadata or symbol is not None
                if _normalize_view_for_comparison(normalized_view) == View.CROSS_SECTIONAL.value and symbol_key_present:
                    raise ValueError(
                        f"SCOPE VIOLATION: symbol key present for CROSS_SECTIONAL view. "
                        f"symbol={symbol_from_meta}, view={normalized_view}, stage={stage_normalized}, "
                        f"target={target}, cohort_id={cohort_id}. "
                        f"CS metadata must not have symbol key at all (not even null)."
                    )
                if _normalize_view_for_comparison(normalized_view) == View.SYMBOL_SPECIFIC.value and not symbol_from_meta:
                    raise ValueError(
                        f"SCOPE VIOLATION: symbol required for SYMBOL_SPECIFIC view but was None/empty. "
                        f"view={normalized_view}, stage={stage_normalized}, target={target}, "
                        f"cohort_id={cohort_id}. Either provide symbol or use CROSS_SECTIONAL view."
                    )
            
            # Invariant 3: universe_sig required (enforced based on strict mode)
            if not universe_sig:
                if strict_mode:
                    raise ValueError(
                        f"SCOPE VIOLATION: universe_sig missing (strict mode enabled). "
                        f"view={normalized_view}, stage={stage_normalized}, target={target}, "
                        f"symbol={symbol_from_meta}, cohort_id={cohort_id}. "
                        f"Ensure resolved_data_config['universe_sig'] is propagated."
                    )
                else:
                    logger.warning(
                        f"Missing universe_sig for {stage_normalized}/{target}. "
                        f"view={normalized_view}, symbol={symbol_from_meta}. "
                        f"Enable strict_scope_partitioning=true to enforce."
                    )
            
            # ========================================================================
            # END HARD INVARIANTS
            # ========================================================================
            
            # Determine if we have all required metadata for full OutputLayout validation
            has_required_metadata = bool(normalized_view and universe_sig and target)
            if normalized_view == "SYMBOL_SPECIFIC" and not symbol_from_meta:
                has_required_metadata = False
            
            # If metadata has required fields, validate using OutputLayout
            if has_required_metadata:
                try:
                    layout = OutputLayout(
                        output_root=self._repro_base_dir,
                        target=target,
                        view=normalized_view,
                        universe_sig=universe_sig,
                        symbol=symbol_from_meta,
                        cohort_id=cohort_id,
                        stage=stage_normalized,  # Pass stage from context for proper path scoping
                    )
                    # Validate cohort_id matches view using OutputLayout
                    layout.validate_cohort_id(cohort_id)
                    
                    # Invariant 4: symbol param and metadata symbol must agree (if both present)
                    if symbol and symbol_from_meta and symbol != symbol_from_meta:
                        raise ValueError(
                            f"SCOPE VIOLATION: symbol mismatch - param symbol={symbol}, metadata symbol={symbol_from_meta}. "
                            f"stage={stage_normalized}, target={target}, cohort_id={cohort_id}. "
                            f"This indicates dirty/mutated metadata dict."
                        )
                    
                    logger.debug(f"OutputLayout validation passed: view={normalized_view}, cohort_id={cohort_id}")
                except ValueError as e:
                    # This is a scope violation - always error
                    logger.error(f"SCOPE VIOLATION: {e}")
                    raise
            else:
                # Legacy fallback behavior - build detailed missing list
                missing = []
                
                if not raw_view:
                    missing.append("view")
                elif not normalized_view:
                    missing.append(f"view (invalid: {raw_view})")
                
                if not universe_sig:
                    missing.append("universe_sig")
                
                if not target:
                    missing.append("target")
                
                # If view is valid and symbol-specific, require symbol
                if _normalize_view_for_comparison(normalized_view) == View.SYMBOL_SPECIFIC.value and not symbol_from_meta:
                    missing.append("symbol")
                
                if strict_mode:
                    # Hard error in strict mode
                    raise ValueError(
                        f"Missing required metadata for OutputLayout (strict mode enabled): {missing}. "
                        f"Metadata keys: {list(cohort_metadata.keys())}. "
                        f"Set safety.output_layout.strict_scope_partitioning=false to allow legacy fallback."
                    )
                else:
                    # Warn and fall back to legacy path construction
                    logger.warning(
                        f"Missing {missing} in metadata for {stage}/{target}. "
                        f"Falling back to legacy path construction. "
                        f"Metadata keys: {list(cohort_metadata.keys())}. "
                        f"Enable safety.output_layout.strict_scope_partitioning=true to enforce strict validation."
                    )
                    
                    # SCOPE VIOLATION DETECTOR: Telemetry even when view is missing/invalid
                    # In strict mode, raise on cohort prefix/view mismatch
                    if cohort_id:
                        prefix = "sy" if cohort_id.startswith("sy_") else "cs" if cohort_id.startswith("cs_") else "unknown"
                        # Check for prefix/view mismatch
                        prefix_view_mismatch = (
                            (_normalize_view_for_comparison(normalized_view) == View.CROSS_SECTIONAL.value and prefix == "sy") or
                            (_normalize_view_for_comparison(normalized_view) == View.SYMBOL_SPECIFIC.value and prefix == "cs")
                        )
                        if prefix_view_mismatch and strict_mode:
                            raise ValueError(
                                f"SCOPE VIOLATION: cohort_prefix={prefix}_ but view={normalized_view}. "
                                f"cohort_id={cohort_id}, stage={stage}, target={target}. "
                                f"This indicates the view was not properly propagated from SST. "
                                f"Set safety.output_layout.strict_scope_partitioning=false to allow legacy fallback."
                            )
                        elif prefix_view_mismatch or normalized_view is None:
                            logger.error(
                                f"SCOPE VIOLATION RISK: view={normalized_view or 'UNKNOWN'} raw_view={raw_view or 'None'} "
                                f"cohort_prefix={prefix} cohort_id={cohort_id} stage={stage} item={target} "
                                f"view={view} symbol={symbol_from_meta}"
                            )
        # ========================================================================
        # END PR1 FIREWALL
        # ========================================================================
        
        # FIX: Normalize cs_config before hashing to ensure consistent structure
        # Always include all keys (even if None) to prevent different hashes for same config
        cs_config_for_hash = (cohort_metadata.get('cs_config') or {}).copy()
        # Ensure all expected keys are present (with None if missing)
        expected_keys = ['min_cs', 'max_cs_samples', 'leakage_filter_version', 'universe_sig']
        for key in expected_keys:
            if key not in cs_config_for_hash:
                cs_config_for_hash[key] = None
        # SST: Use canonical_json and sha256_short for consistent config hashing
        from TRAINING.common.utils.config_hashing import canonical_json, sha256_short
        config_hash = sha256_short(canonical_json(cs_config_for_hash), 8)
        
        full_metadata = {
            "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
            "cohort_id": cohort_id,
            "run_id": run_id_clean,
            "stage": stage_normalized,  # Already normalized to uppercase
            "view": view_value,  # Set for TARGET_RANKING, FEATURE_SELECTION, and TRAINING stages
            "target": target,  # Changed from "target" to match finalize_run() expectations
            "n_effective": cohort_metadata.get('n_effective_cs', 0),  # Changed from "n_effective" to match finalize_run() expectations
            "n_symbols": cohort_metadata.get('n_symbols', 0),
            "symbols": symbols_list,  # Sorted, deduplicated list of symbols
            "date_start": (cohort_metadata.get('date_range') or {}).get('start_ts'),  # Changed from "date_start" to match finalize_run() expectations
            "date_end": (cohort_metadata.get('date_range') or {}).get('end_ts'),  # Changed from "date_end" to match finalize_run() expectations
            # FIX: Single assignment with proper fallback - _normalize_universe_sig checks both top-level and cs_config
            "universe_sig": _normalize_universe_sig(cohort_metadata) if _normalize_universe_sig else ((cohort_metadata.get('cs_config') or {}).get('universe_sig') or cohort_metadata.get('universe_sig')),
            "min_cs": (cohort_metadata.get('cs_config') or {}).get('min_cs'),
            "max_cs_samples": (cohort_metadata.get('cs_config') or {}).get('max_cs_samples'),
            "leakage_filter_version": (cohort_metadata.get('cs_config') or {}).get('leakage_filter_version', 'v1'),
            "config_hash": config_hash,
            "seed": run_data.get('seed') or (additional_data.get('seed') if additional_data else None),
            "git_commit": self._get_git_commit(),
            "created_at": datetime.now().isoformat()
        }
        
        # SST: Fallback extraction from run_identity if primary sources are null
        # This ensures authoritative values from RunIdentity are used when cohort_metadata is incomplete
        if run_identity is not None:
            # universe_sig fallback from dataset_signature
            if full_metadata.get('universe_sig') is None and hasattr(run_identity, 'dataset_signature') and run_identity.dataset_signature:
                # dataset_signature is the full 64-char hash, extract first 12 chars for universe_sig format
                full_metadata['universe_sig'] = run_identity.dataset_signature[:12]
            # seed fallback from train_seed
            if full_metadata.get('seed') is None and hasattr(run_identity, 'train_seed') and run_identity.train_seed is not None:
                full_metadata['seed'] = run_identity.train_seed
        
        # Schema v2: Omit non-applicable fields instead of null
        # Only include symbol if view is SYMBOL_SPECIFIC
        route_normalized = view.upper() if view else None
        # Normalize route_normalized to enum for comparison
        route_enum = View.from_string(route_normalized) if isinstance(route_normalized, str) else route_normalized
        # Check additional_data view (may be string from JSON)
        additional_view = additional_data.get('view') if additional_data else None
        additional_view_enum = View.from_string(additional_view) if isinstance(additional_view, str) and additional_view else None
        if symbol and (route_enum == View.SYMBOL_SPECIFIC or 
                      (stage_enum == Stage.TARGET_RANKING and additional_data and 
                       (additional_view_enum == View.SYMBOL_SPECIFIC or (isinstance(additional_view, str) and additional_view == 'LOSO')))):
            full_metadata["symbol"] = symbol
        # Otherwise omit (cross-sectional doesn't have a single symbol)
        
        # Only include model_family if specified
        if model_family:
            full_metadata["model_family"] = model_family
        # Otherwise omit (not applicable for multi-model or unspecified)
        
        # Add audit-grade fields: data fingerprint and per-symbol stats
        if cohort_metadata.get('data_fingerprint'):
            full_metadata['data_fingerprint'] = cohort_metadata['data_fingerprint']
        
        if cohort_metadata.get('per_symbol_stats'):
            full_metadata['per_symbol_stats'] = cohort_metadata['per_symbol_stats']
        
        # CRITICAL: Add config fingerprints from resolved config for SST consistency
        # This ensures resolved_metadata has config fingerprints available for snapshot creation
        # Try to load from config.resolved.json (SST source of truth)
        # Use retry logic with timeout to handle race conditions where file might not exist immediately
        try:
            import time
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(self._repro_base_dir)
            resolved_config_path = globals_dir / "config.resolved.json"
            
            # Retry logic: wait up to 5 seconds for file to appear (handles concurrent creation)
            max_retries = 10
            retry_delay = 0.5  # 500ms between retries
            resolved_config = None
            
            for attempt in range(max_retries):
                if resolved_config_path.exists():
                    try:
                        with open(resolved_config_path, 'r') as f:
                            resolved_config = json.load(f)
                        # Successfully loaded, break retry loop
                        break
                    except (json.JSONDecodeError, IOError) as e:
                        # File exists but not readable yet (might be being written)
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.warning(
                                f"Could not read config.resolved.json after {max_retries} attempts: {e}. "
                                f"File may be corrupted or locked."
                            )
                            resolved_config = None
                            break
                else:
                    # File doesn't exist yet, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.debug(
                            f"config.resolved.json not found after {max_retries} attempts "
                            f"(waited {max_retries * retry_delay:.1f}s). "
                            f"Fallback to loading in diff_telemetry will handle this."
                        )
                        resolved_config = None
                        break
            
            # Add both fingerprints to full_metadata for SST consistency
            if resolved_config:
                if 'config_fingerprint' in resolved_config:
                    full_metadata['config_fingerprint'] = resolved_config['config_fingerprint']
                if 'deterministic_config_fingerprint' in resolved_config:
                    full_metadata['deterministic_config_fingerprint'] = resolved_config['deterministic_config_fingerprint']
        except Exception as e:
            # Non-critical: fallback to loading in diff_telemetry will handle this
            logger.debug(f"Could not load config fingerprints from config.resolved.json: {e}")
        
        # Add CV details from additional_data
        if additional_data:
            cv_details = {}
            
            # CV method and parameters
            cv_enabled = True
            if 'cv_method' in additional_data:
                cv_details['cv_method'] = additional_data['cv_method']
                if additional_data['cv_method'] in ('none', None):
                    cv_enabled = False
            elif 'cv_scheme' in additional_data:
                cv_details['cv_method'] = additional_data['cv_scheme']
                if additional_data['cv_scheme'] in ('none', None):
                    cv_enabled = False
            elif 'cv_skipped' in additional_data and additional_data['cv_skipped']:
                cv_enabled = False
                cv_details['cv_method'] = 'none'
            else:
                cv_details['cv_method'] = 'purged_kfold'  # Default assumption
            
            # Add explicit enabled flag
            cv_details['enabled'] = cv_enabled
            
            # Horizon, purge, embargo
            if 'horizon_minutes' in additional_data:
                cv_details['horizon_minutes'] = additional_data['horizon_minutes']
            if 'purge_minutes' in additional_data:
                cv_details['purge_minutes'] = additional_data['purge_minutes']
            elif 'purge_time' in additional_data:
                # Extract minutes from Timedelta string
                try:
                    purge_str = str(additional_data['purge_time'])
                    if 'days' in purge_str:
                        # Parse Timedelta string
                        import re
                        match = re.search(r'(\d+)\s*days?\s*(\d+):(\d+):(\d+)', purge_str)
                        if match:
                            days, hours, minutes, seconds = map(int, match.groups())
                            cv_details['purge_minutes'] = days * 24 * 60 + hours * 60 + minutes + seconds / 60
                    else:
                        # Try to extract minutes directly
                        match = re.search(r'(\d+)\s*min', purge_str, re.I)
                        if match:
                            cv_details['purge_minutes'] = int(match.group(1))
                except Exception:
                    pass
            
            # Schema v2: embargo_minutes as tagged union
            # FIX: If CV is enabled (cv_method is set), embargo should be explicitly set to 0 if None,
            # not marked as "not_applicable". Only mark as "not_applicable" if CV is actually disabled.
            if 'embargo_minutes' in additional_data:
                embargo_val = additional_data['embargo_minutes']
                if embargo_val is None:
                    # Check if embargo is per-target-feature (has feature_time_meta_map)
                    if 'feature_time_meta_map' in additional_data and additional_data['feature_time_meta_map']:
                        # Per-target-feature: store reference to artifact
                        embargo_map_path = None
                        embargo_map_sha256 = None
                        # Try to find embargo map artifact
                        if 'embargo_map_path' in additional_data:
                            embargo_map_path = additional_data['embargo_map_path']
                        if 'embargo_map_sha256' in additional_data:
                            embargo_map_sha256 = additional_data['embargo_map_sha256']
                        
                        # Compute rollup stats if available
                        rollup = None
                        if embargo_map_path or embargo_map_sha256:
                            # Try to compute rollup from feature_time_meta_map
                            embargo_values = []
                            for feat_meta in additional_data['feature_time_meta_map'].values():
                                if hasattr(feat_meta, 'embargo_minutes'):
                                    embargo_values.append(feat_meta.embargo_minutes)
                            if embargo_values:
                                import numpy as np
                                rollup = {
                                    "min": float(np.min(embargo_values)),
                                    "p50": float(np.median(embargo_values)),
                                    "max": float(np.max(embargo_values)),
                                    "unique_count": len(set(embargo_values))
                                }
                        
                        cv_details['embargo_minutes'] = make_tagged_per_target_feature(
                            ref_path=embargo_map_path,
                            ref_sha256=embargo_map_sha256,
                            rollup=rollup
                        )
                    else:
                        # CRITICAL FIX: If CV method is set (CV is enabled), embargo should be explicitly 0,
                        # not "not_applicable". Only mark as "not_applicable" if CV is actually disabled.
                        cv_method = cv_details.get('cv_method', '')
                        cv_skipped = additional_data.get('cv_skipped', False)
                        if cv_method and cv_method != 'none' and not cv_skipped:
                            # CV is enabled but embargo is None -> explicitly set to 0 (disabled)
                            cv_details['embargo_minutes'] = make_tagged_scalar(0.0)
                            cv_details['embargo_enabled'] = False
                        else:
                            # CV is disabled or not applicable
                            cv_details['embargo_minutes'] = make_tagged_not_applicable(reason="cv_disabled_or_not_applicable")
                else:
                    # Scalar value
                    cv_details['embargo_minutes'] = make_tagged_scalar(embargo_val)
                    # If embargo is explicitly set (non-zero), mark as enabled
                    if embargo_val != 0:
                        cv_details['embargo_enabled'] = True
                    else:
                        cv_details['embargo_enabled'] = False
            
            # Schema v2: folds as tagged union
            if 'folds' in additional_data:
                folds_val = additional_data['folds']
            elif 'n_splits' in additional_data:
                folds_val = additional_data['n_splits']
            else:
                folds_val = None
            
            if folds_val is not None:
                # Check if it was auto-computed
                if 'folds_auto' in additional_data and additional_data.get('folds_auto', False):
                    cv_details['folds'] = make_tagged_auto(value=folds_val)
                else:
                    cv_details['folds'] = make_tagged_scalar(folds_val)
            elif 'cv_skipped' in additional_data and additional_data['cv_skipped']:
                cv_details['folds'] = make_tagged_not_applicable(reason="cv_disabled")
            # Otherwise omit (not computed yet)
            
            # Fold boundaries hash
            if 'fold_boundaries' in additional_data:
                fold_boundaries = additional_data['fold_boundaries']
                try:
                    fold_boundaries_str = json.dumps(fold_boundaries, sort_keys=True)
                    cv_details['fold_boundaries_hash'] = hashlib.sha256(fold_boundaries_str.encode()).hexdigest()[:16]
                    # Also store the actual boundaries (for debugging)
                    cv_details['fold_boundaries'] = fold_boundaries
                except Exception as e:
                    # CRITICAL: Hash computation affects artifact metadata - use centralized error handling
                    from TRAINING.common.exceptions import handle_error_with_policy
                    handle_error_with_policy(
                        error=e,
                        stage="REPRODUCIBILITY",  # Generic stage for metadata extraction
                        error_type="fold_boundaries_hash",
                        affects_artifact=True,
                        fallback_value=None,  # Continue without hash if fail-open
                        logger_instance=logger
                    )
            elif 'fold_timestamps' in additional_data:
                # Use fold_timestamps to compute hash
                fold_timestamps = additional_data['fold_timestamps']
                try:
                    fold_timestamps_str = json.dumps(fold_timestamps, sort_keys=True, default=str)
                    cv_details['fold_boundaries_hash'] = hashlib.sha256(fold_timestamps_str.encode()).hexdigest()[:16]
                    # Also store the timestamps (for debugging) - convert Timestamps to ISO strings
                    if fold_timestamps:
                        cv_details['fold_timestamps'] = _sanitize_for_json(fold_timestamps)
                except Exception as e:
                    # CRITICAL: Hash computation affects artifact metadata - use centralized error handling
                    from TRAINING.common.exceptions import handle_error_with_policy
                    handle_error_with_policy(
                        error=e,
                        stage="REPRODUCIBILITY",  # Generic stage for metadata extraction
                        error_type="fold_timestamps_hash",
                        affects_artifact=True,
                        fallback_value=None,  # Continue without hash if fail-open
                        logger_instance=logger
                    )
            
            # Feature lookback max minutes
            if 'feature_lookback_max_minutes' in additional_data:
                cv_details['feature_lookback_max_minutes'] = additional_data['feature_lookback_max_minutes']
            elif 'max_feature_lookback_minutes' in additional_data:
                cv_details['feature_lookback_max_minutes'] = additional_data['max_feature_lookback_minutes']
            
            # Label definition hash
            if 'label_definition_hash' in additional_data:
                cv_details['label_definition_hash'] = additional_data['label_definition_hash']
            elif 'target_config_hash' in additional_data:
                cv_details['label_definition_hash'] = additional_data['target_config_hash']
            
            # Splitter implementation (if available)
            if 'splitter_impl' in additional_data:
                cv_details['splitter_impl'] = additional_data['splitter_impl']
            elif 'cv_splitter_class' in additional_data:
                cv_details['splitter_impl'] = additional_data['cv_splitter_class']
            
            if cv_details:
                full_metadata['cv_details'] = cv_details
        
        # Add trend metadata (if computed)
        if trend_metadata:
            full_metadata['trend'] = trend_metadata
        
        # Add feature registry hash
        if additional_data and 'feature_registry_hash' in additional_data:
            full_metadata['feature_registry_hash'] = additional_data['feature_registry_hash']
        elif additional_data and 'feature_names' in additional_data:
            # Compute hash from feature names (sorted for stability)
            try:
                feature_names = additional_data['feature_names']
                if isinstance(feature_names, (list, tuple)):
                    feature_names_sorted = sorted([str(f) for f in feature_names])
                    feature_registry_str = "|".join(feature_names_sorted)
                    full_metadata['feature_registry_hash'] = hashlib.sha256(feature_registry_str.encode()).hexdigest()[:16]
            except Exception as e:
                # CRITICAL: Hash computation affects artifact metadata - use centralized error handling
                from TRAINING.common.exceptions import handle_error_with_policy
                handle_error_with_policy(
                    error=e,
                    stage="REPRODUCIBILITY",  # Generic stage for metadata extraction
                    error_type="feature_registry_hash",
                    affects_artifact=True,
                    fallback_value=None,  # Continue without hash if fail-open
                    logger_instance=logger
                )
        
        # Add sample size bin metadata (for directory organization, NOT series identity)
        # Compute from n_effective if not provided, ensuring consistency
        # This allows backward compatibility and binning scheme versioning
        # CRITICAL: Use SST accessor for n_effective
        n_effective = extract_n_effective(full_metadata)
        if n_effective and n_effective > 0:
            # Use provided bin info if available, otherwise compute from n_effective
            if additional_data and 'sample_size_bin' in additional_data:
                full_metadata['sample_size_bin'] = additional_data['sample_size_bin']
            else:
                # Compute bin info using same logic as IntelligentTrainer
                # This ensures consistency even if bin info wasn't passed through
                bin_info = ReproducibilityTracker._compute_sample_size_bin(n_effective)
                if bin_info:
                    full_metadata['sample_size_bin'] = bin_info
        
        # NEW: Add dropped features metadata (if provided)
        if additional_data and 'dropped_features' in additional_data:
            full_metadata['dropped_features'] = additional_data['dropped_features']
        
        # NEW: Add environment information (audit-grade metadata)
        try:
            env_info = collect_environment_info()
            if env_info:
                full_metadata['environment'] = env_info
        except Exception as e:
            logger.debug(f"Failed to collect environment info: {e}")
        
        # NEW: Add data source details (if available)
        data_source_info = {}
        if additional_data:
            if 'data_source' in additional_data:
                data_source_info['source'] = additional_data['data_source']
            if 'dataset_id' in additional_data:
                data_source_info['dataset_id'] = additional_data['dataset_id']
            elif 'dataset_manifest_hash' in additional_data:
                data_source_info['dataset_manifest_hash'] = additional_data['dataset_manifest_hash']
            if 'bar_size' in additional_data:
                data_source_info['bar_size'] = additional_data['bar_size']
            elif 'data_interval_minutes' in additional_data:
                data_source_info['bar_size'] = f"{additional_data['data_interval_minutes']}m"
            elif 'timeframe' in additional_data:
                data_source_info['bar_size'] = additional_data['timeframe']
            if 'timezone' in additional_data:
                data_source_info['timezone'] = additional_data['timezone']
            if 'market_calendar' in additional_data:
                data_source_info['market_calendar'] = additional_data['market_calendar']
            elif 'session_filters' in additional_data:
                data_source_info['session_filters'] = additional_data['session_filters']
        
        if data_source_info:
            full_metadata['data_source'] = data_source_info
        
        # NEW: Add evaluation details
        evaluation_info = {}
        if additional_data:
            if 'target_definition' in additional_data:
                evaluation_info['target_definition'] = additional_data['target_definition']
            elif 'target_config' in additional_data:
                # Store a hash or summary of target config
                try:
                    target_config = additional_data['target_config']
                    if isinstance(target_config, dict):
                        # SST: Use canonical_json and sha256_short for consistent config hashing
                        from TRAINING.common.utils.config_hashing import canonical_json, sha256_short
                        evaluation_info['target_config_hash'] = sha256_short(canonical_json(target_config), 16)
                except Exception as e:
                    # CRITICAL: Hash computation affects artifact metadata - use centralized error handling
                    from TRAINING.common.exceptions import handle_error_with_policy
                    handle_error_with_policy(
                        error=e,
                        stage="REPRODUCIBILITY",  # Generic stage for metadata extraction
                        error_type="target_config_hash",
                        affects_artifact=True,
                        fallback_value=None,  # Continue without hash if fail-open
                        logger_instance=logger
                    )
            
            # Feature counts
            if 'feature_names' in additional_data:
                feature_names = additional_data['feature_names']
                if isinstance(feature_names, (list, tuple)):
                    evaluation_info['n_features'] = len(feature_names)
                    # Count features by family (if feature names follow a pattern)
                    family_counts = {}
                    for feat_name in feature_names:
                        # Common pattern: family_feature or family__feature
                        parts = str(feat_name).split('_', 1)
                        if len(parts) >= 1:
                            family = parts[0]
                            family_counts[family] = family_counts.get(family, 0) + 1
                    if family_counts:
                        evaluation_info['feature_family_counts'] = family_counts
            
            if 'feature_registry_version' in additional_data:
                evaluation_info['feature_registry_version'] = additional_data['feature_registry_version']
        
        if evaluation_info:
            full_metadata['evaluation'] = evaluation_info
        
        # NEW: Add training information (hyperparameters, train_seed) for TRAINING and FEATURE_SELECTION stages
        # CRITICAL: This is needed for comparability - different HPs/seeds = different outcomes
        # FEATURE_SELECTION also uses models (LightGBM, etc.) with hyperparameters that affect feature selection
        if stage_enum in (Stage.TRAINING, Stage.FEATURE_SELECTION) and additional_data:
            training_info = {}
            
            # Extract train_seed (distinct from split_seed)
            train_seed = (
                additional_data.get('train_seed') or
                additional_data.get('seed') or
                run_data.get('train_seed') or
                run_data.get('seed')
            )
            if train_seed is not None:
                try:
                    training_info['train_seed'] = int(train_seed)
                except (ValueError, TypeError):
                    pass
            
            # Extract hyperparameters from training config
            hyperparameters = {}
            if 'training' in additional_data and isinstance(additional_data['training'], dict):
                training_config = additional_data['training']
                # Extract all hyperparameters (exclude model_family, strategy, seeds - those are handled separately)
                excluded_keys = {'model_family', 'strategy', 'split_seed', 'train_seed', 'seed'}
                # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                from TRAINING.common.utils.determinism_ordering import sorted_items
                for key, value in sorted_items(training_config):
                    if key not in excluded_keys and value is not None:
                        hyperparameters[key] = value
            elif 'hyperparameters' in additional_data:
                # Direct hyperparameters dict
                hp_dict = additional_data['hyperparameters']
                if isinstance(hp_dict, dict):
                    hyperparameters = hp_dict
            
            if hyperparameters:
                training_info['hyperparameters'] = hyperparameters
            
            if training_info:
                full_metadata['training'] = training_info
        
        # NEW: Compute comparable_key for run comparison
        try:
            comparable_key = compute_comparable_key(
                stage=stage_normalized,
                target=target,
                view=full_metadata.get('view'),
                symbol=full_metadata.get('symbol'),
                date_start=full_metadata.get('date_start'),
                date_end=full_metadata.get('date_end'),
                cv_details=full_metadata.get('cv_details'),
                feature_registry_hash=full_metadata.get('feature_registry_hash'),
                label_definition_hash=full_metadata.get('cv_details', {}).get('label_definition_hash') if full_metadata.get('cv_details') else None,
                min_cs=full_metadata.get('min_cs'),
                max_cs_samples=full_metadata.get('max_cs_samples'),
                universe_sig=full_metadata.get('universe_sig')
            )
            if comparable_key:
                full_metadata['comparable_key'] = comparable_key
        except Exception as e:
            logger.debug(f"Failed to compute comparable_key: {e}")
        
        # CRITICAL: Add metrics to full_metadata before finalize_run() is called
        # This ensures _compute_metrics_digest() can find metrics via resolved_metadata['metrics']
        # Extract metrics from run_data (metrics are passed in run_data for TARGET_RANKING/FEATURE_SELECTION)
        if run_data.get('metrics'):
            # Metrics are already in run_data - add to full_metadata for resolved_metadata
            full_metadata['metrics'] = run_data['metrics']
        elif additional_data and 'metrics' in additional_data:
            # Fallback: check additional_data
            full_metadata['metrics'] = additional_data['metrics']
        else:
            # Fallback: Reconstruct metrics from top-level keys (for backward compatibility)
            # Known metric keys from build_clean_metrics_dict() structure
            known_metric_keys = {
                'schema', 'scope', 'primary_metric', 'coverage', 'features', 
                'y_stats', 'label_stats', 'models', 'score', 'fold_timestamps',
                'leakage', 'mismatch_telemetry', 'metrics_schema_version', 
                'scoring_schema_version', 'n_effective', 'metric_name'
            }
            # Check if any known metric keys exist at top level
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            from TRAINING.common.utils.determinism_ordering import sorted_items
            top_level_metrics = {k: v for k, v in sorted_items(run_data) if k in known_metric_keys}
            if top_level_metrics:
                full_metadata['metrics'] = top_level_metrics
                logger.debug(f"Reconstructed metrics from top-level keys: {list(top_level_metrics.keys())}")
        
        # CRITICAL: Initialize telemetry if not already initialized
        # Telemetry is needed for diff tracking and should be available for all runs
        # Check if telemetry exists as instance variable or needs to be created
        # NOTE: We use self._repro_base_dir here since cohort_dir/target_cohort_dir may not be defined yet
        # The actual cohort_dir will be passed to finalize_run() later
        # CRITICAL: Always compute current run directory from self.output_dir (don't rely on stale telemetry)
        # This ensures we use the CURRENT run even if tracker is reused
        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root, normalize_run_root
        current_run_dir = get_run_root(self.output_dir)
        if current_run_dir is None:
            # Fallback to _repro_base_dir if run_root fails (defensive)
            logger.warning(f"Cannot resolve run root from self.output_dir={self.output_dir}, using _repro_base_dir={self._repro_base_dir}")
            current_run_dir = get_run_root(self._repro_base_dir) or self._repro_base_dir
        else:
            current_run_dir = normalize_run_root(current_run_dir)
            logger.debug(f"Detected current run directory: {current_run_dir} (computed from self.output_dir={self.output_dir})")
        
        # Check if telemetry exists and if it's for the current run
        telemetry = getattr(self, '_telemetry', None)
        if telemetry is None or not hasattr(telemetry, 'run_dir') or not telemetry.run_dir:
            # Initialize telemetry with current run directory
            try:
                from TRAINING.orchestration.utils.diff_telemetry import DiffTelemetry
                telemetry = DiffTelemetry(output_dir=current_run_dir)
                self._telemetry = telemetry
                logger.debug(f"Initialized telemetry with current run directory: {current_run_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize diff telemetry: {e}")
                import traceback
                logger.debug(f"Telemetry initialization traceback: {traceback.format_exc()}")
                telemetry = None
                self._telemetry = None
        else:
            # Verify telemetry is for current run - re-initialize if stale
            telemetry_run_resolved = Path(telemetry.run_dir).resolve()
            current_run_resolved = Path(current_run_dir).resolve()
            if telemetry_run_resolved != current_run_resolved:
                logger.warning(
                    f"Telemetry run_dir ({telemetry.run_dir}) does not match current run directory ({current_run_dir}). "
                    f"Re-initializing telemetry with current run directory."
                )
                try:
                    from TRAINING.orchestration.utils.diff_telemetry import DiffTelemetry
                    telemetry = DiffTelemetry(output_dir=current_run_dir)
                    self._telemetry = telemetry
                except Exception as e:
                    logger.warning(f"⚠️  Failed to re-initialize telemetry: {e}")
                    telemetry = None
        
        # PRIORITY 0: Validate current_run_dir against telemetry's run_dir (authoritative source)
        # If telemetry exists and has run_dir, use it as the authoritative current run directory
        # This prevents stale self.output_dir from causing wrong cohort_dir construction
        if telemetry and hasattr(telemetry, 'run_dir') and telemetry.run_dir:
            telemetry_run_dir = Path(telemetry.run_dir).resolve()
            current_run_resolved = Path(current_run_dir).resolve()
            if telemetry_run_dir != current_run_resolved:
                logger.warning(
                    f"current_run_dir ({current_run_dir}) does not match telemetry.run_dir ({telemetry.run_dir}). "
                    f"Using telemetry.run_dir as authoritative current run directory."
                )
                # Use telemetry's run_dir as authoritative source
                current_run_dir = str(telemetry_run_dir)
                logger.debug(f"Updated current_run_dir to match telemetry: {current_run_dir}")
        
        # Determine target-first directory (for TARGET_RANKING and FEATURE_SELECTION stages)
        # CRITICAL: Do this BEFORE finalize_run() so we can pass the correct cohort_dir
        target_cohort_dir = None
        if stage_enum in (Stage.TARGET_RANKING, Stage.FEATURE_SELECTION):
            try:
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_reproducibility_dir, ensure_target_structure
                )
                
                # CRITICAL FIX: Check symbol FIRST - if symbol is set, ALWAYS use SYMBOL_SPECIFIC
                # This must happen before any view determination to prevent symbol-specific data
                # from being written to CROSS_SECTIONAL directories
                if symbol:
                    view_for_target = View.SYMBOL_SPECIFIC.value
                    logger.debug(f"Symbol-specific data detected (symbol={symbol}), forcing SYMBOL_SPECIFIC view")
                elif cohort_id and cohort_id.startswith("sy_"):
                    view_for_target = View.SYMBOL_SPECIFIC.value
                    logger.debug(f"Detected symbol-specific cohort from cohort_id prefix: {cohort_id}")
                else:
                    # Only determine view from parameters if symbol is NOT set
                    view_for_target = None
                    if stage_enum == Stage.TARGET_RANKING:
                        # For TARGET_RANKING, view comes from view or additional_data
                        # Normalize view to enum for comparison
                        if view:
                            try:
                                view_enum = View.from_string(view) if isinstance(view, str) else view
                                view_for_target = view_enum.value  # Use enum value
                            except ValueError:
                                # Handle LOSO (not a View enum value)
                                if isinstance(view, str) and view.upper() == "LOSO":
                                    view_for_target = View.SYMBOL_SPECIFIC.value
                                else:
                                    view_enum = View.from_string(view) if isinstance(view, str) else view
                                    view_for_target = view_enum.value
                        elif additional_data and 'view' in additional_data:
                            # Normalize view from additional_data
                            view_from_data = additional_data['view']
                            view_enum = View.from_string(view_from_data) if isinstance(view_from_data, str) else view_from_data
                            view_for_target = view_enum.value if isinstance(view_enum, View) else str(view_enum).upper()
                    elif stage_enum == Stage.FEATURE_SELECTION:
                        # For FEATURE_SELECTION, map view to view
                        if view:
                            # Normalize view to enum
                            view_enum = View.from_string(view) if isinstance(view, str) else view
                            if view_enum == View.CROSS_SECTIONAL:
                                view_for_target = View.CROSS_SECTIONAL.value
                            elif view_enum == View.SYMBOL_SPECIFIC:
                                view_for_target = View.SYMBOL_SPECIFIC.value
                        elif additional_data and 'view' in additional_data:
                            view_for_target = additional_data['view'].upper()
                    
                    # Default to CROSS_SECTIONAL only if still None after all stage-specific checks
                    if not view_for_target:
                        view_for_target = View.CROSS_SECTIONAL.value  # Default
                
                # CRITICAL VALIDATION: Ensure view_for_target is never None before path construction
                if not view_for_target:
                    logger.warning(f"view_for_target is None for {stage}:{target}, defaulting to CROSS_SECTIONAL")
                    view_for_target = View.CROSS_SECTIONAL.value
                
                # Additional validation: ensure it's a valid View enum value
                if view_for_target not in (View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value):
                    logger.warning(f"Invalid view_for_target={view_for_target}, normalizing to CROSS_SECTIONAL")
                    view_for_target = View.CROSS_SECTIONAL.value
                
                # CRITICAL: Use current_run_dir computed above (telemetry initialization already handled this)
                # This ensures we always use the CURRENT run, not a stale one
                # current_run_dir was computed from self.output_dir and normalized above
                base_output_dir = current_run_dir if 'current_run_dir' in locals() else None
                if base_output_dir is None:
                    # Fallback: compute again (shouldn't happen, but defensive)
                    from TRAINING.orchestration.utils.target_first_paths import run_root, normalize_run_root
                    base_output_dir = normalize_run_root(run_root(self.output_dir))
                    if base_output_dir is None:
                        raise ValueError(
                            f"Cannot determine current run directory for target_cohort_dir. "
                            f"self.output_dir={self.output_dir}, _repro_base_dir={self._repro_base_dir}"
                        )
                logger.debug(f"Using current run directory for target_cohort_dir: {base_output_dir}")
                
                # Ensure target structure exists
                ensure_target_structure(base_output_dir, target)
                
                # Build target-first reproducibility path with stage scoping:
                # For CROSS_SECTIONAL: targets/<target>/reproducibility/stage=<stage>/CROSS_SECTIONAL/cohort=<cohort_id>/
                # For SYMBOL_SPECIFIC: targets/<target>/reproducibility/stage=<stage>/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<cohort_id>/
                target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage_normalized)
                # CRITICAL: view_for_target is already validated above to be a valid View enum value string
                # No need to normalize again - use it directly for path construction
                view_for_target_str = view_for_target
                
                # Extract attempt_id from additional_data (defaults to 0)
                attempt_id = (additional_data.get('attempt_id') if additional_data else None) or 0
                
                # Extract universe_sig for CROSS_SECTIONAL batch_ level
                # Check additional_data first, then cohort_metadata
                universe_sig = None
                if additional_data:
                    # Use SST accessor for universe_sig
                    from TRAINING.orchestration.utils.reproducibility.utils import extract_universe_sig
                    universe_sig = extract_universe_sig(additional_data)
                if not universe_sig and cohort_metadata:
                    universe_sig = cohort_metadata.get('universe_sig')
                    if not universe_sig and 'cs_config' in cohort_metadata:
                        universe_sig = cohort_metadata['cs_config'].get('universe_sig')
                
                # Use single path builder (SST) - always includes attempt_{attempt_id}/ including attempt_0
                from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                target_cohort_dir = build_target_cohort_dir(
                    base_output_dir=base_output_dir,
                    target=target,
                    stage=stage_normalized,
                    view=view_for_target_str,
                    cohort_id=cohort_id,
                    symbol=symbol,
                    attempt_id=attempt_id,  # Always include, even if 0
                    universe_sig=universe_sig  # Required for CROSS_SECTIONAL batch_ level
                )
                
                # PRIORITY 1: Validate target_cohort_dir is within current_run_dir (authoritative current run)
                # If outside, rebuild using same structured inputs but with validated current_run_dir
                # This prevents stale base_output_dir from causing wrong cohort_dir construction
                current_run_resolved = Path(current_run_dir).resolve()
                target_cohort_resolved = Path(target_cohort_dir).resolve()
                try:
                    is_within = target_cohort_resolved.is_relative_to(current_run_resolved)
                except AttributeError:
                    # Python < 3.9 compatibility
                    try:
                        target_cohort_resolved.relative_to(current_run_resolved)
                        is_within = True
                    except ValueError:
                        is_within = False
                
                if not is_within:
                    old_cohort_dir = target_cohort_dir
                    old_run_root = base_output_dir
                    logger.warning(
                        f"target_cohort_dir ({target_cohort_dir}) is outside current run directory ({current_run_dir}). "
                        f"Rebuilding with current run's base_output_dir. "
                        f"Old run root: {old_run_root}, New run root: {current_run_dir}"
                    )
                    # Rebuild using same structured inputs but with validated current_run_dir
                    target_cohort_dir = build_target_cohort_dir(
                        base_output_dir=current_run_dir,  # Use validated current run
                        target=target,  # Same inputs
                        stage=stage_normalized,
                        view=view_for_target_str,
                        cohort_id=cohort_id,
                        symbol=symbol,
                        attempt_id=attempt_id,
                        universe_sig=universe_sig
                    )
                    logger.info(
                        f"Rebuilt target_cohort_dir: {old_cohort_dir} -> {target_cohort_dir} "
                        f"(old run: {old_run_root}, new run: {current_run_dir})"
                    )
                
                target_cohort_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created target-first cohort directory: {target_cohort_dir}")
            except Exception as e:
                # Don't fail if target-first structure creation fails - old structure is primary
                # But log at INFO level so we can see if there are issues
                logger.info(f"⚠️ Failed to create target-first structure for {target}/{view_for_target}/cohort={cohort_id} (non-critical): {e}")
                import traceback
                logger.debug(f"Target-first structure creation traceback: {traceback.format_exc()}")
                target_cohort_dir = None
        
        # CRITICAL: Always use target_cohort_dir (canonical path) - don't fall back to legacy
        # If target_cohort_dir is None, that's an error - legacy paths create wrong structure
        cohort_dir = target_cohort_dir
        skip_diff_emission = False  # Initialize flag for skip logic
        if cohort_dir is None:
            logger.warning(
                f"⚠️  Cannot determine cohort_dir for {target}/{stage}/{cohort_id}: "
                f"target-first structure creation failed. Legacy paths are not allowed."
            )
        else:
            # CRITICAL: Validate target_cohort_dir is within current run's directory
            # Use telemetry's run_dir as source of truth (if available)
            # If still outside after self-healing, skip diff emission to prevent cross-run contamination
            if telemetry and hasattr(telemetry, 'run_dir') and telemetry.run_dir:
                current_run_dir = Path(telemetry.run_dir).resolve()
                target_cohort_resolved = Path(cohort_dir).resolve()
                try:
                    is_within = target_cohort_resolved.is_relative_to(current_run_dir)
                except AttributeError:
                    # Python < 3.9 compatibility
                    try:
                        target_cohort_resolved.relative_to(current_run_dir)
                        is_within = True
                    except ValueError:
                        is_within = False
                
                if not is_within:
                    logger.error(
                        f"CRITICAL: target_cohort_dir ({cohort_dir}) is still outside current run directory "
                        f"({telemetry.run_dir}) after self-healing attempt. "
                        f"Skipping diff emission to prevent cross-run contamination. "
                        f"Computed base_output_dir was: {base_output_dir if 'base_output_dir' in locals() else 'unknown'}"
                    )
                    skip_diff_emission = True
                    # Write deterministic marker file for debugging
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                        globals_dir = get_globals_dir(current_run_dir)
                        marker_file = globals_dir / "PATH_REPAIR_FAILED.json"
                        from TRAINING.common.utils.file_utils import write_atomic_json
                        write_atomic_json(marker_file, {
                            "cohort_dir": str(cohort_dir),
                            "current_run_dir": str(telemetry.run_dir),
                            "target": target,
                            "stage": stage_normalized,
                            "cohort_id": cohort_id
                        })
                    except Exception:
                        pass  # Marker file is best-effort
                else:
                    logger.debug(f"Validated target_cohort_dir is within current run directory: {cohort_dir}")
        
        # CRITICAL: Call finalize_run() BEFORE adding diff_telemetry to full_metadata
        # This ensures snapshot/diff computation uses the exact same resolved_metadata that will be written
        # Pass full_metadata (without diff_telemetry) as resolved_metadata for SST consistency
        diff_telemetry_data = None
        diff_finalized = False  # Regression guard: prevent duplicate finalize_run() calls
        if telemetry:
            try:
                # Extract experiment_id if available
                experiment_id = None
                if additional_data and 'experiment_id' in additional_data:
                    experiment_id = additional_data['experiment_id']
                elif run_data.get('additional_data') and 'experiment_id' in run_data.get('additional_data', {}):
                    experiment_id = run_data['additional_data']['experiment_id']
                
                # Add experiment_id to additional_data if not present
                if experiment_id and additional_data and 'experiment_id' not in additional_data:
                    additional_data = additional_data.copy()
                    additional_data['experiment_id'] = experiment_id
                
                # CRITICAL: Pass full_metadata (without diff_telemetry) as resolved_metadata for SST consistency
                # This ensures snapshot/diff computation uses the exact same data that will be written to metadata.json
                # full_metadata is already built above (lines 1077-1292), we just haven't added diff_telemetry yet
                if cohort_dir is None:
                    logger.warning(
                        f"⚠️  Cannot call finalize_run() for {target}/{stage_normalized}: cohort_dir is None. "
                        f"Snapshot and diff files will not be created."
                    )
                elif skip_diff_emission:
                    logger.warning(
                        f"⚠️  Skipping finalize_run() for {target}/{stage_normalized}: cohort_dir is outside current run. "
                        f"Diff files will not be created to prevent cross-run contamination."
                    )
                    diff_telemetry_data = None
                else:
                    # Regression guard: prevent duplicate finalize_run() calls
                    if diff_finalized:
                        logger.error(
                            f"CRITICAL: finalize_run() already called for {target}/{stage_normalized} - skipping duplicate. "
                            f"This indicates a code path bug (duplicate call site)."
                        )
                        diff_telemetry_data = None
                    else:
                        diff_telemetry_data = telemetry.finalize_run(
                            stage=stage_normalized,
                            run_data=run_data,
                            cohort_dir=cohort_dir,
                            cohort_metadata=cohort_metadata,
                            additional_data=additional_data,
                            resolved_metadata=full_metadata,  # CRITICAL: Pass in-memory metadata for SST consistency
                            run_identity=run_identity,  # NEW: Pass RunIdentity for authoritative signatures
                            prediction_fingerprint=prediction_fingerprint,  # NEW: Pass prediction fingerprint
                        )
                        # Set guard flag after successful call
                        if diff_telemetry_data is not None:
                            diff_finalized = True
                    
                    # FIX: Validate that required files were created after finalize_run()
                    # CRITICAL: Use same SST path resolver as save_diff() to eliminate path mismatches
                    from TRAINING.orchestration.utils.target_first_paths import (
                        build_target_cohort_dir, run_root
                    )
                    
                    # CRITICAL: Use resolved_cohort_dir returned from finalize_run() if available
                    # Otherwise, resolve from metadata (but handle missing gracefully)
                    resolved_cohort_dir = None
                    snapshot_path = None
                    diff_prev_path = None
                    
                    if diff_telemetry_data and 'resolved_cohort_dir' in diff_telemetry_data:
                        # Use resolved path from finalize_run() (SST source of truth)
                        resolved_cohort_dir = Path(diff_telemetry_data['resolved_cohort_dir'])
                        snapshot_path = resolved_cohort_dir / "snapshot.json"
                        diff_prev_path = resolved_cohort_dir / "diff_prev.json"
                    else:
                        # Fallback: resolve from metadata (only if all required fields present)
                        target = cohort_metadata.get('target_name') or target
                        stage = stage_normalized
                        view = view if view else cohort_metadata.get('view')
                        symbol = symbol if symbol else cohort_metadata.get('symbol')
                        cohort_id = cohort_id  # Already extracted
                        attempt_id = attempt_id if 'attempt_id' in locals() else 0
                        universe_sig = cohort_metadata.get('universe_sig')
                        
                        # CRITICAL: Check if required identifiers are present before resolving
                        missing_identifiers = []
                        if not target:
                            missing_identifiers.append('target')
                        if not stage:
                            missing_identifiers.append('stage')
                        if not view:
                            missing_identifiers.append('view')
                        if not cohort_id:
                            missing_identifiers.append('cohort_id')
                        
                        if missing_identifiers:
                            # Cannot validate - log single warning with known values
                            logger.warning(
                                f"⚠️  Cannot validate required files: missing identifiers {missing_identifiers}. "
                                f"Known values: target={target}, stage={stage}, view={view}, cohort_id={cohort_id}, "
                                f"symbol={symbol}, attempt_id={attempt_id}, universe_sig={universe_sig}. "
                                f"This may indicate incomplete metadata."
                            )
                            # Skip validation (don't crash)
                        else:
                            # Resolve canonical paths using SST helper (same as save_diff uses)
                            # CRITICAL: Use run_root() helper, not .parent (SST)
                            base_output_dir = run_root(cohort_dir) if cohort_dir else self.output_dir
                            if base_output_dir is None:
                                logger.warning(f"Cannot resolve run root from cohort_dir: {cohort_dir}")
                            else:
                                resolved_cohort_dir = build_target_cohort_dir(
                                    base_output_dir=base_output_dir,
                                    target=target,
                                    stage=stage,
                                    view=view,
                                    cohort_id=cohort_id,
                                    symbol=symbol,
                                    attempt_id=attempt_id,
                                    universe_sig=universe_sig
                                )
                                # Derive paths from resolved_cohort_dir (not recomputing)
                                snapshot_path = resolved_cohort_dir / "snapshot.json"
                                diff_prev_path = resolved_cohort_dir / "diff_prev.json"
                    
                    # Validate against resolved paths (SST source of truth)
                    if resolved_cohort_dir and snapshot_path and diff_prev_path:
                        missing_files = []
                        if not snapshot_path.exists():
                            missing_files.append('snapshot.json')
                        if not diff_prev_path.exists():
                            missing_files.append('diff_prev.json')
                        
                        if missing_files:
                            logger.warning(
                                f"⚠️  finalize_run() completed but required files are missing (SST resolved paths): "
                                f"{missing_files}. Resolved paths: snapshot={snapshot_path}, diff_prev={diff_prev_path}. "
                                f"This may indicate silent failures in snapshot/diff creation."
                            )
                        else:
                            logger.debug(f"✅ finalize_run() created required files (SST validated): snapshot={snapshot_path}, diff_prev={diff_prev_path}")
                
                # Store diff telemetry data for integration into metadata/metrics
                if diff_telemetry_data:
                    if additional_data is None:
                        additional_data = {}
                    additional_data['diff_telemetry'] = diff_telemetry_data
            except Exception as e:
                # FIX: Log at error level with more context - this is critical for reproducibility
                logger.error(
                    f"❌ Diff telemetry finalize_run() failed for {target}/{stage_normalized}: {e}. "
                    f"This will prevent snapshot.json and diff files from being created, breaking reproducibility tracking."
                )
                import traceback
                logger.debug(f"Diff telemetry traceback: {traceback.format_exc()}")
        
        # NEW: Add diff telemetry to metadata (full audit trail)
        # CRITICAL: diff_telemetry is optional - older runs may not have it
        # CRITICAL: Only process if diff_telemetry_data was successfully computed (not None)
        if additional_data and 'diff_telemetry' in additional_data and diff_telemetry_data is not None:
            diff_telemetry = additional_data['diff_telemetry']
            snapshot = diff_telemetry.get('snapshot', {})
            diff = diff_telemetry.get('diff', {})
            
            # Extract comparison group key
            comparison_group = snapshot.get('comparison_group')
            stage = snapshot.get('stage', 'TRAINING')  # Get stage from snapshot
            comparison_group_key = None
            if isinstance(comparison_group, dict):
                # Construct key from dict fields (matching ComparisonGroup.to_key() logic)
                comparison_group_key = _construct_comparison_group_key_from_dict(comparison_group, stage)
            elif hasattr(comparison_group, 'to_key'):
                # CRITICAL: to_key() now requires stage parameter
                comparison_group_key = comparison_group.to_key(stage, strict=False)
            
            # Extract diff telemetry metadata (full detail for audit trail)
            # CRITICAL: Always write valid diff_telemetry object, even if not comparable
            # Schema mismatches should still produce valid structure with comparability.reason set
            diff_telemetry_blob = {
                'fingerprint_schema_version': snapshot.get('fingerprint_schema_version'),
                'comparison_group_key': comparison_group_key,
                'comparison_group': comparison_group if isinstance(comparison_group, dict) else (
                    comparison_group.to_dict() if hasattr(comparison_group, 'to_dict') else None
                ),
                'fingerprints': {
                    'config_fingerprint': snapshot.get('config_fingerprint'),
                    'data_fingerprint': snapshot.get('data_fingerprint'),
                    'feature_fingerprint': snapshot.get('feature_fingerprint'),
                    'target_fingerprint': snapshot.get('target_fingerprint')
                },
                'fingerprint_sources': snapshot.get('fingerprint_sources', {}),
                'comparability': {
                    'comparable': diff.get('comparable', False),
                    'comparability_reason': diff.get('comparability_reason') or None,  # Explicit None for clarity
                    'prev_run_id': diff.get('prev_run_id')  # Can be None for first run
                },
                'excluded_factors': {
                    'changed': bool(diff.get('excluded_factors_changed', {})),
                    'summary': diff.get('summary', {}).get('excluded_factors_summary'),
                    'changes': diff.get('excluded_factors_changed', {})  # Full payload - empty dict if no changes
                }
            }
            
            # Compute digest of diff_telemetry blob for integrity verification
            # Algorithm: SHA256 hash of canonical JSON (sorted keys, strict JSON-primitive-only)
            # Digest is full SHA256 hash (64 hex characters, 256 bits of entropy)
            # 
            # CRITICAL: diff_telemetry must contain only JSON-primitive types (str/int/float/bool/null/lists/dicts).
            # If non-primitive types are present, this indicates a normalization bug upstream.
            # We fail fast (raise) rather than silently coerce to strings, to catch bugs early.
            try:
                # Strict serialization - will raise TypeError if non-primitive types present
                # This ensures we catch normalization bugs immediately rather than hiding them
                canonical_json = json.dumps(diff_telemetry_blob, sort_keys=True)
                
                # Use full SHA256 hash (64 hex characters, 256 bits) for maximum collision resistance
                digest = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
                diff_telemetry_blob['diff_telemetry_digest'] = digest
            except (TypeError, ValueError) as e:
                # Non-JSON-primitive types detected - this is a normalization bug
                # CRITICAL: Don't raise - still write metadata.json without diff_telemetry rather than failing completely
                logger.error(f"diff_telemetry contains non-JSON-primitive types (normalization bug): {e}")
                logger.error("Cannot compute diff_telemetry_digest - diff_telemetry must contain only JSON-primitive types")
                logger.error("Writing metadata.json without diff_telemetry to prevent complete failure")
                # Clear diff_telemetry_blob so it's not added to metadata
                diff_telemetry_blob = None
            except Exception as e:
                # CRITICAL: Don't raise - still write metadata.json without diff_telemetry rather than failing completely
                logger.error(f"Unexpected error computing diff_telemetry_digest: {e}")
                logger.error("Writing metadata.json without diff_telemetry to prevent complete failure")
                # Clear diff_telemetry_blob so it's not added to metadata
                diff_telemetry_blob = None
            
            # CRITICAL: Only add diff_telemetry if blob was successfully created (not None)
            # If digest computation failed, diff_telemetry_blob is set to None and we skip it
            # This ensures metadata.json is still written even if diff telemetry has issues
            if diff_telemetry_blob is not None and 'diff_telemetry_digest' in diff_telemetry_blob:
                full_metadata['diff_telemetry'] = diff_telemetry_blob
            else:
                logger.warning("Skipping diff_telemetry in metadata.json due to computation failure")
        
        # NOTE: target_cohort_dir was already created above (before finalize_run() call)
        # This section is kept for backward compatibility and to ensure it exists for metadata saving
        
        # Save metadata.json to target-first structure only
        if target_cohort_dir:
            try:
                target_metadata_file = target_cohort_dir / "metadata.json"
                _write_atomic_json_with_lock(target_metadata_file, full_metadata)
                # Log at INFO level so it's visible
                main_logger = _get_main_logger()
                try:
                    # Try to get a relative path for readability using SST helper
                    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                    run_base = get_run_root(target_cohort_dir)
                    rel_path = target_cohort_dir.relative_to(run_base) if run_base.exists() else target_cohort_dir
                    log_msg = f"📁 Reproducibility: Writing cohort data to {rel_path}"
                except (ValueError, AttributeError):
                    log_msg = f"📁 Reproducibility: Writing cohort data to {target_cohort_dir}"
                
                if main_logger != logger:
                    main_logger.info(log_msg)
                    main_logger.info(f"✅ Reproducibility: Saved metadata.json to {target_metadata_file.name} in {target_metadata_file.parent.name}/")
                else:
                    logger.info(log_msg)
                    logger.info(f"✅ Reproducibility: Saved metadata.json to {target_metadata_file.name} in {target_metadata_file.parent.name}/")
            except (IOError, OSError) as e:
                logger.warning(f"Failed to save metadata.json to {target_metadata_file}: {e}, error_type=IO_ERROR")
                self._increment_error_counter("write_failures", "IO_ERROR")
                raise  # Re-raise to prevent silent failure
        else:
            logger.warning(f"Target cohort directory not available for {target}/{stage_normalized}, cannot save metadata.json")
        
        # Write metrics sidecar files (if enabled)
        if self.metrics:
            # Normalize view - ensure it's one of the expected values
            # FIX: Don't shadow the view parameter passed to the function
            metrics_view = view.upper() if view else None
            # Normalize metrics_view to enum for validation
            if metrics_view:
                try:
                    metrics_view_enum = View.from_string(metrics_view) if isinstance(metrics_view, str) else metrics_view
                    if metrics_view_enum not in (View.CROSS_SECTIONAL, View.SYMBOL_SPECIFIC):
                        # Invalid view, will be handled below
                        pass
                except ValueError:
                    # Invalid view, will be handled below
                    pass
            if metrics_view and metrics_view not in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                metrics_view = None  # Invalid view, will use "UNKNOWN" in write call
            
            # Determine target for metrics (for TARGET_RANKING and FEATURE_SELECTION stages)
            metrics_target = target if stage_enum in (Stage.TARGET_RANKING, Stage.FEATURE_SELECTION) else None
            
            # SST: Normalize metrics_view to string (handle enum inputs) before using in baseline_key or write_cohort_metrics
            metrics_view_str = None
            if metrics_view:
                if isinstance(metrics_view, View):
                    metrics_view_str = metrics_view.value
                elif hasattr(metrics_view, 'value'):
                    metrics_view_str = metrics_view.value
                else:
                    metrics_view_str = str(metrics_view)
            else:
                metrics_view_str = "UNKNOWN"
            
            # Generate baseline key for drift comparison: (stage, view, target[, symbol])
            # For FEATURE_SELECTION, use view as view (CROSS_SECTIONAL or INDIVIDUAL)
            baseline_key = None
            if metrics_target:
                # For TARGET_RANKING, view comes from metrics_view (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
                # For FEATURE_SELECTION, view is CROSS_SECTIONAL or INDIVIDUAL (maps to view)
                if stage_enum == Stage.TARGET_RANKING and metrics_view_str:
                    baseline_key = f"{stage_normalized}:{metrics_view_str}:{metrics_target}"
                    if symbol and metrics_view_str == View.SYMBOL_SPECIFIC.value:
                        baseline_key += f":{symbol}"
                elif stage_enum == Stage.FEATURE_SELECTION and metrics_view_str:
                    # Map view to view for FEATURE_SELECTION
                    fs_view = metrics_view_str if metrics_view_str in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value] else View.CROSS_SECTIONAL.value
                    baseline_key = f"{stage_normalized}:{fs_view}:{metrics_target}"
                    if symbol and metrics_view_str == View.SYMBOL_SPECIFIC.value:
                        baseline_key += f":{symbol}"
            
            logger.debug(f"📊 Writing metrics for stage={stage_normalized}, target={metrics_target}, view={metrics_view_str}, target_cohort_dir={target_cohort_dir}")
            
            # Write metrics sidecar files in cohort directory
            # Note: metrics will create target_cohort_dir if it doesn't exist, or fall back to target level
            metrics_written = False
            try:
                # Extract diff telemetry data if available
                diff_telemetry = None
                if additional_data and 'diff_telemetry' in additional_data:
                    diff_telemetry = additional_data['diff_telemetry']
                
                # Write metrics to target-first structure only
                if target_cohort_dir:
                    # FIX: Extract nested metrics if present, otherwise use run_data (but exclude non-metric keys)
                    # This prevents duplication where metrics appear both nested and at root level
                    metrics_to_write = run_data.get('metrics')
                    if not metrics_to_write:
                        # Fallback: use run_data but exclude non-metric keys to prevent duplication
                        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                        from TRAINING.common.utils.determinism_ordering import sorted_items
                        metrics_to_write = {k: v for k, v in sorted_items(run_data)
                                            if k not in ['timestamp', 'cohort_metadata', 'additional_data', 'stage', 'target']}
                    
                    self.metrics.write_cohort_metrics(
                        cohort_dir=target_cohort_dir,
                        stage=stage_normalized,
                        view=metrics_view_str,  # Now guaranteed to be string
                        target=metrics_target,
                        symbol=symbol,
                        run_id=run_id_clean,
                        metrics=metrics_to_write,
                        baseline_key=baseline_key,
                        diff_telemetry=diff_telemetry
                    )
                    metrics_written = True
                else:
                    logger.warning(f"Target cohort directory not available for metrics write: {metrics_target}/{stage_normalized}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to write metrics metadata to cohort directory: {e}")
                import traceback
                logger.debug(f"Metrics write traceback: {traceback.format_exc()}")
            
            # Safety fallback: If cohort-level write failed and we have target/view info, try target-level write
            if not metrics_written and metrics_target and metrics_view:
                try:
                    fallback_dir = self.metrics._get_fallback_metrics_dir(stage_normalized, metrics_view, metrics_target, symbol)
                    if fallback_dir:
                        logger.info(f"📁 Attempting metrics fallback write to: {fallback_dir}")
                        self.metrics._write_metrics(fallback_dir, run_id_clean, run_data, stage=stage_normalized, reproducibility_mode="COHORT_AWARE")
                        if baseline_key:
                            self.metrics._write_drift(
                                fallback_dir, stage_normalized, metrics_view, metrics_target, symbol, run_id_clean, run_data, baseline_key
                            )
                        logger.info(f"✅ Metrics written to fallback location: {fallback_dir}")
                except Exception as e2:
                    logger.warning(f"⚠️  Metrics fallback write also failed: {e2}")
            
            # Aggregate metrics facts table (append-only, after all cohorts saved)
            # This is called per-cohort, but we'll aggregate at the end of the run
            # For now, we'll aggregate on-demand or at end of stage
            
            # NOTE: PHASE 3 diff telemetry block removed - duplicate of main telemetry block (lines 2305-2446)
            # The main telemetry block already handles finalize_run() with proper validation and skip_diff_emission flag
        
        # PHASE 2: Unified schema - build metrics_data for _update_index (always needed)
        # Metrics writer writes metrics.json/parquet, but we still need metrics_data for index
        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
        from TRAINING.common.utils.determinism_ordering import sorted_items
        metrics_data = {
            "run_id": run_id_clean,
            "timestamp": datetime.now().isoformat(),
            "reproducibility_mode": "COHORT_AWARE",  # Track which mode was used
            "stage": stage_normalized,  # Ensure consistent uppercase naming
            **{k: v for k, v in sorted_items(run_data)
               if k not in ['timestamp', 'cohort_metadata', 'additional_data']}
        }
        
        # PHASE 2: Only write as fallback if metrics failed and we don't have metrics.json yet
        # Use target_cohort_dir instead of legacy cohort_dir
        if target_cohort_dir:
            metrics_file = target_cohort_dir / "metrics.json"
        else:
            metrics_file = None
        
        # Always write/update metrics.json on every run (not just when missing)
        # CRITICAL: Include run_id for attribution, decide merge vs overwrite semantics
        if metrics_file and not metrics_written:
            # Read existing metrics if present (for merge semantics)
            existing_metrics = {}
            if metrics_file.exists():
                try:
                    # json is already imported at module level (line 27)
                    with open(metrics_file, 'r') as f:
                        existing_metrics = json.load(f)
                    logger.debug(f"Read existing metrics.json for merge: {len(existing_metrics)} keys")
                except Exception as e:
                    logger.debug(f"Could not read existing metrics.json for merge: {e}, will overwrite")
            
            # Merge: update deterministic keys, preserve historical fields
            # Overwrite: use latest snapshot only (current behavior)
            # For now: overwrite (latest snapshot only) - document explicitly
            # TODO: Consider merge semantics if historical fields need preservation
            metrics_data['run_id'] = run_id_clean  # Ensure run_id is always included for attribution
            metrics_data['timestamp'] = datetime.now().isoformat()  # Update timestamp
            
            # Write metrics.json using unified canonical schema (atomically)
            # Write directly to target-first structure (metrics_file is already target_cohort_dir / "metrics.json")
            try:
                _write_atomic_json_with_lock(metrics_file, metrics_data)
                # Also write metrics.parquet for consistency
                try:
                    # SST: Use safe_dataframe_from_dict to handle enums
                    from TRAINING.common.utils.file_utils import safe_dataframe_from_dict
                    df_metrics = safe_dataframe_from_dict(metrics_data)
                    metrics_parquet = target_cohort_dir / "metrics.parquet"
                    df_metrics.to_parquet(metrics_parquet, index=False, engine='pyarrow', compression='snappy')
                    logger.debug(f"✅ Saved metrics.json/parquet to target-first structure")
                except Exception as e_parquet:
                    logger.debug(f"Failed to write metrics.parquet fallback: {e_parquet}")
                # Log at INFO level so it's visible
                main_logger = _get_main_logger()
                if main_logger != logger:
                    main_logger.info(f"✅ Reproducibility: Saved metrics.json (fallback) to {metrics_file.name} in {metrics_file.parent.name}/")
                else:
                    logger.info(f"✅ Reproducibility: Saved metrics.json (fallback) to {metrics_file.name} in {metrics_file.parent.name}/")
            except (IOError, OSError) as e:
                logger.warning(f"Failed to save metrics.json (fallback) to {metrics_file}: {e}, error_type=IO_ERROR")
                self._increment_error_counter("write_failures", "IO_ERROR")
                # Don't raise - metrics might have written it, or we'll try again
        elif metrics_written and target_cohort_dir:
            # Metrics were already written by MetricsWriter to target-first structure
            # No need to duplicate - MetricsWriter already writes to target_cohort_dir
            logger.debug(f"✅ Metrics already written to target-first structure by MetricsWriter")
        
        # Update index.parquet (use target_cohort_dir if available, otherwise None)
        try:
            self._update_index(
                stage, target, view, symbol, model_family,
                cohort_id, run_id_clean, full_metadata, metrics_data, target_cohort_dir  # Use target-first structure
            )
        except Exception as e:
            error_type = "IO_ERROR" if isinstance(e, (IOError, OSError)) else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError)) else "UNKNOWN_ERROR"
            logger.warning(f"Failed to update index.parquet: {e}, error_type={error_type}")
            self._increment_error_counter("index_update_failures", error_type)
            # Don't re-raise - index update failure shouldn't break the run
        
        # Post-run decision hook: Evaluate and persist decisions
        try:
            from TRAINING.decisioning.decision_engine import DecisionEngine
            from TRAINING.ranking.utils.resolved_config import get_cfg
            
            # Read index from globals/ first, then fall back to legacy REPRODUCIBILITY/
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(self._repro_base_dir)
            index_file = globals_dir / "index.parquet"
            if not index_file.exists():
                # Fallback to legacy
                repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
                index_file = repro_dir / "index.parquet"
            if index_file.exists():
                # Check if Bayesian policy is enabled
                use_bayesian = get_cfg("training.decisions.use_bayesian", default=False, config_name="training_config")
                
                engine = DecisionEngine(
                    index_file,
                    apply_mode=False,  # Assist mode by default
                    use_bayesian=use_bayesian,
                    base_dir=self.output_dir.parent
                )
                # Get segment_id from index if available
                segment_id_for_decision = None
                try:
                    if index_file.exists():
                        df_temp = pd.read_parquet(index_file)
                        mask = df_temp['cohort_id'] == cohort_id
                        if mask.any():
                            segment_id_for_decision = df_temp[mask]['segment_id'].iloc[-1] if 'segment_id' in df_temp.columns else None
                except Exception:
                    pass
                decision_result = engine.evaluate(cohort_id, run_id_clean, segment_id=segment_id_for_decision)
                engine.persist(decision_result, self.output_dir.parent)
                
                # Extract metrics from run_data for decision engine (metrics may be nested or top-level)
                current_metrics = run_data.get('metrics', {})
                if not current_metrics:
                    # Fallback: reconstruct from top-level keys if metrics were spread
                    known_metric_keys = {
                        'schema', 'scope', 'primary_metric', 'coverage', 'features', 
                        'y_stats', 'label_stats', 'models', 'score', 'fold_timestamps',
                        'leakage', 'mismatch_telemetry', 'metrics_schema_version', 
                        'scoring_schema_version', 'n_effective', 'metric_name'
                    }
                    # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                    from TRAINING.common.utils.determinism_ordering import sorted_items
                    current_metrics = {k: v for k, v in sorted_items(run_data) if k in known_metric_keys}
                
                # Update Bayesian state if enabled
                if use_bayesian and engine.bayesian_policy:
                    try:
                        # Get applied patch template from decision result
                        applied_patch_template = None
                        if decision_result.policy_results and 'bayesian_patch' in decision_result.policy_results:
                            bayesian_result = decision_result.policy_results['bayesian_patch']
                            applied_patch_template = bayesian_result.get('recommended_patch')
                        
                        # Update Bayesian state with observed reward
                        engine.update_bayesian_state(
                            decision_result=decision_result,
                            current_run_metrics=current_metrics,
                            applied_patch_template=applied_patch_template
                        )
                    except Exception as e:
                        logger.debug(f"Bayesian state update failed (non-critical): {e}")
                
                # Store decision fields in metrics for index update
                if current_metrics:
                    current_metrics['decision_level'] = decision_result.decision_level
                    current_metrics['decision_action_mask'] = decision_result.decision_action_mask
                    current_metrics['decision_reason_codes'] = decision_result.decision_reason_codes
                    # Update run_data with modified metrics
                    run_data['metrics'] = current_metrics
                if decision_result.decision_level > 0:
                    logger.info(f"📊 Decision: level={decision_result.decision_level}, actions={decision_result.decision_action_mask}, reasons={decision_result.decision_reason_codes}")
                    # Log Bayesian metadata if available
                    if decision_result.policy_results and 'bayesian_metadata' in decision_result.policy_results:
                        bayes_meta = decision_result.policy_results['bayesian_metadata']
                        logger.info(f"🎲 Bayesian: confidence={bayes_meta.get('confidence', 0):.3f}, expected_gain={bayes_meta.get('expected_gain', 0):.4f}")
        except Exception as e:
            logger.debug(f"Decision evaluation failed (non-critical): {e}")
            # Don't re-raise - decision evaluation is optional
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash. Delegates to SST module."""
        from TRAINING.common.utils.git_utils import get_git_commit
        return get_git_commit(short=True)
    
    def _increment_error_counter(self, counter_name: str, error_type: str = "UNKNOWN") -> None:
        """
        Increment error counter in stats.json.
        
        Args:
            counter_name: Name of the counter (e.g., "write_failures", "index_update_failures")
            error_type: Type of error (e.g., "IO_ERROR", "SERIALIZATION_ERROR")
        """
        try:
            # Load existing stats
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                except (json.JSONDecodeError, IOError):
                    stats = {}
            else:
                stats = {}
            
            # Initialize counters if needed
            if "errors" not in stats:
                stats["errors"] = {}
            if counter_name not in stats["errors"]:
                stats["errors"][counter_name] = {}
            if error_type not in stats["errors"][counter_name]:
                stats["errors"][counter_name][error_type] = 0
            
            # Increment
            stats["errors"][counter_name][error_type] = stats["errors"][counter_name].get(error_type, 0) + 1
            stats["last_updated"] = datetime.now().isoformat()
            
            # Save
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(self.stats_file, stats)
        except Exception as e:
            # Don't log here to avoid recursion - stats are best-effort
            pass
    
    def _increment_mode_counter(self, mode: str) -> None:
        """Increment mode usage counter (COHORT_AWARE vs LEGACY)."""
        try:
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                except (json.JSONDecodeError, IOError):
                    stats = {}
            else:
                stats = {}
            
            if "modes" not in stats:
                stats["modes"] = {}
            if mode not in stats["modes"]:
                stats["modes"][mode] = 0
            
            stats["modes"][mode] = stats["modes"].get(mode, 0) + 1
            stats["last_updated"] = datetime.now().isoformat()
            
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(self.stats_file, stats)
        except Exception as e:
            # CRITICAL: Stats file write affects artifacts - use centralized error handling
            from TRAINING.common.exceptions import handle_error_with_policy
            handle_error_with_policy(
                error=e,
                stage="REPRODUCIBILITY",
                error_type="stats_file_write",
                affects_artifact=True,
                logger_instance=logger
            )
    
    def generate_trend_summary(
        self,
        view: str = "STRICT",
        min_runs_for_trend: int = 2  # Minimum 2 runs for trend (slope requires 2 points)
    ) -> Dict[str, Any]:
        """
        Generate trend summary for all series in the reproducibility directory.
        
        This can be called at the end of a run to show overall trend status.
        
        Args:
            view: "STRICT" or "PROGRESS"
            min_runs_for_trend: Minimum runs required for trend fitting
        
        Returns:
            Dict with trend summary statistics
        """
        if not _AUDIT_AVAILABLE:
            return {"status": "trend_analyzer_not_available"}
        
        try:
            from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView
            
            # Check for target-first structure first (targets/ and globals/)
            # Fallback to legacy REPRODUCIBILITY structure for backward compatibility
            repro_base = None
            comparison_group_dir = None
            
            # Try target-first structure: check for targets/ or globals/ directories
            temp_dir = self._repro_base_dir
            for _ in range(10):  # Limit depth
                if (temp_dir / "targets").exists() or (temp_dir / "globals").exists():
                    repro_base = temp_dir
                    break
                if not temp_dir.parent.exists():
                    break
                temp_dir = temp_dir.parent
            
            # If we found a run directory, check if it's in a comparison group structure
            # Structure: RESULTS/runs/cg-*/run_name/
            if repro_base:
                temp_dir = repro_base
                for _ in range(5):  # Limit depth
                    if temp_dir.parent.name == "runs" and temp_dir.parent.parent.name == "RESULTS":
                        # We're in RESULTS/runs/cg-*/run_name/
                        comparison_group_dir = temp_dir.parent
                        # Pass the comparison group directory to TrendAnalyzer so it searches all runs
                        repro_base = comparison_group_dir
                        logger.info(f"Found comparison group directory: {comparison_group_dir.name}, will search across all runs")
                        break
                    if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                        break
                    temp_dir = temp_dir.parent
            
            # Fallback to legacy REPRODUCIBILITY structure
            if repro_base is None:
                repro_base = self._repro_base_dir / "REPRODUCIBILITY"
                if not repro_base.exists():
                    # Try alternative location
                    repro_base = self.output_dir / "REPRODUCIBILITY"
            
            # Check if we have either target-first or legacy structure
            has_target_first = (self._repro_base_dir / "targets").exists() or (self._repro_base_dir / "globals").exists()
            has_legacy = repro_base.exists() if repro_base else False
            
            if not has_target_first and not has_legacy:
                return {"status": "reproducibility_directory_not_found"}
            
            # If repro_base is a comparison group directory, TrendAnalyzer will search within it
            # If repro_base is a run directory, TrendAnalyzer will search for targets/globals/REPRODUCIBILITY
            # Pass the appropriate directory (comparison group or run directory)
            trend_analyzer = TrendAnalyzer(
                reproducibility_dir=repro_base,
                half_life_days=7.0,
                min_runs_for_trend=min_runs_for_trend
            )
            
            # Analyze trends
            series_view = SeriesView(view.upper() if view.upper() in ["STRICT", "PROGRESS"] else "STRICT")
            all_trends = trend_analyzer.analyze_all_series(view=series_view)
            
            # Generate summary
            summary = {
                "status": "ok",
                "view": series_view.value,
                "n_series": len(all_trends),
                "n_trends": sum(len(t) for t in all_trends.values()),
                "series_with_trends": [],
                "alerts": [],
                "declining_trends": []
            }
            
            for series_key_str, trend_list in all_trends.items():
                for trend in trend_list:
                    if trend.status == "ok":
                        series_info = {
                            "series_key": series_key_str[:100],  # Truncate for readability
                            "metric": trend.metric_name,
                            "slope_per_day": trend.slope_per_day,
                            "current_estimate": trend.current_estimate,
                            "n_runs": trend.n_runs
                        }
                        summary["series_with_trends"].append(series_info)
                        
                        # Collect alerts
                        if trend.alerts:
                            summary["alerts"].extend(trend.alerts)
                        
                        # Flag declining trends
                        if trend.slope_per_day and trend.slope_per_day < -0.001:
                            summary["declining_trends"].append({
                                "metric": trend.metric_name,
                                "slope": trend.slope_per_day,
                                "series": series_key_str[:100]
                            })
            
            # Log summary
            logger.info(f"📊 Trend Summary ({series_view.value}): {summary['n_series']} series, {summary['n_trends']} trends")
            if summary["declining_trends"]:
                logger.warning(f"  ⚠️  {len(summary['declining_trends'])} declining trends detected")
                for decl in summary["declining_trends"][:5]:  # Show first 5
                    logger.warning(f"    - {decl['metric']}: slope={decl['slope']:.6f}/day")
            if summary["alerts"]:
                logger.info(f"  ℹ️  {len(summary['alerts'])} trend alerts")
            
            return summary
        except Exception as e:
            logger.debug(f"Could not generate trend summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_metrics_rollups(
        self,
        stage: Union[str, Stage],  # SST: Accept both string and Stage enum
        run_id: str
    ) -> None:
        """
        Generate view-level and stage-level metrics rollups.

        Should be called after all cohorts for a stage are saved.

        Args:
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, etc.) - string or Stage enum
            run_id: Current run identifier
        """
        if not self.metrics:
            return
        
        # SST: Normalize stage enum to string for JSON serialization
        if isinstance(stage, Stage):
            stage = stage.value
        elif hasattr(stage, 'value'):
            stage = stage.value
        else:
            stage = str(stage)
        
        # Rollups are now generated per-target in targets/<target>/metrics/
        # This method is kept for backward compatibility but does nothing
        # (rollups are handled by MetricsWriter in target-first structure)
        return
        
        # Aggregate metrics facts table (append to Parquet)
        try:
            from TRAINING.common.utils.metrics import aggregate_metrics_facts
            aggregate_metrics_facts(repro_dir)
        except Exception as e:
            logger.debug(f"Failed to aggregate metrics facts table: {e}")
