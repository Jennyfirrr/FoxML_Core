# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Logging API Mixin for ReproducibilityTracker.

Contains the main public API methods for logging comparisons and runs.
Extracted from reproducibility_tracker.py for maintainability.

SST COMPLIANCE:
- Uses extract_n_effective(), extract_universe_sig() helpers
- Uses Stage/View enums for consistent handling
- Uses sorted_items() for deterministic dict iteration
- Uses write_atomic_json for atomic writes

DETERMINISM:
- All dict iterations use sorted_items()
- Enum comparisons instead of string comparisons
- Sample-adjusted statistics for robust comparisons
"""

import fcntl
import hashlib
import json
import logging
import math
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json
from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json

# SST: Import Stage and View enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

# SST: Import extraction helpers
from TRAINING.orchestration.utils.reproducibility.utils import (
    extract_n_effective,
    extract_auc,
    extract_universe_sig,
    _get_main_logger,
)

# Import WriteScope for scope-safe writes (optional)
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

# Import RunContext and AuditEnforcer for automated audit-grade tracking
try:
    from TRAINING.orchestration.utils.run_context import RunContext
    from TRAINING.common.utils.audit_enforcer import AuditEnforcer, AuditMode
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False
    RunContext = None
    AuditMode = None
    AuditEnforcer = None

# Schema version for reproducibility files
from TRAINING.orchestration.utils.reproducibility_tracker import REPRODUCIBILITY_SCHEMA_VERSION

if TYPE_CHECKING:
    from TRAINING.orchestration.utils.metrics_writer import MetricsWriter

logger = logging.getLogger(__name__)


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


def _normalize_view_for_comparison(view: Union[str, View, None]) -> str:
    """Normalize view to string for safe comparisons."""
    if view is None:
        return View.CROSS_SECTIONAL.value
    if isinstance(view, View):
        return view.value
    return str(view).upper()


class LoggingAPIMixin:
    """
    Mixin class providing the main logging API for ReproducibilityTracker.

    This mixin contains the primary public methods:
    - log_comparison: Compare current run to previous run and log comparison
    - log_run: Automated audit-grade reproducibility tracking using RunContext

    Methods in this mixin expect the following attributes on self:
    - output_dir: Path - Output directory for this tracker
    - log_file: Path - Log file path
    - cohort_aware: bool - Whether cohort-aware mode is enabled
    - n_ratio_threshold: float - Threshold for N ratio comparisons
    - thresholds: Dict[str, Dict[str, float]] - Classification thresholds
    - _repro_base_dir: Path - Base directory for reproducibility artifacts
    - audit_enforcer: Optional[AuditEnforcer] - Audit enforcer instance
    - metrics: Optional[MetricsWriter] - Metrics writer instance

    Methods in this mixin call the following methods on self:
    - _extract_cohort_metadata: From CohortManagerMixin
    - _compute_cohort_id: From CohortManagerMixin
    - _find_matching_cohort: From ComparisonEngineMixin
    - get_last_comparable_run: From ComparisonEngineMixin
    - _compare_within_cohort: From ComparisonEngineMixin
    - _compute_drift: From ComparisonEngineMixin
    - _classify_diff: From main class
    - _extract_view: From main class
    - _extract_symbol: From main class
    - _extract_model_family: From main class
    - _increment_mode_counter: From main class
    - _increment_error_counter: From main class
    - load_previous_run: From main class
    - save_run: From main class
    - _save_to_cohort: From main class
    """

    # Type hints for expected attributes (set by the main class)
    output_dir: Path
    log_file: Path
    cohort_aware: bool
    n_ratio_threshold: float
    thresholds: Dict[str, Dict[str, float]]
    _repro_base_dir: Path
    audit_enforcer: Any  # Optional[AuditEnforcer]
    metrics: Any  # Optional[MetricsWriter]

    def log_comparison(
        self,
        stage: Union[str, Stage],
        target: str,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None,
        view: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        cohort_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,
        prediction_fingerprint: Optional[Dict] = None,
    ) -> None:
        """
        Compare current run to previous run and log the comparison for reproducibility verification.

        Uses tolerance bands with STABLE/DRIFTING/DIVERGED classification. Only escalates to
        warnings for meaningful differences (DIVERGED).

        If cohort_aware=True and cohort metadata is available, organizes runs by cohort and
        only compares within the same cohort using sample-adjusted statistics.

        This method should never raise exceptions - all errors are logged and handled gracefully.

        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            target: Name of the item (e.g., target name, symbol name)
            metrics: Dictionary of metrics to track and compare
            additional_data: Optional additional data to store with the run
            view: DEPRECATED - use `view` instead
            symbol: Optional symbol name (for symbol-specific views)
            model_family: Optional model family (for training stage)
            view: Modeling granularity ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
            run_identity: Optional RunIdentity SST object with authoritative signatures
            prediction_fingerprint: Optional prediction fingerprint dict for predictions_sha256
        """
        # CRITICAL FIX: Extract run_id from run_identity (SST canonical source) and ensure it's in additional_data
        # This ensures all downstream code (extract_run_id, _save_to_cohort) can find it
        if run_identity is not None:
            run_id_from_identity = None
            # Try multiple ways to extract run_id from run_identity object
            if hasattr(run_identity, 'run_id'):
                run_id_from_identity = getattr(run_identity, 'run_id', None)
            elif hasattr(run_identity, 'timestamp'):
                run_id_from_identity = getattr(run_identity, 'timestamp', None)
            elif isinstance(run_identity, dict):
                run_id_from_identity = run_identity.get('run_id') or run_identity.get('timestamp')

            # Ensure additional_data exists and has run_id/timestamp
            if run_id_from_identity:
                if additional_data is None:
                    additional_data = {}
                # Only add if not already present (don't overwrite existing)
                if 'run_id' not in additional_data and 'timestamp' not in additional_data:
                    additional_data['run_id'] = str(run_id_from_identity)

        # Normalize stage to enum for internal use
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage

        # SST: view takes precedence over view
        if view is not None:
            view = view
        try:
            # Extract cohort metadata if available
            # CRITICAL: If cohort_metadata is already provided (e.g., from log_run()), use it directly
            # This avoids redundant extraction and ensures we use the same metadata that was extracted from RunContext
            if cohort_metadata is None:
                try:
                    cohort_metadata = self._extract_cohort_metadata(metrics, additional_data)
                except Exception as e:
                    logger.warning(f"Failed to extract cohort metadata for {stage}:{target}: {e}. Falling back to legacy mode.")
                    logger.debug(f"Cohort metadata extraction traceback: {traceback.format_exc()}")
                    cohort_metadata = None

            # Cohort-aware mode is the default - use it if enabled and metadata is available
            use_cohort_aware = self.cohort_aware and cohort_metadata is not None
            if self.cohort_aware and not use_cohort_aware:
                # Use INFO level so it's visible - this is important for debugging
                main_logger = _get_main_logger()
                msg = (f"Reproducibility: Cohort-aware mode enabled (default) but insufficient metadata for {stage}:{target}. "
                       f"Falling back to legacy mode. "
                       f"Metrics keys: {list(metrics.keys())}, "
                       f"Additional data keys: {list(additional_data.keys()) if additional_data else 'None'}. "
                       f"To enable cohort-aware mode, pass n_effective_cs, n_symbols, date_range, and cs_config in metrics/additional_data.")
                if main_logger != logger:
                    main_logger.info(msg)
                else:
                    logger.info(msg)
            elif use_cohort_aware:
                # Log when cohort-aware mode is successfully used
                main_logger = _get_main_logger()
                n_info = f"N={cohort_metadata.get('n_effective_cs', '?')}, symbols={cohort_metadata.get('n_symbols', '?')}"
                msg = f"Reproducibility: Using cohort-aware mode for {stage}:{target} ({n_info})"
                if main_logger != logger:
                    main_logger.debug(msg)
                else:
                    logger.debug(msg)

            # Extract view, symbol, model_family
            # Use provided parameters if available, otherwise extract from additional_data
            # For TARGET_RANKING, view comes from "view" field in additional_data
            # For FEATURE_SELECTION, map view to view (CROSS_SECTIONAL -> CROSS_SECTIONAL, SYMBOL_SPECIFIC -> INDIVIDUAL)
            if view is None:
                # Normalize stage to enum for comparison
                stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
                if stage_enum == Stage.TARGET_RANKING:
                    view = additional_data.get("view") if additional_data else None
                    if view:
                        view = view.upper()  # Normalize to uppercase
                elif stage_enum == Stage.FEATURE_SELECTION:
                    # FIX: Map view to view for FEATURE_SELECTION (ensures proper metrics scoping)
                    view = additional_data.get("view") if additional_data else None
                    if view:
                        view_str = _normalize_view_for_comparison(view)
                        # Normalize view_str to enum
                        view_enum = View.from_string(view_str) if isinstance(view_str, str) else view_str
                        if view_enum == View.CROSS_SECTIONAL:
                            view = View.CROSS_SECTIONAL
                        elif view_enum == View.SYMBOL_SPECIFIC:
                            view = View.SYMBOL_SPECIFIC
                    if not view:
                        # Fallback to extraction method
                        view = self._extract_view(additional_data)
                else:
                    # Use normalized stage_enum for comparison
                    view = self._extract_view(additional_data) if stage_enum in (Stage.FEATURE_SELECTION, Stage.TRAINING) else None

            if symbol is None:
                symbol = self._extract_symbol(additional_data)

            if model_family is None:
                model_family = self._extract_model_family(additional_data)

            if use_cohort_aware:
                # Cohort-aware path: find matching cohort
                main_logger = _get_main_logger()
                n_info = f"N={cohort_metadata.get('n_effective_cs', '?')}, symbols={cohort_metadata.get('n_symbols', '?')}"
                if main_logger != logger:
                    main_logger.debug(f"Reproducibility: Searching for matching cohort for {stage}:{target} ({n_info})")
                else:
                    logger.debug(f"Reproducibility: Searching for matching cohort for {stage}:{target} ({n_info})")

                cohort_id = self._find_matching_cohort(stage, target, cohort_metadata, view, symbol, model_family)

                if cohort_id is None:
                    # New cohort - save as baseline
                    # Derive view from view for cohort_id computation
                    view_for_cohort = "CROSS_SECTIONAL"
                    if view:
                        rt_upper = view.upper()
                        if rt_upper == "SYMBOL_SPECIFIC":
                            view_for_cohort = "SYMBOL_SPECIFIC"
                    cohort_id = self._compute_cohort_id(cohort_metadata, view=view_for_cohort)

                    # Extract run_id before creating run_data (SST pattern: read from manifest)
                    # This ensures normalize_snapshot() gets run_id from run_data, not just timestamp
                    extracted_run_id = None
                    if run_identity is not None:
                        try:
                            from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                            extracted_run_id = derive_run_id_from_identity(run_identity=run_identity)
                        except Exception:
                            pass

                    # Fallback: Read from manifest.json if available
                    if not extracted_run_id and self.output_dir:
                        try:
                            from TRAINING.orchestration.utils.manifest import read_run_id_from_manifest
                            from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                            run_root = get_run_root(self.output_dir)
                            if run_root:
                                manifest_path = run_root / "manifest.json"
                                extracted_run_id = read_run_id_from_manifest(manifest_path)
                        except Exception:
                            pass

                    # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                    from TRAINING.common.utils.determinism_ordering import sorted_items
                    run_data = {
                        "run_id": extracted_run_id,  # Include run_id from manifest (SST pattern)
                        "timestamp": datetime.now().isoformat(),
                        "stage": stage,
                        "target": target,
                        "metrics": metrics,  # NEW: Add metrics key for _normalize_outputs() and _compute_metrics_digest()
                        **{k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in sorted_items(metrics)},  # Keep top-level for backward compat
                        "cohort_metadata": cohort_metadata
                    }
                    if additional_data:
                        run_data["additional_data"] = additional_data

                    # FIX: Pass symbol and model_family to _save_to_cohort so symbol subdirectory is created
                    # NEW: Also pass run_identity and prediction_fingerprint for snapshot creation
                    self._save_to_cohort(stage, target, cohort_id, cohort_metadata, run_data, view, symbol, model_family, additional_data, run_identity=run_identity, prediction_fingerprint=prediction_fingerprint)
                    self._increment_mode_counter("COHORT_AWARE")

                    main_logger = _get_main_logger()
                    n_info = f"N={cohort_metadata['n_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                    if cohort_metadata.get('date_range', {}).get('start_ts'):
                        date_info = f", date_range={cohort_metadata['date_range']['start_ts']}->{cohort_metadata['date_range'].get('end_ts', '')}"
                    else:
                        date_info = ""

                    msg = f"Reproducibility: First run for {stage}:{target} (new cohort: {n_info}{date_info})"
                    if main_logger != logger:
                        main_logger.info(msg)
                    else:
                        logger.info(msg)
                    return

                # Load previous run from index (only same cohort)
                previous = self.get_last_comparable_run(
                    stage=stage,
                    target=target,
                    view=view,
                    symbol=symbol,
                    model_family=model_family,
                    cohort_id=cohort_id,  # Key: only same cohort
                    current_N=cohort_metadata.get('n_effective_cs', 0),
                    n_ratio_threshold=self.n_ratio_threshold
                )

                if previous is None:
                    # First run in this cohort
                    # Extract run_id before creating run_data (SST pattern: read from manifest)
                    extracted_run_id = None
                    if run_identity is not None:
                        try:
                            from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                            extracted_run_id = derive_run_id_from_identity(run_identity=run_identity)
                        except Exception:
                            pass

                    # Fallback: Read from manifest.json if available
                    if not extracted_run_id and self.output_dir:
                        try:
                            from TRAINING.orchestration.utils.manifest import read_run_id_from_manifest
                            from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                            run_root = get_run_root(self.output_dir)
                            if run_root:
                                manifest_path = run_root / "manifest.json"
                                extracted_run_id = read_run_id_from_manifest(manifest_path)
                        except Exception:
                            pass

                    run_data = {
                        "run_id": extracted_run_id,  # Include run_id from manifest (SST pattern)
                        "timestamp": datetime.now().isoformat(),
                        "stage": stage,
                        "target": target,
                        **{k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in metrics.items()},
                        "cohort_metadata": cohort_metadata
                    }
                    if additional_data:
                        run_data["additional_data"] = additional_data

                    # NEW: Pass run_identity and prediction_fingerprint for snapshot creation
                    self._save_to_cohort(stage, target, cohort_id, cohort_metadata, run_data, view, symbol, model_family, additional_data, run_identity=run_identity, prediction_fingerprint=prediction_fingerprint)
                    self._increment_mode_counter("COHORT_AWARE")

                    main_logger = _get_main_logger()
                    n_info = f"N={cohort_metadata['n_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                    route_info = f" [{view}]" if view else ""
                    symbol_info = f" symbol={symbol}" if symbol else ""
                    model_info = f" model={model_family}" if model_family else ""
                    msg = f"Reproducibility: First run in cohort for {stage}:{target}{route_info}{symbol_info}{model_info} ({n_info})"
                    if main_logger != logger:
                        main_logger.info(msg)
                    else:
                        logger.info(msg)
                    return

                # Extract metrics for comparison (only reached if previous exists)
                metric_name = metrics.get("metric_name", "Score")
                current_mean = float(extract_auc(metrics) or 0.0)  # Handles both old and new structures
                previous_mean = float(extract_auc(previous) or 0.0)  # Handles both old and new structures

                # Try new structure first, then fallback to old
                current_std = float((metrics.get("primary_metric", {}).get("std") or
                                   metrics.get("primary_metric", {}).get("skill_se") or
                                   metrics.get("std_score")) or 0.0)
                previous_std = float((previous.get("primary_metric", {}).get("std") or
                                     previous.get("primary_metric", {}).get("skill_se") or
                                     previous.get("std_score")) or 0.0)

                # Compare importance if present
                current_importance = float(metrics.get("mean_importance", 0.0))
                previous_importance = float(previous.get("mean_importance", 0.0))

                # Compare composite score if present
                current_composite = float(metrics.get("composite_score", current_mean))
                previous_composite = float(previous.get("composite_score", previous_mean))

                # Compute route_changed and route_entropy for regression tracking
                # SST: 'view' is the canonical key
                prev_route = previous.get('view')
                curr_route = additional_data.get('view') if additional_data else None
                if curr_route is None:
                    curr_route = view
                route_changed = 1 if (prev_route and curr_route and prev_route != curr_route) else 0

                # Compute route_entropy from route history (if we have access to index)
                route_entropy = None
                try:
                    repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
                    index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        df = pd.read_parquet(index_file)
                        # Get route history for this cohort/target
                        # Derive view from view for cohort_id computation
                        view_for_cohort = "CROSS_SECTIONAL"
                        if view:
                            rt_upper = view.upper()
                            if rt_upper == "SYMBOL_SPECIFIC":
                                view_for_cohort = "SYMBOL_SPECIFIC"
                        cohort_id = self._compute_cohort_id(cohort_metadata, view=view_for_cohort)
                        mask = (df['cohort_id'] == cohort_id) & (df['target'] == target)
                        if view:
                            mask &= (df['mode'] == view.upper())
                        route_history = df[mask]['route'].dropna().tolist()
                        if len(route_history) >= 3:
                            # Compute entropy: -sum(p * log2(p))
                            route_counts = Counter(route_history)
                            total = len(route_history)
                            entropy = -sum((count / total) * math.log2(count / total)
                                         for count in route_counts.values() if count > 0)
                            route_entropy = float(entropy)
                except Exception:
                    pass  # Non-critical, continue without entropy

                # Use sample-adjusted comparison if cohort-aware and within same cohort
                # Prepare current run data for comparison
                # Store route_changed and route_entropy in metrics for index update
                if route_changed is not None:
                    metrics['route_changed'] = route_changed
                if route_entropy is not None:
                    metrics['route_entropy'] = route_entropy
                if curr_route:
                    metrics['route'] = curr_route

                curr_run_data = {
                    **metrics,
                    'n_effective_cs': cohort_metadata.get('n_effective_cs'),
                    'n_samples': cohort_metadata.get('n_effective_cs'),
                    'sample_size': cohort_metadata.get('n_effective_cs'),
                    'route': curr_route,
                    'route_changed': route_changed,
                    'route_entropy': route_entropy
                }
                prev_run_data = {
                    **previous,
                    'n_effective_cs': previous.get('cohort_metadata', {}).get('n_effective_cs') or previous.get('n_effective_cs'),
                    'n_samples': previous.get('cohort_metadata', {}).get('n_effective_cs') or previous.get('n_samples'),
                    'sample_size': previous.get('cohort_metadata', {}).get('n_effective_cs') or previous.get('sample_size'),
                    'route': prev_route
                }

                mean_class, mean_abs, mean_rel, mean_z, mean_stats = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'roc_auc'
                )
                composite_class, composite_abs, composite_rel, composite_z, _ = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'composite'
                )
                importance_class, importance_abs, importance_rel, importance_z, _ = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'importance'
                )
            else:
                # Legacy path: use flat structure
                main_logger = _get_main_logger()
                if main_logger != logger:
                    main_logger.info(f"Reproducibility: Using legacy mode for {stage}:{target} (files in {self.log_file.parent.name}/)")
                else:
                    logger.info(f"Reproducibility: Using legacy mode for {stage}:{target} (files in {self.log_file.parent.name}/)")

                previous = self.load_previous_run(stage, target)

                if previous is None:
                    # Use main logger if available for better visibility
                    main_logger = _get_main_logger()
                    # Only log once - use main logger if available, otherwise use module logger
                    if main_logger != logger:
                        main_logger.info(f"Reproducibility: First run for {stage}:{target} (no previous run to compare)")
                    else:
                        logger.info(f"Reproducibility: First run for {stage}:{target} (no previous run to compare)")
                    # Save current run for next time
                    self.save_run(stage, target, metrics, additional_data)
                    self._increment_mode_counter("LEGACY")
                    return

                # Extract metrics for comparison (only reached if previous exists)
                metric_name = metrics.get("metric_name", "Score")
                current_mean = float(extract_auc(metrics) or 0.0)  # Handles both old and new structures
                previous_mean = float(extract_auc(previous) or 0.0)  # Handles both old and new structures

                # Try new structure first, then fallback to old
                current_std = float((metrics.get("primary_metric", {}).get("std") or
                                   metrics.get("primary_metric", {}).get("skill_se") or
                                   metrics.get("std_score")) or 0.0)
                previous_std = float((previous.get("primary_metric", {}).get("std") or
                                     previous.get("primary_metric", {}).get("skill_se") or
                                     previous.get("std_score")) or 0.0)

                # Compare importance if present
                current_importance = float(metrics.get("mean_importance", 0.0))
                previous_importance = float(previous.get("mean_importance", 0.0))

                # Compare composite score if present
                current_composite = float(metrics.get("composite_score", current_mean))
                previous_composite = float(previous.get("composite_score", previous_mean))

                # Legacy comparison
                mean_class, mean_abs, mean_rel, mean_z = self._classify_diff(
                    previous_mean, current_mean, previous_std, 'roc_auc'
                )
                composite_class, composite_abs, composite_rel, composite_z = self._classify_diff(
                    previous_composite, current_composite, None, 'composite'
                )
                importance_class, importance_abs, importance_rel, importance_z = self._classify_diff(
                    previous_importance, current_importance, None, 'importance'
                )
                mean_stats = {}

            # Overall classification: use worst case
            if 'DIVERGED' in [mean_class, composite_class, importance_class]:
                overall_class = 'DIVERGED'
            elif 'DRIFTING' in [mean_class, composite_class, importance_class]:
                overall_class = 'DRIFTING'
            else:
                overall_class = 'STABLE'

            # Determine log level and emoji based on classification
            if overall_class == 'STABLE':
                log_level = logger.info
                emoji = "INFO"
            elif overall_class == 'DRIFTING':
                log_level = logger.info
                emoji = "INFO"
            else:  # DIVERGED
                log_level = logger.warning
                emoji = "WARN"

            # Use main logger if available for better visibility
            main_logger = _get_main_logger()

            # Build comparison log message
            mean_diff = current_mean - previous_mean
            composite_diff = current_composite - previous_composite
            importance_diff = current_importance - previous_importance

            # Format z-score if available
            z_info = ""
            if mean_z is not None:
                z_info = f", z={mean_z:.2f}"

            # Main status line
            cohort_info = ""
            if use_cohort_aware and cohort_metadata:
                n_info = f"N={cohort_metadata['n_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                if mean_stats.get('sample_adjusted'):
                    cohort_info = f" [cohort: {n_info}, sample-adjusted]"
                else:
                    cohort_info = f" [cohort: {n_info}]"

            status_msg = f"[{emoji}] Reproducibility: {overall_class}{cohort_info}"
            if overall_class == 'STABLE':
                status_msg += f" (Delta {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); within tolerance)"
            elif overall_class == 'DRIFTING':
                status_msg += f" (Delta {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); small drift detected)"
            else:  # DIVERGED
                status_msg += f" (Delta {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); exceeds tolerance)"

            # Only log once - use main logger if available, otherwise use module logger
            if main_logger != logger:
                if overall_class == 'DIVERGED':
                    main_logger.warning(status_msg)
                else:
                    main_logger.info(status_msg)
            else:
                log_level(status_msg)

            # Detailed comparison (always log for traceability)
            prev_n_info = ""
            curr_n_info = ""
            if use_cohort_aware and cohort_metadata:
                cohort_meta = previous.get('cohort_metadata', {})
                prev_n = extract_n_effective(cohort_meta) or extract_n_effective(previous)
                curr_n = cohort_metadata.get('n_effective_cs')
                if prev_n:
                    prev_n_info = f", N={int(prev_n)}"
                if curr_n:
                    curr_n_info = f", N={int(curr_n)}"

            prev_msg = f"   Previous: {metric_name}={previous_mean:.3f}+/-{previous_std:.3f}{prev_n_info}, " \
                       f"importance={previous_importance:.2f}, composite={previous_composite:.3f}"
            if main_logger != logger:
                main_logger.info(prev_msg)
            else:
                logger.info(prev_msg)

            curr_msg = f"   Current:  {metric_name}={current_mean:.3f}+/-{current_std:.3f}{curr_n_info}, " \
                       f"importance={current_importance:.2f}, composite={current_composite:.3f}"
            if main_logger != logger:
                main_logger.info(curr_msg)
            else:
                logger.info(curr_msg)

            # Diff line with classifications
            diff_parts = [f"{metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{', z=' + f'{mean_z:.2f}' if mean_z else ''}) [{mean_class}]"]
            diff_parts.append(f"composite={composite_diff:+.4f} ({composite_rel:+.2f}%) [{composite_class}]")
            diff_parts.append(f"importance={importance_diff:+.2f} ({importance_rel:+.2f}%) [{importance_class}]")
            diff_msg = f"   Diff:     {', '.join(diff_parts)}"
            if main_logger != logger:
                main_logger.info(diff_msg)
            else:
                logger.info(diff_msg)

            # Warning only for DIVERGED
            if overall_class == 'DIVERGED':
                warn_msg = f"   [WARN] Results differ significantly from previous run - check for non-deterministic behavior, config changes, or data differences"
                if main_logger != logger:
                    main_logger.warning(warn_msg)
                else:
                    logger.warning(warn_msg)

            # Save current run for next time
            if use_cohort_aware:
                # Extract run_id before creating run_data (SST pattern: read from manifest)
                extracted_run_id = None
                if run_identity is not None:
                    try:
                        from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                        extracted_run_id = derive_run_id_from_identity(run_identity=run_identity)
                    except Exception:
                        pass

                # Fallback: Read from manifest.json if available
                if not extracted_run_id and self.output_dir:
                    try:
                        from TRAINING.orchestration.utils.manifest import read_run_id_from_manifest
                        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                        run_root = get_run_root(self.output_dir)
                        if run_root:
                            manifest_path = run_root / "manifest.json"
                            extracted_run_id = read_run_id_from_manifest(manifest_path)
                    except Exception:
                        pass

                # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                from TRAINING.common.utils.determinism_ordering import sorted_items
                run_data = {
                    "run_id": extracted_run_id,  # Include run_id from manifest (SST pattern)
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage,
                    "target": target,
                    "metrics": metrics,  # NEW: Add metrics key for _normalize_outputs() and _compute_metrics_digest()
                    **{k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in sorted_items(metrics)},  # Keep top-level for backward compat
                    "cohort_metadata": cohort_metadata
                }
                if additional_data:
                    run_data["additional_data"] = additional_data

                # Derive view from view for cohort_id computation
                view_for_cohort = "CROSS_SECTIONAL"
                if view:
                    rt_upper = view.upper()
                    if rt_upper == "SYMBOL_SPECIFIC":
                        view_for_cohort = "SYMBOL_SPECIFIC"
                cohort_id = self._compute_cohort_id(cohort_metadata, view=view_for_cohort)
                # NEW: Pass run_identity and prediction_fingerprint for snapshot creation
                self._save_to_cohort(stage, target, cohort_id, cohort_metadata, run_data, view, symbol, model_family, additional_data, run_identity=run_identity, prediction_fingerprint=prediction_fingerprint)
                self._increment_mode_counter("COHORT_AWARE")

                # Compute trend analysis for this series (if enough runs exist)
                # NOTE: This code is in log_run() method, not _save_to_cohort()
                # Variables available: stage, target, cohort_id, cohort_metadata, run_data, view, symbol, model_family
                trend_metadata = None
                try:
                    if _AUDIT_AVAILABLE:
                        from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView

                        # Get reproducibility base directory from self._repro_base_dir
                        repro_base = self._repro_base_dir
                        if repro_base.exists():
                            trend_analyzer = TrendAnalyzer(
                                reproducibility_dir=repro_base,
                                half_life_days=7.0,
                                min_runs_for_trend=2  # Minimum 2 runs for trend (slope requires 2 points)
                            )

                            # Analyze STRICT series
                            all_trends = trend_analyzer.analyze_all_series(view=SeriesView.STRICT)

                            # Use normalized stage_enum for trend matching
                            stage_for_trend = str(stage_enum).replace("MODEL_TRAINING", "TRAINING") if stage_enum else (str(stage).upper().replace("MODEL_TRAINING", "TRAINING") if stage else "UNKNOWN")

                            # Find trend for this series
                            for series_key_str, trend_list in all_trends.items():
                                # Check if this series matches
                                if any(t.series_key.target == target and
                                       t.series_key.stage == stage_for_trend for t in trend_list):
                                    # NOTE: Trend writing happens in _save_to_cohort() where target_cohort_dir is available
                                    # This code path is for analysis only, not writing
                                    logger.debug(f"Found trend for {stage_for_trend}:{target} (trend writing happens in _save_to_cohort)")
                                    break
                except Exception as e:
                    logger.debug(f"Trend analysis failed (non-critical): {e}")
                    logger.debug(f"Could not compute trend analysis: {e}")

                # Compute and save drift.json if previous run exists
                if previous:
                    # FIX: Extract run_id from multiple sources (run_data, additional_data, metrics) to handle scoping issues
                    # This avoids relying solely on run_data which may not be constructed in all code paths
                    run_id_or_timestamp = None
                    # Try run_data first (if it exists and is a dict)
                    try:
                        if 'run_data' in locals() and isinstance(run_data, dict):
                            run_id_or_timestamp = run_data.get('run_id') or run_data.get('timestamp')
                    except (NameError, TypeError):
                        pass  # run_data not defined or not accessible
                    # Fallback to additional_data
                    if not run_id_or_timestamp and additional_data:
                        run_id_or_timestamp = additional_data.get('run_id') or additional_data.get('timestamp')
                    # Fallback to metrics dict
                    if not run_id_or_timestamp and isinstance(metrics, dict):
                        run_id_or_timestamp = metrics.get('run_id') or metrics.get('timestamp')
                    # Final fallback: generate new timestamp
                    if not run_id_or_timestamp or not isinstance(run_id_or_timestamp, str) or not run_id_or_timestamp.strip():
                        run_id_or_timestamp = datetime.now().isoformat()
                    # Ensure it's a non-empty string before .replace()
                    if not run_id_or_timestamp or not isinstance(run_id_or_timestamp, str) or not run_id_or_timestamp.strip():
                        run_id_or_timestamp = datetime.now().isoformat()
                    # Now guaranteed to be a non-empty string - safe to call .replace()
                    run_id_clean = run_id_or_timestamp.replace(':', '-').replace('.', '-').replace('T', '_')
                    try:
                        drift_data = self._compute_drift(
                            previous, run_data, cohort_metadata,
                            stage, target, view, symbol, model_family,
                            cohort_id, run_id_clean
                        )
                        # Write drift.json to target-first structure only
                        try:
                            from TRAINING.orchestration.utils.target_first_paths import (
                                get_target_reproducibility_dir, ensure_target_structure
                            )
                            base_output_dir = self._repro_base_dir
                            ensure_target_structure(base_output_dir, target)
                            target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage)

                            # Determine view
                            view_str = _normalize_view_for_comparison(view) if view else View.CROSS_SECTIONAL.value
                            if view_str not in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                                view = View.SYMBOL_SPECIFIC  # Normalize legacy values
                            else:
                                view = View.CROSS_SECTIONAL if view_str == View.CROSS_SECTIONAL.value else View.SYMBOL_SPECIFIC

                            # CRITICAL FIX: If symbol is set, force SYMBOL_SPECIFIC
                            # This must happen before path construction to prevent wrong directory
                            if symbol:
                                view = View.SYMBOL_SPECIFIC
                                view_str = View.SYMBOL_SPECIFIC.value
                            elif cohort_id and cohort_id.startswith("sy_"):
                                view = View.SYMBOL_SPECIFIC
                                view_str = View.SYMBOL_SPECIFIC.value
                            else:
                                # Normalize view to string for path construction
                                view_str = view.value if isinstance(view, View) else (view if isinstance(view, str) else View.CROSS_SECTIONAL.value)

                            # Use build_target_cohort_dir() for canonical path construction (includes batch_ and attempt_ levels)
                            # Extract attempt_id and universe_sig from additional_data or cohort_metadata
                            attempt_id = (additional_data.get('attempt_id') if additional_data else None) or 0
                            universe_sig = None
                            if additional_data:
                                universe_sig = extract_universe_sig(additional_data)
                            if not universe_sig and cohort_metadata:
                                universe_sig = cohort_metadata.get('universe_sig')
                                if not universe_sig and 'cs_config' in cohort_metadata:
                                    universe_sig = cohort_metadata['cs_config'].get('universe_sig')

                            from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                            target_cohort_dir = build_target_cohort_dir(
                                base_output_dir=base_output_dir,
                                target=target,
                                stage=stage,
                                view=view_str,
                                cohort_id=cohort_id,
                                symbol=symbol if view_str == View.SYMBOL_SPECIFIC.value else None,
                                attempt_id=attempt_id,  # Always include, even if 0
                                universe_sig=universe_sig  # Required for CROSS_SECTIONAL batch_ level
                            )

                            drift_file = target_cohort_dir / "drift.json"
                            target_cohort_dir.mkdir(parents=True, exist_ok=True)
                            # SST: Use write_atomic_json for atomic write with canonical serialization
                            from TRAINING.common.utils.file_utils import write_atomic_json
                            write_atomic_json(drift_file, drift_data)
                        except (IOError, OSError) as e:
                            logger.warning(f"Failed to save drift.json to target-first structure: {e}, error_type=IO_ERROR")
                            self._increment_error_counter("write_failures", "IO_ERROR")
                        except Exception as e:
                            logger.debug(f"Could not write drift.json to target-first structure: {e}")
                            # Don't re-raise - drift file failure shouldn't break the run
                    except Exception as e:
                        logger.warning(f"Failed to compute drift for {stage}:{target}: {e}")
                        logger.debug(f"Drift computation traceback: {traceback.format_exc()}")
            else:
                # CRITICAL: Even in legacy mode, try to write metadata.json and metrics.json to cohort directory
                # This ensures files are written for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
                # Build minimal cohort metadata from available data
                minimal_cohort_metadata = {}
                if metrics:
                    # Try to extract n_effective from metrics - use SST accessor
                    n_effective = extract_n_effective(metrics)
                    if n_effective:
                        minimal_cohort_metadata['n_effective_cs'] = int(n_effective)

                if additional_data:
                    n_symbols = additional_data.get('n_symbols')
                    if n_symbols:
                        minimal_cohort_metadata['n_symbols'] = int(n_symbols)

                    # Extract date range if available
                    date_range = {}
                    if 'date_range' in additional_data:
                        date_range = additional_data['date_range']
                    elif 'start_ts' in additional_data or 'end_ts' in additional_data:
                        date_range = {
                            'start_ts': additional_data.get('start_ts'),
                            'end_ts': additional_data.get('end_ts')
                        }
                    if date_range:
                        minimal_cohort_metadata['date_range'] = date_range

                    # Extract cs_config if available
                    if 'cs_config' in additional_data:
                        minimal_cohort_metadata['cs_config'] = additional_data['cs_config']
                    elif 'min_cs' in additional_data or 'max_cs_samples' in additional_data:
                        minimal_cohort_metadata['cs_config'] = {
                            'min_cs': additional_data.get('min_cs'),
                            'max_cs_samples': additional_data.get('max_cs_samples'),
                            'leakage_filter_version': additional_data.get('leakage_filter_version', 'v1')
                        }

                # If we have minimal cohort metadata, try to write to cohort directory
                if minimal_cohort_metadata.get('n_effective_cs'):
                    try:
                        # Compute cohort_id from minimal metadata
                        # Derive view from view for cohort_id computation
                        view_for_cohort = "CROSS_SECTIONAL"
                        if view:
                            rt_upper = view.upper()
                            if rt_upper == "SYMBOL_SPECIFIC":
                                view_for_cohort = "SYMBOL_SPECIFIC"
                        minimal_cohort_id = self._compute_cohort_id(minimal_cohort_metadata, view=view_for_cohort)

                        # Extract run_id before creating run_data (SST pattern: read from manifest)
                        extracted_run_id = None
                        if run_identity is not None:
                            try:
                                from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                                extracted_run_id = derive_run_id_from_identity(run_identity=run_identity)
                            except Exception:
                                pass

                        # Fallback: Read from manifest.json if available
                        if not extracted_run_id and self.output_dir:
                            try:
                                from TRAINING.orchestration.utils.manifest import read_run_id_from_manifest
                                from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                                run_root = get_run_root(self.output_dir)
                                if run_root:
                                    manifest_path = run_root / "manifest.json"
                                    extracted_run_id = read_run_id_from_manifest(manifest_path)
                            except Exception:
                                pass

                        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                        from TRAINING.common.utils.determinism_ordering import sorted_items
                        run_data = {
                            "run_id": extracted_run_id,  # Include run_id from manifest (SST pattern)
                            "timestamp": datetime.now().isoformat(),
                            "stage": stage,
                            "target": target,
                            "metrics": metrics,  # NEW: Add metrics key for _normalize_outputs() and _compute_metrics_digest()
                            **{k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in sorted_items(metrics)},  # Keep top-level for backward compat
                            "cohort_metadata": minimal_cohort_metadata
                        }
                        if additional_data:
                            run_data["additional_data"] = additional_data

                        # Try to write to cohort directory (even with minimal metadata)
                        # NEW: Pass run_identity and prediction_fingerprint for snapshot creation
                        self._save_to_cohort(stage, target, minimal_cohort_id, minimal_cohort_metadata, run_data, view, symbol, model_family, additional_data, run_identity=run_identity, prediction_fingerprint=prediction_fingerprint)
                        self._increment_mode_counter("LEGACY_WITH_COHORT_WRITE")
                        logger.info(f"Reproducibility: Wrote metadata.json/metrics.json to cohort directory (legacy mode with minimal metadata)")
                    except Exception as e:
                        # If cohort write fails, fall back to legacy save_run
                        logger.warning(f"Failed to write to cohort directory in legacy mode: {e}. Falling back to legacy save_run.")
                        logger.debug(f"Cohort write traceback: {traceback.format_exc()}")
                        self.save_run(stage, target, metrics, additional_data)
                        self._increment_mode_counter("LEGACY")
                else:
                    # No cohort metadata available - use legacy save_run
                    self.save_run(stage, target, metrics, additional_data)
                    self._increment_mode_counter("LEGACY")
        except Exception as e:
            # Final safety net - ensure log_comparison never raises
            # NOTE: This is appropriate for telemetry - reproducibility tracking is non-critical
            # and should never break the main pipeline. This is documented fail-open behavior.
            error_type = "IO_ERROR" if isinstance(e, (IOError, OSError)) else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError)) else "UNKNOWN_ERROR"

            logger.error(
                f"Reproducibility tracking failed completely for {stage}:{target}. "
                f"error_type={error_type}, reason={str(e)}"
            )
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Update stats counter
            self._increment_error_counter("total_failures", error_type)

            # Don't re-raise - reproducibility tracking should never break the main pipeline
            # This is documented fail-open behavior for telemetry/non-critical paths

    def log_run(
        self,
        ctx: Any,  # RunContext (using Any to avoid circular import issues)
        metrics: Dict[str, Any],
        additional_data_override: Optional[Dict[str, Any]] = None,
        prediction_fingerprint: Optional[Dict] = None,
        run_identity: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Automated audit-grade reproducibility tracking using RunContext.

        This is the recommended API for new code. It:
        1. Extracts all metadata from RunContext automatically
        2. Validates with AuditEnforcer
        3. Saves metadata/metrics
        4. Compares to previous run and writes audit report

        Args:
            ctx: RunContext containing all run data and configuration
            metrics: Dictionary of metrics (auc, std_score, etc.)
            additional_data_override: Optional dict to merge into additional_data
            prediction_fingerprint: Optional prediction fingerprint dict for predictions_sha256
            run_identity: Optional SST RunIdentity object for authoritative signatures

        Returns:
            Dict with audit_report and saved metadata paths

        Raises:
            ValueError: If required fields are missing (in COHORT_AWARE mode) or audit validation fails (in strict mode)
        """
        # Normalize ctx.stage to Stage enum at the beginning
        stage_enum = None
        if hasattr(ctx, 'stage') and ctx.stage:
            stage_enum = Stage.from_string(ctx.stage) if isinstance(ctx.stage, str) else ctx.stage

        if not _AUDIT_AVAILABLE:
            # Fallback to legacy API
            logger.warning("RunContext not available, falling back to legacy log_comparison API")
            self.log_comparison(
                stage=ctx.stage,
                target=ctx.target or ctx.target_column or "unknown",
                metrics=metrics,
                additional_data=ctx.to_dict(),
                run_identity=run_identity,  # Pass through SST identity
            )
            return {"mode": "legacy_fallback"}

        # 1. Validate required fields
        # FIX: If required fields are missing in COHORT_AWARE mode, downgrade to NON_COHORT and log warning
        if self.cohort_aware:
            missing = ctx.validate_required_fields("COHORT_AWARE")
            if missing:
                # Check if this is a fallback scenario (all core data fields are None)
                # In fallback scenarios, this is expected behavior, so use debug level
                # FIX 2: Safely check for None without triggering numpy array boolean ambiguity
                is_fallback = (
                    (ctx.X is None or (isinstance(ctx.X, (np.ndarray, list)) and len(ctx.X) == 0)) and
                    (ctx.y is None or (isinstance(ctx.y, (np.ndarray, list)) and len(ctx.y) == 0)) and
                    (ctx.time_vals is None or (isinstance(ctx.time_vals, (np.ndarray, list)) and len(ctx.time_vals) == 0))
                )

                if is_fallback:
                    logger.debug(
                        f"Missing required fields for COHORT_AWARE mode: {missing}. "
                        f"Downgrading to NON_COHORT mode (expected in fallback scenario). "
                        f"RunContext should contain: {ctx.get_required_fields('COHORT_AWARE')}"
                    )
                else:
                    logger.warning(
                        f"[WARN] Missing required fields for COHORT_AWARE mode: {missing}. "
                        f"Downgrading to NON_COHORT mode for this run. "
                        f"RunContext should contain: {ctx.get_required_fields('COHORT_AWARE')}"
                    )
                # Downgrade to NON_COHORT mode for this run (don't fail)
                use_cohort_aware = False
            else:
                use_cohort_aware = True
        else:
            use_cohort_aware = False

        # 2. Auto-derive purge/embargo if not set
        if ctx.purge_minutes is None and ctx.horizon_minutes is not None:
            purge_min, embargo_min = ctx.derive_purge_embargo()
            ctx.purge_minutes = purge_min
            if ctx.embargo_minutes is None:
                ctx.embargo_minutes = embargo_min
            logger.info(f"Auto-derived purge={purge_min:.1f}m, embargo={embargo_min:.1f}m from horizon={ctx.horizon_minutes}m")

        # 3. Extract metadata from RunContext (only if use_cohort_aware is True)
        from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker

        if use_cohort_aware:
            cohort_metadata = extract_cohort_metadata(
                X=ctx.X,
                y=ctx.y,
                symbols=ctx.symbols,
                time_vals=ctx.time_vals,
                mtf_data=ctx.mtf_data,
                min_cs=ctx.min_cs,
                max_cs_samples=ctx.max_cs_samples,
                leakage_filter_version=ctx.leakage_filter_version,
                universe_sig=ctx.universe_sig,
                compute_data_fingerprint=True,
                compute_per_symbol_stats=True
            )

            # Format for tracker
            cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
        else:
            # NON_COHORT mode: use minimal metadata but still preserve critical scope fields
            # FIX: Extract universe_sig, view, and symbols from RunContext to prevent scope warnings
            cohort_metrics = {}
            cohort_additional_data = {}
            cohort_metadata = None  # FIX: Initialize to None for NON_COHORT mode to avoid UnboundLocalError
            # Build minimal cs_config with universe_sig for scope tracking
            # FIX 2 (Location 2): Safely check ctx.symbols (may be numpy array)
            has_universe_sig = ctx.universe_sig is not None
            has_symbols = ctx.symbols is not None and (
                (isinstance(ctx.symbols, (np.ndarray, list)) and len(ctx.symbols) > 0) or
                (not isinstance(ctx.symbols, (np.ndarray, list)) and bool(ctx.symbols))
            )

            if has_universe_sig or has_symbols:
                minimal_cs_config = {}
                if ctx.universe_sig:
                    minimal_cs_config['universe_sig'] = ctx.universe_sig
                if ctx.min_cs is not None:
                    minimal_cs_config['min_cs'] = ctx.min_cs
                if ctx.max_cs_samples is not None:
                    minimal_cs_config['max_cs_samples'] = ctx.max_cs_samples
                cohort_additional_data['cs_config'] = minimal_cs_config
                if has_symbols:
                    cohort_additional_data['symbols'] = ctx.symbols
                    cohort_additional_data['n_symbols'] = len(ctx.symbols) if isinstance(ctx.symbols, (np.ndarray, list)) else (len(ctx.symbols) if hasattr(ctx.symbols, '__len__') else 1)

        # Build additional_data with CV details
        additional_data = {
            **cohort_additional_data,
            "cv_method": ctx.cv_method,
            "folds": ctx.folds,
            "horizon_minutes": ctx.horizon_minutes,
            "purge_minutes": ctx.purge_minutes,
            "embargo_minutes": ctx.embargo_minutes,
            "feature_lookback_max_minutes": ctx.feature_lookback_max_minutes,
            "data_interval_minutes": ctx.data_interval_minutes,
            "feature_names": ctx.feature_names,
            "seed": ctx.seed,
            "train_seed": ctx.seed  # Also pass as train_seed for FEATURE_SELECTION/TRAINING
        }

        # Merge additional_data_override if provided (e.g., hyperparameters for FEATURE_SELECTION)
        if additional_data_override:
            additional_data.update(additional_data_override)

        # Add fold timestamps if available
        if ctx.fold_timestamps:
            additional_data["fold_timestamps"] = ctx.fold_timestamps

        # Add label definition hash
        if ctx.target_column:
            label_def_str = f"{ctx.target_column}|{ctx.target or ctx.target_column}"
            additional_data["label_definition_hash"] = hashlib.sha256(label_def_str.encode()).hexdigest()[:16]

        # Add view metadata for TARGET_RANKING
        # FIX: Add view to additional_data for both TARGET_RANKING and FEATURE_SELECTION
        # This ensures proper metrics scoping (features compared per-target, per-view, per-symbol)
        if hasattr(ctx, 'view') and ctx.view:
            additional_data["view"] = ctx.view
        # Also add symbol for SYMBOL_SPECIFIC/INDIVIDUAL views
        if hasattr(ctx, 'symbol') and ctx.symbol:
            additional_data["symbol"] = ctx.symbol
        # FIX: Add universe_sig at top level for scope tracking
        if hasattr(ctx, 'universe_sig') and ctx.universe_sig:
            additional_data["universe_sig"] = ctx.universe_sig

        # Merge metrics
        metrics_with_cohort = {**metrics, **cohort_metrics}

        # 4. Load previous run metadata for comparison
        # FIX: For FEATURE_SELECTION, map view to view (ensures proper metrics scoping)
        view_for_cohort = ctx.view if hasattr(ctx, 'view') else None
        # Use normalized stage_enum for comparison
        if stage_enum == Stage.TARGET_RANKING and hasattr(ctx, 'view') and ctx.view:
            view_for_cohort = ctx.view
        elif stage_enum == Stage.FEATURE_SELECTION and hasattr(ctx, 'view') and ctx.view:
            # Map view to view for FEATURE_SELECTION
            # FIX: Use SYMBOL_SPECIFIC directly (not INDIVIDUAL) to match directory structure
            # Normalize ctx.view to enum for comparison
            ctx_view_enum = View.from_string(ctx.view) if isinstance(ctx.view, str) else ctx.view
            if ctx_view_enum == View.CROSS_SECTIONAL:
                view_for_cohort = View.CROSS_SECTIONAL.value
            else:
                view_for_cohort = View.SYMBOL_SPECIFIC.value

        # Normalize view_for_cohort to enum (it may be string from JSON)
        if view_for_cohort:
            try:
                view_for_cohort_enum = View.from_string(view_for_cohort) if isinstance(view_for_cohort, str) else view_for_cohort
                view_for_cohort = view_for_cohort_enum.value if isinstance(view_for_cohort_enum, View) else str(view_for_cohort_enum).upper()
            except ValueError:
                view_for_cohort = View.CROSS_SECTIONAL.value  # Default if invalid
        else:
            view_for_cohort = View.CROSS_SECTIONAL.value  # Default if None
        # Final validation
        if view_for_cohort not in (View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value):
            view_for_cohort = View.CROSS_SECTIONAL.value  # Default if unexpected value
        cohort_id = self._compute_cohort_id(cohort_metadata, view=view_for_cohort)
        previous_metadata = None
        target_cohort_dir = None
        try:
            # Use target-first structure for reading previous metadata
            from TRAINING.orchestration.utils.target_first_paths import (
                get_target_reproducibility_dir
            )
            base_output_dir = self._repro_base_dir
            target = ctx.target or ctx.target_column or "unknown"
            # Pass stage for stage-scoped path lookup (falls back to legacy if stage is None)
            stage_for_path = ctx.stage if hasattr(ctx, 'stage') and ctx.stage else None
            target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage_for_path)

            # Determine view
            # CRITICAL FIX: Check symbol FIRST - if symbol is set, ALWAYS use SYMBOL_SPECIFIC
            if ctx.symbol:
                view_str = View.SYMBOL_SPECIFIC.value
            elif view_for_cohort:
                view_str = _normalize_view_for_comparison(view_for_cohort)
                if view_str not in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                    view_str = View.SYMBOL_SPECIFIC.value  # Normalize legacy values
            else:
                view_str = View.CROSS_SECTIONAL.value  # Default

            # CRITICAL: Use canonical path builder to ensure cohorts are in correct structure
            # Extract universe_sig, attempt_id, and use build_target_cohort_dir()
            attempt_id = additional_data.get('attempt_id', 0) if additional_data else 0
            universe_sig = None
            if additional_data:
                universe_sig = extract_universe_sig(additional_data)
            if not universe_sig and cohort_metadata:
                universe_sig = cohort_metadata.get('universe_sig')
                if not universe_sig and 'cs_config' in cohort_metadata:
                    universe_sig = cohort_metadata['cs_config'].get('universe_sig')

            from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
            target_cohort_dir = build_target_cohort_dir(
                base_output_dir=base_output_dir,
                target=target,
                stage=stage_for_path,
                view=view_str,
                cohort_id=cohort_id,
                symbol=ctx.symbol,
                attempt_id=attempt_id,
                universe_sig=universe_sig
            )

            # CRITICAL: Always use target_cohort_dir (canonical path) - don't fall back to legacy
            # If it doesn't exist, we'll create it below. Legacy paths create wrong structure.
            cohort_dir = target_cohort_dir

            if cohort_dir.exists():
                metadata_file = cohort_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        previous_metadata = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load previous metadata: {e}")

        # 5. Validate with AuditEnforcer (before saving)
        if self.audit_enforcer:
            # Build temporary metadata for validation
            temp_metadata = {
                "cohort_id": cohort_id,
                "data_fingerprint": cohort_metadata.get("data_fingerprint") if cohort_metadata else None,
                "feature_registry_hash": None,  # Will be computed in _save_to_cohort
                "cv_details": {
                    "cv_method": ctx.cv_method,
                    "horizon_minutes": ctx.horizon_minutes,
                    "purge_minutes": ctx.purge_minutes,
                    "embargo_minutes": ctx.embargo_minutes,
                    "folds": ctx.folds,
                    "feature_lookback_max_minutes": ctx.feature_lookback_max_minutes
                }
            }
            if ctx.fold_timestamps:
                fold_str = json.dumps(ctx.fold_timestamps, sort_keys=True, default=str)
                temp_metadata["cv_details"]["fold_boundaries_hash"] = hashlib.sha256(fold_str.encode()).hexdigest()[:16]

            is_valid, audit_report = self.audit_enforcer.validate(temp_metadata, metrics_with_cohort, previous_metadata)

            if not is_valid and self.audit_enforcer.mode == AuditMode.STRICT:
                # Already raised by enforcer, but be explicit
                raise ValueError(f"Audit validation failed: {audit_report}")
        else:
            audit_report = {"mode": "off", "violations": [], "warnings": []}

        # 6. Save using existing log_comparison (which handles cohort-aware saving)
        # For TARGET_RANKING, pass view as view
        view_for_log = ctx.view if hasattr(ctx, 'view') else None
        # Use normalized stage_enum for comparison
        if stage_enum == Stage.TARGET_RANKING and hasattr(ctx, 'view') and ctx.view:
            view_for_log = ctx.view

        # CRITICAL: Wrap log_comparison in try/except to ensure we can still write audit report
        # even if log_comparison fails. log_comparison itself has exception handling, but
        # we want to be defensive here.
        try:
            # CRITICAL: Pass the already-extracted cohort_metadata directly to log_comparison()
            # This ensures we use the same metadata that was extracted from RunContext, avoiding redundant extraction
            self.log_comparison(
                stage=ctx.stage,
                target=ctx.target or ctx.target_column or "unknown",
                metrics=metrics_with_cohort,
                additional_data=additional_data,
                view=view_for_log,  # SST: use view parameter
                symbol=ctx.symbol,
                cohort_metadata=cohort_metadata,  # Pass pre-extracted cohort_metadata from RunContext
                prediction_fingerprint=prediction_fingerprint,  # FIX: Pass through for predictions_sha256
                run_identity=run_identity,  # SST: Pass through authoritative identity
            )
        except Exception as e:
            # log_comparison should never raise (it has its own exception handling),
            # but if it does, log it and continue so we can still write audit report
            logger.error(f"log_comparison raised unexpected exception (this should not happen): {e}")
            logger.debug(f"log_comparison exception traceback: {traceback.format_exc()}")
            # Continue - we'll still write audit report below

        # 7. Write audit report
        audit_report_path = None
        cohort_dir = None
        try:
            # FIX: Use view as view for TARGET_RANKING and FEATURE_SELECTION when getting cohort directory
            view_for_cohort_dir = view_for_log  # Use same as log_comparison
            # Use normalized stage_enum for comparison
            if stage_enum == Stage.TARGET_RANKING and hasattr(ctx, 'view') and ctx.view:
                view_for_cohort_dir = ctx.view
            elif stage_enum == Stage.FEATURE_SELECTION and hasattr(ctx, 'view') and ctx.view:
                # Map view to view for FEATURE_SELECTION
                # FIX: Use SYMBOL_SPECIFIC directly (not INDIVIDUAL) to match directory structure
                ctx_view_str = _normalize_view_for_comparison(ctx.view) if hasattr(ctx, 'view') and ctx.view else View.CROSS_SECTIONAL.value
                if ctx_view_str == View.CROSS_SECTIONAL.value:
                    view_for_cohort_dir = View.CROSS_SECTIONAL
                else:
                    view_for_cohort_dir = View.SYMBOL_SPECIFIC

            # Use target-first structure with stage scoping
            from TRAINING.orchestration.utils.target_first_paths import (
                get_target_reproducibility_dir, ensure_target_structure
            )
            base_output_dir = self._repro_base_dir
            target = ctx.target or ctx.target_column or "unknown"
            ensure_target_structure(base_output_dir, target)
            # Pass stage for stage-scoped path (falls back to legacy if stage is None)
            stage_for_path = ctx.stage if hasattr(ctx, 'stage') and ctx.stage else None
            target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage_for_path)

            # Determine view - use view from run context (SST) if available
            view_str = _normalize_view_for_comparison(view_for_cohort_dir) if view_for_cohort_dir else View.CROSS_SECTIONAL.value
            if view_str not in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                view = View.SYMBOL_SPECIFIC  # Normalize legacy values
            else:
                view = View.CROSS_SECTIONAL if view_str == View.CROSS_SECTIONAL.value else View.SYMBOL_SPECIFIC

            # Try to load view from run context (SST) and use it if available
            try:
                from TRAINING.orchestration.utils.run_context import get_view
                view = get_view(self._repro_base_dir)
                if view:
                    # Use view instead of inferred view
                    view = view
                    logger.debug(f"Using view={view} from run context (SST) for cohort directory")
            except Exception as e:
                logger.debug(f"Could not load view from run context: {e}, using inferred view={view}")

            # CRITICAL FIX: If symbol is set, force SYMBOL_SPECIFIC
            # This must happen before path construction to prevent wrong directory
            if ctx.symbol:
                view_str = View.SYMBOL_SPECIFIC.value
            elif cohort_id and cohort_id.startswith("sy_"):
                view_str = View.SYMBOL_SPECIFIC.value
            else:
                # Normalize view to string for path construction
                view_str = _normalize_view_for_comparison(view_for_cohort_dir) if view_for_cohort_dir else View.CROSS_SECTIONAL.value
                if view_str not in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                    view_str = View.SYMBOL_SPECIFIC.value  # Normalize legacy values
                else:
                    # Ensure it's a valid View enum value string
                    view_str = view_str  # Already validated above

            # CRITICAL: Use canonical path builder to ensure cohorts are in correct structure
            # Extract universe_sig, attempt_id, and use build_target_cohort_dir()
            attempt_id = (additional_data.get('attempt_id') if additional_data else None) or 0
            universe_sig = None
            if additional_data:
                universe_sig = extract_universe_sig(additional_data)
            if not universe_sig and cohort_metadata:
                universe_sig = cohort_metadata.get('universe_sig')
                if not universe_sig and 'cs_config' in cohort_metadata:
                    universe_sig = cohort_metadata['cs_config'].get('universe_sig')

            from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
            target_cohort_dir = build_target_cohort_dir(
                base_output_dir=base_output_dir,
                target=target,
                stage=ctx.stage,
                view=view_str,
                cohort_id=cohort_id,
                symbol=ctx.symbol,
                attempt_id=attempt_id,
                universe_sig=universe_sig
            )

            # CRITICAL: Always use target_cohort_dir (canonical path) - don't fall back to legacy
            # Legacy paths create wrong structure (missing batch_ and attempt_ levels)
            # If it doesn't exist, we'll create it below
            cohort_dir = target_cohort_dir

            # CRITICAL: Ensure target-first cohort_dir exists (it should have been created by _save_to_cohort)
            if not target_cohort_dir.exists():
                target_cohort_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"[WARN] Target-first cohort directory {target_cohort_dir.name}/ did not exist - created it. This may indicate _save_to_cohort() was not called.")

            # Verify metadata files exist (should have been written by _save_to_cohort)
            metadata_file = cohort_dir / "metadata.json"
            metrics_file = cohort_dir / "metrics.json"
            if not metadata_file.exists() or not metrics_file.exists():
                logger.warning(
                    f"[WARN] Metadata files missing in {cohort_dir.name}/: "
                    f"metadata.json={'missing' if not metadata_file.exists() else 'exists'}, "
                    f"metrics.json={'missing' if not metrics_file.exists() else 'exists'}. "
                    f"Attempting to write them now as fallback."
                )

                # CRITICAL FALLBACK: If _save_to_cohort() didn't write the files, write them here
                # This ensures metadata.json and metrics.json are always written
                try:
                    # Build minimal metadata from available data
                    # CRITICAL: Use NEW field names to match finalize_run() expectations
                    minimal_metadata = {
                        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
                        "cohort_id": cohort_id,
                        "run_id": ctx.run_id if hasattr(ctx, 'run_id') else None,
                        "stage": ctx.stage,
                        "view": view_for_cohort_dir,
                        "target": ctx.target or ctx.target_column or "unknown",
                        "n_effective": cohort_metadata.get('n_effective_cs', 0) if cohort_metadata else 0,
                        "n_symbols": cohort_metadata.get('n_symbols', 0) if cohort_metadata else 0,
                        "date_start": cohort_metadata.get('date_range', {}).get('start_ts') if cohort_metadata else None,
                        "date_end": cohort_metadata.get('date_range', {}).get('end_ts') if cohort_metadata else None,
                        "created_at": datetime.now().isoformat()
                    }

                    # Write metadata.json if missing (to target-first structure)
                    target_metadata_file = target_cohort_dir / "metadata.json"
                    if not target_metadata_file.exists():
                        _write_atomic_json_with_lock(target_metadata_file, minimal_metadata)
                        logger.info(f"Wrote metadata.json (fallback) to {target_cohort_dir.name}/")

                    # Write metrics.json if missing (to target-first structure)
                    target_metrics_file = target_cohort_dir / "metrics.json"
                    if not target_metrics_file.exists() and self.metrics:
                        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                        from TRAINING.common.utils.determinism_ordering import sorted_items
                        minimal_metrics = {
                            "run_id": minimal_metadata.get("run_id"),
                            "timestamp": minimal_metadata.get("created_at"),
                            "stage": ctx.stage,
                            **{k: v for k, v in sorted_items(metrics_with_cohort) if k not in ['timestamp', 'cohort_metadata', 'additional_data']}
                        }
                        # Write metrics to target-first structure
                        # SST: Normalize stage and view to strings (handle enum inputs)
                        stage_str = ctx.stage.value if isinstance(ctx.stage, Stage) else (ctx.stage if hasattr(ctx, 'stage') else "UNKNOWN")
                        view_str_local = ctx.view.value if hasattr(ctx, 'view') and isinstance(ctx.view, View) else (ctx.view if hasattr(ctx, 'view') else "UNKNOWN")
                        self.metrics.write_cohort_metrics(
                            cohort_dir=target_cohort_dir,  # Use target-first, not legacy
                            stage=stage_str,
                            view=view_str_local,
                            target=ctx.target or ctx.target_column or "unknown",
                            symbol=ctx.symbol if hasattr(ctx, 'symbol') else None,
                            run_id=minimal_metadata.get("run_id") or datetime.now().isoformat(),
                            metrics=minimal_metrics
                        )
                        logger.info(f"Wrote metrics.json (fallback) to {target_cohort_dir.name}/")
                except Exception as e:
                    logger.error(f"Failed to write fallback metadata/metrics: {e}")
                    logger.debug(f"Fallback write traceback: {traceback.format_exc()}")

                # Write audit report to target-first structure only
                audit_report_path = target_cohort_dir / "audit_report.json"
                try:
                    # SST: Use write_atomic_json for atomic write with canonical serialization
                    from TRAINING.common.utils.file_utils import write_atomic_json
                    write_atomic_json(audit_report_path, audit_report)
                except Exception as e:
                    logger.debug(f"Could not write audit report to target-first structure: {e}")
        except Exception as e:
            logger.debug(f"Could not write audit report: {e}")

        # 8. Compute trend analysis for this series (if enough runs exist)
        trend_summary = None
        try:
            if _AUDIT_AVAILABLE:
                from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView

                # Get reproducibility base directory (use target-first structure)
                # Walk up from target_cohort_dir to find run directory
                if target_cohort_dir and target_cohort_dir.exists():
                    repro_base = target_cohort_dir
                    # Walk up to find run directory (should have targets/ or RESULTS/)
                    for _ in range(5):
                        if (repro_base / "targets").exists() or repro_base.name in ["RESULTS", "intelligent_output"]:
                            break
                        if not repro_base.parent.exists():
                            break
                        repro_base = repro_base.parent
                else:
                    # Fallback: use output_dir (should have targets/ structure)
                    repro_base = self._repro_base_dir

                if repro_base.exists():
                    trend_analyzer = TrendAnalyzer(
                        reproducibility_dir=repro_base,
                        half_life_days=7.0,
                        min_runs_for_trend=2  # Minimum 2 runs for trend (slope requires 2 points)
                    )

                    # Analyze STRICT series for this specific target
                    all_trends = trend_analyzer.analyze_all_series(view=SeriesView.STRICT)

                    # Compute stage string for trend matching (reuse throughout this section)
                    stage_str_for_trend = str(stage_enum) if stage_enum else (ctx.stage.upper() if hasattr(ctx, 'stage') and ctx.stage else "UNKNOWN")

                    # Find trend for this series
                    series_key_str = None
                    for sk, trend_list in all_trends.items():
                        # Check if this series matches
                        if any(t.series_key.target == (ctx.target or ctx.target_column) and
                               t.series_key.stage == stage_str_for_trend for t in trend_list):
                            series_key_str = sk
                            break

                    if series_key_str and series_key_str in all_trends:
                        trends = all_trends[series_key_str]

                        # Write trend.json to cohort directory (similar to metadata.json and metrics.json)
                        if cohort_dir and cohort_dir.exists():
                            try:
                                target = ctx.target or ctx.target_column or "unknown"
                                trend_analyzer.write_cohort_trend(
                                    cohort_dir=cohort_dir,
                                    stage=stage_str_for_trend,
                                    target=target,
                                    trends={series_key_str: trends}  # Pass pre-computed trends
                                )

                                # Also write across-runs timeseries to trend_reports/
                                try:
                                    # Find RESULTS directory (walk up from reproducibility_dir)
                                    results_dir = self._repro_base_dir.parent if hasattr(self, '_repro_base_dir') else None
                                    if results_dir is None:
                                        # Try to find RESULTS by walking up from cohort_dir
                                        current = Path(cohort_dir)
                                        for _ in range(10):
                                            if current.name == "RESULTS":
                                                results_dir = current
                                                break
                                            if not current.parent.exists():
                                                break
                                            current = current.parent

                                    if results_dir and results_dir.name == "RESULTS":
                                        stage_str_for_trend = str(stage_enum) if stage_enum else (ctx.stage.upper() if hasattr(ctx, 'stage') and ctx.stage else "UNKNOWN")
                                        trend_analyzer.write_across_runs_timeseries(
                                            results_dir=results_dir,
                                            target=target,
                                            stage=stage_str_for_trend,
                                            view=str(ctx.view) if hasattr(ctx, 'view') and ctx.view else View.CROSS_SECTIONAL.value
                                        )

                                        # Write run snapshot
                                        if hasattr(ctx, 'run_id') and ctx.run_id:
                                            trend_analyzer.write_run_snapshot(
                                                results_dir=results_dir,
                                                run_id=ctx.run_id,
                                                trends={series_key_str: trends}
                                            )
                                except Exception as e2:
                                    logger.debug(f"Failed to write across-runs timeseries: {e2}")
                            except Exception as e:
                                logger.debug(f"Failed to write trend.json: {e}")

                        # Find trend for the primary metric
                        primary_metric = metrics.get("metric_name", "auc")
                        if primary_metric:
                            # Try to find matching metric trend
                            for trend in trends:
                                if trend.metric_name in ["auc_mean", "auc", primary_metric.lower()]:
                                    if trend.status == "ok":
                                        trend_summary = {
                                            "slope_per_day": trend.slope_per_day,
                                            "current_estimate": trend.current_estimate,
                                            "ewma_value": trend.ewma_value,
                                            "n_runs": trend.n_runs,
                                            "residual_std": trend.residual_std,
                                            "alerts": trend.alerts
                                        }

                                        # Log trend summary
                                        slope_str = f"{trend.slope_per_day:+.6f}" if trend.slope_per_day else "N/A"
                                        logger.info(
                                            f"Trend ({trend.metric_name}): slope={slope_str}/day, "
                                            f"current={trend.current_estimate:.4f}, "
                                            f"ewma={trend.ewma_value:.4f}, "
                                            f"n={trend.n_runs} runs"
                                        )

                                        # Log alerts if any
                                        if trend.alerts:
                                            for alert in trend.alerts:
                                                if alert.get('severity') == 'warning':
                                                    logger.warning(f"  [WARN] {alert['message']}")
                                                else:
                                                    logger.info(f"  [INFO] {alert['message']}")
                                    break
        except Exception as e:
            logger.debug(f"Could not compute trend analysis: {e}")
            # Don't fail if trend analysis fails

        return {
            "audit_report": audit_report,
            "audit_report_path": str(audit_report_path) if audit_report_path else None,
            "cohort_id": cohort_id,
            "metadata_path": str(target_cohort_dir / "metadata.json") if target_cohort_dir and target_cohort_dir.exists() else None,
            "trend_summary": trend_summary
        }
