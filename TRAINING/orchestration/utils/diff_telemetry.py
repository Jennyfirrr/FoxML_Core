# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Diff Telemetry System

First-class telemetry with strict SST (Stable, Sortable, Typed) rules for tracking
changes across runs. Provides:
- Normalized snapshots for diffing
- Delta tracking (prev vs baseline)
- Comparison groups and comparability checks
- Blame assignment for drift
- Regression detection

Key principle: Only diff things that are canonically normalized and hash-addressed.
"""

import json
import logging
import hashlib
import fcntl
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd

# SST: Import View and Stage enums for consistent view/stage handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import sorted_unique, iterdir_sorted, glob_sorted, rglob_sorted

logger = logging.getLogger(__name__)

# Fingerprint schema version - increment when fingerprint computation changes
FINGERPRINT_SCHEMA_VERSION = "1.0"


# Import from modular components
from TRAINING.orchestration.utils.diff_telemetry.types import (
    ChangeSeverity,
    ComparabilityStatus,
    ResolvedRunContext,
    ComparisonGroup,
    NormalizedSnapshot,
    DiffResult,
    BaselineState
)

# Import run hash functions from submodule
from TRAINING.orchestration.utils.diff_telemetry.run_hash import (
    compute_full_run_hash,
    compute_run_hash_with_changes,
    save_run_hash,
    _can_runs_be_compared,
    _normalize_run_id_for_comparison,
    _extract_deterministic_fields,
    _load_manifest_comparability_flags,
)

# Import mixins for various methods
from TRAINING.orchestration.utils.diff_telemetry.diff_engine import DiffEngineMixin
from TRAINING.orchestration.utils.diff_telemetry.fingerprint_mixin import FingerprintMixin
from TRAINING.orchestration.utils.diff_telemetry.comparison_group_mixin import ComparisonGroupMixin
from TRAINING.orchestration.utils.diff_telemetry.normalization_mixin import NormalizationMixin
from TRAINING.orchestration.utils.diff_telemetry.digest_mixin import DigestMixin
from TRAINING.orchestration.utils.diff_telemetry.context_builder_mixin import ContextBuilderMixin

# Keep FINGERPRINT_SCHEMA_VERSION constant for backward compatibility
FINGERPRINT_SCHEMA_VERSION = "1.0"

# All dataclasses and enums are now in diff_telemetry/types.py
# Import them above (lines 52-60)

# Import atomic JSON write from centralized utilities
from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert pandas Timestamp objects and enums to JSON-serializable types.
    
    This ensures all Timestamp objects are converted to ISO format strings and enums
    are converted to their string values before writing to JSON files.
    
    DETERMINISM: Uses sorted_items() for deterministic dict iteration and normalizes keys to strings.
    
    Args:
        obj: Object to sanitize (can be dict, list, tuple, Timestamp, Enum, or other types)
    
    Returns:
        Sanitized object with all Timestamp objects converted to ISO strings and enums to strings
    """
    import pandas as pd
    from enum import Enum
    from TRAINING.common.utils.determinism_ordering import sorted_items
    
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        # Convert enum to string value for JSON serialization
        return obj.value
    elif isinstance(obj, dict):
        # DETERMINISM: Use sorted_items() for deterministic iteration + normalize keys to strings
        out = {}
        for k, v in sorted_items(obj):
            # CRITICAL: Normalize key to string (prevents TypeError on mixed-type keys)
            out[str(k)] = _sanitize_for_json(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, set):
        # DETERMINISM: Convert sets to sorted lists for deterministic serialization
        return [_sanitize_for_json(v) for v in sorted(obj, key=str)]
    else:
        return obj


def _write_atomic_json_with_lock(
    file_path: Path,
    data: Dict[str, Any],
    lock_timeout: float = 30.0
) -> None:
    """
    Write JSON file atomically with file locking to prevent race conditions.
    
    Uses fcntl.flock with LOCK_EX to ensure exclusive access during write.
    This prevents concurrent writes from multiple processes/threads.
    
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
    
    try:
        # Try to acquire lock with timeout
        with open(lock_file, 'w') as lock_f:
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
            
            # Lock is automatically released when file is closed
    except Exception as e:
        if lock_acquired:
            # Release lock on error (if we had it)
            try:
                with open(lock_file, 'w') as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        raise IOError(f"Failed to write locked JSON to {file_path}: {e}") from e


def _run_id_part(run_id: Optional[str]) -> str:
    """
    Get run_id part for string formatting (handles None).
    
    Returns:
        run_id if present, "__LEGACY__" if None/empty
    """
    if not run_id or (isinstance(run_id, str) and not run_id.strip()):
        return "__LEGACY__"
    return run_id.strip() if isinstance(run_id, str) else str(run_id)


# DiffTelemetry class definition starts here
class DiffTelemetry(DiffEngineMixin, FingerprintMixin, ComparisonGroupMixin, NormalizationMixin, DigestMixin, ContextBuilderMixin):
    """
    First-class diff telemetry system with SST rules.
    
    Tracks:
    - Normalized snapshots per run
    - Diffs against previous comparable runs
    - Diffs against baseline (regression point)
    - Comparison groups and comparability
    - Blame assignment for drift
    """
    
    def __init__(
        self,
        output_dir: Path,
        min_runs_for_baseline: int = 5,
        baseline_window_size: int = 10
    ):
        """
        Initialize diff telemetry.
        
        Args:
            output_dir: Base output directory (RESULTS/ or RESULTS/{run}/)
            min_runs_for_baseline: Minimum runs before establishing baseline
            baseline_window_size: Rolling window size for baseline
        """
        self.output_dir = Path(output_dir)
        self.min_runs_for_baseline = min_runs_for_baseline
        self.baseline_window_size = baseline_window_size
        
        # Find RESULTS directory and determine structure
        results_dir = self.output_dir
        run_dir = None
        bin_dir = None
        
        # Walk up to find RESULTS directory and identify run/bin structure
        temp_dir = self.output_dir
        for _ in range(10):  # Limit depth
            if temp_dir.name == "RESULTS":
                results_dir = temp_dir
                break
            # Check if we're in a run directory (has REPRODUCIBILITY subdirectory)
            if (temp_dir / "REPRODUCIBILITY").exists():
                run_dir = temp_dir
            # Check if we're in a sample size bin directory
            if temp_dir.name.startswith("sample_") and temp_dir.parent.name == "RESULTS":
                bin_dir = temp_dir
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        # If we couldn't find RESULTS, try to infer from output_dir
        if results_dir.name != "RESULTS":
            # Try to find RESULTS by looking for sample_* directories
            temp_dir = self.output_dir
            for _ in range(10):
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                if any((temp_dir / d).is_dir() and d.startswith("sample_") for d in iterdir_sorted(temp_dir) if d.is_dir()):
                    results_dir = temp_dir
                    break
                if not temp_dir.parent.exists():
                    break
                temp_dir = temp_dir.parent
        
        # Run-specific snapshot index: stored in run's globals/ (target-first structure)
        # Find run directory using SST helper
        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
        run_dir_for_globals = get_run_root(self.output_dir)
        
        if run_dir_for_globals:
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(run_dir_for_globals)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
        else:
            # Fallback: use output_dir/globals/ if we can't find run directory
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(self.output_dir)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
        
        # Baselines are stored per-cohort (in cohort directory), not in a global index
        # This ensures only exactly the same runs (same cohort, stage, target, etc.) share baselines
        # We'll load baselines on-demand from cohort directories when needed
        
        # Store run_dir for later use when saving baselines
        self.run_dir = run_dir if run_dir else self.output_dir
        self.results_dir = results_dir
        
        # Track if run has been moved to comparison group directory
        self._run_moved = False
        
        # Load existing indices
        self._snapshots: Dict[str, NormalizedSnapshot] = {}
        self._baselines: Dict[str, BaselineState] = {}  # Cache for current session, but saved per-cohort
    
    @staticmethod
    def parse_snapshot_key(key: str) -> Dict[str, Any]:
        """
        Parse snapshot key with tolerant handling of variable length.
        
        Handles:
        - Legacy: run_id (1 token)
        - Legacy: run_id:stage (2 tokens)
        - Legacy: run_id:stage:target:view (4 tokens)
        - Legacy: run_id:stage:target:view:symbol (5 tokens)
        - New: run_id:stage:target:view:symbol:attempt_id (6 tokens)
        
        Returns dict with all fields, defaults attempt_id=0 if missing.
        """
        parts = key.split(":")
        result = {
            'run_id': parts[0] if len(parts) > 0 else None,
            'stage': parts[1] if len(parts) > 1 else None,
            'target': parts[2] if len(parts) > 2 else None,
            'view': parts[3] if len(parts) > 3 else None,
            'symbol': parts[4] if len(parts) > 4 else None,
            'attempt_id': int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0
        }
        return result
    
    def _load_indices(self):
        """
        Load snapshot index (baselines loaded on-demand from cohort directories).
        
        Handles both old format (run_id key) and new format (run_id:stage:target:view:symbol:attempt_id) for backwards compatibility.
        """
        if self.snapshot_index.exists():
            try:
                with open(self.snapshot_index) as f:
                    data = json.load(f)
                    # DETERMINISM: Use sorted_items() for deterministic iteration order
                    # VERIFIED SAFE: Loads snapshots into dict keyed by canonical_key (no "first wins" semantics)
                    from TRAINING.common.utils.determinism_ordering import sorted_items
                    for key, snap_data in sorted_items(data):
                        snap = self._deserialize_snapshot(snap_data)
                        # Handle old formats:
                        # - run_id:stage (legacy, 1 colon)
                        # - run_id:stage:target:view (previous fix, 3 colons)
                        # - run_id:stage:target:view:symbol (current format, 4 colons)
                        # Use tolerant parser to handle all key formats
                        parsed = self.parse_snapshot_key(key)
                        # Build canonical key with all components (including attempt_id)
                        from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                        target_clean = normalize_target_name(snap.target or parsed.get('target') or "unknown")
                        view_clean = snap.view or parsed.get('view') or "UNKNOWN"
                        symbol_clean = normalize_target_name(snap.symbol or parsed.get('symbol') or "NONE")
                        attempt_id = getattr(snap, 'attempt_id', parsed.get('attempt_id', 0))
                        canonical_key = f"{_run_id_part(snap.run_id)}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}:{attempt_id}"
                        self._snapshots[canonical_key] = snap
            except Exception as e:
                logger.warning(f"Failed to load snapshot index: {e}")
        
        # Baselines are loaded on-demand from cohort directories (see _load_baseline_from_cohort)
    
    def _save_indices(self):
        """
        Save snapshot index per-run (not one mega file).
        
        CRITICAL: Uses (run_id, stage, target, view, symbol) as key to prevent overwrites.
        A single run_id can produce multiple stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING),
        and each stage can process multiple targets and symbols, so we need all five components in the key.
        
        Index is stored per-run in: {run_dir}/globals/snapshot_index.json (target-first structure)
        This keeps indices correlated by run and prevents one mega file from growing unbounded.
        
        Merges with existing snapshots in the file to prevent overwriting snapshots from other targets/symbols.
        """
        if not self.run_dir or not self.snapshot_index:
            return
        
        try:
            # Load existing snapshots from file to merge with new ones
            existing_snapshots = {}
            if self.snapshot_index.exists():
                try:
                    with open(self.snapshot_index) as f:
                        existing_data = json.load(f)
                        # Handle old formats:
                        # - run_id:stage (legacy)
                        # - run_id:stage:target:view (previous fix)
                        # - run_id:stage:target:view:symbol (current format)
                        # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
                        from TRAINING.common.utils.determinism_ordering import sorted_items
                        for key, snap_data in sorted_items(existing_data):
                            existing_snapshots[key] = snap_data
                except Exception as e:
                    logger.debug(f"Failed to load existing snapshot index for merge: {e}")
            
            # Merge existing snapshots with new ones (new ones take precedence)
            run_snapshots = existing_snapshots.copy()
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            from TRAINING.common.utils.determinism_ordering import sorted_items
            for snapshot_key, snap in sorted_items(self._snapshots):
                # Build key: run_id:stage:target:view:symbol
                from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                target_clean = normalize_target_name(snap.target or "unknown")
                view_clean = snap.view or "UNKNOWN"
                symbol_clean = normalize_target_name(snap.symbol or "NONE")
                key = f"{_run_id_part(snap.run_id)}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                run_snapshots[key] = snap.to_dict()
            
            _write_atomic_json(self.snapshot_index, _sanitize_for_json(run_snapshots))
        except Exception as e:
            logger.warning(f"Failed to save snapshot index: {e}")
    
    def _load_baseline_from_cohort(self, cohort_dir: Path, comparison_group_key: str) -> Optional[BaselineState]:
        """Load baseline from cohort directory."""
        baseline_file = Path(cohort_dir) / "baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
                    # Verify it matches the comparison group
                    if data.get('comparison_group_key') == comparison_group_key:
                        return BaselineState(**data)
            except Exception as e:
                logger.debug(f"Failed to load baseline from {baseline_file}: {e}")
        return None
    
    def _save_baseline_to_cohort(self, cohort_dir: Path, baseline: BaselineState):
        """Save baseline to cohort directory."""
        cohort_dir = Path(cohort_dir)
        cohort_dir_str = str(cohort_dir)
        # NEVER create legacy REPRODUCIBILITY directories - only use target-first structure
        # Check for uppercase REPRODUCIBILITY (legacy) but allow lowercase reproducibility (target-first)
        if "REPRODUCIBILITY" in cohort_dir_str and "reproducibility" not in cohort_dir_str.lower():
            logger.warning(f"⚠️ Skipping baseline save to legacy REPRODUCIBILITY path: {cohort_dir}")
            return
        cohort_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = cohort_dir / "baseline.json"
        try:
            _write_atomic_json(baseline_file, baseline.to_dict(), default=str)
            logger.debug(f"✅ Saved baseline.json to {baseline_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save baseline to {baseline_file}: {e}")
            import traceback
            logger.debug(f"Baseline save traceback: {traceback.format_exc()}")
    
    def _deserialize_snapshot(self, data: Dict[str, Any]) -> NormalizedSnapshot:
        """Deserialize snapshot from dict."""
        comp_group = None
        if 'comparison_group' in data:
            comp_group_data = data['comparison_group']
            # Handle backward compatibility: old snapshots might not have new fields
            if 'n_effective' not in comp_group_data:
                comp_group_data['n_effective'] = None
            if 'model_family' not in comp_group_data:
                comp_group_data['model_family'] = None
            if 'feature_signature' not in comp_group_data:
                comp_group_data['feature_signature'] = None
            comp_group = ComparisonGroup(**comp_group_data)
        
        return NormalizedSnapshot(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            stage=data['stage'],
            view=data.get('view'),
            target=data.get('target'),
            symbol=data.get('symbol'),
            experiment_id=data.get('experiment_id'),  # May be None for old snapshots
            snapshot_seq=data.get('snapshot_seq'),  # May be None for old snapshots
            attempt_id=data.get('attempt_id', 0),  # Default to 0 for legacy snapshots
            fingerprint_schema_version=data.get('fingerprint_schema_version', '1.0'),  # Default for old snapshots
            metrics_schema_version=data.get('metrics_schema_version', '1.0'),  # Default to 1.0 for old snapshots
            scoring_schema_version=data.get('scoring_schema_version', '1.0'),  # Default to 1.0 for old snapshots
            config_fingerprint=data.get('config_fingerprint'),
            deterministic_config_fingerprint=data.get('deterministic_config_fingerprint'),
            data_fingerprint=data.get('data_fingerprint'),
            feature_fingerprint=data.get('feature_fingerprint'),
            target_fingerprint=data.get('target_fingerprint'),
            metrics_sha256=data.get('metrics_sha256'),  # May be None for old snapshots
            artifacts_manifest_sha256=data.get('artifacts_manifest_sha256'),  # May be None for old snapshots
            predictions_sha256=data.get('predictions_sha256'),  # May be None for old snapshots
            fingerprint_sources=data.get('fingerprint_sources', {}),
            inputs=data.get('inputs', {}),
            process=data.get('process', {}),
            outputs=data.get('outputs', {}),
            comparison_group=comp_group
        )

    # Methods extracted to ContextBuilderMixin:
    # - _build_resolved_context
    # - _get_required_fields_for_stage
    # - _validate_stage_schema

    def normalize_snapshot(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint dict
    ) -> NormalizedSnapshot:
        """
        Create normalized snapshot from run data.
        
        CRITICAL: This now uses ResolvedRunContext to ensure all required fields are
        non-null. Missing required fields will cause validation failure.
        
        For SST consistency, prefer `resolved_metadata` (in-memory metadata dict) over
        reading from filesystem. This ensures snapshot computation uses the exact same
        data that will be persisted.
        
        Args:
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
            run_data: Run data dict (from reproducibility tracker)
            cohort_metadata: Cohort metadata (fallback)
            additional_data: Additional data dict
            cohort_dir: Cohort directory (fallback - only used if resolved_metadata not provided)
            resolved_metadata: In-memory metadata dict (SST - preferred source)
            run_identity: RunIdentity SST object with authoritative signatures
            prediction_fingerprint: Prediction fingerprint dict for predictions_sha256
        
        Returns:
            NormalizedSnapshot
        
        Raises:
            ValueError: If required fields for stage are missing
        """
        # Build resolved context from all available sources (prefer resolved_metadata for SST)
        ctx = self._build_resolved_context(
            stage, run_data, cohort_metadata, additional_data, cohort_dir, resolved_metadata
        )
        
        # Validate stage-specific schema
        is_valid, reason = self._validate_stage_schema(stage, ctx)
        if not is_valid:
            raise ValueError(f"Cannot create snapshot for {stage}: {reason}")
        
        # Extract core identifiers
        run_id = run_data.get('run_id') or run_data.get('timestamp', datetime.now().isoformat())
        timestamp = run_data.get('timestamp', datetime.now().isoformat())
        # SST: Normalize view to string (handle enum inputs)
        view_raw = ctx.view
        view = view_raw.value if isinstance(view_raw, View) else (view_raw if isinstance(view_raw, str) else str(view_raw))
        target = ctx.target
        symbol = additional_data.get('symbol') if additional_data else None
        
        # Extract universe_sig for comparison scoping (CRITICAL for CS runs)
        # Check top-level first, then cs_config for backward compatibility
        universe_sig = None
        if additional_data:
            universe_sig = additional_data.get('universe_sig')
            if not universe_sig and 'cs_config' in additional_data:
                universe_sig = additional_data['cs_config'].get('universe_sig')
        
        # Build fingerprints (using resolved context)
        # Extract both deterministic and full config fingerprints
        config_fp_result = self._compute_config_fingerprint_from_context(ctx, additional_data, resolved_metadata, cohort_dir)
        
        # Initialize variables to handle all code paths
        deterministic_config_fp = None
        full_config_fp = None
        config_fp = None
        
        if isinstance(config_fp_result, dict):
            # New format: returns dict with both fingerprints
            deterministic_config_fp = config_fp_result.get('deterministic_config_fingerprint')
            full_config_fp = config_fp_result.get('config_fingerprint')
            # Use deterministic for comparison, fallback to full if deterministic not available
            config_fp = deterministic_config_fp or full_config_fp
        else:
            # Legacy format: single fingerprint
            config_fp = config_fp_result
            full_config_fp = config_fp_result
            deterministic_config_fp = config_fp_result  # Legacy: same fingerprint for both
        
        data_fp = self._compute_data_fingerprint_from_context(ctx)
        feature_fp = self._compute_feature_fingerprint_from_context(ctx)
        target_fp = self._compute_target_fingerprint_from_context(ctx)
        
        # Store fingerprints in context for comparison group
        ctx.data_fingerprint = data_fp
        ctx.target_fingerprint = target_fp
        ctx.feature_fingerprint = feature_fp
        
        # Extract hyperparameters and train_seed from process data (before normalization)
        # We need these for the comparison group, so extract them early
        hyperparameters_signature = None
        train_seed = None
        
        # Try to get hyperparameters from resolved_metadata (metadata.json), additional_data, or run_data
        training_data = {}
        if resolved_metadata and 'training' in resolved_metadata:
            training_data = resolved_metadata['training']
        elif additional_data and 'training' in additional_data:
            training_data = additional_data['training']
        elif run_data.get('additional_data') and 'training' in run_data.get('additional_data', {}):
            training_data = run_data['additional_data']['training']
        elif run_data.get('training'):
            training_data = run_data['training']
        
        # Extract hyperparameters (exclude model_family, strategy, seeds - those are handled separately)
        hyperparameters = {}
        excluded_keys = {'model_family', 'strategy', 'split_seed', 'train_seed', 'seed'}
        for key, value in training_data.items():
            if key not in excluded_keys and value is not None:
                hyperparameters[key] = value
        
        # Compute hyperparameters signature if we have any
        if hyperparameters:
            # Sort keys for stable hash
            hp_str = "|".join(f"{k}={v}" for k, v in sorted(hyperparameters.items()))
            hyperparameters_signature = hashlib.sha256(hp_str.encode()).hexdigest()[:16]
        
        # Extract train_seed
        train_seed = (
            training_data.get('train_seed') or
            training_data.get('seed') or
            (additional_data.get('train_seed') if additional_data else None) or
            (additional_data.get('seed') if additional_data else None) or
            (run_data.get('train_seed')) or
            (run_data.get('seed'))
        )
        if train_seed is not None:
            try:
                train_seed = int(train_seed)
            except (ValueError, TypeError):
                train_seed = None
        
        # Extract library versions and compute signature (CRITICAL: different versions = different outcomes)
        library_versions_signature = None
        library_versions = ctx.library_versions
        if not library_versions:
            # Try to get from resolved_metadata (metadata.json), additional_data, or run_data
            if resolved_metadata and 'environment' in resolved_metadata and 'library_versions' in resolved_metadata['environment']:
                library_versions = resolved_metadata['environment']['library_versions']
            elif additional_data and 'library_versions' in additional_data:
                library_versions = additional_data['library_versions']
            elif run_data.get('additional_data') and 'library_versions' in run_data.get('additional_data', {}):
                library_versions = run_data['additional_data']['library_versions']
            elif run_data.get('library_versions'):
                library_versions = run_data['library_versions']
        
        if library_versions and isinstance(library_versions, dict):
            # Sort keys for stable hash, include python_version if available
            lib_parts = []
            if ctx.python_version:
                lib_parts.append(f"python={ctx.python_version}")
            # Sort library versions for stable hash
            for key in sorted(library_versions.keys()):
                lib_parts.append(f"{key}={library_versions[key]}")
            if lib_parts:
                lib_str = "|".join(lib_parts)
                library_versions_signature = hashlib.sha256(lib_str.encode()).hexdigest()[:16]
        
        # Build comparison group (using resolved context, stage-aware)
        # CRITICAL: Pass symbol and universe_sig for proper comparison scoping
        # NEW: Pass run_identity for authoritative signatures
        comparison_group = self._build_comparison_group_from_context(
            stage, ctx, config_fp, data_fp, target_fp, hyperparameters_signature, train_seed, library_versions_signature,
            symbol=symbol, universe_sig=universe_sig, run_identity=run_identity
        )
        
        # Normalize inputs (using resolved context - no nulls for required fields)
        inputs = self._normalize_inputs_from_context(stage, ctx)
        
        # Normalize process (using resolved context)
        process = self._normalize_process_from_context(ctx)
        
        # Normalize outputs
        outputs = self._normalize_outputs(run_data, additional_data, cohort_dir, resolved_metadata)
        
        # CRITICAL: Compute output digests for artifact/metric reproducibility verification
        # These enable comparison of outputs across reruns for reproducibility tracking
        metrics_sha256 = self._compute_metrics_digest(outputs, resolved_metadata, cohort_dir)
        
        # CRITICAL: Validate that metrics exist for stages that require them
        # Only error if ALL paths failed (outputs, resolved_metadata, cohort_dir files, reference pointers, legacy)
        # If metrics were found via any path, metrics_sha256 will be set and this check won't fire
        stages_requiring_metrics = ["TARGET_RANKING", "FEATURE_SELECTION"]
        if stage in stages_requiring_metrics and not metrics_sha256:
            # Check if metrics exist in outputs or resolved_metadata (even if digest computation failed)
            has_metrics_in_outputs = bool(outputs.get('metrics'))
            has_metrics_in_resolved = bool(resolved_metadata and resolved_metadata.get('metrics'))
            has_metrics_files = False
            if cohort_dir:
                cohort_path = Path(cohort_dir)
                has_metrics_files = (cohort_path / "metrics.json").exists() or (cohort_path / "metrics.parquet").exists()
            
            # Only error if truly no metrics found anywhere
            if not has_metrics_in_outputs and not has_metrics_in_resolved and not has_metrics_files:
                logger.error(
                    f"❌ CRITICAL: metrics_sha256 cannot be computed for {stage} stage "
                    f"(target={target}, view={view}). No metrics found in outputs, resolved_metadata, or cohort_dir. "
                    f"Metrics are required for reproducibility verification. "
                    f"Snapshot will be saved but reproducibility verification will be incomplete."
                )
            else:
                # Metrics exist but digest computation failed - this is a bug, log as warning
                logger.warning(
                    f"⚠️  metrics_sha256 cannot be computed for {stage} stage "
                    f"(target={target}, view={view}) despite metrics being available. "
                    f"This may indicate a bug in _compute_metrics_digest(). "
                    f"outputs.has_metrics={has_metrics_in_outputs}, "
                    f"resolved_metadata.has_metrics={has_metrics_in_resolved}, "
                    f"cohort_dir.has_files={has_metrics_files}"
                )
            # Don't fail snapshot creation, but log for visibility
        
        artifacts_manifest_sha256 = self._compute_artifacts_manifest_digest(cohort_dir, stage)
        # NEW: Pass prediction_fingerprint for authoritative prediction hash
        predictions_sha256 = self._compute_predictions_digest(cohort_dir, stage, prediction_fingerprint)
        
        # Build fingerprint source descriptions (for auditability)
        # These document what each fingerprint represents for reproducibility tracking
        fingerprint_sources = {}
        
        # Document config fingerprint source
        if config_fp:
            fingerprint_sources['config_fingerprint'] = (
                "hash of pipeline.determinism and safety.output_layout config sections"
            )
        
        # Document data fingerprint source
        if data_fp:
            fingerprint_sources['data_fingerprint'] = (
                "hash of n_samples, symbols list, and date_range (start/end)"
            )
        
        # Document feature fingerprint source
        if feature_fp:
            fingerprint_sources['feature_fingerprint'] = (
                "hash of sorted feature spec list resolved from registry"
            )
        
        # Document target fingerprint source
        if target_fp:
            fingerprint_sources['target_fingerprint'] = (
                "hash of target name, view, horizon_minutes, and labeling_impl_hash"
            )
        
        # Document fold assignment hash source (if available)
        if ctx.fold_assignment_hash:
            fingerprint_sources['fold_assignment_hash'] = (
                (additional_data.get('fold_assignment_hash_source') if additional_data else None) or
                "hash over row_id→fold_id mapping"
            )
        
        # Extract schema versions from metrics (for full parity with TrainingSnapshot/FeatureSelectionSnapshot)
        # SST: Default to current versions (matching get_scoring_schema_version() which returns "1.2")
        metrics_schema_version = "1.1"  # Default to current version
        scoring_schema_version = "1.2"  # Default to current version (matches get_scoring_schema_version())
        
        # Try to extract from outputs.metrics first (most common)
        if outputs.get('metrics'):
            # Check nested schema.scoring path (current structure from build_clean_metrics_dict)
            if 'schema' in outputs['metrics'] and isinstance(outputs['metrics']['schema'], dict):
                if 'scoring' in outputs['metrics']['schema']:
                    scoring_schema_version = outputs['metrics']['schema']['scoring']
                if 'metrics' in outputs['metrics']['schema']:
                    metrics_schema_version = outputs['metrics']['schema']['metrics']
            # Fallback to top-level for backward compatibility
            if 'metrics_schema_version' in outputs['metrics']:
                metrics_schema_version = outputs['metrics']['metrics_schema_version']
            if 'scoring_schema_version' in outputs['metrics']:
                scoring_schema_version = outputs['metrics']['scoring_schema_version']
        
        # Fallback to run_data metrics
        if metrics_schema_version == "1.1" and run_data.get('metrics'):
            # Check nested schema path first
            if 'schema' in run_data['metrics'] and isinstance(run_data['metrics']['schema'], dict):
                if 'scoring' in run_data['metrics']['schema']:
                    scoring_schema_version = run_data['metrics']['schema']['scoring']
                if 'metrics' in run_data['metrics']['schema']:
                    metrics_schema_version = run_data['metrics']['schema']['metrics']
            # Fallback to top-level
            if 'metrics_schema_version' in run_data['metrics']:
                metrics_schema_version = run_data['metrics']['metrics_schema_version']
            if 'scoring_schema_version' in run_data['metrics']:
                scoring_schema_version = run_data['metrics']['scoring_schema_version']
        
        # Fallback to additional_data metrics
        if metrics_schema_version == "1.1" and additional_data and additional_data.get('metrics'):
            # Check nested schema path first
            if 'schema' in additional_data['metrics'] and isinstance(additional_data['metrics']['schema'], dict):
                if 'scoring' in additional_data['metrics']['schema']:
                    scoring_schema_version = additional_data['metrics']['schema']['scoring']
                if 'metrics' in additional_data['metrics']['schema']:
                    metrics_schema_version = additional_data['metrics']['schema']['metrics']
            # Fallback to top-level
            if 'metrics_schema_version' in additional_data['metrics']:
                metrics_schema_version = additional_data['metrics']['metrics_schema_version']
            if 'scoring_schema_version' in additional_data['metrics']:
                scoring_schema_version = additional_data['metrics']['scoring_schema_version']
        
        # Ensure string type and valid defaults
        if not isinstance(metrics_schema_version, str):
            metrics_schema_version = "1.1"
        if not isinstance(scoring_schema_version, str):
            scoring_schema_version = "1.2"  # Default to current version
        
        # SST: Normalize stage to string (handle enum inputs)
        stage_str = stage.value if isinstance(stage, Stage) else (stage if isinstance(stage, str) else str(stage))
        
        # Extract attempt_id from additional_data (defaults to 0)
        attempt_id = (additional_data.get('attempt_id') if additional_data else None) or 0
        
        # Extract experiment_id from additional_data or comparison_group
        experiment_id = None
        if additional_data and 'experiment_id' in additional_data:
            experiment_id = additional_data['experiment_id']
        elif comparison_group and comparison_group.experiment_id:
            experiment_id = comparison_group.experiment_id
        
        return NormalizedSnapshot(
            run_id=run_id,
            timestamp=timestamp,
            stage=stage_str,  # Normalized to string
            view=view,  # Already normalized above
            target=target,
            symbol=symbol,
            experiment_id=experiment_id,  # Extract from additional_data or comparison_group
            attempt_id=attempt_id,  # Extract from additional_data
            config_fingerprint=full_config_fp,  # Full fingerprint (includes run_id/timestamp) - for metadata
            deterministic_config_fingerprint=deterministic_config_fp,  # Deterministic fingerprint (excludes run_id/timestamp) - for comparison
            data_fingerprint=data_fp,
            feature_fingerprint=feature_fp,
            target_fingerprint=target_fp,
            metrics_sha256=metrics_sha256,
            artifacts_manifest_sha256=artifacts_manifest_sha256,
            predictions_sha256=predictions_sha256,
            fingerprint_sources=fingerprint_sources,
            inputs=inputs,
            process=process,
            outputs=outputs,
            comparison_group=comparison_group,
            metrics_schema_version=metrics_schema_version,
            scoring_schema_version=scoring_schema_version
        )
    
    
    def _normalize_value_for_hash(self, val: Any) -> Any:
        """Normalize value for hashing (round floats, sort lists, etc.)."""
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 6)
        elif isinstance(val, (list, tuple)):
            try:
                return sorted([self._normalize_value_for_hash(v) for v in val])
            except TypeError:
                return [self._normalize_value_for_hash(v) for v in val]
        elif isinstance(val, dict):
            # DETERMINISM: Use sorted_items() for consistency with other normalization functions
            from TRAINING.common.utils.determinism_ordering import sorted_items
            return {k: self._normalize_value_for_hash(v) for k, v in sorted_items(val)}
        else:
            return val
    
    def save_snapshot(
        self,
        snapshot: NormalizedSnapshot,
        cohort_dir: Path
    ) -> Path:
        """
        Save normalized snapshot to cohort directory.

        Automatically organizes run by comparison group metadata on first snapshot.

        Args:
            snapshot: Normalized snapshot
            cohort_dir: Cohort directory (where metadata.json lives)

        Returns:
            Path: The resolved cohort directory where snapshot was actually saved
                  (may differ from input if path validation/repair was needed)
        """
        cohort_dir = Path(cohort_dir)
        
        # CRITICAL: Ensure we only write to target-first structure
        # If cohort_dir is in legacy REPRODUCIBILITY structure, find/create target-first equivalent
        # Also handle paths that are already in target-first format (reproducibility/...)
        target_cohort_dir = cohort_dir
        cohort_dir_str = str(cohort_dir)
        is_legacy_path = "REPRODUCIBILITY" in cohort_dir_str
        is_target_first_path = "reproducibility" in cohort_dir_str.lower() and not is_legacy_path
        
        # If path is already in target-first format, use it directly
        if is_target_first_path:
            # Path is already correct, just ensure it exists
            target_cohort_dir = Path(cohort_dir)
            target_cohort_dir.mkdir(parents=True, exist_ok=True)
        elif is_legacy_path:
            # Extract identifiers from cohort_dir path and create target-first path
            try:
                parts = Path(cohort_dir).parts
                stage = None
                view = None
                target = None
                cohort_id = None
                symbol_for_target = None
                
                for i, part in enumerate(parts):
                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING']:
                        stage = part
                        if i + 1 < len(parts) and parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                            view = parts[i+1]
                            if i + 2 < len(parts) and not parts[i+2].startswith('cohort='):
                                target = parts[i+2]
                        # Find cohort_id and symbol
                        for j in range(i, len(parts)):
                            if parts[j].startswith('cohort='):
                                cohort_id = parts[j].replace('cohort=', '')
                            elif parts[j].startswith('symbol='):
                                symbol_for_target = parts[j].replace('symbol=', '')
                            elif parts[j].startswith('model_family='):
                                # Skip model_family in path
                                continue
                
                # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
                if stage in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target and cohort_id:
                    # Find base output directory (run directory) using SST helper
                    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                    base_output_dir = get_run_root(cohort_dir)
                    
                    if base_output_dir:
                        from TRAINING.orchestration.utils.target_first_paths import (
                            get_target_reproducibility_dir, ensure_target_structure
                        )
                        
                        # CRITICAL FIX: Prefer snapshot's view/symbol over path parsing
                        # This fixes CROSS_SECTIONAL vs SYMBOL_SPECIFIC path organization bug
                        snapshot_view = getattr(snapshot, 'view', None)
                        snapshot_symbol = getattr(snapshot, 'symbol', None)
                        
                        # Use snapshot's view if available, otherwise fallback to path-parsed view
                        view_for_target = snapshot_view if snapshot_view else view
                        
                        # Normalize view for FEATURE_SELECTION (INDIVIDUAL -> SYMBOL_SPECIFIC)
                        if stage == 'FEATURE_SELECTION' and view_for_target == 'INDIVIDUAL':
                            view_for_target = View.SYMBOL_SPECIFIC.value
                        
                        # Normalize view to enum for comparison
                        view_for_target_enum = View.from_string(view_for_target) if isinstance(view_for_target, str) else view_for_target
                        # Convert to string for path construction
                        view_for_target_str = view_for_target_enum.value if isinstance(view_for_target_enum, View) else str(view_for_target_enum)
                        
                        # Use snapshot's symbol if available, otherwise fallback to path-parsed symbol
                        symbol_for_target_final = snapshot_symbol if snapshot_symbol else symbol_for_target
                        
                        # Ensure target structure exists
                        ensure_target_structure(base_output_dir, target)
                        
                        # Extract attempt_id from snapshot (defaults to 0)
                        attempt_id = getattr(snapshot, 'attempt_id', 0)
                        
                        # Extract universe_sig for CROSS_SECTIONAL batch_ level
                        # Try snapshot.comparison_group first, then parse from cohort_dir path
                        universe_sig = None
                        if snapshot.comparison_group and hasattr(snapshot.comparison_group, 'universe_sig'):
                            universe_sig = snapshot.comparison_group.universe_sig
                        if not universe_sig:
                            # Fallback: parse from cohort_dir path
                            from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                            parsed = parse_reproducibility_path(cohort_dir)
                            universe_sig = parsed.get('universe_sig')
                        
                        # Use single path builder (SST) - always includes attempt_{attempt_id}/ including attempt_0
                        from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                        target_cohort_dir = build_target_cohort_dir(
                            base_output_dir=base_output_dir,
                            target=target,
                            stage=stage,
                            view=view_for_target_str,
                            cohort_id=cohort_id,
                            symbol=symbol_for_target_final,
                            attempt_id=attempt_id,  # Always include, even if 0
                            universe_sig=universe_sig  # Required for CROSS_SECTIONAL batch_ level
                        )
                        target_cohort_dir.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"✅ Created target-first cohort directory for snapshot: {target_cohort_dir}")
            except Exception as e:
                logger.warning(f"Failed to create target-first structure for snapshot: {e}")
                # Don't fall back to legacy path - fail instead to prevent REPRODUCIBILITY creation
                target_cohort_dir = None
        
        # Use target-first directory for all writes - NEVER use legacy REPRODUCIBILITY paths
        if target_cohort_dir is None:
            logger.error(f"❌ Cannot save snapshot: failed to create target-first structure and legacy paths are not allowed")
            return
        
        cohort_dir = target_cohort_dir
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # NOTE: Run organization by comparison group is now done at startup (config load time)
        # in IntelligentTrainer.__init__(). This ensures runs are organized by metadata from the start.
        # We no longer move runs here - they're already in the correct location.
        # Keeping this as a fallback for edge cases where startup organization didn't happen.
        if not self._run_moved and snapshot.comparison_group and self.run_dir and self.results_dir:
            # Only organize if run is in a sample size bin or _pending (not already organized)
            run_parent = self.run_dir.parent.name
            if run_parent.startswith("sample_") or run_parent == "_pending":
                self._organize_run_by_comparison_group(snapshot)
            else:
                # Already organized, mark as moved
                self._run_moved = True
        
        # Assign monotonic sequence number for correct ordering (concurrency-safe)
        # This ensures "prev run" selection is correct regardless of mtime/timestamp quirks
        # CRITICAL: Must be done under lock to prevent two concurrent writers from picking same seq
        if snapshot.snapshot_seq is None:
            # Use cohort-level lock file for sequence assignment
            cohort_lock_file = cohort_dir / ".snapshot_seq.lock"
            
            with open(cohort_lock_file, 'w') as lock_f:
                try:
                    # Acquire exclusive lock (blocks until available)
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    
                    # Re-read snapshots to get latest sequence (another process may have updated it)
                    # Load from run-level snapshot index to get all snapshots for this run
                    run_snapshot_index = self.snapshot_index
                    if run_snapshot_index and run_snapshot_index.exists():
                        try:
                            with open(run_snapshot_index) as f:
                                index_data = json.load(f)
                                for key, snap_data in index_data.items():
                                    snap = self._deserialize_snapshot(snap_data)
                                    # Handle old formats:
                                    # - run_id:stage (legacy, 1 colon)
                                    # - run_id:stage:target:view (previous fix, 3 colons)
                                    # - run_id:stage:target:view:symbol (current format, 4 colons)
                                    # Use tolerant parser to handle all key formats
                                    parsed = self.parse_snapshot_key(key)
                                    # Build canonical key with all components (including attempt_id)
                                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                                    target_clean = normalize_target_name(snap.target or parsed.get('target') or "unknown")
                                    view_clean = snap.view or parsed.get('view') or "UNKNOWN"
                                    symbol_clean = normalize_target_name(snap.symbol or parsed.get('symbol') or "NONE")
                                    attempt_id = getattr(snap, 'attempt_id', parsed.get('attempt_id', 0))
                                    canonical_key = f"{_run_id_part(snap.run_id)}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}:{attempt_id}"
                                    if canonical_key not in self._snapshots:
                                        self._snapshots[canonical_key] = snap
                        except Exception:
                            pass
                    
                    # Get next sequence number (max existing + 1, or 1 if none)
                    max_seq = 0
                    for snap in self._snapshots.values():
                        if snap.snapshot_seq and snap.snapshot_seq > max_seq:
                            max_seq = snap.snapshot_seq
                    snapshot.snapshot_seq = max_seq + 1
                    
                    # Lock is automatically released when file is closed
                except Exception as e:
                    # Release lock on error
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    raise
        
        # Save full snapshot atomically
        snapshot_file = cohort_dir / "snapshot.json"
        try:
            snapshot_dict = snapshot.to_dict()
            
            # CRITICAL: Extract from snapshot_dict (serialized dict), not snapshot.outputs (dataclass field)
            # This ensures we work with the exact shape that will be written to JSON
            # SEMANTICS: These signatures affect run identity hashing (compute_full_run_hash),
            # NOT comparison group eligibility. They describe outputs, not inputs.
            outputs = snapshot_dict.get("outputs")
            if isinstance(outputs, dict):
                metrics = outputs.get("metrics")
                if isinstance(metrics, dict):
                    # Extract scoring_signature from nested outputs.metrics.score.signature
                    score = metrics.get("score")
                    scoring_sig = None
                    if isinstance(score, dict):
                        scoring_sig = score.get("signature")
                    
                    # Extract selection_signature from nested outputs.metrics.selection.signature (for FEATURE_SELECTION)
                    selection = metrics.get("selection")
                    selection_sig = None
                    if isinstance(selection, dict):
                        selection_sig = selection.get("signature")
                    
            # Add top-level fields for compute_full_run_hash() compatibility
            # Use is not None (not truthiness) to handle empty strings correctly
            if scoring_sig is not None:
                snapshot_dict['scoring_signature'] = scoring_sig
            if selection_sig is not None:
                snapshot_dict['selection_signature'] = selection_sig
            
            # CRITICAL: Normalize library_versions_signature null vs missing
            # Only omit library_versions_signature if None (other nulls like symbol:null for CROSS_SECTIONAL are valid)
            # This ensures null vs missing hash identically in compute_full_run_hash()
            if 'comparison_group' in snapshot_dict and isinstance(snapshot_dict['comparison_group'], dict):
                cg = snapshot_dict['comparison_group']
                # CRITICAL: Check if key exists AND is None (not just if get() returns None)
                # This distinguishes between missing key (get returns None) and key with None value
                if 'library_versions_signature' in cg and cg['library_versions_signature'] is None:
                    # Omit key if it exists with None value (both missing and null hash identically)
                    del cg['library_versions_signature']
            
            # Also normalize top-level library_versions_signature if present
            if 'library_versions_signature' in snapshot_dict and snapshot_dict['library_versions_signature'] is None:
                del snapshot_dict['library_versions_signature']
            
            _write_atomic_json_with_lock(snapshot_file, snapshot_dict)
            logger.debug(f"✅ Saved snapshot.json to {snapshot_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save snapshot.json to {snapshot_file}: {e}")
            import traceback
            logger.debug(f"Snapshot save traceback: {traceback.format_exc()}")
            raise  # Re-raise to prevent silent failure
        
        # Also write to target-first structure (for TARGET_RANKING and FEATURE_SELECTION stages)
        # NOTE: This is now redundant since we already write to target-first structure above,
        # but keeping for backward compatibility and to ensure snapshot.json is written
        try:
            # Extract identifiers from snapshot (preferred) or cohort_dir path
            # Prefer snapshot's stage/view/target over path parsing
            stage = snapshot.stage
            view = getattr(snapshot, 'view', None)
            target = snapshot.target
            cohort_id = None
            
            # Extract cohort_id from path if not available from snapshot
            parts = Path(cohort_dir).parts
            for part in parts:
                if part.startswith('cohort='):
                    cohort_id = part.replace('cohort=', '')
                    break
            
            # If we couldn't get identifiers from snapshot, try parsing path (legacy support)
            if not stage or not target:
                for i, part in enumerate(parts):
                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING']:
                        if not stage:
                            stage = part
                        if i + 1 < len(parts) and parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                            if not view:
                                view = parts[i+1]
                            if i + 2 < len(parts) and not parts[i+2].startswith('cohort='):
                                if not target:
                                    target = parts[i+2]
                        # Find cohort_id
                        for j in range(i, len(parts)):
                            if parts[j].startswith('cohort='):
                                cohort_id = parts[j].replace('cohort=', '')
                                break
                        break
            
            # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
            if stage in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target and cohort_id:
                # Find base output directory (run directory)
                temp_dir = cohort_dir
                for _ in range(10):  # Limit depth
                    if (temp_dir / "targets").exists() or (temp_dir.parent / "targets").exists():
                        # Found run directory
                        if (temp_dir / "targets").exists():
                            base_output_dir = temp_dir
                        else:
                            base_output_dir = temp_dir.parent
                        break
                    if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                        break
                    temp_dir = temp_dir.parent
                
                if base_output_dir:
                    from TRAINING.orchestration.utils.target_first_paths import (
                        get_target_reproducibility_dir, ensure_target_structure
                    )
                    
                    # CRITICAL FIX: Prefer snapshot's view/symbol over path parsing
                    snapshot_view = getattr(snapshot, 'view', None)
                    snapshot_symbol = getattr(snapshot, 'symbol', None)
                    
                    # Use snapshot's view if available, otherwise fallback to path-parsed view
                    view_for_target = snapshot_view if snapshot_view else view
                    
                    # Normalize view for FEATURE_SELECTION (INDIVIDUAL -> SYMBOL_SPECIFIC)
                    if stage == 'FEATURE_SELECTION' and view_for_target == 'INDIVIDUAL':
                        view_for_target = 'SYMBOL_SPECIFIC'
                    
                    # Use snapshot's symbol if available
                    symbol_for_target = snapshot_symbol
                    
                    # Ensure target structure exists
                    ensure_target_structure(base_output_dir, target)
                    
                    # Build target-first reproducibility path with stage scoping
                    target_repro_dir = get_target_reproducibility_dir(base_output_dir, target, stage=stage)
                    # CRITICAL: Use canonical path builder to ensure cohorts are in correct structure
                    # Extract attempt_id and universe_sig from snapshot
                    attempt_id = getattr(snapshot, 'attempt_id', 0)
                    universe_sig = None
                    if snapshot.comparison_group and hasattr(snapshot.comparison_group, 'universe_sig'):
                        universe_sig = snapshot.comparison_group.universe_sig
                    if not universe_sig:
                        # Fallback: parse from cohort_dir path
                        from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                        parsed = parse_reproducibility_path(cohort_dir)
                        universe_sig = parsed.get('universe_sig')
                    
                    # CRITICAL FIX: Ensure symbol is included in path for SYMBOL_SPECIFIC view
                    # Check symbol_for_target FIRST to prevent symbol-specific data going to CROSS_SECTIONAL
                    if symbol_for_target:
                        # If symbol is present, force SYMBOL_SPECIFIC view
                        view_for_target = "SYMBOL_SPECIFIC"
                    elif view_for_target == "SYMBOL_SPECIFIC":
                        # SYMBOL_SPECIFIC view but no symbol - this is an error condition
                        logger.warning(f"SYMBOL_SPECIFIC view but no symbol for snapshot {target}, using CROSS_SECTIONAL path")
                        view_for_target = "CROSS_SECTIONAL"
                    
                    # Use canonical path builder
                    from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                    target_cohort_dir = build_target_cohort_dir(
                        base_output_dir=base_output_dir,
                        target=target,
                        stage=stage,
                        view=view_for_target,
                        cohort_id=cohort_id,
                        symbol=symbol_for_target,
                        attempt_id=attempt_id,
                        universe_sig=universe_sig
                    )
                    target_cohort_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Write snapshot.json to target-first structure
                    target_snapshot_file = target_cohort_dir / "snapshot.json"
                    _write_atomic_json_with_lock(target_snapshot_file, snapshot_dict)
                    logger.debug(f"✅ Also saved snapshot.json to target-first structure")
        except Exception as e:
            logger.debug(f"Failed to save snapshot.json to target-first structure (non-critical): {e}")
        
        # Update index (keyed by run_id:stage:target:view:symbol:attempt_id for uniqueness)
        # Include target, view, symbol, and attempt_id in key to prevent overwrites
        from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
        target_clean = normalize_target_name(snapshot.target or "unknown")
        view_clean = snapshot.view or "UNKNOWN"
        symbol_clean = normalize_target_name(snapshot.symbol or "NONE")
        attempt_id = getattr(snapshot, 'attempt_id', 0)
        snapshot_key = f"{_run_id_part(snapshot.run_id)}:{snapshot.stage}:{target_clean}:{view_clean}:{symbol_clean}:{attempt_id}"
        self._snapshots[snapshot_key] = snapshot
        self._save_indices()

        logger.debug(f"✅ Saved snapshot to {snapshot_file}")

        # Return the resolved cohort_dir so caller can use it for subsequent operations
        # This ensures save_diff uses the same path as save_snapshot
        return cohort_dir

    def _organize_run_by_comparison_group(self, snapshot: NormalizedSnapshot) -> None:
        """
        Move run directory to comparison group-based organization.
        
        Structure: RESULTS/{comparison_group_dir}/{run_name}/
        
        This ensures runs with exactly the same outcome-influencing metadata
        are stored together for human auditability.
        
        Args:
            snapshot: First snapshot with comparison group metadata
        """
        if not snapshot.comparison_group:
            return
        
        try:
            # Determine target directory based on comparison group
            # Structure: RESULTS/runs/{comparison_group_dir}/{run_name}/
            comparison_group_dir = snapshot.comparison_group.to_dir_name()
            runs_dir = self.results_dir / "runs"
            target_dir = runs_dir / comparison_group_dir / self.run_dir.name
            
            # Skip if already in correct location
            if self.run_dir.resolve() == target_dir.resolve():
                self._run_moved = True
                return
            
            # Skip if target already exists (another run with same comparison group)
            if target_dir.exists():
                logger.warning(f"⚠️  Target directory already exists: {target_dir}")
                logger.warning(f"   Run will remain in current location: {self.run_dir}")
                self._run_moved = True  # Mark as moved to prevent retry
                return
            
            # Create target directory parent
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the entire run directory
            import shutil
            logger.info(f"📁 Organizing run by comparison group metadata...")
            logger.info(f"   From: {self.run_dir}")
            logger.info(f"   To:   {target_dir}")
            group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
            if group_key:
                logger.info(f"   Group: {group_key}")
            else:
                logger.warning(f"   Group: <invalid - missing required fields>")
            
            shutil.move(str(self.run_dir), str(target_dir))
            
            # Update internal references
            self.run_dir = target_dir
            
            # Use target-first structure (globals/) instead of legacy REPRODUCIBILITY/
            # REPRODUCIBILITY should only exist within run directories, not at RESULTS root
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(target_dir)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
            
            # Update output_dir if it was pointing to run_dir
            if self.output_dir.resolve() == self.run_dir.resolve():
                self.output_dir = target_dir
            
            self._run_moved = True
            logger.info(f"✅ Run organized by comparison group metadata")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to organize run by comparison group: {e}")
            logger.warning(f"   Run will remain in current location: {self.run_dir}")
            # Mark as moved to prevent retry loops
            self._run_moved = True

    def find_previous_comparable(
        self,
        snapshot: NormalizedSnapshot
    ) -> Optional[NormalizedSnapshot]:
        """Find previous comparable snapshot.
        
        Searches across runs in the same sample size bin to find previous comparable runs.
        """
        if not snapshot.comparison_group:
            return None
        
        # CRITICAL: to_key() now requires stage and may return None for invalid groups
        group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        if group_key is None:
            logger.warning(f"Snapshot {snapshot.run_id} has invalid comparison group, cannot find comparable")
            return None
        
        # First, search in current run's snapshots
        # NOTE: We only use snapshots from snapshot_index.json search below (which verifies file existence)
        # In-memory snapshots from self._snapshots might be stale if runs were deleted
        candidates = []
        # Skip in-memory cache search - rely on snapshot_index.json which verifies file existence
        
        # Search across ALL runs in RESULTS to find previous comparable runs
        # This ensures we find exactly the same runs (same comparison_group) regardless of bin
        # CRITICAL: Also search in comparison group directories (cg-*_n-*_fam-*)
        if hasattr(self, 'run_dir') and self.run_dir:
            # Find RESULTS directory
            results_dir = self.run_dir
            while results_dir.parent.exists() and results_dir.name != "RESULTS":
                results_dir = results_dir.parent
                if results_dir.name == "RESULTS":
                    break
            
            if results_dir.name == "RESULTS":
                # Search both sample_* bins and comparison group directories (cg-*)
                # Handle both old structure (RESULTS/sample_*/) and new structure (RESULTS/runs/cg-*/)
                search_dirs = []
                runs_dir = results_dir / "runs"
                if runs_dir.exists() and runs_dir.is_dir():
                    # New structure: RESULTS/runs/cg-*/
                    search_dirs = [runs_dir]
                else:
                    # Old structure: RESULTS/sample_*/ or RESULTS/cg-*/
                    search_dirs = [results_dir]
                
                for base_dir in search_dirs:
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for bin_dir in iterdir_sorted(base_dir):
                        if not bin_dir.is_dir():
                            continue
                        
                        # Search all runs in this bin/directory
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for run_subdir in iterdir_sorted(bin_dir):
                            if not run_subdir.is_dir() or run_subdir.name == "METRICS":
                                continue
                            
                            # Check both target-first (globals/) and legacy (REPRODUCIBILITY/METRICS/)
                            # Prioritize target-first structure
                            run_snapshot_index = None
                            globals_snapshot_index = run_subdir / "globals" / "snapshot_index.json"
                            legacy_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"
                            
                            if globals_snapshot_index.exists():
                                run_snapshot_index = globals_snapshot_index
                            elif legacy_snapshot_index.exists():
                                run_snapshot_index = legacy_snapshot_index
                            
                            if run_snapshot_index and run_snapshot_index.exists():
                                try:
                                    with open(run_snapshot_index) as f:
                                        data = json.load(f)
                                    
                                    for key, snap_data in data.items():
                                        # Handle both old format (run_id key) and new format (run_id:stage key)
                                        if ':' in key:
                                            run_id = key.split(':', 1)[0]
                                        else:
                                            run_id = key
                                        
                                        # CRITICAL: Never pick the same run_id (even if different stage)
                                        if run_id == snapshot.run_id:
                                            continue
                                        
                                        try:
                                            snap = self._deserialize_snapshot(snap_data)
                                            # Double-check run_id (defense in depth)
                                            if snap.run_id == snapshot.run_id:
                                                continue
                                            
                                            # CRITICAL: Verify snapshot matches stage and view for exact matching
                                            if snap.stage != snapshot.stage:
                                                continue
                                            if snap.view != snapshot.view:
                                                continue
                                            
                                            # Only add if same comparison_group (exactly the same runs)
                                            snap_key = snap.comparison_group.to_key(snap.stage, strict=False) if snap.comparison_group else None
                                            if (snap.comparison_group and 
                                                snap_key is not None and snap_key == group_key):
                                                comparable, reason = self._check_comparability(snapshot, snap)
                                                if comparable:
                                                    # CRITICAL: Verify snapshot file actually exists on disk
                                                    # The snapshot_index.json might reference a snapshot from a deleted run
                                                    # We need to verify the snapshot.json file exists before using it
                                                    snapshot_file_exists = False
                                                    cohort_subdir_found = None
                                                    if snap.stage and snap.view and snap.target:
                                                        from TRAINING.orchestration.utils.target_first_paths import (
                                                            find_cohort_dirs, parse_attempt_id_from_cohort_dir
                                                        )
                                                        # Use existing discovery primitive (rglob handles nested structures)
                                                        cohort_dirs = find_cohort_dirs(
                                                            base_output_dir=run_subdir,
                                                            target=snap.target,
                                                            stage=snap.stage,
                                                            view=snap.view
                                                        )
                                                        # Check each cohort directory for matching snapshot
                                                        for cohort_dir in cohort_dirs:
                                                            snapshot_file = cohort_dir / "snapshot.json"
                                                            if snapshot_file.exists():
                                                                try:
                                                                    with open(snapshot_file, 'r') as f:
                                                                        snapshot_data = json.load(f)
                                                                        if snapshot_data.get('run_id') == snap.run_id:
                                                                            # Parse attempt_id from path for verification
                                                                            parsed_attempt_id = parse_attempt_id_from_cohort_dir(cohort_dir)
                                                                            snap_attempt_id = getattr(snap, 'attempt_id', 0)
                                                                            # Match attempt_id if available, otherwise accept any
                                                                            if parsed_attempt_id == snap_attempt_id or snap_attempt_id == 0:
                                                                                snapshot_file_exists = True
                                                                                cohort_subdir_found = cohort_dir
                                                                                break
                                                                except Exception:
                                                                    continue
                                                        
                                                        # Fallback to legacy structure: REPRODUCIBILITY/<stage>/<view>/<target>/cohort=<cohort_id>/
                                                        if not snapshot_file_exists:
                                                            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                                                            target_clean_legacy = normalize_target_name(snap.target or "unknown")
                                                            stage_dir = run_subdir / "REPRODUCIBILITY" / snap.stage / snap.view / target_clean_legacy
                                                            if stage_dir.exists():
                                                                # Search for cohort directories
                                                                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                                                for cohort_subdir in iterdir_sorted(stage_dir):
                                                                    if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                        snapshot_file = cohort_subdir / "snapshot.json"
                                                                        if snapshot_file.exists():
                                                                            # Verify this snapshot.json matches the run_id
                                                                            try:
                                                                                with open(snapshot_file, 'r') as f:
                                                                                    snapshot_data = json.load(f)
                                                                                    if snapshot_data.get('run_id') == snap.run_id:
                                                                                        snapshot_file_exists = True
                                                                                        cohort_subdir_found = cohort_subdir
                                                                                        break
                                                                            except Exception:
                                                                                continue
                                                            
                                                    # If we found a matching snapshot file, use it
                                                    if snapshot_file_exists and cohort_subdir_found:
                                                        # CRITICAL: Reload metrics from actual metrics.json file
                                                        # The snapshot might have been saved before we fixed _normalize_outputs
                                                        # Try target-first metrics location first, then legacy
                                                        metrics_file = None
                                                        metrics_json = None
                                                        try:
                                                            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                                                            metrics_dir = get_metrics_path_from_cohort_dir(cohort_subdir_found, base_output_dir=run_subdir)
                                                            if metrics_dir:
                                                                metrics_file = metrics_dir / "metrics.json"
                                                                if not metrics_file.exists():
                                                                    metrics_parquet = metrics_dir / "metrics.parquet"
                                                                    if metrics_parquet.exists():
                                                                        # Handle parquet separately
                                                                        import pandas as pd
                                                                        df = pd.read_parquet(metrics_parquet)
                                                                        if len(df) > 0:
                                                                            metrics_json = df.iloc[0].to_dict()
                                                                        metrics_file = None  # Mark as handled
                                                                    else:
                                                                        metrics_file = None
                                                        except Exception as e:
                                                            logger.debug(f"Failed to get metrics path from cohort_dir: {e}")
                                                        
                                                        # Fallback to legacy location
                                                        if not metrics_file and not metrics_json:
                                                            metrics_file = cohort_subdir_found / "metrics.json"
                                                        
                                                        if metrics_file and metrics_file.exists():
                                                            try:
                                                                if metrics_file.suffix == '.parquet':
                                                                    # Handle parquet
                                                                    import pandas as pd
                                                                    df = pd.read_parquet(metrics_file)
                                                                    if len(df) > 0:
                                                                        metrics_json = df.iloc[0].to_dict()
                                                                else:
                                                                    with open(metrics_file, 'r') as f:
                                                                        metrics_json = json.load(f)
                                                            except Exception as e:
                                                                logger.debug(f"Failed to reload metrics from {metrics_file}: {e}")
                                                        
                                                        # Process metrics_json if we have it
                                                        if metrics_json:
                                                            # Extract all numeric metrics (same logic as _normalize_outputs)
                                                            metrics_data = {
                                                                k: v for k, v in metrics_json.items()
                                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                                           'composite_version', 'leakage', 'leakage_flag']
                                                                and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                                                            }
                                                            # Update snapshot's outputs.metrics
                                                            if metrics_data:
                                                                snap.outputs['metrics'] = metrics_data
                                                                logger.debug(f"Reloaded metrics for snapshot {snap.run_id} from {metrics_file or 'parquet'}")
                                                            
                                                            # Mark source for auditability
                                                            if bin_dir.name.startswith("cg-"):
                                                                snap._comparison_source = "comparison_group_directory"
                                                            else:
                                                                snap._comparison_source = "snapshot_index"
                                                            candidates.append((snap.timestamp, snap))
                                                        else:
                                                            logger.debug(f"Skipping snapshot {snap.run_id} from {run_snapshot_index} - snapshot.json file not found on disk (run may have been deleted)")
                                                    else:
                                                        # If we can't determine the path, skip it (safety first)
                                                        logger.debug(f"Skipping snapshot {snap.run_id} - missing stage/view/target for path reconstruction")
                                        except Exception as e:
                                            logger.debug(f"Failed to deserialize snapshot from {run_snapshot_index}: {e}")
                                            continue
                                except Exception as e:
                                    logger.debug(f"Failed to read snapshot index {run_snapshot_index}: {e}")
                                    continue
        
        if not candidates:
            return None
        
        # Return most recent by monotonic sequence number (snapshot_seq)
        # This is the correct ordering method - mtime can change for unrelated reasons
        # (file copies, post-processing, filesystem quirks, coarse timestamp resolution)
        candidates_with_seq = []
        for timestamp, snap in candidates:
            # Use snapshot_seq if available (assigned at save time), fallback to timestamp
            seq = snap.snapshot_seq if hasattr(snap, 'snapshot_seq') and snap.snapshot_seq is not None else 0
            # If no seq, use timestamp as fallback (but log warning)
            if seq == 0:
                logger.debug(f"Snapshot {snap.run_id} has no snapshot_seq, using timestamp fallback")
                # Convert timestamp to numeric for comparison (ISO format)
                try:
                    from datetime import datetime
                    # Defensive: Ensure timestamp is a string before calling .replace()
                    if timestamp and isinstance(timestamp, str):
                        ts_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        seq = ts_dt.timestamp()
                    else:
                        seq = 0
                except Exception:
                    seq = 0
            
            candidates_with_seq.append((seq, snap))
        
        # Deterministic ordering: sort by (attempt_id, snapshot_seq) ascending
        # Within same run: compare attempt_k to attempt_{k-1}
        # Cross-run: compare attempt_0 to attempt_0
        current_attempt_id = getattr(snapshot, 'attempt_id', 0)
        
        # Group candidates by run_id
        candidates_by_run = {}
        for seq, cand in candidates_with_seq:
            run_id = cand.run_id
            if run_id not in candidates_by_run:
                candidates_by_run[run_id] = []
            candidates_by_run[run_id].append((seq, cand))
        
        # For same run: find attempt_{k-1}
        prev_snapshot = None
        if snapshot.run_id in candidates_by_run:
            # Same run - find previous attempt
            same_run_candidates = candidates_by_run[snapshot.run_id]
            same_run_sorted = sorted(
                same_run_candidates,
                key=lambda x: (
                    getattr(x[1], 'attempt_id', 0),
                    x[0] or 0
                )
            )
            # Find attempt_id = current_attempt_id - 1
            for seq, cand in same_run_sorted:
                cand_attempt_id = getattr(cand, 'attempt_id', 0)
                if cand_attempt_id == current_attempt_id - 1:
                    prev_snapshot = cand
                    break
            # If no previous attempt found, use most recent attempt_0 from same run
            if not prev_snapshot:
                attempt_0_same_run = [(seq, cand) for seq, cand in same_run_sorted 
                                     if getattr(cand, 'attempt_id', 0) == 0]
                if attempt_0_same_run:
                    # DETERMINISTIC: snapshot_seq is already unique within same run, but sort for consistency
                    attempt_0_same_run.sort(key=lambda x: x[0] or 0, reverse=True)
                    prev_snapshot = attempt_0_same_run[0][1]
        
        # Cross-run: use most recent attempt_0 from different runs
        # CRITICAL: snapshot_seq is per-run (starts at 1 each run), so cannot use for cross-run ordering
        # Use timestamp for cross-run comparisons to find the most recent run (i-1, not i-2)
        if not prev_snapshot:
            attempt_0_candidates = []
            for run_id, run_candidates in candidates_by_run.items():
                if run_id != snapshot.run_id:  # Different run
                    for seq, cand in run_candidates:
                        if getattr(cand, 'attempt_id', 0) == 0:  # attempt_0 only
                            attempt_0_candidates.append((seq, cand))
            if attempt_0_candidates:
                # CRITICAL: Sort by timestamp (most recent first) for cross-run comparisons
                # snapshot_seq is per-run and resets to 1, so cannot be used for cross-run ordering
                # Use timestamp to find run i-1 (previous run), not run i-2
                from datetime import datetime
                def get_timestamp_sort_key(item):
                    seq, cand = item
                    # Primary: use timestamp field
                    try:
                        if hasattr(cand, 'timestamp') and cand.timestamp and isinstance(cand.timestamp, str):
                            ts_dt = datetime.fromisoformat(cand.timestamp.replace('Z', '+00:00'))
                            return ts_dt.timestamp()
                    except Exception:
                        pass
                    # Fallback: use run_id if it contains timestamp (e.g., intelligent_output_20260112_044406)
                    try:
                        if hasattr(cand, 'run_id') and cand.run_id:
                            run_id = cand.run_id
                            # Try to extract timestamp from run_id format: intelligent_output_YYYYMMDD_HHMMSS
                            if '_' in run_id:
                                parts = run_id.split('_')
                                if len(parts) >= 2:
                                    date_part = parts[-2]  # YYYYMMDD
                                    time_part = parts[-1]  # HHMMSS
                                    if len(date_part) == 8 and len(time_part) == 6:
                                        ts_str = f"{date_part}_{time_part}"
                                        ts_dt = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                                        return ts_dt.timestamp()
                    except Exception:
                        pass
                    return 0  # Last resort: unknown timestamp
                
                attempt_0_candidates.sort(key=get_timestamp_sort_key, reverse=True)
                prev_snapshot = attempt_0_candidates[0][1]  # Most recent (i-1)
                logger.debug(
                    f"Selected previous run for comparison: {prev_snapshot.run_id} "
                    f"(current: {snapshot.run_id}, timestamp: {prev_snapshot.timestamp})"
                )
            elif candidates_with_seq:
                # Fallback: use most recent overall by timestamp
                # CRITICAL: Use timestamp, not snapshot_seq, for cross-run ordering
                from datetime import datetime
                def get_timestamp_sort_key_cross(item):
                    seq, cand = item
                    # Primary: use timestamp field
                    try:
                        if hasattr(cand, 'timestamp') and cand.timestamp and isinstance(cand.timestamp, str):
                            ts_dt = datetime.fromisoformat(cand.timestamp.replace('Z', '+00:00'))
                            return ts_dt.timestamp()
                    except Exception:
                        pass
                    # Fallback: use run_id if it contains timestamp (e.g., intelligent_output_20260112_044406)
                    try:
                        if hasattr(cand, 'run_id') and cand.run_id:
                            run_id = cand.run_id
                            # Try to extract timestamp from run_id format: intelligent_output_YYYYMMDD_HHMMSS
                            if '_' in run_id:
                                parts = run_id.split('_')
                                if len(parts) >= 2:
                                    date_part = parts[-2]  # YYYYMMDD
                                    time_part = parts[-1]  # HHMMSS
                                    if len(date_part) == 8 and len(time_part) == 6:
                                        ts_str = f"{date_part}_{time_part}"
                                        ts_dt = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                                        return ts_dt.timestamp()
                    except Exception:
                        pass
                    return 0  # Last resort: unknown timestamp
                
                candidates_with_seq.sort(key=get_timestamp_sort_key_cross, reverse=True)
                prev_snapshot = candidates_with_seq[0][1]  # Most recent by timestamp
                logger.debug(
                    f"Selected previous run (fallback): {prev_snapshot.run_id} "
                    f"(current: {snapshot.run_id}, timestamp: {prev_snapshot.timestamp})"
                )
        
        # CRITICAL: Reload metrics from actual metrics.json file for the previous snapshot
        # The snapshot might have been saved before we fixed _normalize_outputs, so metrics might be incomplete
        # We need to reload from the actual metrics.json file to ensure we have all metrics for comparison
        prev_snapshot = self._reload_snapshot_metrics(prev_snapshot)
        
        return prev_snapshot
    
    def get_or_establish_baseline(
        self,
        snapshot: NormalizedSnapshot,
        metrics: Dict[str, float],
        cohort_dir: Path
    ) -> Tuple[Optional[BaselineState], bool]:
        """
        Get or establish baseline for comparison group.
        
        Baselines are stored per-cohort to ensure only exactly the same runs share baselines.
        
        Args:
            snapshot: Normalized snapshot
            metrics: Metrics dict
            cohort_dir: Cohort directory where baseline will be stored
        
        Returns:
            (BaselineState or None, is_new_baseline)
        """
        if not snapshot.comparison_group:
            return None, False
        
        # CRITICAL: to_key() now requires stage and may return None for invalid groups
        group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        if group_key is None:
            logger.warning(f"Snapshot {snapshot.run_id} has invalid comparison group, cannot get/establish baseline")
            return None, False
        
        # Check cache first
        if group_key in self._baselines:
            return self._baselines[group_key], False
        
        # Load from cohort directory
        baseline = self._load_baseline_from_cohort(cohort_dir, group_key)
        if baseline:
            self._baselines[group_key] = baseline  # Cache it
            return baseline, False
        
        # Count comparable runs (search across runs in same bin)
        # CRITICAL: Never include the same run_id (even if different stage)
        comparable_runs = [
            snap for snap in self._snapshots.values()
            if (snap.comparison_group and 
                snap.comparison_group.to_key(snap.stage, strict=False) == group_key and
                snap.run_id != snapshot.run_id)  # Exclude same run_id
        ]
        
        # Search across ALL runs in RESULTS to find comparable runs with same comparison_group
        # This ensures baselines are established from exactly the same runs
        if hasattr(self, 'run_dir') and self.run_dir:
            # Find RESULTS directory
            results_dir = self.run_dir
            while results_dir.parent.exists() and results_dir.name != "RESULTS":
                results_dir = results_dir.parent
                if results_dir.name == "RESULTS":
                    break
            
            if results_dir.name == "RESULTS":
                # Search all sample_* bins
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                for bin_dir in iterdir_sorted(results_dir):
                    if bin_dir.is_dir() and bin_dir.name.startswith("sample_"):
                        # Search all runs in this bin
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for run_subdir in iterdir_sorted(bin_dir):
                            if run_subdir.is_dir() and run_subdir.name != "METRICS":
                                # Check both target-first (globals/) and legacy (REPRODUCIBILITY/METRICS/)
                                # Prioritize target-first structure
                                run_snapshot_index = None
                                globals_snapshot_index = run_subdir / "globals" / "snapshot_index.json"
                                legacy_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"
                                
                                if globals_snapshot_index.exists():
                                    run_snapshot_index = globals_snapshot_index
                                elif legacy_snapshot_index.exists():
                                    run_snapshot_index = legacy_snapshot_index
                                
                                if run_snapshot_index and run_snapshot_index.exists():
                                    try:
                                        with open(run_snapshot_index) as f:
                                            data = json.load(f)
                                            for key, snap_data in data.items():
                                                # Handle both old format (run_id key) and new format (run_id:stage key)
                                                if ':' in key:
                                                    run_id = key.split(':', 1)[0]
                                                else:
                                                    run_id = key
                                                
                                                # CRITICAL: Never pick the same run_id (even if different stage)
                                                if run_id == snapshot.run_id:
                                                    continue
                                                
                                                try:
                                                    snap = self._deserialize_snapshot(snap_data)
                                                    # Double-check run_id (defense in depth)
                                                    if snap.run_id == snapshot.run_id:
                                                        continue
                                                    # Only add if same comparison_group (exactly the same runs)
                                                    snap_key = snap.comparison_group.to_key(snap.stage, strict=False) if snap.comparison_group else None
                                                    if (snap.comparison_group and 
                                                        snap_key is not None and snap_key == group_key):
                                                        comparable_runs.append(snap)
                                                except Exception:
                                                    continue
                                    except Exception:
                                        continue
        
        if len(comparable_runs) < self.min_runs_for_baseline:
            return None, False
        
        # Establish baseline (use best metric run)
        best_run = None
        best_score = None
        
        for snap in comparable_runs:
            if snap.outputs.get('metrics', {}).get('auc'):
                score = snap.outputs['metrics']['auc']
                if best_score is None or score > best_score:
                    best_score = score
                    best_run = snap
        
        if best_run:
            baseline = BaselineState(
                comparison_group_key=group_key,
                baseline_run_id=best_run.run_id,
                baseline_timestamp=best_run.timestamp,
                baseline_metrics=best_run.outputs.get('metrics', {}),
                established_at=datetime.now().isoformat()
            )
            self._baselines[group_key] = baseline  # Cache it
            # Save to cohort directory (not global index)
            self._save_baseline_to_cohort(cohort_dir, baseline)
            return baseline, True
        
        return None, False
    
    def save_diff(
        self,
        diff: DiffResult,
        baseline_diff: Optional[DiffResult],
        cohort_dir: Path,
        view: Optional[str] = None,  # NEW: Structured input (preferred over parsing)
        target: Optional[str] = None,  # NEW: Structured input (preferred over parsing)
        symbol: Optional[str] = None,  # NEW: Structured input (preferred over parsing)
        stage: Optional[str] = None,  # NEW: Structured input (preferred over parsing)
        base_output_dir: Optional[Path] = None,  # NEW: Explicit run root (SST)
    ) -> Path:  # NEW: Return resolved cohort_dir path
        """
        Save diff results to cohort directory.
        
        Args:
            diff: Diff against previous run
            baseline_diff: Diff against baseline (if available)
            cohort_dir: Cohort directory
            view: View type (CROSS_SECTIONAL or SYMBOL_SPECIFIC) - preferred over parsing
            target: Target name - preferred over parsing
            symbol: Symbol name (for SYMBOL_SPECIFIC) - preferred over parsing
            stage: Stage name - preferred over parsing
            base_output_dir: Explicit run root directory (SST). If None, will resolve from cohort_dir.
        
        Returns:
            Resolved cohort_dir path (SST) where files were written
        """
        cohort_dir = Path(cohort_dir)
        cohort_dir_str = str(cohort_dir)
        # NEVER create legacy REPRODUCIBILITY directories - only use target-first structure
        # Check for uppercase REPRODUCIBILITY (legacy) but allow lowercase reproducibility (target-first)
        if "REPRODUCIBILITY" in cohort_dir_str and "reproducibility" not in cohort_dir_str.lower():
            logger.warning(f"⚠️ Skipping diff save to legacy REPRODUCIBILITY path: {cohort_dir}")
            # CRITICAL: Function must return Path, not None. Return cohort_dir as fallback.
            return Path(cohort_dir)
        
        # ========================================================================
        # CRITICAL: Derive paths from structured inputs (preferred) instead of parsing
        # Priority 1: Use explicit parameters (view, target, symbol, stage) - SST pattern
        # Priority 2: Use DiffResult fields (explicit, preferred over parsing)
        # Priority 3: Use SST functions to parse target-first path (fallback only)
        # Priority 4: Parse legacy path (last resort)
        # ========================================================================
        # CRITICAL: Use structured inputs (preferred) - NEVER use diff.prev_* for path resolution
        # diff.prev_* fields are for diff CONTENT (comparing runs), NOT for path resolution
        # Use explicit None checks (not 'or') to handle falsy values correctly (e.g., stage=0, view="")
        stage_resolved = stage if stage is not None else "UNKNOWN"
        view_resolved = view if (view is not None and view != "") else "UNKNOWN"
        target_resolved = target or "UNKNOWN"
        symbol_resolved = symbol  # May be None for CROSS_SECTIONAL
        
        # Log structured inputs at start for debugging
        logger.debug(
            f"save_diff() called: view={view_resolved}, target={target_resolved}, symbol={symbol_resolved}, "
            f"stage={stage_resolved}, cohort_dir={cohort_dir}"
        )
        
        cohort_id = None
        # CRITICAL: Use parameter if provided, normalize it, only resolve if None
        # Never overwrite the parameter - that causes path duplication bugs
        if base_output_dir is not None:
            from TRAINING.orchestration.utils.target_first_paths import normalize_run_root
            base_output_dir = normalize_run_root(base_output_dir)
        elif self.run_dir is not None:
            from TRAINING.orchestration.utils.target_first_paths import normalize_run_root, run_root
            base_output_dir = normalize_run_root(run_root(self.run_dir) or self.run_dir)
            logger.debug(f"Resolved base_output_dir from self.run_dir: {base_output_dir}")
        else:
            # Will be handled below with proper error
            base_output_dir = None
        target_cohort_dir = None
        
        VALID_VIEWS = ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']
        
        # Extract cohort_id from path (still needed even with structured inputs)
        parts = Path(cohort_dir).parts
        for part in parts:
            if part.startswith('cohort='):
                cohort_id = part.replace('cohort=', '')
                break
        
        # If structured inputs incomplete, try to fill from DiffResult or parse (fallback only)
        # CRITICAL: Never use diff.prev_* for path resolution - these are from previous run
        if target_resolved == "UNKNOWN" or view_resolved == "UNKNOWN" or stage_resolved == "UNKNOWN":
            # Fallback: Try DiffResult.target field (current run's target, not prev)
            if target_resolved == "UNKNOWN" and hasattr(diff, 'target') and diff.target:
                target_resolved = diff.target
            
            # If still incomplete, parse from path (last resort)
            # NOTE: We do NOT use diff.prev_view or diff.prev_stage here - those are from previous run
            if target_resolved == "UNKNOWN" or view_resolved == "UNKNOWN" or stage_resolved == "UNKNOWN":
                logger.debug(f"Structured inputs incomplete, parsing from path: view={view_resolved}, target={target_resolved}, stage={stage_resolved}")
                try:
                    from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                    parsed = parse_reproducibility_path(Path(cohort_dir))
                    if parsed.get("target") and target_resolved == "UNKNOWN":
                        target_resolved = parsed["target"]
                    if parsed.get("view") and view_resolved == "UNKNOWN":
                        view_resolved = parsed["view"]
                    if parsed.get("stage") and stage_resolved == "UNKNOWN":
                        stage_resolved = parsed["stage"]
                    if parsed.get("cohort_id") and not cohort_id:
                        cohort_id = parsed["cohort_id"]
                except Exception as e:
                    logger.debug(f"Path parsing failed: {e}")
        
        # If still UNKNOWN after all attempts, log warning with caller context
        if target_resolved == "UNKNOWN" or view_resolved == "UNKNOWN" or stage_resolved == "UNKNOWN":
            logger.warning(
                f"Could not determine complete identifiers for diff save (cohort_dir={cohort_dir}). "
                f"Structured inputs: view={view}, target={target}, symbol={symbol}, stage={stage}. "
                f"Resolved: view={view_resolved}, target={target_resolved}, stage={stage_resolved}. "
                f"Path parsing also failed. Files will be written with UNKNOWN identifiers."
            )
        
        # CRITICAL: base_output_dir should already be set above (from parameter or self.run_dir)
        # This is a defensive check in case the above logic didn't set it
        if base_output_dir is None:
            if self.run_dir is None:
                raise ValueError(
                    f"Cannot determine current run directory: self.run_dir is None. "
                    f"Cannot save diff files. Pass base_output_dir explicitly or ensure DiffTelemetry is initialized with run_dir."
                )
            from TRAINING.orchestration.utils.target_first_paths import normalize_run_root, run_root
            base_output_dir = normalize_run_root(run_root(self.run_dir) or self.run_dir)
            logger.debug(f"Using self.run_dir as base_output_dir (fallback): {base_output_dir}")
        
        # PRIORITY 2: Validate that provided cohort_dir is within current run directory
        # If outside, rebuild using structured inputs (preferred) or parse path (fallback)
        # This is a defensive backstop - the real fix is in reproducibility_tracker.py
        cohort_dir_original = cohort_dir
        if cohort_dir:
            cohort_resolved = Path(cohort_dir).resolve()
            base_resolved = Path(base_output_dir).resolve()
            try:
                is_within = cohort_resolved.is_relative_to(base_resolved)
            except AttributeError:
                # Python < 3.9 compatibility
                try:
                    cohort_resolved.relative_to(base_resolved)
                    is_within = True
                except ValueError:
                    is_within = False
            
            if not is_within:
                old_cohort_dir = cohort_dir
                old_run_root = str(cohort_resolved)
                # Try to find run root from old path
                try:
                    from TRAINING.orchestration.utils.target_first_paths import run_root, normalize_run_root
                    old_run_root = normalize_run_root(run_root(Path(cohort_dir))) or str(cohort_resolved)
                except Exception:
                    pass
                
                logger.warning(
                    f"Provided cohort_dir ({cohort_dir}) is outside current run directory ({base_output_dir}). "
                    f"Rebuilding with current run's base_output_dir. "
                    f"Old run root: {old_run_root}, New run root: {base_output_dir}"
                )
                
                # Prefer rebuilding from structured inputs if available
                # Extract cohort_id from path if not already available
                if not cohort_id:
                    parts = Path(cohort_dir).parts
                    for part in parts:
                        if part.startswith('cohort='):
                            cohort_id = part.replace('cohort=', '')
                            break
                
                # If we have structured inputs (target, stage, view, symbol, cohort_id), rebuild
                # Check that they're not "UNKNOWN" (which means not resolved)
                if (target_resolved and target_resolved != "UNKNOWN" and 
                    stage_resolved and stage_resolved != "UNKNOWN" and 
                    view_resolved and view_resolved != "UNKNOWN" and 
                    cohort_id):
                    # Extract attempt_id and universe_sig from path (will be parsed below if needed)
                    attempt_id = None
                    universe_sig = None
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import (
                            parse_attempt_id_from_cohort_dir, parse_reproducibility_path
                        )
                        attempt_id = parse_attempt_id_from_cohort_dir(Path(cohort_dir))
                        parsed = parse_reproducibility_path(Path(cohort_dir))
                        universe_sig = parsed.get('universe_sig') if parsed else None
                    except Exception:
                        pass
                    
                    # Rebuild using structured inputs with current run's base_output_dir
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                        # Determine symbol if not provided
                        symbol_for_rebuild = symbol_resolved
                        if symbol_for_rebuild is None:
                            # Try to extract from path
                            parts = Path(cohort_dir).parts
                            for part in parts:
                                if part.startswith('symbol='):
                                    symbol_for_rebuild = part.replace('symbol=', '')
                                    break
                        
                        # Normalize view
                        view_for_rebuild = view_resolved
                        if symbol_for_rebuild:
                            view_for_rebuild = 'SYMBOL_SPECIFIC'
                        elif view_for_rebuild == 'INDIVIDUAL' and stage_resolved == 'FEATURE_SELECTION':
                            view_for_rebuild = 'SYMBOL_SPECIFIC'
                        
                        cohort_dir = build_target_cohort_dir(
                            base_output_dir=base_output_dir,
                            target=target_resolved,
                            stage=stage_resolved,
                            view=view_for_rebuild,
                            cohort_id=cohort_id,
                            symbol=symbol_for_rebuild,
                            attempt_id=attempt_id or 0,
                            universe_sig=universe_sig
                        )
                        logger.info(
                            f"Rebuilt cohort_dir: {old_cohort_dir} -> {cohort_dir} "
                            f"(old run: {old_run_root}, new run: {base_output_dir})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to rebuild cohort_dir from structured inputs: {e}. "
                            f"Will continue with original path (may cause issues)."
                        )
                        # Fall back to original (will fail validation later, but at least we tried)
                        cohort_dir = cohort_dir_original
                else:
                    logger.warning(
                        f"Cannot rebuild cohort_dir - missing structured inputs "
                        f"(target={target_resolved}, stage={stage_resolved}, view={view_resolved}, cohort_id={cohort_id}). "
                        f"Will continue with original path (may cause issues)."
                    )
                    # Keep original - will be handled by downstream validation
                    cohort_dir = cohort_dir_original
            else:
                logger.debug(f"Validated cohort_dir is within current run directory: {cohort_dir}")
        
        # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
        if stage_resolved in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target_resolved != "UNKNOWN" and cohort_id:
            if base_output_dir:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import (
                        get_target_reproducibility_dir, ensure_target_structure
                    )
                    
                    # CRITICAL: Use structured inputs (preferred) or extract symbol from path (fallback)
                    # If symbol provided as parameter, use it; otherwise try to extract from path
                    symbol_for_target = symbol_resolved
                    if symbol_for_target is None:
                        # Fallback: try to extract from path
                        for part in parts:
                            if part.startswith('symbol='):
                                symbol_for_target = part.replace('symbol=', '')
                                logger.debug(f"Extracted symbol from path: {symbol_for_target}")
                                break
                    
                    # Normalize view: if symbol exists, it's SYMBOL_SPECIFIC
                    # Otherwise, use provided view with INDIVIDUAL->SYMBOL_SPECIFIC normalization
                    if symbol_for_target:
                        view_for_target = 'SYMBOL_SPECIFIC'
                        logger.debug(f"Symbol present ({symbol_for_target}), forcing SYMBOL_SPECIFIC view")
                    else:
                        view_for_target = view_resolved
                        if stage_resolved == 'FEATURE_SELECTION' and view_resolved == 'INDIVIDUAL':
                            view_for_target = 'SYMBOL_SPECIFIC'
                    
                    # Ensure target structure exists
                    ensure_target_structure(base_output_dir, target_resolved)
                        
                    # CRITICAL: Use canonical path builder to ensure cohorts are in correct structure
                    # Extract attempt_id and universe_sig from cohort_dir path using SST functions
                    from TRAINING.orchestration.utils.target_first_paths import (
                        parse_attempt_id_from_cohort_dir
                    )
                    # Use SST functions which already handle batch_*/attempt_* structure
                    attempt_id = parse_attempt_id_from_cohort_dir(Path(cohort_dir))
                    # Get universe_sig from parsed result if available
                    universe_sig = None
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                        parsed = parse_reproducibility_path(Path(cohort_dir))
                        universe_sig = parsed.get('universe_sig') if parsed else None
                    except Exception:
                        universe_sig = None
                    
                    from TRAINING.orchestration.utils.target_first_paths import build_target_cohort_dir
                    target_cohort_dir = build_target_cohort_dir(
                        base_output_dir=base_output_dir,
                        target=target_resolved,
                        stage=stage_resolved,
                        view=view_for_target,
                        cohort_id=cohort_id,
                        symbol=symbol_for_target,
                        attempt_id=attempt_id,
                        universe_sig=universe_sig
                    )
                    logger.debug(
                        f"Built target_cohort_dir from structured inputs: {target_cohort_dir} "
                        f"(target={target_resolved}, view={view_for_target}, symbol={symbol_for_target}, stage={stage_resolved})"
                    )
                    target_cohort_dir.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created target-first cohort directory for diffs: {target_cohort_dir}")
                except Exception as e:
                    logger.debug(f"Failed to create target-first structure for diffs (non-critical): {e}")
                    target_cohort_dir = None
        
        # CRITICAL: Use cohort_dir directly (same pattern as save_snapshot) - don't rebuild from structured inputs
        # Snapshots work because they use cohort_dir directly, so we should too
        resolved_cohort_dir = Path(cohort_dir).resolve()
        
        # CRITICAL: Validate resolved_cohort_dir is within base_output_dir (current run)
        # If still outside after self-healing, skip diff emission to prevent cross-run contamination
        if base_output_dir:
            base = Path(base_output_dir).resolve()
            out = Path(resolved_cohort_dir).resolve()
            try:
                is_within = out.is_relative_to(base)
            except AttributeError:
                # Python < 3.9 compatibility: use relative_to() in try/except
                try:
                    out.relative_to(base)
                    is_within = True
                except ValueError:
                    is_within = False
            
            if not is_within:
                logger.error(
                    f"CRITICAL: resolved_cohort_dir ({resolved_cohort_dir}) is still outside "
                    f"base_output_dir ({base_output_dir}) after self-healing attempt. "
                    f"Skipping diff emission to prevent cross-run contamination. "
                    f"Structured inputs: stage={stage}, view={view}, target={target}, symbol={symbol}"
                )
                # Write deterministic marker file for debugging
                try:
                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                    globals_dir = get_globals_dir(base_output_dir)
                    marker_file = globals_dir / "PATH_REPAIR_FAILED.json"
                    from TRAINING.common.utils.file_utils import write_atomic_json
                    write_atomic_json(marker_file, {
                        "resolved_cohort_dir": str(resolved_cohort_dir),
                        "base_output_dir": str(base_output_dir),
                        "stage": stage,
                        "view": view,
                        "target": target,
                        "symbol": symbol
                    })
                except Exception:
                    pass  # Marker file is best-effort
                # Return early - skip diff writing (but return path for caller)
                return resolved_cohort_dir
            else:
                logger.debug(f"Validated resolved_cohort_dir is within base_output_dir: {resolved_cohort_dir}")
        
        # Ensure resolved directory exists
        resolved_cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # Derive paths from resolved_cohort_dir (not recomputing)
        diff_prev_path = resolved_cohort_dir / "diff_prev.json"
        snapshot_path = resolved_cohort_dir / "snapshot.json"  # For reference
        
        # Log resolved path once (SST source of truth)
        logger.debug(f"Resolved paths (SST): cohort_dir={resolved_cohort_dir}, diff_prev={diff_prev_path}")
        
        # Tier A: Summary in diff_prev.json (lightweight, always present)
        # This includes: metric_deltas_count, impact_label, top_regressions, top_improvements
        prev_diff_dict = diff.to_dict()
        
        # Tier B: Structured per-metric deltas in metric_deltas.json (detailed, includes all deltas)
        # This is the "real diff" with full structured deltas for each metric
        # Create metric_deltas.json whenever there's a previous snapshot to compare against
        # (even if all metrics are identical, no metrics were compared, or all deltas are below threshold)
        metric_deltas_file_path = None
        metric_deltas_total = diff.summary.get('metric_deltas_total', 0)
        if diff.comparable:
            metric_deltas_file = resolved_cohort_dir / "metric_deltas.json"
            # Use relative path from cohort_dir for portability
            metric_deltas_file_path = "metric_deltas.json"
            
            # Structure: keyed by metric name with full delta info
            metric_deltas_data = {
                'run_id': diff.current_run_id,
                'prev_run_id': diff.prev_run_id,
                'timestamp': datetime.now().isoformat(),
                # Identifiers for downstream joining
                'stage': stage_resolved,
                'view': view_resolved,
                'target': target_resolved,
                'metric_deltas': diff.metric_deltas,  # Now includes ALL deltas, not just significant ones
                'summary': {
                    'total_metrics': len(diff.metric_deltas),  # Total metrics in dict (all deltas)
                    'total_compared': metric_deltas_total,  # Total metrics compared
                    'significant_count': diff.summary.get('metric_deltas_significant', 0),  # Only significant ones
                    'impact_label': diff.summary.get('impact_label', 'none'),
                    'top_regressions': diff.summary.get('top_regressions', []),
                    'top_improvements': diff.summary.get('top_improvements', [])
                }
            }
            # Write to target-first structure
            try:
                _write_atomic_json_with_lock(metric_deltas_file, metric_deltas_data)
                logger.debug(f"✅ Saved metric_deltas.json to {metric_deltas_file} ({len(diff.metric_deltas)} metrics, {diff.summary.get('metric_deltas_significant', 0)} significant)")
            except Exception as e:
                logger.error(f"❌ Failed to save metric_deltas.json to {metric_deltas_file}: {e}")
                import traceback
                logger.debug(f"Metric deltas save traceback: {traceback.format_exc()}")
        
        # Add reference to metric_deltas.json in diff_prev.json summary (before writing)
        if metric_deltas_file_path:
            prev_diff_dict['summary']['metric_deltas_file'] = metric_deltas_file_path
        
        # Ensure summary includes impact classification (already computed in compute_diff)
        # Write to resolved path
        # CRITICAL: Always create diff_prev.json (even if no previous run - it's a summary of current state)
        try:
            # Single log message about where diff file is written (remove duplicates)
            logger.info(f"📝 Writing diff_prev.json to: {diff_prev_path} (resolved_cohort_dir={resolved_cohort_dir})")
            _write_atomic_json_with_lock(diff_prev_path, prev_diff_dict)
            
            # CRITICAL: If write reports success but file missing, that's a bug (not warning)
            if not diff_prev_path.exists():
                from TRAINING.common.determinism import is_strict_mode
                error_msg = f"❌ diff_prev.json write reported success but file does not exist: {diff_prev_path}"
                logger.error(error_msg)
                if is_strict_mode():
                    raise RuntimeError(error_msg)  # Fail-closed in strict mode
        except Exception as e:
            from TRAINING.common.determinism import is_strict_mode
            logger.error(f"❌ Failed to save diff_prev.json to {diff_prev_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            if is_strict_mode():
                raise  # Fail-closed in strict mode
            # Best-effort: log and continue
        
        # Tier C: Full raw metrics remain in metrics.json (already written by MetricsWriter)
        # We don't duplicate them here - just reference the path
        
        # Save baseline diff if available (atomically) to resolved path
        if baseline_diff:
            try:
                baseline_diff_dict = baseline_diff.to_dict()
                baseline_diff_file = resolved_cohort_dir / "diff_baseline.json"
                _write_atomic_json_with_lock(baseline_diff_file, baseline_diff_dict)
                logger.debug(f"✅ Saved diff_baseline.json to {baseline_diff_file}")
            except Exception as e:
                logger.error(f"❌ Failed to save diff_baseline.json to {baseline_diff_file}: {e}")
                import traceback
                logger.debug(f"Baseline diff save traceback: {traceback.format_exc()}")

        logger.debug(f"✅ Saved diffs to {resolved_cohort_dir} (view={view_resolved}, target={target_resolved}, symbol={symbol_resolved})")
        
        # Return resolved path so caller can validate against same path
        return resolved_cohort_dir
    
    def _emit_trend_time_series(
        self,
        snapshot: NormalizedSnapshot,
        metrics: Dict[str, Any],
        cohort_dir: Path
    ) -> None:
        """
        Emit time series data for trend/drift analysis.
        
        Emits one row per metric key with:
        - identifiers: run_id, timestamp, comparison_group, stage, view, target, metric_name
        - values: auc, std_score, composite_score, mean_importance, pos_rate, n_effective_cs, n_models
        - derived: (rolling baseline, drift_z, ema, cusum computed later by trend analyzer)
        
        Stores in metrics_timeseries.parquet at target level (one level up from cohort_dir)
        for aggregation across runs.
        """
        if not metrics:
            return
        
        # Get comparison group key
        comparison_group_key = None
        if snapshot.comparison_group:
            comparison_group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        
        # Extract target from target (for TARGET_RANKING) or from other sources
        target = snapshot.target or "unknown"
        
        # Build time series rows - one per metric
        # Key metrics to track
        metric_fields = [
            'auc', 'std_score', 'composite_score', 'mean_importance',
            'pos_rate', 'n_effective_cs', 'n_models'
        ]
        
        rows = []
        for metric_name in metric_fields:
            if metric_name not in metrics:
                continue
            
            value = metrics.get(metric_name)
            # Skip non-numeric values
            if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                continue
            
            row = {
                'run_id': snapshot.run_id,
                'timestamp': snapshot.timestamp,
                'comparison_group': comparison_group_key,
                'stage': snapshot.stage,
                'view': snapshot.view or 'UNKNOWN',
                'target': target,
                'metric_name': metric_name,
                'value': float(value),
                # Include key context metrics for trend analysis (same values across all rows for this run)
                'auc': metrics.get('auc') if 'auc' in metrics else None,
                'std_score': metrics.get('std_score') if 'std_score' in metrics else None,
                'composite_score': metrics.get('composite_score') if 'composite_score' in metrics else None,
                'mean_importance': metrics.get('mean_importance') if 'mean_importance' in metrics else None,
                'pos_rate': metrics.get('pos_rate') if 'pos_rate' in metrics else None,
                'n_effective_cs': metrics.get('n_effective_cs') if 'n_effective_cs' in metrics else None,
                'n_models': metrics.get('n_models') if 'n_models' in metrics else None
            }
            rows.append(row)
        
        if not rows:
            return
        
        # Store in metrics/ folder (not reproducibility/) for consistency
        # Map cohort_dir to metrics/ folder using helper function
        try:
            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
            metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir)
            if metrics_dir:
                timeseries_file = metrics_dir / "metrics_timeseries.parquet"
            else:
                # Fallback: try to construct path manually
                logger.warning(f"Could not map cohort_dir to metrics path, using fallback: {cohort_dir}")
                # Extract target and view from cohort_dir using SST-aware parser
                from TRAINING.orchestration.utils.target_first_paths import parse_reproducibility_path
                parsed = parse_reproducibility_path(Path(cohort_dir))
                target = parsed.get("target")
                view = parsed.get("view")
                symbol = parsed.get("symbol")
                # stage = parsed.get("stage")  # Available if needed
                
                if target and view:
                    # Find base_output_dir using SST helper (if we're in targets/, go up one level first)
                    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                    base_output_dir = Path(cohort_dir)
                    if base_output_dir.name == "targets":
                        base_output_dir = base_output_dir.parent
                    base_output_dir = get_run_root(base_output_dir)
                    
                    from TRAINING.orchestration.utils.target_first_paths import get_target_metrics_dir
                    metrics_dir = get_target_metrics_dir(base_output_dir, target) / f"view={view}"
                    if symbol:
                        metrics_dir = metrics_dir / f"symbol={symbol}"
                    timeseries_file = metrics_dir / "metrics_timeseries.parquet"
                else:
                    # Last resort: use old location
                    target_dir = cohort_dir.parent
                    timeseries_file = target_dir / "metrics_timeseries.parquet"
        except Exception as e:
            logger.warning(f"Failed to map cohort_dir to metrics path: {e}, using fallback")
            # Fallback to old location
            target_dir = cohort_dir.parent
            timeseries_file = target_dir / "metrics_timeseries.parquet"
        
        try:
            # Read existing data if file exists
            existing_df = None
            if timeseries_file.exists():
                try:
                    existing_df = pd.read_parquet(timeseries_file)
                except Exception as e:
                    logger.debug(f"Could not read existing timeseries file {timeseries_file}: {e}")
            
            # Create new DataFrame from current rows
            new_df = pd.DataFrame(rows)
            
            # Append to existing data
            if existing_df is not None and len(existing_df) > 0:
                # Combine and deduplicate by (run_id, metric_name) to avoid duplicates
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates, keeping the most recent (last) entry
                combined_df = combined_df.drop_duplicates(
                    subset=['run_id', 'metric_name'],
                    keep='last'
                )
                # Sort by timestamp for easier querying
                combined_df = combined_df.sort_values('timestamp')
            else:
                combined_df = new_df
            
            # Write back to parquet
            timeseries_file.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(
                timeseries_file,
                index=False,
                engine='pyarrow',
                compression='snappy'
            )
            
            logger.debug(f"✅ Emitted {len(rows)} time series rows to {timeseries_file}")
        except Exception as e:
            logger.warning(f"Failed to emit trend time series to {timeseries_file}: {e}")
            # Don't fail the run if trend emission fails
    
    def finalize_run(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_dir: Path,
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint dict
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize run: create snapshot, compute diffs, update baseline.
        
        This is the main entry point - call this after each run completes.
        
        CRITICAL: For SST consistency, pass `resolved_metadata` (the in-memory metadata dict
        that will be written to metadata.json). This ensures the snapshot/diff computation uses
        the exact same data that gets persisted, preventing coherence drift.
        
        Args:
            stage: Pipeline stage
            run_data: Run data from reproducibility tracker
            cohort_dir: Cohort directory
            cohort_metadata: Cohort metadata (fallback if resolved_metadata not provided)
            additional_data: Additional data
            resolved_metadata: In-memory metadata dict (SST - use this if available)
            run_identity: RunIdentity SST object with authoritative signatures
            prediction_fingerprint: Prediction fingerprint dict for predictions_sha256
        
        Returns:
            Diff telemetry data dict
        """
        # CRITICAL: Validate resolved_metadata matches current stage to prevent cross-stage contamination
        # Note: We don't strictly validate run_id format - run_ids are identifiers, not reproducibility factors.
        # What matters for reproducibility are fingerprints (data, config, feature, target) and comparison groups.
        if resolved_metadata:
            resolved_stage = resolved_metadata.get("stage")
            
            if resolved_stage != stage:
                raise ValueError(
                    f"Stage mismatch: resolved_metadata stage={resolved_stage}, current={stage}. "
                    f"This indicates cross-stage metadata contamination. Ensure full_metadata is stage-scoped."
                )
            # Run ID format differences (e.g., underscores vs T) don't affect reproducibility
            # Only log a debug message if formats differ significantly, but don't fail
            resolved_run_id = resolved_metadata.get("run_id")
            current_run_id = run_data.get('run_id') or run_data.get('timestamp')
            if resolved_run_id and current_run_id and resolved_run_id != current_run_id:
                # Normalize both to check if they represent the same timestamp
                # If they're just format differences, it's fine
                logger.debug(f"Run ID format differs (non-critical): resolved={resolved_run_id}, current={current_run_id}")
            
            # CRITICAL: Validate required fields are present and non-null for this stage
            # This ensures we catch incomplete SST before snapshot computation
            # BUT: Allow fallback extraction from run_data/additional_data for fields that might be set later
            required_fields = self._get_required_fields_for_stage(stage)
            missing_fields = []
            null_fields = []
            
            # Try to fill in missing fields from run_data or additional_data as fallback
            # This handles cases where metadata is built incrementally
            for field in required_fields:
                if field not in resolved_metadata or resolved_metadata[field] is None:
                    # Try fallback sources
                    fallback_value = None
                    if field == "target":
                        # For TARGET_RANKING, try multiple fallback sources
                        # target is passed to log_comparison() but might not be in run_data
                        fallback_value = (
                            run_data.get('target') or
                            run_data.get('target') or
                            run_data.get('target') or
                            (additional_data.get('target') if additional_data else None) or
                            (additional_data.get('target') if additional_data else None) or
                            (additional_data.get('target') if additional_data else None)
                        )
                        # Last resort: try to extract from cohort_dir path (e.g., .../TARGET_RANKING/SYMBOL_SPECIFIC/{target}/...)
                        if not fallback_value and cohort_dir:
                            try:
                                parts = Path(cohort_dir).parts
                                # Look for target in path (usually after view name)
                                for i, part in enumerate(parts):
                                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING'] and i + 2 < len(parts):
                                        # Next part might be view, then target
                                        if parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                                            if i + 2 < len(parts) and not parts[i+2].startswith('symbol=') and not parts[i+2].startswith('cohort='):
                                                fallback_value = parts[i+2]
                                                break
                                        elif not parts[i+1].startswith('symbol=') and not parts[i+1].startswith('cohort='):
                                            # No view, target is next
                                            fallback_value = parts[i+1]
                                            break
                            except Exception:
                                pass  # Don't fail if path parsing fails
                    
                    if fallback_value:
                        resolved_metadata[field] = fallback_value
                        logger.debug(f"Filled missing {field} from fallback: {fallback_value}")
                    elif field not in resolved_metadata:
                        missing_fields.append(field)
                    elif resolved_metadata[field] is None:
                        null_fields.append(field)
            
            if missing_fields or null_fields:
                error_parts = []
                if missing_fields:
                    error_parts.append(f"missing: {', '.join(missing_fields)}")
                if null_fields:
                    error_parts.append(f"null: {', '.join(null_fields)}")
                raise ValueError(
                    f"Incomplete resolved_metadata for stage={stage}: {', '.join(error_parts)}. "
                    f"Required fields must be present and non-null before finalize_run(). "
                    f"Ensure full_metadata is built AFTER all required fields are finalized."
                )
        
        # Create normalized snapshot (prefer resolved_metadata for SST consistency)
        # NEW: Pass run_identity and prediction_fingerprint for authoritative signatures
        snapshot = self.normalize_snapshot(
            stage=stage,
            run_data=run_data,
            cohort_metadata=cohort_metadata,
            additional_data=additional_data,
            cohort_dir=cohort_dir,
            resolved_metadata=resolved_metadata,
            run_identity=run_identity,
            prediction_fingerprint=prediction_fingerprint
        )
        
        # Save snapshot and get resolved cohort_dir
        # CRITICAL: save_snapshot may resolve/repair cohort_dir to target-first structure
        # We must use the returned path for save_diff to ensure both files end up in same directory
        resolved_snapshot_cohort_dir = self.save_snapshot(snapshot, cohort_dir)
        # Use resolved path for subsequent operations
        cohort_dir = resolved_snapshot_cohort_dir

        # Find previous comparable run
        prev_snapshot = self.find_previous_comparable(snapshot)
        
        # Try to find previous snapshot's cohort directory for trend comparison
        prev_cohort_dir = None
        if prev_snapshot and prev_snapshot.stage and prev_snapshot.view and prev_snapshot.target:
            # Reconstruct path similar to find_previous_comparable
            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
            target_clean = normalize_target_name(prev_snapshot.target)
            if hasattr(self, 'run_dir') and self.run_dir:
                results_dir = self.run_dir
                while results_dir.parent.exists() and results_dir.name != "RESULTS":
                    results_dir = results_dir.parent
                    if results_dir.name == "RESULTS":
                        break
                
                if results_dir.name == "RESULTS":
                    # Search for the previous run's cohort directory using discovery primitive
                    runs_dir = results_dir / "runs"
                    if runs_dir.exists():
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for cg_dir in iterdir_sorted(runs_dir):
                            if not cg_dir.is_dir() or not cg_dir.name.startswith("cg-"):
                                continue
                            # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                            for run_dir in iterdir_sorted(cg_dir):
                                if not run_dir.is_dir():
                                    continue
                                # Use existing discovery primitive (rglob handles nested structures including attempt dirs)
                                from TRAINING.orchestration.utils.target_first_paths import (
                                    find_cohort_dirs, parse_attempt_id_from_cohort_dir
                                )
                                cohort_dirs = find_cohort_dirs(
                                    base_output_dir=run_dir,
                                    target=target_clean,
                                    stage=prev_snapshot.stage,
                                    view=prev_snapshot.view
                                )
                                # Check each cohort directory for matching snapshot
                                for cohort_dir in cohort_dirs:
                                    snapshot_file = cohort_dir / "snapshot.json"
                                    if snapshot_file.exists():
                                        try:
                                            with open(snapshot_file, 'r') as f:
                                                snapshot_data = json.load(f)
                                                if snapshot_data.get('run_id') == prev_snapshot.run_id:
                                                    # Parse attempt_id from path and verify match
                                                    parsed_attempt_id = parse_attempt_id_from_cohort_dir(cohort_dir)
                                                    snap_attempt_id = getattr(prev_snapshot, 'attempt_id', 0)
                                                    if parsed_attempt_id == snap_attempt_id or snap_attempt_id == 0:
                                                        prev_cohort_dir = cohort_dir
                                                        break
                                        except Exception:
                                            continue
                                if prev_cohort_dir:
                                    break
                            if prev_cohort_dir:
                                break
                                if prev_cohort_dir:
                                    break
                            if prev_cohort_dir:
                                break
        
        # Compute diff against previous
        if prev_snapshot:
            diff = self.compute_diff(snapshot, prev_snapshot, prev_cohort_dir=prev_cohort_dir, curr_cohort_dir=cohort_dir)
        else:
            # First run / no previous run: return stable shape with empty excluded factors
            # CRITICAL: If not comparable, severity must be CRITICAL with reason
            diff = DiffResult(
                prev_run_id=None,  # Use None instead of "none" for clarity
                current_run_id=snapshot.run_id,
                comparable=False,
                comparability_reason="No previous comparable run found",
                prev_timestamp=None,
                prev_snapshot_seq=None,
                prev_stage=None,
                prev_view=None,
                comparison_source=None,
                severity=ChangeSeverity.CRITICAL,
                severity_reason="No previous comparable run found - cannot determine changes",
                excluded_factors_changed={},  # Empty but present
                summary={
                    'total_changes': 0,
                    'input_changes': 0,
                    'process_changes': 0,
                    'output_changes': 0,
                    'metric_deltas_total': 0,  # Explicitly set (was missing)
                    'metric_deltas_count': 0,
                    'metric_deltas_significant': 0,  # Also add this for consistency
                    'excluded_factors_changed': False,
                    'excluded_factors_summary': None
                }
            )
        
        # Get or establish baseline (stored per-cohort for exact matching)
        metrics = snapshot.outputs.get('metrics', {})
        baseline_state, is_new = self.get_or_establish_baseline(snapshot, metrics, cohort_dir)
        
        # Compute diff against baseline
        baseline_diff = None
        if baseline_state:
            # Load baseline snapshot - search by run_id since keys now include target/view
            baseline_snapshot = None
            for snap in self._snapshots.values():
                if snap.run_id == baseline_state.baseline_run_id:
                    baseline_snapshot = snap
                    break
            if baseline_snapshot:
                # Try to find baseline snapshot's cohort directory
                baseline_cohort_dir = None
                if baseline_snapshot.stage and baseline_snapshot.view and baseline_snapshot.target:
                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                    target_clean = normalize_target_name(baseline_snapshot.target)
                    if hasattr(self, 'run_dir') and self.run_dir:
                        results_dir = self.run_dir
                        while results_dir.parent.exists() and results_dir.name != "RESULTS":
                            results_dir = results_dir.parent
                            if results_dir.name == "RESULTS":
                                break
                        
                        if results_dir.name == "RESULTS":
                            runs_dir = results_dir / "runs"
                            if runs_dir.exists():
                                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                for cg_dir in iterdir_sorted(runs_dir):
                                    if not cg_dir.is_dir() or not cg_dir.name.startswith("cg-"):
                                        continue
                                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                    for run_dir in iterdir_sorted(cg_dir):
                                        if not run_dir.is_dir():
                                            continue
                                        # Try target-first structure first
                                        targets_dir = run_dir / "targets"
                                        if targets_dir.exists() and (targets_dir / target_clean).exists():
                                            target_dir = targets_dir / target_clean
                                            repro_dir = target_dir / "reproducibility"
                                            if repro_dir.exists():
                                                view_dir = repro_dir / baseline_snapshot.view
                                                if view_dir.exists():
                                                    # Check for symbol-specific path
                                                    if baseline_snapshot.symbol:
                                                        symbol_dir = view_dir / f"symbol={baseline_snapshot.symbol}"
                                                        if symbol_dir.exists():
                                                            # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                                            for cohort_subdir in iterdir_sorted(symbol_dir):
                                                                if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                    snapshot_file = cohort_subdir / "snapshot.json"
                                                                    if snapshot_file.exists():
                                                                        try:
                                                                            with open(snapshot_file, 'r') as f:
                                                                                snapshot_data = json.load(f)
                                                                                if snapshot_data.get('run_id') == baseline_snapshot.run_id:
                                                                                    baseline_cohort_dir = cohort_subdir
                                                                                    break
                                                                        except Exception:
                                                                            continue
                                                    # Check for cross-sectional path (no symbol)
                                                    if not baseline_cohort_dir:
                                                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                                        for cohort_subdir in iterdir_sorted(view_dir):
                                                            if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                snapshot_file = cohort_subdir / "snapshot.json"
                                                                if snapshot_file.exists():
                                                                    try:
                                                                        with open(snapshot_file, 'r') as f:
                                                                            snapshot_data = json.load(f)
                                                                            if snapshot_data.get('run_id') == baseline_snapshot.run_id:
                                                                                baseline_cohort_dir = cohort_subdir
                                                                                break
                                                                    except Exception:
                                                                        continue
                                        # Fallback to legacy REPRODUCIBILITY structure
                                        if not baseline_cohort_dir:
                                            stage_dir = run_dir / "REPRODUCIBILITY" / baseline_snapshot.stage / baseline_snapshot.view / target_clean
                                            if stage_dir.exists():
                                                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                                                for cohort_subdir in iterdir_sorted(stage_dir):
                                                    if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                        snapshot_file = cohort_subdir / "snapshot.json"
                                                        if snapshot_file.exists():
                                                            try:
                                                                with open(snapshot_file, 'r') as f:
                                                                    snapshot_data = json.load(f)
                                                                    if snapshot_data.get('run_id') == baseline_snapshot.run_id:
                                                                        baseline_cohort_dir = cohort_subdir
                                                                        break
                                                            except Exception:
                                                                continue
                                        if baseline_cohort_dir:
                                            break
                                    if baseline_cohort_dir:
                                        break
                
                baseline_diff = self.compute_diff(snapshot, baseline_snapshot, prev_cohort_dir=baseline_cohort_dir, curr_cohort_dir=cohort_dir)
        
        # CRITICAL: Use self.run_dir (current run) instead of walking up from cohort_dir
        # Walking up from cohort_dir could find a previous run's directory if cohort_dir is wrong
        # self.run_dir is the authoritative current run directory
        base_output_dir = self.run_dir
        if base_output_dir is None:
            raise ValueError(f"Cannot determine current run directory: self.run_dir is None. Cannot save diff files.")
        
        # Save diffs - pass structured inputs from snapshot AND explicit base_output_dir
        resolved_cohort_dir = self.save_diff(
            diff, baseline_diff, cohort_dir,
            view=snapshot.view,
            target=snapshot.target,
            symbol=snapshot.symbol,
            stage=snapshot.stage,
            base_output_dir=base_output_dir  # NEW: Pass explicit run root (SST)
        )
        
        # CRITICAL: Validate resolved_cohort_dir is within current run before emitting trends
        # If save_diff() returned early due to path repair failure, skip trend emission
        if base_output_dir:
            base = Path(base_output_dir).resolve()
            out = Path(resolved_cohort_dir).resolve()
            try:
                is_within = out.is_relative_to(base)
            except AttributeError:
                try:
                    out.relative_to(base)
                    is_within = True
                except ValueError:
                    is_within = False
            
            if not is_within:
                logger.warning(
                    f"Skipping trend time series emission: resolved_cohort_dir ({resolved_cohort_dir}) "
                    f"is outside base_output_dir ({base_output_dir}). This prevents cross-run contamination."
                )
            else:
                # Emit trend time series data for trend/drift analysis
                # Use resolved_cohort_dir (corrected path) instead of original cohort_dir
                metrics = snapshot.outputs.get('metrics', {})
                if metrics:
                    self._emit_trend_time_series(snapshot, metrics, resolved_cohort_dir)
        else:
            # If base_output_dir is None, skip trend emission (shouldn't happen, but defensive)
            logger.warning("Skipping trend time series emission: base_output_dir is None")
        
        # Return diff data for integration into metadata/metrics
        diff_telemetry_data = {
            'diff': diff.to_dict(),
            'baseline_diff': baseline_diff.to_dict() if baseline_diff else None,
            'snapshot': snapshot.to_dict()
        }
        
        # Store resolved path in return dict (backward compatible - adds key to existing dict)
        diff_telemetry_data['resolved_cohort_dir'] = str(resolved_cohort_dir)  # Convert to string for JSON serialization
        
        logger.info(f"✅ Telemetry finalized for {stage}:{snapshot.target or 'unknown'}")
        if diff.comparable:
            logger.info(f"   Changes: {len(diff.changed_keys)} keys, severity={diff.severity.value}")
            if diff.metric_deltas:
                for metric, delta in diff.metric_deltas.items():
                    logger.info(f"   {metric}: {delta['delta_abs']:+.4f} ({delta['delta_pct']:+.2f}%)")
            # Surface excluded factors loudly
            if diff.excluded_factors_changed and diff.summary.get('excluded_factors_summary'):
                logger.warning(f"   ⚠️  Excluded factors changed: {diff.summary['excluded_factors_summary']}")
        
        return diff_telemetry_data
    
    def get_run_hash_with_changes(
        self,
        run_id: Optional[str] = None,
        prev_run_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get run hash with change detection for this run.
        
        Args:
            run_id: Optional run ID to filter snapshots
            prev_run_id: Optional previous run ID for change detection
        
        Returns:
            Dict with run_hash, run_id, changes summary, or None if no snapshots found
        """
        return compute_run_hash_with_changes(
            output_dir=self.output_dir,
            run_id=run_id,
            prev_run_id=prev_run_id,
            diff_telemetry=self
        )


# NOTE: Run hash functions have been extracted to the run_hash submodule.
# The following functions are now imported from diff_telemetry/run_hash.py:
#   - _extract_deterministic_fields()
#   - compute_full_run_hash()
#   - _load_manifest_comparability_flags()
#   - _normalize_run_id_for_comparison()
#   - _can_runs_be_compared()
#   - compute_run_hash_with_changes()
#   - save_run_hash()
# See the import statement at the top of this file.

