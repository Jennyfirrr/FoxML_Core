# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Manifest Generation Utilities

Creates and updates manifest.json at run root with run metadata and target index.
"""

import json
import logging
import hashlib
import sys
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import subprocess
import numpy as np
import uuid
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Import git utilities from SST module
from TRAINING.common.utils.git_utils import get_git_commit as get_git_sha

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View

# Import SST accessor functions
from TRAINING.orchestration.utils.reproducibility.utils import (
    extract_auc,
    extract_n_effective,
    extract_date_range,
    collect_environment_info,
)

# Import canonicalization from SST
from TRAINING.common.utils.config_hashing import canonicalize, canonical_json, sha256_full
# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import iterdir_sorted, rglob_sorted


def derive_run_id_from_identity(
    run_identity: Optional[Any] = None
) -> str:
    """
    Derive deterministic run_id from RunIdentity (pure function, SST pattern).
    
    Pure function: only works when identity exists, raises if missing.
    No mode parameter - pure ID derivation only.
    
    Pattern:
    - Uses hash-based derivation to avoid prefix collisions
    - run_id = f"ridv1_{sha256(strict_key + ':' + replicate_key)[:20]}"
    
    Args:
        run_identity: RunIdentity object (required for stable run_id)
    
    Returns:
        Deterministic run_id string in format: ridv1_{20-char-hash}
    
    Raises:
        ValueError: If run_identity not provided, not finalized, or missing keys
    """
    # Extract keys from RunIdentity
    strict_key = None
    replicate_key = None
    
    if run_identity is not None:
        # Handle RunIdentity object
        if hasattr(run_identity, 'is_final') and run_identity.is_final:
            if hasattr(run_identity, 'strict_key') and hasattr(run_identity, 'replicate_key'):
                strict_key = run_identity.strict_key
                replicate_key = run_identity.replicate_key
        # Handle dict with keys
        elif isinstance(run_identity, dict):
            if run_identity.get('is_final') and run_identity.get('strict_key') and run_identity.get('replicate_key'):
                strict_key = run_identity['strict_key']
                replicate_key = run_identity['replicate_key']
    
    # Validate keys are available
    if not strict_key or not replicate_key:
        raise ValueError(
            "Cannot derive run_id without finalized RunIdentity. "
            "Provide run_identity with is_final=True, strict_key, and replicate_key."
        )
    
    # Hash-based derivation: sha256(strict_key + ":" + replicate_key)[:20]
    combined = f"{strict_key}:{replicate_key}"
    hash_digest = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:20]
    
    return f"ridv1_{hash_digest}"


def derive_unstable_run_id(
    run_instance_id: str
) -> str:
    """
    Derive unstable run_id from run instance ID (pure function).
    
    Used when identity is unavailable. Returns deterministic ID based on instance ID.
    
    Pattern:
    - run_id = f"rid_unstable_{instance_id}"
    
    Args:
        run_instance_id: Run instance identifier (e.g., from generate_run_instance_id())
    
    Returns:
        Unstable run_id string in format: rid_unstable_{instance_id}
    """
    return f"rid_unstable_{run_instance_id}"


def read_run_id_from_manifest(manifest_path: Path) -> Optional[str]:
    """
    Read run_id from manifest.json (SST helper).
    
    Centralized manifest reading ensures format changes are single-point.
    
    Args:
        manifest_path: Path to manifest.json
    
    Returns:
        run_id string if found and valid, None otherwise
    """
    if not manifest_path.exists():
        logger.debug(f"Manifest not found: {manifest_path}")
        return None
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            run_id = manifest.get('run_id')
            # Validate it's a non-empty string
            if run_id and isinstance(run_id, str) and run_id.strip():
                logger.debug(f"Read run_id from manifest: {run_id}")
                return run_id.strip()
            else:
                logger.debug(f"Manifest exists but run_id is missing/empty/invalid: {manifest_path}")
    except json.JSONDecodeError as e:
        logger.debug(f"Manifest JSON parse error: {e} (path: {manifest_path})")
    except Exception as e:
        logger.debug(f"Error reading manifest: {e} (path: {manifest_path})")
    
    return None


def assess_comparability(
    run_identity: Optional[Any] = None,
    dataset_snapshot_hash: Optional[str] = None,
    mode: str = "best-effort"
) -> tuple[bool, str]:
    """
    Assess run comparability and determine run_id kind (pure function).
    
    Separates ID derivation from comparability assessment (prevents drift).
    
    Strict mode: if ANY comparability prerequisite missing (run_identity OR required dataset snapshot hash) → raise
    Best-effort mode: if prerequisites missing → (False, "unstable"), else → (True, "stable")
    
    Args:
        run_identity: RunIdentity object (optional)
        dataset_snapshot_hash: Data snapshot hash from data layer (parquet manifest, feature store commit, etc.) (optional)
        mode: "strict" or "best-effort" (default: "best-effort")
    
    Returns:
        Tuple of (is_comparable: bool, run_id_kind: "stable" | "unstable")
    
    Raises:
        ValueError: In strict mode if run_identity is missing or not finalized
        ValueError: In strict mode if dataset_snapshot_hash is required but missing
    """
    # Check if run_identity is available and finalized
    has_identity = False
    if run_identity is not None:
        if hasattr(run_identity, 'is_final') and run_identity.is_final:
            if hasattr(run_identity, 'strict_key') and hasattr(run_identity, 'replicate_key'):
                has_identity = True
        elif isinstance(run_identity, dict):
            if run_identity.get('is_final') and run_identity.get('strict_key') and run_identity.get('replicate_key'):
                has_identity = True
    
    # RI-006: Check for dataset_snapshot_hash
    has_dataset_snapshot = dataset_snapshot_hash is not None and dataset_snapshot_hash.strip() != ""

    # Strict mode: require run_identity and optionally dataset_snapshot_hash
    if mode == "strict":
        if not has_identity:
            raise ValueError(
                "Cannot assess comparability in strict mode without finalized RunIdentity. "
                "Provide run_identity with is_final=True, strict_key, and replicate_key."
            )
        # RI-006: Enforce dataset_snapshot_hash in strict mode
        # Check if strict enforcement is enabled (defaults to warning-only during transition)
        try:
            from CONFIG.config_loader import get_cfg
            require_snapshot = get_cfg("pipeline.determinism.require_dataset_snapshot_strict", default=False)
        except Exception:
            require_snapshot = False

        if not has_dataset_snapshot:
            if require_snapshot:
                raise ValueError(
                    "Cannot assess comparability in strict mode without dataset snapshot hash. "
                    "Provide dataset_snapshot_hash from data layer (parquet manifest, feature store commit, etc.). "
                    "To disable this check, set pipeline.determinism.require_dataset_snapshot_strict=false"
                )
            else:
                # Graduated enforcement: warn but don't fail during transition period
                import logging
                logging.getLogger(__name__).warning(
                    "⚠️ RI-006: dataset_snapshot_hash not provided in strict mode. "
                    "Run comparability may be affected. Set pipeline.determinism.require_dataset_snapshot_strict=true "
                    "to enforce this requirement."
                )
        return (True, "stable")

    # Best-effort mode: allow missing prerequisites but track status
    if has_identity:
        # With identity but no snapshot, we're stable for identity comparison
        # but cannot guarantee data hasn't changed between runs
        if has_dataset_snapshot:
            return (True, "stable")
        else:
            # Still return stable for identity-based comparison, but note limitation
            return (True, "stable")
    else:
        return (False, "unstable")


@dataclass
class RunInstanceParts:
    """Parsed components of a run instance directory name."""
    prefix: str  # "intelligent_output" or "intelligent-output"
    date_str: str  # "YYYYMMDD"
    time_str: str  # "HHMMSS"
    uuid_suffix: Optional[str] = None  # Optional UUID suffix (8 chars)


def generate_run_instance_id() -> str:
    """
    Generate unique instance ID for run directory name.
    
    Format: intelligent_output_YYYYMMDD_HHMMSS_{uuid4()[:8]}
    
    Used for directory name only, never for snapshot matching.
    
    Returns:
        Run instance ID string
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    uuid_suffix = uuid.uuid4().hex[:8]
    return f"intelligent_output_{date_str}_{time_str}_{uuid_suffix}"


def parse_run_instance_dirname(dirname: str) -> Optional[RunInstanceParts]:
    """
    Parse run instance directory name with tolerant handling of multiple formats.
    
    Handles:
    - Old underscore form: intelligent_output_YYYYMMDD_HHMMSS
    - Old dash form: intelligent-output-YYYYMMDD-HHMMSS
    - New suffix form: intelligent_output_YYYYMMDD_HHMMSS_{uuid8}
    
    Returns None for invalid formats (rejects garbage cleanly).
    
    Args:
        dirname: Directory name to parse
    
    Returns:
        RunInstanceParts dataclass with parsed components, or None if invalid
    """
    if not dirname or not isinstance(dirname, str):
        return None
    
    # Pattern 1: New suffix form: intelligent_output_YYYYMMDD_HHMMSS_{uuid8}
    pattern1 = r'^intelligent_output_(\d{8})_(\d{6})_([a-f0-9]{8})$'
    match1 = re.match(pattern1, dirname)
    if match1:
        return RunInstanceParts(
            prefix="intelligent_output",
            date_str=match1.group(1),
            time_str=match1.group(2),
            uuid_suffix=match1.group(3)
        )
    
    # Pattern 2: Old underscore form: intelligent_output_YYYYMMDD_HHMMSS
    pattern2 = r'^intelligent_output_(\d{8})_(\d{6})$'
    match2 = re.match(pattern2, dirname)
    if match2:
        return RunInstanceParts(
            prefix="intelligent_output",
            date_str=match2.group(1),
            time_str=match2.group(2),
            uuid_suffix=None
        )
    
    # Pattern 3: Old dash form: intelligent-output-YYYYMMDD-HHMMSS
    pattern3 = r'^intelligent-output-(\d{8})-(\d{6})$'
    match3 = re.match(pattern3, dirname)
    if match3:
        return RunInstanceParts(
            prefix="intelligent-output",
            date_str=match3.group(1),
            time_str=match3.group(2),
            uuid_suffix=None
        )
    
    # No match - invalid format
    return None


def create_manifest(
    output_dir: Path,
    run_id: Optional[str] = None,
    config_digest: Optional[str] = None,
    targets: Optional[List[str]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    run_identity: Optional[Any] = None,  # NEW: RunIdentity for deterministic run_id
    dataset_snapshot_hash: Optional[str] = None,  # NEW: Data snapshot hash from data layer
    run_instance_id: Optional[str] = None,  # NEW: Run instance ID for directory name
    mode: str = "best-effort"  # NEW: "strict" or "best-effort" for comparability assessment
) -> Path:
    """
    Create manifest.json at run root with comprehensive run metadata.
    
    Args:
        output_dir: Base run output directory
        run_id: Run identifier (if not provided, derived from run_identity or generated)
        config_digest: Config digest/hash for reproducibility
        targets: List of targets processed
        experiment_config: Optional experiment config dict (name, data_dir, symbols, etc.)
        run_metadata: Optional additional run metadata (data_dir, symbols, n_effective, etc.)
        run_identity: RunIdentity object for deterministic run_id derivation
        dataset_snapshot_hash: Data snapshot hash from data layer (parquet manifest, feature store commit, etc.)
        run_instance_id: Run instance ID for directory name (if not provided, generated)
        mode: "strict" or "best-effort" for comparability assessment (default: "best-effort")
    
    Returns:
        Path to manifest.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if strict mode
    try:
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            mode = "strict"
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Could not check strict mode: {e}")
        # Use provided mode
    
    # Assess comparability and determine run_id kind
    try:
        is_comparable, run_id_kind = assess_comparability(
            run_identity=run_identity,
            dataset_snapshot_hash=dataset_snapshot_hash,
            mode=mode
        )
    except ValueError:
        # Strict mode raised - mark as unstable
        is_comparable = False
        run_id_kind = "unstable"
    
    # Generate run_id if not provided
    if run_id is None:
        if is_comparable and run_id_kind == "stable":
            # Use stable ID derivation
            try:
                run_id = derive_run_id_from_identity(run_identity=run_identity)
            except ValueError:
                # Fallback to unstable if derivation fails
                if run_instance_id is None:
                    run_instance_id = generate_run_instance_id()
                run_id = derive_unstable_run_id(run_instance_id)
                is_comparable = False
                run_id_kind = "unstable"
        else:
            # Use unstable ID derivation
            if run_instance_id is None:
                run_instance_id = generate_run_instance_id()
            run_id = derive_unstable_run_id(run_instance_id)
    
    # Generate run_instance_id if not provided (for directory name tracking)
    if run_instance_id is None:
        # Try to extract from output_dir name, or generate new
        parsed = parse_run_instance_dirname(output_dir.name)
        if parsed:
            # Reconstruct from parsed parts
            if parsed.uuid_suffix:
                run_instance_id = f"{parsed.prefix}_{parsed.date_str}_{parsed.time_str}_{parsed.uuid_suffix}"
            else:
                run_instance_id = f"{parsed.prefix}_{parsed.date_str}_{parsed.time_str}"
        else:
            run_instance_id = generate_run_instance_id()
    
    # Get git SHA
    git_sha = get_git_sha()
    
    # Build manifest structure - always include core fields
    manifest = {
        "run_id": run_id,
        "run_instance_id": run_instance_id,  # NEW: Separate instance ID for directory name
        "is_comparable": is_comparable,  # NEW: Comparability flag
        "run_id_kind": run_id_kind,  # NEW: "stable" or "unstable"
        "git_sha": git_sha,
        "config_digest": config_digest,
        "created_at": datetime.now().isoformat(),
        "targets": targets or []
    }
    
    # Store full identity keys in metadata if available (for debugging/audit)
    if run_identity is not None:
        if hasattr(run_identity, 'strict_key') and hasattr(run_identity, 'replicate_key'):
            manifest["run_identity"] = {
                "strict_key": run_identity.strict_key,
                "replicate_key": run_identity.replicate_key,
                "is_final": getattr(run_identity, 'is_final', False)
            }
        elif isinstance(run_identity, dict):
            manifest["run_identity"] = {
                "strict_key": run_identity.get('strict_key'),
                "replicate_key": run_identity.get('replicate_key'),
                "is_final": run_identity.get('is_final', False)
            }
    
    # Store dataset snapshot hash if available
    if dataset_snapshot_hash:
        manifest["dataset_snapshot_hash"] = dataset_snapshot_hash
    
    # Add experiment config if provided (always include experiment section for consistency)
    if experiment_config:
        manifest["experiment"] = {
            "name": experiment_config.get("name") or experiment_config.get("experiment", {}).get("name"),
            "description": experiment_config.get("description") or experiment_config.get("experiment", {}).get("description"),
            "data_dir": str(experiment_config.get("data_dir")) if experiment_config.get("data_dir") else None,
            "symbols": experiment_config.get("symbols") or experiment_config.get("data", {}).get("symbols"),
            "interval": experiment_config.get("interval") or experiment_config.get("data", {}).get("bar_interval"),
            "max_samples_per_symbol": experiment_config.get("max_samples_per_symbol") or experiment_config.get("data", {}).get("max_rows_per_symbol")
        }
    
    # Always include run_metadata section (even if empty/null) for consistency
    manifest["run_metadata"] = {}
    if run_metadata:
        # DETERMINISM: Use sorted_items() for deterministic iteration order
        from TRAINING.common.utils.determinism_ordering import sorted_items
        manifest["run_metadata"] = {
            k: v for k, v in sorted_items(run_metadata)
            if k not in ["experiment_config", "targets"]  # Avoid duplication
        }
    
    # Always include target_index section (even if empty) for consistency
    manifest["target_index"] = {}
    targets_dir = output_dir / "targets"
    if targets_dir.exists():
        target_index = {}
        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        for target_dir in iterdir_sorted(targets_dir):
            if target_dir.is_dir():
                target = target_dir.name
                target_info = {
                    "decision": _find_files(target_dir / "decision"),
                    "models": _find_model_families(target_dir / "models"),
                    "metrics": _find_files(target_dir / "metrics"),
                    "trends": _find_files(target_dir / "trends"),
                    "reproducibility": _find_files(target_dir / "reproducibility")
                }
                target_index[target] = target_info
        manifest["target_index"] = target_index
    
    # Add routing and training plan hashes for fast change detection
    manifest["plan_hashes"] = _compute_plan_hashes(output_dir)
    
    # Add trend reports references if they exist
    # trend_reports is at RESULTS/trend_reports/ (outside run directories)
    # Find RESULTS directory by walking up from output_dir
    results_dir = output_dir
    for _ in range(10):
        if results_dir.name == "RESULTS":
            break
        if not results_dir.parent.exists():
            break
        results_dir = results_dir.parent
    
    if results_dir.name == "RESULTS":
        trend_reports_dir = results_dir / "trend_reports"
        if trend_reports_dir.exists():
            trend_index = {}
            by_target_dir = trend_reports_dir / "by_target"
            if by_target_dir.exists():
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                for target_dir in iterdir_sorted(by_target_dir):
                    if target_dir.is_dir():
                        target = target_dir.name
                        trend_files = {}
                        for trend_file in ["performance_timeseries.parquet", "routing_score_timeseries.parquet", 
                                         "feature_importance_timeseries.parquet"]:
                            trend_path = target_dir / trend_file
                            if trend_path.exists():
                                # Store relative path from RESULTS directory
                                trend_files[trend_file.replace(".parquet", "")] = str(trend_path.relative_to(results_dir))
                        if trend_files:
                            trend_index[target] = trend_files
            
            # Also add by_run references
            by_run_dir = trend_reports_dir / "by_run"
            if by_run_dir.exists():
                run_snapshots = {}
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                for run_dir in iterdir_sorted(by_run_dir):
                    if run_dir.is_dir():
                        snapshot_file = run_dir / f"{run_dir.name}_summary.json"
                        if snapshot_file.exists():
                            run_snapshots[run_dir.name] = str(snapshot_file.relative_to(results_dir))
                if run_snapshots:
                    trend_index["_by_run"] = run_snapshots
            
            if trend_index:
                manifest["trend_reports"] = trend_index
    
    # Add run hash if available
    globals_dir = output_dir / "globals"
    run_hash_file = globals_dir / "run_hash.json"
    if run_hash_file.exists():
        try:
            with open(run_hash_file, 'r') as f:
                run_hash_data = json.load(f)
                manifest["run_hash"] = run_hash_data.get("run_hash")
                manifest["run_id"] = run_hash_data.get("run_id")
                if run_hash_data.get("changes"):
                    manifest["run_changes"] = {
                        "severity": run_hash_data["changes"].get("severity_summary"),
                        "changed_snapshots_count": len(run_hash_data["changes"].get("changed_snapshots", [])),
                    }
        except Exception as e:
            logger.debug(f"Failed to load run hash: {e}")
    
    # Write manifest
    # SST: Sanitize manifest data to normalize enums to strings before JSON serialization
    from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json
    sanitized_manifest = _sanitize_for_json(manifest)
    
    manifest_path = output_dir / "manifest.json"
    # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
    from TRAINING.common.utils.file_utils import write_atomic_json
    write_atomic_json(manifest_path, sanitized_manifest, default=str)
    
    logger.info(f"✅ Created manifest.json: {manifest_path}")
    return manifest_path


def update_manifest(
    output_dir: Path,
    target: str,
    selected_feature_set_digest: Optional[str] = None,
    split_ids: Optional[List[str]] = None,
    model_artifact_paths: Optional[Dict[str, List[str]]] = None,
    decision_paths: Optional[Dict[str, str]] = None
) -> None:
    """
    Update manifest.json with target-specific information.
    
    Args:
        output_dir: Base run output directory
        target: Target name
        selected_feature_set_digest: Feature set digest
        split_ids: List of split IDs
        model_artifact_paths: Dict mapping family -> list of artifact paths
        decision_paths: Dict mapping decision type -> path
    """
    manifest_path = output_dir / "manifest.json"
    
    if not manifest_path.exists():
        # Create initial manifest if it doesn't exist
        create_manifest(output_dir)
    
    # Load existing manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Initialize target_index if needed
    if "target_index" not in manifest:
        manifest["target_index"] = {}
    
    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
    target_clean = normalize_target_name(target)
    
    # Update or create target entry
    if target_clean not in manifest["target_index"]:
        manifest["target_index"][target_clean] = {}
    
    target_entry = manifest["target_index"][target_clean]
    
    # Update target-specific fields
    if selected_feature_set_digest:
        target_entry["selected_feature_set_digest"] = selected_feature_set_digest
    if split_ids:
        target_entry["split_ids"] = split_ids
    if model_artifact_paths:
        target_entry["model_artifact_paths"] = model_artifact_paths
    if decision_paths:
        target_entry["decision_paths"] = decision_paths
    
    # Update timestamp
    manifest["updated_at"] = datetime.now().isoformat()

    # SST: Sanitize manifest data to normalize enums to strings before JSON serialization
    from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json
    sanitized_manifest = _sanitize_for_json(manifest)
    
    # Write updated manifest
    # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
    from TRAINING.common.utils.file_utils import write_atomic_json
    write_atomic_json(manifest_path, sanitized_manifest, default=str)
    
    logger.debug(f"Updated manifest.json for target {target_clean}")


def update_manifest_with_plan_hashes(output_dir: Path) -> None:
    """
    Update manifest.json with routing and training plan hashes.
    
    This should be called after routing and training plans are generated and saved.
    
    Args:
        output_dir: Base run output directory
    """
    manifest_path = output_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.debug("Manifest not found, skipping plan_hashes update")
        return
    
    try:
        # Load existing manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Compute plan hashes (may be null if plans don't exist yet)
        plan_hashes = _compute_plan_hashes(output_dir)
        
        # Update manifest with plan hashes
        manifest["plan_hashes"] = plan_hashes
        
        # Write updated manifest
        # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
        from TRAINING.common.utils.file_utils import write_atomic_json
        write_atomic_json(manifest_path, manifest, default=str)
        
        # Log which hashes were found
        if plan_hashes.get("routing_plan_hash"):
            logger.debug(f"Updated manifest with routing_plan_hash: {plan_hashes['routing_plan_hash'][:16]}...")
        else:
            logger.debug("routing_plan_hash is null (routing plan may not exist yet)")
        
        if plan_hashes.get("training_plan_hash"):
            logger.debug(f"Updated manifest with training_plan_hash: {plan_hashes['training_plan_hash'][:16]}...")
        else:
            logger.debug("training_plan_hash is null (training plan may not exist yet)")
        
        logger.debug("Updated manifest.json with plan_hashes")
    except Exception as e:
        logger.warning(f"Failed to update manifest with plan_hashes: {e}")


def update_manifest_with_run_hash(output_dir: Path) -> None:
    """
    Update manifest.json with run_hash and run_changes from globals/run_hash.json.
    
    This should be called after run_hash is computed and saved.
    
    Args:
        output_dir: Base run output directory
    """
    manifest_path = output_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.debug("Manifest not found, skipping run_hash update")
        return
    
    try:
        # Load existing manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Load run_hash if available
        globals_dir = output_dir / "globals"
        run_hash_file = globals_dir / "run_hash.json"
        if run_hash_file.exists():
            try:
                with open(run_hash_file, 'r') as f:
                    run_hash_data = json.load(f)
                    manifest["run_hash"] = run_hash_data.get("run_hash")
                    # Update run_id from run_hash if available (more authoritative)
                    if run_hash_data.get("run_id"):
                        manifest["run_id"] = run_hash_data.get("run_id")
                    if run_hash_data.get("changes"):
                        manifest["run_changes"] = {
                            "severity": run_hash_data["changes"].get("severity_summary"),
                            "changed_snapshots_count": len(run_hash_data["changes"].get("changed_snapshots", [])),
                        }
            except Exception as e:
                logger.debug(f"Failed to load run hash for manifest update: {e}")
        
        # Also refresh target_index and other fields that may have changed
        targets_dir = output_dir / "targets"
        if targets_dir.exists():
            target_index = {}
            target_names = []
            # DETERMINISM: Use iterdir_sorted for deterministic iteration order
            for target_dir in iterdir_sorted(targets_dir):
                if target_dir.is_dir():
                    target = target_dir.name
                    target_names.append(target)
                    target_info = {
                        "decision": _find_files(target_dir / "decision"),
                        "models": _find_model_families(target_dir / "models"),
                        "metrics": _find_files(target_dir / "metrics"),
                        "trends": _find_files(target_dir / "trends"),
                        "reproducibility": _find_files(target_dir / "reproducibility")
                    }
                    target_index[target] = target_info
            manifest["target_index"] = target_index
            # Update targets list if it was empty or outdated
            if not manifest.get("targets") or len(manifest.get("targets", [])) != len(target_names):
                manifest["targets"] = sorted(target_names)
        
        # Ensure run_metadata exists if we have the data (preserve existing or add if missing)
        # This ensures run_metadata is populated even if it wasn't in initial manifest
        if "run_metadata" not in manifest:
            # Try to extract from experiment section or add placeholder
            run_metadata = {}
            if manifest.get("experiment"):
                exp = manifest["experiment"]
                if exp.get("data_dir"):
                    run_metadata["data_dir"] = exp["data_dir"]
                if exp.get("symbols"):
                    run_metadata["symbols"] = exp["symbols"]
            if run_metadata:
                manifest["run_metadata"] = run_metadata
        
        # Update timestamp
        manifest["updated_at"] = datetime.now().isoformat()

        # SST: Sanitize manifest data to normalize enums to strings before JSON serialization
        from TRAINING.orchestration.utils.diff_telemetry import _sanitize_for_json
        sanitized_manifest = _sanitize_for_json(manifest)
        
        # Write updated manifest
        # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
        from TRAINING.common.utils.file_utils import write_atomic_json
        write_atomic_json(manifest_path, sanitized_manifest, default=str)
        
        logger.debug("Updated manifest.json with run_hash and refreshed target_index")
    except Exception as e:
        logger.warning(f"Failed to update manifest with run_hash: {e}")


def _find_files(directory: Path) -> List[str]:
    """Find all files in a directory recursively."""
    if not directory.exists():
        return []
    files = []
    # DETERMINISM: Use rglob_sorted for deterministic iteration order
    for path in rglob_sorted(directory, "*"):
        if path.is_file():
            files.append(str(path.relative_to(directory.parent.parent)))
    return sorted(files)


def _compute_plan_hashes(output_dir: Path) -> Dict[str, Optional[str]]:
    """
    Compute SHA256 hashes of routing and training plans for fast change detection.
    
    Args:
        output_dir: Base run output directory
    
    Returns:
        Dict with 'routing_plan_hash' and 'training_plan_hash' (None if files don't exist)
    """
    hashes = {
        "routing_plan_hash": None,
        "training_plan_hash": None
    }
    
    globals_dir = output_dir / "globals"
    
    # Hash routing plan (globals/routing_plan/routing_plan.json)
    routing_plan_path = globals_dir / "routing_plan" / "routing_plan.json"
    if routing_plan_path.exists():
        try:
            with open(routing_plan_path, 'rb') as f:
                routing_content = f.read()
            hashes["routing_plan_hash"] = sha256_full(routing_content)
        except Exception as e:
            logger.debug(f"Failed to hash routing plan: {e}")
    
    # Hash training plan (globals/training_plan/master_training_plan.json or training_plan.json)
    training_plan_dir = globals_dir / "training_plan"
    training_plan_path = training_plan_dir / "master_training_plan.json"
    if not training_plan_path.exists():
        # Fallback to training_plan.json
        training_plan_path = training_plan_dir / "training_plan.json"
    
    if training_plan_path.exists():
        try:
            with open(training_plan_path, 'rb') as f:
                training_content = f.read()
            hashes["training_plan_hash"] = sha256_full(training_content)
        except Exception as e:
            logger.debug(f"Failed to hash training plan: {e}")
    
    return hashes


def _find_model_families(models_dir: Path) -> Dict[str, List[str]]:
    """Find model families and their artifacts."""
    if not models_dir.exists():
        return {}
    
    families = {}
    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
    for family_dir in iterdir_sorted(models_dir):
        if family_dir.is_dir():
            family_name = family_dir.name
            artifacts = []
            # DETERMINISM: Use rglob_sorted for deterministic iteration order
            for artifact_path in rglob_sorted(family_dir, "*"):
                if artifact_path.is_file():
                    artifacts.append(str(artifact_path.relative_to(models_dir.parent.parent)))
            if artifacts:
                families[family_name] = sorted(artifacts)
    
    return families


def create_target_metadata(
    output_dir: Path,
    target: str
) -> Path:
    """
    Create per-target metadata.json that aggregates information from all cohorts.
    
    This file provides a single source of truth for all metadata related to a target,
    aggregating information from:
    - All cohort directories (CROSS_SECTIONAL and SYMBOL_SPECIFIC)
    - Decision files
    - Metrics summaries
    - Feature selection results
    
    Args:
        output_dir: Base run output directory
        target: Target name (will be cleaned)
    
    Returns:
        Path to target metadata.json
    """
    import json
    from TRAINING.orchestration.utils.target_first_paths import (
        get_target_reproducibility_dir, get_target_decision_dir, get_target_metrics_dir
    )
    
    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
    target_clean = normalize_target_name(target)
    target_dir = output_dir / "targets" / target_clean
    
    if not target_dir.exists():
        logger.warning(f"Target directory does not exist: {target_dir}")
        return None
    
    target_metadata = {
        "target": target,
        "target_clean": target_clean,
        "created_at": datetime.now().isoformat(),
        "cohorts": {},
        "views": {},
        "decisions": {},
        "metrics_summary": {}
    }
    
    # Collect cohort metadata from reproducibility directory
    repro_dir = get_target_reproducibility_dir(output_dir, target_clean)
    if repro_dir.exists():
        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        for view_dir in iterdir_sorted(repro_dir):
            if view_dir.is_dir() and view_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
                view_name = view_dir.name
                target_metadata["views"][view_name] = {
                    "cohorts": []
                }
                
                # DETERMINISTIC: Use rglob_sorted to find all cohort directories deterministically
                # This works for both CROSS_SECTIONAL (batch_*/attempt_*/cohort=*) and SYMBOL_SPECIFIC (symbol=*/attempt_*/cohort=*)
                for cohort_dir in rglob_sorted(view_dir, "cohort=*"):
                    if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                        cohort_id = cohort_dir.name.replace("cohort=", "")
                        metadata_file = cohort_dir / "metadata.json"
                        
                        # Extract symbol from path for SYMBOL_SPECIFIC (if present)
                        symbol = None
                        for part in cohort_dir.parts:
                            if part.startswith("symbol="):
                                symbol = part.replace("symbol=", "")
                                break
                        
                        cohort_info = {
                            "cohort_id": cohort_id,
                            "path": str(cohort_dir.relative_to(output_dir))
                        }
                        if symbol:
                            cohort_info["symbol"] = symbol
                        
                        # Load metadata if available
                        if metadata_file.exists():
                            try:
                                with open(metadata_file) as f:
                                    cohort_metadata = json.load(f)
                                    date_start, date_end = extract_date_range(cohort_metadata)
                                    from TRAINING.orchestration.utils.reproducibility.utils import extract_view, extract_universe_sig
                                    cohort_info["metadata"] = {
                                        "stage": cohort_metadata.get("stage"),
                                        "view": extract_view(cohort_metadata),  # SST: Use view instead of view
                                        "n_effective": extract_n_effective(cohort_metadata),
                                        "n_symbols": cohort_metadata.get("n_symbols"),
                                        "date_start": date_start,
                                        "date_end": date_end,
                                        "universe_sig": extract_universe_sig(cohort_metadata),  # SST: Use universe_sig
                                        "min_cs": cohort_metadata.get("min_cs"),
                                        "max_cs_samples": cohort_metadata.get("max_cs_samples")
                                    }
                            except Exception as e:
                                logger.debug(f"Failed to load metadata from {metadata_file}: {e}")
                        
                        target_metadata["views"][view_name]["cohorts"].append(cohort_info)
                        cohort_entry = {
                            "view": view_name,
                            "path": str(cohort_dir.relative_to(output_dir))
                        }
                        if symbol:
                            cohort_entry["symbol"] = symbol
                        target_metadata["cohorts"][cohort_id] = cohort_entry
    
    # Collect decision files
    decision_dir = get_target_decision_dir(output_dir, target_clean)
    if decision_dir.exists():
        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        for decision_file in iterdir_sorted(decision_dir):
            if decision_file.is_file() and decision_file.suffix in [".json", ".yaml"]:
                decision_type = decision_file.stem
                target_metadata["decisions"][decision_type] = {
                    "path": str(decision_file.relative_to(output_dir)),
                    "format": decision_file.suffix[1:]  # Remove leading dot
                }
    
    # Collect metrics summary
    metrics_dir = get_target_metrics_dir(output_dir, target_clean)
    if metrics_dir.exists():
        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        for view_dir in iterdir_sorted(metrics_dir):
            if view_dir.is_dir() and view_dir.name.startswith("view="):
                view_name = view_dir.name.replace("view=", "")
                metrics_file = view_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics_data = json.load(f)
                            target_metadata["metrics_summary"][view_name] = {
                                "path": str(metrics_file.relative_to(output_dir)),
                                # Use SST accessors to handle both old and new structures
                                "auc": extract_auc(metrics_data) if 'extract_auc' in globals() else metrics_data.get("auc"),
                                "std_score": (metrics_data.get("primary_metric", {}).get("std") or 
                                            metrics_data.get("primary_metric", {}).get("skill_se") or 
                                            metrics_data.get("std_score")),
                                "composite_score": (metrics_data.get("score", {}).get("composite") or 
                                                   metrics_data.get("composite_score")),
                                "metric_name": metrics_data.get("metric_name")
                            }
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from {metrics_file}: {e}")
    
    # Write target metadata
    # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
    from TRAINING.common.utils.file_utils import write_atomic_json
    target_metadata_file = target_dir / "metadata.json"
    write_atomic_json(target_metadata_file, target_metadata, default=str)
    
    logger.debug(f"Created target metadata: {target_metadata_file}")
    return target_metadata_file


# =============================================================================
# Resolved Config Persistence (for run reproduction)
# =============================================================================

def normalize_numeric_types(obj: Any) -> Any:
    """
    Normalize numeric types for deterministic JSON serialization.
    
    - Counts (n_estimators, n_features, etc.) → int
    - Floats → rounded to 6 decimal places
    - NaN/Inf → None (excluded)
    
    Args:
        obj: Object to normalize
        
    Returns:
        Normalized object
    """
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), 6)
    if isinstance(obj, dict):
        # DETERMINISM: Use sorted_items() for consistency
        from TRAINING.common.utils.determinism_ordering import sorted_items
        return {k: normalize_numeric_types(v) for k, v in sorted_items(obj)}
    if isinstance(obj, (list, tuple)):
        return [normalize_numeric_types(v) for v in obj]
    return obj


def normalize_paths(obj: Any, base_path: Optional[Path] = None) -> Any:
    """
    Normalize paths to relative paths or content hashes.
    
    Args:
        obj: Object that may contain paths
        base_path: Base path for relative path computation
        
    Returns:
        Object with normalized paths
    """
    if isinstance(obj, Path):
        if base_path:
            try:
                return str(obj.relative_to(base_path))
            except ValueError:
                # Path not relative to base, use absolute as string
                return str(obj)
        return str(obj)
    if isinstance(obj, dict):
        # DETERMINISM: Use sorted_items() for consistency
        from TRAINING.common.utils.determinism_ordering import sorted_items
        return {k: normalize_paths(v, base_path) for k, v in sorted_items(obj)}
    if isinstance(obj, (list, tuple)):
        return [normalize_paths(v, base_path) for v in obj]
    return obj


def canonicalize_json_for_config(obj: Any) -> Any:
    """
    Canonicalize object for config JSON with special handling.
    
    Uses existing canonicalize() but also:
    - Normalizes numeric types
    - Normalizes paths
    - Removes None values
    
    Args:
        obj: Object to canonicalize
        
    Returns:
        Canonicalized object
    """
    # First normalize numeric types and paths
    normalized = normalize_numeric_types(obj)
    normalized = normalize_paths(normalized)
    
    # Then use existing canonicalize (handles sorting, nested structures, etc.)
    canonicalized = canonicalize(normalized)
    
    return canonicalized


def compute_config_fingerprint(resolved_config: Dict[str, Any]) -> str:
    """
    Compute SHA256 fingerprint of canonical resolved config.
    
    Includes all fields including run_id and timestamp for metadata/tracking.
    
    Args:
        resolved_config: Resolved config dictionary
        
    Returns:
        64-character hexadecimal hash
    """
    # Use canonical JSON (compact, sorted keys, deterministic)
    canonical_str = canonical_json(resolved_config)
    return sha256_full(canonical_str)


def compute_deterministic_config_fingerprint(resolved_config: Dict[str, Any]) -> str:
    """
    Compute SHA256 fingerprint of canonical resolved config excluding run_id/timestamp.
    
    This fingerprint is deterministic - same settings/data produce same hash,
    enabling comparison between runs with identical configurations.
    
    Excludes (non-deterministic fields):
    - run.run_id
    - run.timestamp
    - git.dirty (working directory state can change between runs)
    
    Keeps (deterministic settings):
    - run.seed_global
    - run.mode
    - git.commit (code version - deterministic)
    - All other config fields
    
    Args:
        resolved_config: Resolved config dictionary
        
    Returns:
        64-character hexadecimal hash
    """
    # Create a copy to avoid mutating the original
    import copy
    deterministic_config = copy.deepcopy(resolved_config)
    
    # Remove run_id and timestamp from run section (if present)
    if 'run' in deterministic_config and isinstance(deterministic_config['run'], dict):
        deterministic_config['run'] = {
            k: v for k, v in deterministic_config['run'].items()
            if k not in ('run_id', 'timestamp')
        }
    
    # Remove git.dirty from git section (if present) - working directory state is non-deterministic
    if 'git' in deterministic_config and isinstance(deterministic_config['git'], dict):
        git_clean = {k: v for k, v in deterministic_config['git'].items() if k != 'dirty'}
        if git_clean:  # Only keep git section if there are other fields
            deterministic_config['git'] = git_clean
        else:
            # Remove empty git section
            deterministic_config.pop('git', None)
    
    # Remove audit-only fields from registry section (excluded from deterministic fingerprint)
    # CRITICAL: Fingerprint tracks behavior only, not filesystem state or working directory
    if 'registry' in deterministic_config and isinstance(deterministic_config['registry'], dict):
        registry_behavior = {}
        # Include behavior bits: overlay_applied (intent), overlay_loaded (actual)
        if 'overlay_applied' in deterministic_config['registry']:
            registry_behavior['overlay_applied'] = deterministic_config['registry']['overlay_applied']
        if 'overlay_loaded' in deterministic_config['registry']:
            registry_behavior['overlay_loaded'] = deterministic_config['registry']['overlay_loaded']
        # Include overlay_hash only when both applied and loaded (behavior bit)
        if (deterministic_config['registry'].get('overlay_applied') and 
            deterministic_config['registry'].get('overlay_loaded') and
            'overlay_file_hash' in deterministic_config['registry']):
            registry_behavior['overlay_file_hash'] = deterministic_config['registry']['overlay_file_hash']
        # Exclude: overlay_present, overlay_file (absolute paths) - audit only
        if registry_behavior:
            deterministic_config['registry'] = registry_behavior
        else:
            # Remove empty registry section
            deterministic_config.pop('registry', None)
    
    # Use canonical JSON (compact, sorted keys, deterministic)
    canonical_str = canonical_json(deterministic_config)
    return sha256_full(canonical_str)


def _get_git_info() -> Dict[str, Any]:
    """Get git information for config."""
    git_info = {}
    try:
        # Get full commit hash
        from TRAINING.common.utils.git_utils import get_git_commit
        commit = get_git_commit(short=False)  # Full hash
        if commit:
            git_info["commit"] = commit
            git_info["repo"] = "origin"  # Could be enhanced to detect actual remote
        
        # Check if working directory is dirty
        try:
            from TRAINING.common.subprocess_utils import safe_subprocess_run
            result = safe_subprocess_run(['git', 'diff', '--quiet'], timeout=5)
            git_info["dirty"] = result.returncode != 0
        except Exception:
            git_info["dirty"] = None  # Unknown
        
        # Get git tag if available
        try:
            from TRAINING.common.subprocess_utils import safe_subprocess_run
            result = safe_subprocess_run(['git', 'describe', '--tags', '--exact-match', 'HEAD'], timeout=5)
            if result.returncode == 0:
                git_info["tag"] = result.stdout.strip()
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Failed to get git info: {e}")
    
    return git_info


def _get_library_versions() -> Dict[str, str]:
    """Get library versions for config."""
    library_versions = {}
    critical_libs = [
        'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn',
        'lightgbm', 'xgboost', 'catboost',
        'torch', 'tensorflow', 'keras',
        'joblib', 'polars'
    ]
    
    for lib_name in critical_libs:
        try:
            import_name = 'sklearn' if lib_name == 'scikit-learn' else lib_name
            mod = __import__(import_name)
            if hasattr(mod, '__version__'):
                library_versions[lib_name] = mod.__version__
        except (ImportError, AttributeError):
            pass
    
    return library_versions


def _extract_model_hyperparams(
    multi_model_config: Optional[Dict[str, Any]],
    model_families: Optional[List[str]] = None,
    task_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract model hyperparameters from multi_model_config.
    
    Args:
        multi_model_config: Multi-model configuration dict
        model_families: List of enabled model families
        task_type: Task type (regression, binary, multiclass)
        
    Returns:
        List of model config dicts with hyperparameters
    """
    models = []
    
    if not multi_model_config or 'model_families' not in multi_model_config:
        return models
    
    model_families_dict = multi_model_config.get('model_families', {})
    if not isinstance(model_families_dict, dict):
        return models
    
    # Get enabled families
    enabled_families = model_families or [
        name for name, config in model_families_dict.items()
        if isinstance(config, dict) and config.get('enabled', False)
    ]
    
    # Sort for determinism
    enabled_families = sorted(enabled_families)
    
    for family_name in enabled_families:
        if family_name not in model_families_dict:
            continue
        
        family_config = model_families_dict[family_name]
        if not isinstance(family_config, dict):
            continue
        
        # Get model config (hyperparameters)
        model_params = family_config.get('config', {})
        if not isinstance(model_params, dict):
            model_params = {}
        
        # Extract implementation info
        impl_info = {
            "package": family_name,  # e.g., "lightgbm"
        }
        
        # Try to get actual package version
        try:
            if family_name == 'lightgbm':
                import lightgbm as lgb
                if hasattr(lgb, '__version__'):
                    impl_info["version"] = lgb.__version__
                else:
                    impl_info["version"] = "unknown"
                impl_info["class"] = "LGBMRegressor" if task_type == "regression" else "LGBMClassifier"
            elif family_name == 'xgboost':
                import xgboost as xgb
                if hasattr(xgb, '__version__'):
                    impl_info["version"] = xgb.__version__
                else:
                    impl_info["version"] = "unknown"
                impl_info["class"] = "XGBRegressor" if task_type == "regression" else "XGBClassifier"
            elif family_name == 'catboost':
                import catboost as cb
                if hasattr(cb, '__version__'):
                    impl_info["version"] = cb.__version__
                else:
                    impl_info["version"] = "unknown"
                impl_info["class"] = "CatBoostRegressor" if task_type == "regression" else "CatBoostClassifier"
            elif family_name in ['random_forest', 'randomforest']:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                import sklearn
                impl_info["version"] = sklearn.__version__
                impl_info["class"] = "RandomForestRegressor" if task_type == "regression" else "RandomForestClassifier"
            elif family_name in ['neural_network', 'mlp']:
                from sklearn.neural_network import MLPRegressor, MLPClassifier
                import sklearn
                impl_info["version"] = sklearn.__version__
                impl_info["class"] = "MLPRegressor" if task_type == "regression" else "MLPClassifier"
            else:
                impl_info["version"] = "unknown"
                impl_info["class"] = "unknown"
        except Exception as e:
            logger.debug(f"Failed to get version for {family_name}: {e}")
            impl_info["version"] = "unknown"
            impl_info["class"] = "unknown"
        
        # Extract seed (may be in params or resolved separately)
        seed = model_params.get('seed') or model_params.get('random_state')
        
        # Separate params from fit_params
        # fit_params typically include: early_stopping_rounds, eval_metric, verbose, etc.
        fit_param_keys = {'early_stopping_rounds', 'eval_metric', 'verbose', 'verbosity', 'callbacks'}
        params = {}
        fit_params = {}
        
        for key, value in model_params.items():
            if key in fit_param_keys:
                fit_params[key] = value
            elif key not in {'seed', 'random_state'}:  # Seed handled separately
                params[key] = value
        
        model_entry = {
            "name": family_name,
            "impl": impl_info,
            "seed": seed,
            "params": params,
        }
        
        if fit_params:
            model_entry["fit_params"] = fit_params
        
        models.append(model_entry)
    
    return models


def create_resolved_config(
    output_dir: Path,
    run_id: Optional[str] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    multi_model_config: Optional[Dict[str, Any]] = None,
    model_families: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    base_seed: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    run_identity: Optional[Any] = None  # NEW: RunIdentity for deterministic run_id
) -> Dict[str, Any]:
    """
    Create resolved config (authoritative, machine-canonical) for run reproduction.
    
    This is the Single Source of Truth for all hyperparameters and configuration.
    
    Args:
        output_dir: Base run output directory
        run_id: Run identifier
        experiment_config: Experiment configuration dict
        multi_model_config: Multi-model configuration dict
        model_families: List of enabled model families
        task_type: Task type (regression, binary, multiclass)
        base_seed: Base seed for determinism
        overrides: Runtime overrides (CLI args, env vars, patches)
        
    Returns:
        Resolved config dictionary
    """
    output_dir = Path(output_dir)
    globals_dir = output_dir / "globals"
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run_id if not provided - use deterministic derivation from RunIdentity
    if run_id is None:
        try:
            run_id = derive_run_id_from_identity(
                run_identity=run_identity
            )
        except ValueError:
            # Fallback to unstable run_id if identity not available
            run_id = derive_unstable_run_id(generate_run_instance_id())
    
    # Collect environment info
    env_info = collect_environment_info()
    
    # Get git info
    git_info = _get_git_info()
    
    # Get library versions
    library_versions = _get_library_versions()
    
    # Extract model hyperparameters
    models = _extract_model_hyperparams(multi_model_config, model_families, task_type)
    
    # Build resolved config
    resolved_config = {
        "schema_version": "1.2",
        "pipeline_version": "trader-core@1.0.0",  # Could be enhanced to read from __version__
        "git": git_info,
        "build": {
            "python": env_info.get("python_version", sys.version.split()[0]),
            "platform": platform.platform(),
            "cpu_info": env_info.get("platform", {}).get("processor"),
        },
        "deps": library_versions,
        "run": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "mode": "strict",  # Could be enhanced to read from config
            "seed_global": base_seed or 42,
        },
    }
    
    # Add task config
    if task_type:
        task_family = "regression" if task_type == "regression" else "classification"
        primary_metric = "spearman_ic" if task_type == "regression" else "roc_auc"
        resolved_config["task"] = {
            "family": task_family,
            "primary_metric": primary_metric,
            "objective": task_type,
        }
    
    # Add models
    if models:
        resolved_config["models"] = models
    
    # Add data contract
    resolved_config["data_contract"] = {
        "feature_order": "sorted_by_feature_signature",
        "categoricals": "explicit_list_only",
        "missing": "as_is",
    }
    
    # Add experiment config info if available
    if experiment_config:
        resolved_config["experiment"] = {
            "name": experiment_config.get("name"),
            "data_dir": str(experiment_config.get("data_dir")) if experiment_config.get("data_dir") else None,
            "symbols": experiment_config.get("symbols"),
            "interval": experiment_config.get("interval") or experiment_config.get("data", {}).get("bar_interval"),
        }
    
    # Add registry overlay hash if applied (for run identity/config snapshot linkage)
    # CRITICAL: Overlay hash must be in config fingerprint to prevent "same run identity, different behavior"
    # Fingerprint tracks behavior only: overlay_applied (intent) + overlay_loaded (actual)
    try:
        from TRAINING.common.utils.registry_autopatch import get_autopatch
        
        repo_root = Path(__file__).resolve().parents[3]
        overlay_file = repo_root / "CONFIG" / "data" / "overrides" / "feature_registry_overrides.auto.yaml"
        
        overlay_present = overlay_file.exists()
        overlay_applied = False
        overlay_loaded = False
        overlay_hash = None
        overlay_path_relative = None
        
        # Check apply flag (behavior bit - intent)
        try:
            autopatch = get_autopatch()
            overlay_applied = autopatch.apply
        except Exception:
            # If autopatch not available, assume False
            overlay_applied = False
        
        # Check overlay_loaded status (behavior bit - actual)
        # Try to get from FeatureRegistry if available (registry should have loaded overlay)
        if overlay_applied and overlay_present:
            try:
                # Try to get registry instance and check its overlay_loaded status
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
                overlay_loaded = registry.get_overlay_loaded_status()
                
                # If overlay was loaded, compute hash
                if overlay_loaded:
                    import hashlib
                    overlay_hash = hashlib.sha256(overlay_file.read_bytes()).hexdigest()
                    overlay_path_relative = "CONFIG/data/overrides/feature_registry_overrides.auto.yaml"
            except Exception as e:
                # If registry not available, infer: if apply=True and file exists and readable,
                # assume loaded=True (registry would have hard-failed on parse failure)
                try:
                    import hashlib
                    overlay_hash = hashlib.sha256(overlay_file.read_bytes()).hexdigest()
                    # If we got here and apply=True, overlay should have been loaded successfully
                    # (registry would have hard-failed on parse failure)
                    overlay_loaded = True
                    overlay_path_relative = "CONFIG/data/overrides/feature_registry_overrides.auto.yaml"
                except Exception as e2:
                    # If we can't read the file, overlay_loaded=False
                    logger.warning(f"Could not read overlay file for hash: {e2}")
                    overlay_loaded = False
        
        # Build registry section
        # Audit fields (excluded from deterministic fingerprint):
        registry_section = {
            "overlay_present": overlay_present,  # Audit only
            "overlay_applied": overlay_applied,  # Behavior bit (intent)
            "overlay_loaded": overlay_loaded,     # Behavior bit (actual)
        }
        
        # Include overlay_path (relative) for audit, but exclude from fingerprint
        if overlay_path_relative:
            registry_section["overlay_file"] = overlay_path_relative  # Audit only (relative path)
        
        # Include overlay_hash only when both applied and loaded (behavior bit)
        if overlay_applied and overlay_loaded and overlay_hash:
            registry_section["overlay_file_hash"] = overlay_hash  # Behavior bit (included in fingerprint)
        
        resolved_config["registry"] = registry_section
        
    except Exception as e:
        # If overlay check fails, don't break config creation
        logger.debug(f"Could not check registry overlay: {e}")
        resolved_config["registry"] = {
            "overlay_present": False,
            "overlay_applied": False,
            "overlay_loaded": False
        }
    
    # Canonicalize config
    canonicalized = canonicalize_json_for_config(resolved_config)
    
    # Compute both fingerprints
    # Full fingerprint (includes run_id/timestamp) - for metadata/tracking
    config_fingerprint = compute_config_fingerprint(canonicalized)
    # Deterministic fingerprint (excludes run_id/timestamp) - for comparison
    deterministic_config_fingerprint = compute_deterministic_config_fingerprint(canonicalized)
    
    canonicalized["config_fingerprint"] = config_fingerprint
    canonicalized["deterministic_config_fingerprint"] = deterministic_config_fingerprint
    
    # Save to globals/config.resolved.json
    # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
    from TRAINING.common.utils.file_utils import write_atomic_json
    resolved_config_path = globals_dir / "config.resolved.json"
    write_atomic_json(resolved_config_path, canonicalized, default=str)
    
    logger.info(f"✅ Created resolved config: {resolved_config_path}")
    logger.info(f"   Full fingerprint: {config_fingerprint[:16]}...")
    logger.info(f"   Deterministic fingerprint: {deterministic_config_fingerprint[:16]}...")
    
    return canonicalized


def save_user_config(
    output_dir: Path,
    experiment_config_path: Optional[Path] = None,
    experiment_config_dict: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Save user config (config.user.yaml) if available.
    
    Args:
        output_dir: Base run output directory
        experiment_config_path: Path to user's experiment config file
        experiment_config_dict: Experiment config dict (if path not available)
        
    Returns:
        Path to saved config.user.yaml, or None if not saved
    """
    globals_dir = output_dir / "globals"
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    user_config_path = globals_dir / "config.user.yaml"
    
    # Try to copy from source path first
    if experiment_config_path and Path(experiment_config_path).exists():
        try:
            import shutil
            shutil.copy2(experiment_config_path, user_config_path)
            logger.info(f"✅ Saved user config: {user_config_path}")
            return user_config_path
        except Exception as e:
            logger.debug(f"Failed to copy user config from {experiment_config_path}: {e}")
    
    # Fallback: write dict as YAML
    if experiment_config_dict:
        try:
            # DETERMINISM: Use canonical_yaml() for deterministic YAML output
            from TRAINING.common.utils.determinism_serialization import write_canonical_yaml
            write_canonical_yaml(user_config_path, experiment_config_dict)
            logger.info(f"✅ Saved user config: {user_config_path}")
            return user_config_path
        except ImportError:
            logger.debug("PyYAML not available, skipping user config save")
        except Exception as e:
            logger.debug(f"Failed to save user config: {e}")
    
    return None


def save_overrides_config(
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Save runtime overrides (config.overrides.json) if any were applied.
    
    Args:
        output_dir: Base run output directory
        overrides: Runtime overrides dict (CLI args, env vars, patches)
        
    Returns:
        Path to saved config.overrides.json, or None if not saved
    """
    if not overrides:
        return None
    
    globals_dir = output_dir / "globals"
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    overrides_path = globals_dir / "config.overrides.json"
    
    try:
        # Canonicalize overrides
        # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
        from TRAINING.common.utils.file_utils import write_atomic_json
        canonicalized = canonicalize_json_for_config(overrides)
        write_atomic_json(overrides_path, canonicalized, default=str)
        logger.info(f"✅ Saved overrides config: {overrides_path}")
        return overrides_path
    except Exception as e:
        logger.debug(f"Failed to save overrides config: {e}")
        return None


def save_all_configs(
    output_dir: Path,
    experiment_config_name: Optional[str] = None
) -> Optional[Path]:
    """
    Copy all config files from CONFIG directory to globals/configs/ preserving structure.
    
    This creates a complete snapshot of all configuration files used in the run,
    enabling easy run recreation without needing access to the original CONFIG folder.
    
    Args:
        output_dir: Base run output directory
        experiment_config_name: Optional experiment config name (if used)
        
    Returns:
        Path to globals/configs/ directory, or None if failed
    """
    try:
        # Find CONFIG directory (walk up from this file)
        manifest_file = Path(__file__).resolve()
        repo_root = manifest_file.parents[3]  # utils -> orchestration -> TRAINING -> repo root
        config_dir = repo_root / "CONFIG"
        
        if not config_dir.exists():
            logger.warning(f"CONFIG directory not found at {config_dir}, skipping config dump")
            return None
        
        globals_dir = output_dir / "globals"
        configs_dest = globals_dir / "configs"
        configs_dest.mkdir(parents=True, exist_ok=True)
        
        # Collect all YAML files from CONFIG directory
        config_files = []
        # DETERMINISM: Use rglob_sorted for deterministic iteration order
        for yaml_file in rglob_sorted(config_dir, "*.yaml"):
            # Skip archive directory
            if "archive" in yaml_file.parts:
                continue
            
            # Skip Python files and READMEs
            if yaml_file.name.endswith((".py", ".md")):
                continue
            
            config_files.append(yaml_file)
        
        if not config_files:
            logger.warning(f"No config files found in {config_dir}")
            return None
        
        # Copy files preserving directory structure
        copied_count = 0
        failed_count = 0
        config_list = []
        
        for src_file in config_files:
            try:
                # Get relative path from CONFIG directory
                rel_path = src_file.relative_to(config_dir)
                dest_file = configs_dest / rel_path
                
                # Create parent directories
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                import shutil
                shutil.copy2(src_file, dest_file)
                copied_count += 1
                config_list.append(str(rel_path))
            except Exception as e:
                logger.debug(f"Failed to copy {src_file}: {e}")
                failed_count += 1
        
        # Create INDEX.md listing all configs
        index_path = configs_dest / "INDEX.md"
        try:
            with open(index_path, 'w') as f:
                f.write("# Configuration Files Index\n\n")
                f.write(f"This directory contains a snapshot of all configuration files from `CONFIG/` used in this run.\n\n")
                f.write(f"**Total configs:** {copied_count}\n")
                if experiment_config_name:
                    f.write(f"**Experiment config:** `experiments/{experiment_config_name}.yaml`\n")
                f.write("\n## Config Files by Category\n\n")
                
                # Group by category
                categories = {}
                for config_path in sorted(config_list):
                    parts = Path(config_path).parts
                    if len(parts) > 1:
                        category = parts[0]
                    else:
                        category = "root"
                    
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(config_path)
                
                # Write by category
                for category in sorted(categories.keys()):
                    f.write(f"### {category.capitalize()}/\n\n")
                    for config_path in sorted(categories[category]):
                        f.write(f"- `{config_path}`\n")
                    f.write("\n")
                
                f.write("\n## Usage\n\n")
                f.write("To recreate this run, use the configs in this directory as reference.\n")
                f.write("The original CONFIG directory structure is preserved here.\n")
        except Exception as e:
            logger.debug(f"Failed to create INDEX.md: {e}")
        
        if copied_count > 0:
            logger.info(f"✅ Saved {copied_count} config files to {configs_dest}")
            if failed_count > 0:
                logger.warning(f"   ⚠️  Failed to copy {failed_count} config files")
            return configs_dest
        else:
            logger.warning(f"No config files were copied")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to save all configs: {e}")
        import traceback
        logger.debug(f"Config dump traceback: {traceback.format_exc()}")
        return None

