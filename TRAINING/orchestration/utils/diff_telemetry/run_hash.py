# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Run Hash Computation

Functions for computing deterministic hashes of pipeline runs and comparing runs.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# DETERMINISM: Use sorted iterdir for filesystem enumeration
from TRAINING.common.utils.determinism_ordering import iterdir_sorted

if TYPE_CHECKING:
    from TRAINING.orchestration.utils.diff_telemetry import DiffTelemetry

logger = logging.getLogger(__name__)


def _extract_deterministic_fields(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract deterministic fields from snapshot, returning None if missing fingerprint.

    This helper function extracts all deterministic fields needed for run hash computation,
    excluding run_id, timestamp, and snapshot_seq to ensure hash determinism.

    Args:
        snapshot: Snapshot dictionary from index

    Returns:
        Dictionary of deterministic fields, or None if config fingerprint is missing
    """
    config_fp = snapshot.get('deterministic_config_fingerprint') or snapshot.get('config_fingerprint')
    if not config_fp:
        return None

    # Normalize model_family: FEATURE_SELECTION uses 'method', TRAINING uses 'model_family'
    model_family = snapshot.get('model_family') or snapshot.get('method')

    deterministic = {
        'stage': snapshot.get('stage'),
        'target': snapshot.get('target'),
        'view': snapshot.get('view'),
        'symbol': snapshot.get('symbol'),
        'model_family': model_family,
        'config_fingerprint': config_fp,
        'data_fingerprint': snapshot.get('data_fingerprint'),
        'feature_fingerprint': snapshot.get('feature_fingerprint'),
        'target_fingerprint': snapshot.get('target_fingerprint'),
        'scoring_signature': snapshot.get('scoring_signature'),
        'selection_signature': snapshot.get('selection_signature'),
        'hyperparameters_signature': snapshot.get('hyperparameters_signature'),
        'metrics_sha256': snapshot.get('metrics_sha256'),
        'artifacts_manifest_sha256': snapshot.get('artifacts_manifest_sha256'),
        'predictions_sha256': snapshot.get('predictions_sha256'),
        'metrics_schema_version': snapshot.get('metrics_schema_version'),
        'scoring_schema_version': snapshot.get('scoring_schema_version'),
    }

    # Normalize null vs missing - only include if non-None
    lib_sig = snapshot.get('library_versions_signature')
    if lib_sig is not None:
        deterministic['library_versions_signature'] = lib_sig

    # Add comparison_group key fields if available
    if 'comparison_group' in snapshot:
        cg = snapshot['comparison_group']
        deterministic['comparison_group'] = {
            'task_signature': cg.get('task_signature'),
            'feature_signature': cg.get('feature_signature'),
            'split_signature': cg.get('split_signature'),
            'n_effective': cg.get('n_effective'),
            'dataset_signature': cg.get('dataset_signature'),
            'routing_signature': cg.get('routing_signature'),
        }

    return deterministic


def compute_full_run_hash(output_dir: Path, run_id: Optional[str] = None) -> Optional[str]:
    """
    Compute deterministic hash of entire run across all stages.

    This hash represents the full state of a run and can be used for:
    - Run comparison (same hash = identical run)
    - Run deduplication
    - Reproducibility verification

    Args:
        output_dir: Base output directory (should contain globals/)
        run_id: Optional run_id to filter snapshots

    Returns:
        16-character hex digest of run hash, or None if no snapshots found
    """
    globals_dir = output_dir / "globals"
    if not globals_dir.exists():
        return None

    # Load all snapshot indices
    snapshot_indices = {}
    missing_indices = []
    for index_file in ["snapshot_index.json", "fs_snapshot_index.json", "training_snapshot_index.json"]:
        index_path = globals_dir / index_file
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    snapshot_indices[index_file] = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load {index_file}: {e}")
                missing_indices.append(index_file)
        else:
            missing_indices.append(index_file)

    if not snapshot_indices:
        logger.warning(
            f"⚠️ No snapshot indices found in {globals_dir}. Missing: {', '.join(missing_indices)}. "
            f"Run hash cannot be computed. This may indicate snapshots were not saved correctly."
        )
        return None
    elif missing_indices:
        logger.debug(f"Some snapshot indices missing (non-critical): {', '.join(missing_indices)}")

    # One-pass collection: track both all_entries and filtered_entries
    all_entries = []
    filtered_entries = []
    seen_run_ids = []

    # Counters for precise decision logic
    total_snapshots = 0
    match_runid = 0
    with_fp = 0
    match_runid_with_fp = 0

    # Sort indices by name for deterministic order
    sorted_indices = sorted(snapshot_indices.items())

    for index_name, index_data in sorted_indices:
        sorted_snapshots = sorted(index_data.items())

        for key, snapshot in sorted_snapshots:
            total_snapshots += 1
            snapshot_run_id = snapshot.get('run_id')

            if len(seen_run_ids) < 10 and snapshot_run_id:
                seen_run_ids.append(snapshot_run_id)

            matches_runid = (not run_id) or (snapshot_run_id == run_id)
            if matches_runid:
                match_runid += 1

            deterministic = _extract_deterministic_fields(snapshot)
            if deterministic:
                with_fp += 1
                all_entries.append((index_name, key, deterministic))

                if matches_runid:
                    match_runid_with_fp += 1
                    filtered_entries.append((index_name, key, deterministic))

    # Decision logic
    if run_id:
        if match_runid == 0:
            sample_str = ', '.join(seen_run_ids[:5]) if seen_run_ids else 'N/A'
            logger.warning(
                f"⚠️ No snapshots found for run_id={run_id}. "
                f"Total snapshots: {total_snapshots}. Sample run_ids: [{sample_str}]. "
                f"Falling back to all snapshots with fingerprints ({with_fp})."
            )
            chosen_entries = all_entries
        elif match_runid_with_fp == 0:
            logger.warning(
                f"⚠️ Found {match_runid} snapshots for run_id={run_id}, but none have fingerprints. "
                f"Falling back to all snapshots with fingerprints ({with_fp})."
            )
            chosen_entries = all_entries
        else:
            logger.debug(
                f"compute_full_run_hash: Found {match_runid_with_fp} snapshots with fingerprints for run_id={run_id}"
            )
            chosen_entries = filtered_entries
    else:
        chosen_entries = all_entries

    if not chosen_entries:
        logger.warning(
            f"⚠️ No valid snapshots found (total={total_snapshots}, with_fp={with_fp}). "
            f"Run hash cannot be computed."
        )
        return None

    # Sort for deterministic ordering
    chosen_entries.sort(key=lambda x: (x[0], x[1]))

    # Extract just the deterministic dicts
    run_state = [entry[2] for entry in chosen_entries]

    logger.debug(f"compute_full_run_hash: Computing hash from {len(run_state)} snapshots")

    # Compute hash
    canonical_json = json.dumps(run_state, sort_keys=True)
    full_hash = hashlib.sha256(canonical_json.encode()).hexdigest()
    return full_hash[:16]


def _load_manifest_comparability_flags(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load comparability flags from manifest.json.

    Returns dict with is_comparable and run_id_kind, or None if manifest not found/invalid.

    Args:
        output_dir: Run output directory (should contain manifest.json)

    Returns:
        Dict with keys: is_comparable (bool), run_id_kind (str), legacy_inferred (bool)
        or None if manifest not found
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        is_comparable = manifest.get('is_comparable')
        run_id_kind = manifest.get('run_id_kind')

        if is_comparable is not None and run_id_kind:
            return {
                'is_comparable': bool(is_comparable),
                'run_id_kind': str(run_id_kind),
                'legacy_inferred': False
            }

        return None
    except Exception:
        return None


def _normalize_run_id_for_comparison(run_id: Optional[str]) -> Optional[str]:
    """
    Normalize run_id for comparison by converting underscores to dashes.

    LEGACY FUNCTION: Use only when manifest flags are absent.
    New code should check is_comparable flag from manifest instead.

    Args:
        run_id: Run ID string (may contain underscores or dashes)

    Returns:
        Normalized run_id with dashes, or None if input is None/empty
    """
    if not run_id:
        return None
    return run_id.replace("_", "-")


def _can_runs_be_compared(
    run1_dir: Path,
    run2_dir: Path,
    run1_id: Optional[str] = None,
    run2_id: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if two runs can be compared based on comparability flags.

    Legacy precedence rules:
    - If both manifests have flags: trust them (authoritative)
    - Else: use legacy normalization heuristics, mark as legacy_inferred

    Args:
        run1_dir: First run directory
        run2_dir: Second run directory
        run1_id: Optional run1 ID (for legacy fallback)
        run2_id: Optional run2 ID (for legacy fallback)

    Returns:
        Tuple of (can_compare: bool, reason: Optional[str])
    """
    flags1 = _load_manifest_comparability_flags(run1_dir)
    flags2 = _load_manifest_comparability_flags(run2_dir)

    if flags1 and flags2:
        if flags1['run_id_kind'] != flags2['run_id_kind']:
            return (False, f"run_id_kind mismatch: {flags1['run_id_kind']} vs {flags2['run_id_kind']}")

        if not flags1['is_comparable'] or not flags2['is_comparable']:
            return (False, "one or both runs marked as not comparable")

        return (True, None)

    if run1_id and run2_id:
        norm1 = _normalize_run_id_for_comparison(run1_id)
        norm2 = _normalize_run_id_for_comparison(run2_id)
        if norm1 and norm2 and norm1 == norm2:
            return (True, "legacy_inferred")

    return (False, "legacy normalization failed or run_ids missing")


def compute_run_hash_with_changes(
    output_dir: Path,
    run_id: Optional[str] = None,
    prev_run_id: Optional[str] = None,
    diff_telemetry: Optional['DiffTelemetry'] = None
) -> Optional[Dict[str, Any]]:
    """
    Compute run hash and aggregate change information.

    Args:
        output_dir: Base output directory
        run_id: Current run ID
        prev_run_id: Previous run ID for change detection
        diff_telemetry: Optional DiffTelemetry instance for computing diffs

    Returns:
        Dict with run_hash, run_id, changes summary, or None if no snapshots found
    """
    run_hash = compute_full_run_hash(output_dir, run_id)
    if run_hash is None:
        return None

    globals_dir = output_dir / "globals"
    snapshot_count = 0
    for index_file in ["snapshot_index.json", "fs_snapshot_index.json", "training_snapshot_index.json"]:
        index_path = globals_dir / index_file
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                    if run_id:
                        normalized_run_id = _normalize_run_id_for_comparison(run_id)
                        snapshot_count += sum(1 for s in index_data.values()
                                           if _normalize_run_id_for_comparison(s.get('run_id')) == normalized_run_id)
                    else:
                        snapshot_count += len(index_data)
            except Exception:
                continue

    changes = None
    if prev_run_id and diff_telemetry:
        try:
            prev_run_hash = None
            try:
                results_dir = output_dir
                for _ in range(10):
                    if results_dir.name == "RESULTS":
                        break
                    if not results_dir.parent.exists():
                        break
                    results_dir = results_dir.parent

                if results_dir.name == "RESULTS":
                    runs_dir = results_dir / "runs"
                    if runs_dir.exists():
                        search_base = runs_dir

                        for candidate_dir in iterdir_sorted(search_base):
                            if not candidate_dir.is_dir():
                                continue
                            candidate_hash_file = candidate_dir / "globals" / "run_hash.json"
                            if candidate_hash_file.exists():
                                try:
                                    with open(candidate_hash_file, 'r') as f:
                                        candidate_hash_data = json.load(f)
                                        candidate_run_id = candidate_hash_data.get('run_id')
                                        if candidate_run_id and _normalize_run_id_for_comparison(candidate_run_id) == _normalize_run_id_for_comparison(prev_run_id):
                                            prev_run_hash = candidate_hash_data.get('run_hash')
                                            break
                                except Exception:
                                    continue
            except Exception as e:
                logger.debug(f"Failed to search for previous run hash: {e}")

            all_diffs = diff_telemetry.get_all_diffs()
            changed_snapshots = 0
            changed_keys_summary = []
            metric_deltas_summary = {}
            excluded_factors_summary = {}

            for stage_key, diff_result in sorted(all_diffs.items()):
                if diff_result.changed_keys:
                    changed_snapshots += 1
                    changed_keys_summary.extend(diff_result.changed_keys[:5])

                if diff_result.metric_deltas:
                    for metric_name, delta_info in diff_result.metric_deltas.items():
                        if metric_name not in metric_deltas_summary:
                            metric_deltas_summary[metric_name] = {
                                'total_delta': 0.0,
                                'count': 0
                            }
                        if 'delta' in delta_info:
                            metric_deltas_summary[metric_name]['total_delta'] += abs(delta_info['delta'])
                            metric_deltas_summary[metric_name]['count'] += 1

                if diff_result.excluded_factors_changed:
                    for factor_name, factor_val in diff_result.excluded_factors_changed.items():
                        if factor_name not in excluded_factors_summary:
                            excluded_factors_summary[factor_name] = 0
                        excluded_factors_summary[factor_name] += 1

            if changed_snapshots == 0:
                severity_summary = 'none'
            elif any(d.severity.value == 'critical' for d in all_diffs.values()):
                severity_summary = 'critical'
            elif any(d.severity.value == 'major' for d in all_diffs.values()):
                severity_summary = 'major'
            elif any(d.severity.value == 'minor' for d in all_diffs.values()):
                severity_summary = 'minor'
            else:
                severity_summary = 'noise'

            changes = {
                'changed_snapshots': changed_snapshots,
                'changed_keys_summary': sorted(set(changed_keys_summary)),
                'severity_summary': severity_summary,
                'metric_deltas_summary': metric_deltas_summary,
                'excluded_factors_summary': excluded_factors_summary,
            }
        except Exception as e:
            logger.warning(f"Failed to compute run changes: {e}")
            changes = None

    return {
        'run_hash': run_hash,
        'run_id': run_id,
        'prev_run_id': prev_run_id,
        'changes': changes,
        'snapshot_count': snapshot_count,
    }


def save_run_hash(
    output_dir: Path,
    run_id: Optional[str] = None,
    prev_run_id: Optional[str] = None,
    diff_telemetry: Optional['DiffTelemetry'] = None
) -> Optional[Path]:
    """
    Compute and save run hash with changes to globals/run_hash.json.

    Args:
        output_dir: Base output directory
        run_id: Current run ID
        prev_run_id: Previous run ID for change detection
        diff_telemetry: Optional DiffTelemetry instance

    Returns:
        Path to run_hash.json if saved successfully, None otherwise
    """
    # Import write_atomic_json for saving
    from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json

    try:
        run_hash_data = compute_run_hash_with_changes(
            output_dir=output_dir,
            run_id=run_id,
            prev_run_id=prev_run_id,
            diff_telemetry=diff_telemetry
        )

        if run_hash_data is None:
            logger.warning(
                f"⚠️ Run hash computation returned None. This may indicate: "
                f"(1) No snapshot indices found in {output_dir / 'globals'}, "
                f"(2) All snapshots were skipped due to missing fingerprints, or "
                f"(3) globals_dir does not exist. Check logs above for details."
            )
            return None

        globals_dir = output_dir / "globals"
        globals_dir.mkdir(parents=True, exist_ok=True)
        run_hash_file = globals_dir / "run_hash.json"

        run_hash_data['computed_at'] = datetime.now().isoformat()

        _write_atomic_json(run_hash_file, run_hash_data)

        logger.info(f"✅ Saved run hash: {run_hash_data['run_hash']} ({run_hash_data['snapshot_count']} snapshots)")
        if run_hash_data.get('changes'):
            logger.info(f"   Changes detected: {run_hash_data['changes']['severity_summary']} severity, {len(run_hash_data['changes']['changed_snapshots'])} snapshots changed")

        return run_hash_file
    except Exception as e:
        logger.warning(f"Failed to save run hash: {e}")
        return None
