# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Digest Computation Mixin for DiffTelemetry.

Contains methods for computing SHA256 digests of run outputs:
- Metrics digest
- Artifacts manifest digest
- Predictions digest

These digests enable reproducibility verification across reruns.

Extracted from diff_telemetry.py for maintainability.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Import View enum from scope_resolution
from TRAINING.orchestration.utils.scope_resolution import View

# Import determinism helpers
from TRAINING.common.utils.determinism_ordering import glob_sorted

logger = logging.getLogger(__name__)


class DigestMixin:
    """
    Mixin class providing digest computation methods for DiffTelemetry.

    This mixin contains methods related to:
    - Computing metrics digest for reproducibility verification
    - Computing artifacts manifest digest
    - Computing predictions digest

    Methods in this mixin expect the following:
    - self._normalize_value_for_hash(val): Method for normalizing values (from main class)

    IMPORTANT: This mixin requires the main class to have _normalize_value_for_hash method.
    """

    def _compute_metrics_digest(
        self,
        outputs: Dict[str, Any],
        resolved_metadata: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path]
    ) -> Optional[str]:
        """
        Compute SHA256 digest of metrics for reproducibility verification.

        This enables comparison of metrics across reruns with same inputs/process.

        Metrics can be in:
        1. outputs['metrics'] (from run_data)
        2. resolved_metadata['metrics'] (from metadata.json)
        3. metrics.json file in cohort_dir (most common for TARGET_RANKING/FEATURE_SELECTION)
        """
        metrics_data = outputs.get('metrics', {})
        source_path = None  # Track which path successfully found metrics

        if metrics_data:
            source_path = "outputs['metrics']"

        # Check resolved_metadata for metrics if outputs.metrics is empty
        if not metrics_data and resolved_metadata:
            metrics_data = resolved_metadata.get('metrics', {})
            if metrics_data:
                source_path = "resolved_metadata['metrics']"

        # SST Architecture: Read metrics from canonical location (cohort_dir) first
        # Then fall back to reference pointers, then legacy locations
        if not metrics_data and cohort_dir:
            cohort_path = Path(cohort_dir)

            # 1. Try canonical metrics.parquet in cohort directory (SST)
            metrics_parquet = cohort_path / "metrics.parquet"
            if metrics_parquet.exists():
                try:
                    import pandas as pd
                    df_metrics = pd.read_parquet(metrics_parquet)
                    if len(df_metrics) > 0:
                        metrics_dict = df_metrics.iloc[0].to_dict()
                        # Extract key metrics (exclude diff_telemetry, run_id, timestamp, and other metadata for stable hash)
                        metrics_data = {
                            k: v for k, v in metrics_dict.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                        }
                        if metrics_data:
                            source_path = f"cohort_dir/metrics.parquet ({metrics_parquet})"
                        logger.debug(f"Loaded metrics from canonical parquet: {metrics_parquet}")
                except Exception as e:
                    logger.debug(f"Failed to read metrics.parquet from {cohort_dir}: {e}")

            # 2. Fall back to metrics.json in cohort directory (debug export)
            if not metrics_data:
                metrics_json_file = cohort_path / "metrics.json"
                if metrics_json_file.exists():
                    try:
                        with open(metrics_json_file, 'r') as f:
                            metrics_json = json.load(f)
                        metrics_data = {
                            k: v for k, v in metrics_json.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                        }
                        if metrics_data:
                            source_path = f"cohort_dir/metrics.json ({metrics_json_file})"
                        logger.debug(f"Loaded metrics from JSON: {metrics_json_file}")
                    except Exception as e:
                        logger.debug(f"Failed to read metrics.json from {cohort_dir}: {e}")

            # 3. Fallback to reference pointer in metrics/ directory
            if not metrics_data:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import get_target_metrics_dir
                    # Try to extract target from path
                    target = None
                    current = cohort_path
                    for _ in range(10):
                        if current.name.startswith('cohort='):
                            # Walk up to find target
                            parent = current.parent
                            if parent.name in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                                parent = parent.parent
                            if parent.name not in ['reproducibility', View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
                                target = parent.name
                                break
                        if not current.parent.exists():
                            break
                        current = current.parent

                    if target:
                        # Find base output directory using SST helper
                        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                        base_output_dir = get_run_root(cohort_path)

                        if (base_output_dir / "targets").exists():
                            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                            target_clean = normalize_target_name(target)
                            target_metrics_dir = get_target_metrics_dir(base_output_dir, target_clean)

                            # Try to find view from path
                            view = None
                            if View.CROSS_SECTIONAL.value in cohort_path.parts:
                                view = View.CROSS_SECTIONAL
                            elif View.SYMBOL_SPECIFIC.value in cohort_path.parts:
                                view = View.SYMBOL_SPECIFIC

                            if view:
                                view_metrics_dir = target_metrics_dir / f"view={view}"
                                ref_file = view_metrics_dir / "latest_ref.json"
                                if ref_file.exists():
                                    try:
                                        with open(ref_file, 'r') as f:
                                            ref_data = json.load(f)
                                        canonical_path = Path(ref_data.get("canonical_path", ""))
                                        if canonical_path.exists():
                                            from TRAINING.common.utils.metrics import MetricsWriter
                                            metrics_dict = MetricsWriter.export_metrics_json_from_parquet(canonical_path)
                                            metrics_data = {
                                                k: v for k, v in metrics_dict.items()
                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                           'composite_version', 'leakage', 'leakage_flag']
                                            }
                                            if metrics_data:
                                                source_path = f"reference pointer ({canonical_path})"
                                            logger.debug(f"Loaded metrics via reference pointer: {canonical_path}")
                                    except Exception as e:
                                        logger.debug(f"Failed to follow reference pointer: {e}")
                except Exception as e:
                    logger.debug(f"Failed to load metrics via reference: {e}")

            # 4. Last resort: try legacy locations for backward compatibility
            if not metrics_data:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                    metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir)
                    if metrics_dir:
                        legacy_parquet = metrics_dir / "metrics.parquet"
                        if legacy_parquet.exists():
                            import pandas as pd
                            df_metrics = pd.read_parquet(legacy_parquet)
                            if len(df_metrics) > 0:
                                metrics_dict = df_metrics.iloc[0].to_dict()
                                metrics_data = {
                                    k: v for k, v in metrics_dict.items()
                                    if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                               'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                               'composite_version', 'leakage', 'leakage_flag']
                                }
                                if metrics_data:
                                    source_path = f"legacy location parquet ({metrics_dir})"
                        elif (metrics_dir / "metrics.json").exists():
                            with open(metrics_dir / "metrics.json", 'r') as f:
                                metrics_json = json.load(f)
                                metrics_data = {
                                    k: v for k, v in metrics_json.items()
                                    if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                               'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                               'composite_version', 'leakage', 'leakage_flag']
                                }
                                if metrics_data:
                                    source_path = f"legacy location json ({metrics_dir})"
                except Exception as e:
                    logger.debug(f"Failed to load metrics from legacy location: {e}")

        if not metrics_data:
            return None

        # Log which path successfully found metrics (for debugging)
        if source_path:
            logger.debug(f"metrics_sha256 computed from {source_path}")
        else:
            logger.debug(f"metrics_sha256 computed (source path unknown)")

        # Normalize metrics dict for stable hashing (sort keys, round floats)
        normalized = self._normalize_value_for_hash(metrics_data)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_artifacts_manifest_digest(
        self,
        cohort_dir: Optional[Path],
        stage: str
    ) -> Optional[str]:
        """
        Compute SHA256 digest of artifacts manifest for reproducibility verification.

        This enables comparison of artifacts (feature_importances.parquet, etc.) across reruns.
        Creates a manifest of artifact files with their sizes and modification times.

        Artifacts are stored at the view level with view-specific scoping:
        - SYMBOL_SPECIFIC: targets/{target}/reproducibility/SYMBOL_SPECIFIC/symbol={sym}/feature_importances/
        - CROSS_SECTIONAL: targets/{target}/reproducibility/CROSS_SECTIONAL/universe={sig}/feature_importances/
        - TRAINING: targets/{target}/reproducibility/{view}/cohort={cohort_id}/ (artifacts in cohort dir)
        """
        if not cohort_dir or not cohort_dir.exists():
            return None

        # For TARGET_RANKING and FEATURE_SELECTION, artifacts are at target level (one level up from cohort)
        # For TRAINING, artifacts are in the cohort directory itself
        if stage in ['TARGET_RANKING', 'FEATURE_SELECTION']:
            # FIX: Use existing SST functions to find feature_importances directory instead of manual path traversal
            from TRAINING.orchestration.utils.target_first_paths import (
                parse_reproducibility_path, parse_attempt_id_from_cohort_dir, get_scoped_artifact_dir, run_root
            )
            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name

            # Parse path components from cohort_dir using SST function
            path_info = parse_reproducibility_path(cohort_dir)
            attempt_id = parse_attempt_id_from_cohort_dir(cohort_dir)

            # Extract target from path (cohort_dir is under targets/{target}/reproducibility/...)
            target = path_info.get('target')
            if not target:
                # Fallback: extract from cohort_dir path
                parts = cohort_dir.parts
                if 'targets' in parts:
                    target_idx = parts.index('targets')
                    if target_idx + 1 < len(parts):
                        target = normalize_target_name(parts[target_idx + 1])

            # Get base output directory (run root) using SST function
            base_output_dir = run_root(cohort_dir)

            # Use SST function to get feature_importances directory
            feature_importances_dir = None
            if target and base_output_dir:
                try:
                    feature_importances_dir = get_scoped_artifact_dir(
                        base_output_dir=base_output_dir,
                        target=target,
                        artifact_type='feature_importances',
                        view=path_info.get('view'),
                        symbol=path_info.get('symbol'),
                        universe_sig=path_info.get('universe_sig'),
                        stage=stage,
                        attempt_id=attempt_id,
                    )
                except Exception as e:
                    logger.debug(f"Failed to get feature_importances_dir using SST function: {e}")

            # ROOT CAUSE DEBUG: Log when feature_importances_dir is not found
            if not feature_importances_dir or not feature_importances_dir.exists():
                logger.warning(
                    f"feature_importances_dir not found or doesn't exist: {feature_importances_dir} "
                    f"(attempt_id={attempt_id}, target={target}, "
                    f"view={path_info.get('view')}, symbol={path_info.get('symbol')}, "
                    f"universe_sig={path_info.get('universe_sig')}, stage={stage}). "
                    f"This means feature importances were never saved, so artifacts_manifest_sha256 will be null."
                )
                # Don't create directory here - it's read-only. Just note that directory doesn't exist
                feature_importances_dir = None

            # For legacy compatibility, set target_dir to view_dir (extracted from path_info or cohort_dir)
            # This is used for other artifacts like selected_features.txt
            if path_info.get('view'):
                # Try to construct view_dir from path
                parts = cohort_dir.parts
                if 'reproducibility' in parts:
                    repro_idx = parts.index('reproducibility')
                    if 'stage=' in parts[repro_idx + 1] if repro_idx + 1 < len(parts) else False:
                        stage_idx = repro_idx + 1
                        view_idx = stage_idx + 1
                        if view_idx < len(parts):
                            target_dir = Path(*parts[:view_idx + 1])
                        else:
                            target_dir = cohort_dir.parent.parent.parent if cohort_dir.parent.name.startswith('attempt_') else cohort_dir.parent.parent
                    else:
                        view_idx = repro_idx + 1
                        if view_idx < len(parts):
                            target_dir = Path(*parts[:view_idx + 1])
                        else:
                            target_dir = cohort_dir.parent.parent.parent if cohort_dir.parent.name.startswith('attempt_') else cohort_dir.parent.parent
                else:
                    target_dir = cohort_dir.parent.parent.parent if cohort_dir.parent.name.startswith('attempt_') else cohort_dir.parent.parent
            else:
                target_dir = cohort_dir.parent.parent.parent if cohort_dir.parent.name.startswith('attempt_') else cohort_dir.parent.parent
        else:
            # TRAINING: artifacts are in cohort directory
            target_dir = cohort_dir
            feature_importances_dir = None

        # Define artifact file patterns by stage
        artifact_patterns = {
            'TARGET_RANKING': [
                ('feature_importances', feature_importances_dir)  # Directory with CSV files
            ],
            'FEATURE_SELECTION': [
                ('selected_features.txt', target_dir),  # At view level
                ('target_confidence.json', target_dir),  # At view level
                ('feature_importances', feature_importances_dir)  # Directory with CSV files
            ],
            'TRAINING': [
                ('model_hash.txt', cohort_dir),  # In cohort directory
                ('meta_*.json', cohort_dir)  # In cohort directory
            ]
        }

        patterns = artifact_patterns.get(stage, [])
        manifest = []

        for pattern, search_dir in patterns:
            if not search_dir or not search_dir.exists():
                continue

            if pattern == 'feature_importances':
                # Special case: hash all CSV files in feature_importances directory
                if search_dir.is_dir():
                    # DETERMINISM: Use glob_sorted for deterministic iteration order
                    csv_files = glob_sorted(search_dir, '*.csv')
                    for csv_file in csv_files:
                        if csv_file.is_file():
                            # Hash file contents instead of using mtime (volatile)
                            try:
                                with open(csv_file, 'rb') as f:
                                    content_hash = hashlib.sha256(f.read()).hexdigest()
                                manifest.append({
                                    'path': f'feature_importances/{csv_file.name}',
                                    'content_sha256': content_hash
                                })
                            except Exception as e:
                                logger.debug(f"Failed to hash {csv_file}: {e}")
                                # Fallback to size only (no mtime)
                                stat = csv_file.stat()
                                manifest.append({
                                    'path': f'feature_importances/{csv_file.name}',
                                    'size': stat.st_size
                                })
            elif '*' in pattern:
                # Handle glob patterns
                # DETERMINISM: Use glob_sorted for deterministic iteration order
                matches = glob_sorted(search_dir, pattern)
                for match in matches:
                    if match.is_file():
                        # Hash file contents instead of using mtime (volatile)
                        try:
                            with open(match, 'rb') as f:
                                content_hash = hashlib.sha256(f.read()).hexdigest()
                            manifest.append({
                                'path': match.name,
                                'content_sha256': content_hash
                            })
                        except Exception as e:
                            logger.debug(f"Failed to hash {match}: {e}")
                            # Fallback to size only (no mtime)
                            stat = match.stat()
                            manifest.append({
                                'path': match.name,
                                'size': stat.st_size
                            })
            else:
                # Exact file match
                file_path = search_dir / pattern
                if file_path.exists() and file_path.is_file():
                    # Hash file contents instead of using mtime (volatile)
                    try:
                        with open(file_path, 'rb') as f:
                            content_hash = hashlib.sha256(f.read()).hexdigest()
                        manifest.append({
                            'path': pattern,
                            'content_sha256': content_hash
                        })
                    except Exception as e:
                        logger.debug(f"Failed to hash {file_path}: {e}")
                        # Fallback to size only (no mtime)
                        stat = file_path.stat()
                        manifest.append({
                            'path': pattern,
                            'size': stat.st_size
                        })

        # FIX 2: Add logging when manifest is empty
        if not manifest:
            logger.debug(
                f"No artifacts found for {stage} stage "
                f"(cohort_dir={cohort_dir}, "
                f"feature_importances_dir={feature_importances_dir if 'feature_importances_dir' in locals() else 'N/A'})"
            )
            return None

        # Sort manifest for stable hashing
        manifest_sorted = sorted(manifest, key=lambda x: x['path'])
        json_str = json.dumps(manifest_sorted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_predictions_digest(
        self,
        cohort_dir: Optional[Path],
        stage: str,
        prediction_fingerprint: Optional[Dict] = None,  # Prediction fingerprint dict
    ) -> Optional[str]:
        """
        Compute SHA256 digest of predictions for reproducibility verification.

        This enables comparison of predictions across reruns.
        Currently, predictions may not be stored in cohort directories for all stages.

        NEW: If prediction_fingerprint is provided, use its prediction_hash directly.
        This is the authoritative source from the prediction hashing system.
        """
        # If prediction_fingerprint provided, use its hash
        if prediction_fingerprint and prediction_fingerprint.get('prediction_hash'):
            return prediction_fingerprint['prediction_hash']

        if not cohort_dir or not cohort_dir.exists():
            return None

        # Check for predictions files (if they exist)
        predictions_files = []
        for pattern in ['predictions.parquet', 'predictions.csv', 'predictions.json']:
            file_path = cohort_dir / pattern
            if file_path.exists() and file_path.is_file():
                # For large files, hash first N bytes + size instead of full content
                # This is a compromise for performance while still detecting changes
                stat = file_path.stat()
                if stat.st_size > 10 * 1024 * 1024:  # > 10MB
                    # Hash first 1MB + size + mtime
                    with open(file_path, 'rb') as f:
                        first_mb = f.read(1024 * 1024)
                    content_hash = hashlib.sha256(
                        first_mb + str(stat.st_size).encode() + str(stat.st_mtime).encode()
                    ).hexdigest()
                else:
                    # Hash full file for smaller files
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()

                predictions_files.append({
                    'path': pattern,
                    'size': stat.st_size,
                    'hash': content_hash
                })

        if not predictions_files:
            return None

        # Sort for stable hashing
        predictions_sorted = sorted(predictions_files, key=lambda x: x['path'])
        json_str = json.dumps(predictions_sorted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
