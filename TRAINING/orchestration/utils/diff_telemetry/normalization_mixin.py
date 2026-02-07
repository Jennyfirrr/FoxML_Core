# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Normalization Mixin for DiffTelemetry.

Contains methods for normalizing run data into standardized sections:
- inputs (config, data, target, features)
- process (split, training, environment)
- outputs (metrics, stability)

Extracted from diff_telemetry.py for maintainability.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Import types
from .types import ResolvedRunContext

# Import determinism helpers
from TRAINING.common.utils.determinism_ordering import sorted_items, iterdir_sorted

logger = logging.getLogger(__name__)


class NormalizationMixin:
    """
    Mixin class providing normalization methods for DiffTelemetry.

    This mixin contains methods related to:
    - Normalizing inputs section (config, data, target, features)
    - Normalizing process section (split, training, environment)
    - Normalizing outputs section (metrics, stability)

    Methods in this mixin expect the following:
    - self._compute_feature_fingerprint(run_data, additional_data): Method from FingerprintMixin

    IMPORTANT: This mixin must be listed AFTER FingerprintMixin in the class inheritance.
    """

    def _normalize_inputs_from_context(
        self,
        stage: str,
        ctx: ResolvedRunContext
    ) -> Dict[str, Any]:
        """Normalize inputs section from resolved context (no nulls for required fields)."""
        inputs = {}

        # Config (stage-specific - only include applicable fields)
        config = {}
        if ctx.min_cs is not None:
            config['min_cs'] = ctx.min_cs
        if ctx.max_cs_samples is not None:
            config['max_cs_samples'] = ctx.max_cs_samples
        if stage in ["FEATURE_SELECTION", "TRAINING"]:
            if ctx.n_features is not None:
                config['n_features'] = ctx.n_features
        if stage == "TRAINING":
            if ctx.model_family:
                config['model_family'] = ctx.model_family
            if ctx.trainer_strategy:
                config['strategy'] = ctx.trainer_strategy
        if stage in ["TARGET_RANKING", "FEATURE_SELECTION"]:
            if ctx.model_families:
                config['model_families'] = sorted(ctx.model_families)  # Sort for determinism
        if stage == "FEATURE_SELECTION" and ctx.feature_selection:
            # Add feature selection parameters to config
            if 'feature_selection' not in config:
                config['feature_selection'] = {}
            config['feature_selection'].update(ctx.feature_selection)
        if config:
            inputs['config'] = config

        # Data (all required fields, should be non-null)
        inputs['data'] = {
            'n_symbols': ctx.n_symbols,  # Required, validated
            'date_start': ctx.date_start,  # Required, validated
            'date_end': ctx.date_end,  # Required, validated
        }
        if ctx.n_rows_total is not None:
            inputs['data']['n_rows_total'] = ctx.n_rows_total
        if ctx.symbols:
            inputs['data']['symbols'] = ctx.symbols
        if ctx.data_dir:
            inputs['data']['data_dir'] = ctx.data_dir

        # Target (required, should be non-null)
        inputs['target'] = {
            'target': ctx.target,  # Required, validated
            'view': ctx.view,  # Required, validated
        }
        if ctx.horizon_minutes is not None:
            inputs['target']['horizon_minutes'] = ctx.horizon_minutes
        if ctx.labeling_impl_hash:
            inputs['target']['labeling_impl_hash'] = ctx.labeling_impl_hash

        # Feature set (stage-specific)
        if stage in ["FEATURE_SELECTION", "TRAINING"]:
            features = {}
            if ctx.n_features is not None:
                features['n_features'] = ctx.n_features
            if ctx.feature_fingerprint:
                features['feature_fingerprint'] = ctx.feature_fingerprint
            if ctx.feature_names:
                features['feature_names'] = ctx.feature_names
            if features:
                inputs['features'] = features

        return inputs

    def _normalize_inputs(
        self,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize inputs section."""
        inputs = {}

        # Config fingerprint tree
        if additional_data:
            config = {
                'strategy': additional_data.get('strategy'),
                'model_family': additional_data.get('model_family'),
                'n_features': additional_data.get('n_features'),
                'min_cs': additional_data.get('min_cs'),
                'max_cs_samples': additional_data.get('max_cs_samples')
            }
            # Add model_families for TARGET_RANKING/FEATURE_SELECTION
            if additional_data.get('model_families'):
                config['model_families'] = sorted(additional_data.get('model_families'))
            # Add feature_selection parameters for FEATURE_SELECTION stage
            if additional_data.get('feature_selection'):
                if 'feature_selection' not in config:
                    config['feature_selection'] = {}
                config['feature_selection'].update(additional_data.get('feature_selection'))
            # Remove None values
            config = {k: v for k, v in config.items() if v is not None}
            if config:
                inputs['config'] = config

        # Data fingerprint tree
        if cohort_metadata:
            data = {
                'n_symbols': cohort_metadata.get('n_symbols'),
                'date_start': cohort_metadata.get('date_start'),
                'date_end': cohort_metadata.get('date_end'),
                'n_samples': cohort_metadata.get('n_samples')
            }
            # Add symbols and data_dir if available
            if cohort_metadata.get('symbols'):
                data['symbols'] = cohort_metadata.get('symbols')
            if cohort_metadata.get('data_dir'):
                data['data_dir'] = cohort_metadata.get('data_dir')
            elif additional_data and additional_data.get('data_dir'):
                data['data_dir'] = additional_data.get('data_dir')
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            if data:
                inputs['data'] = data

        # Target provenance
        target = None
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('target'):
            parts = run_data['target'].split(':')
            target = parts[0] if parts else None

        if target:
            inputs['target'] = {
                'target': target,
                'view': additional_data.get('view') if additional_data else None
            }

        # Feature set provenance
        if additional_data and 'n_features' in additional_data:
            inputs['features'] = {
                'n_features': additional_data['n_features'],
                'feature_fingerprint': self._compute_feature_fingerprint(run_data, additional_data)
            }

        return inputs

    def _normalize_process_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Dict[str, Any]:
        """Normalize process section from resolved context."""
        process = {}

        # Split integrity (required fields, should be non-null)
        split = {}
        if ctx.min_cs is not None:
            split['min_cs'] = ctx.min_cs
        if ctx.max_cs_samples is not None:
            split['max_cs_samples'] = ctx.max_cs_samples
        if ctx.cv_method:
            split['cv_method'] = ctx.cv_method
        if ctx.purge_minutes is not None:
            split['purge_minutes'] = ctx.purge_minutes
        if ctx.embargo_minutes is not None:
            split['embargo_minutes'] = ctx.embargo_minutes
        if ctx.split_seed is not None:
            split['split_seed'] = ctx.split_seed
        if ctx.fold_assignment_hash:
            split['fold_assignment_hash'] = ctx.fold_assignment_hash
        if split:
            process['split'] = split

        # Training regime (only for TRAINING stage)
        if ctx.trainer_strategy or ctx.model_family:
            training = {}
            if ctx.trainer_strategy:
                training['strategy'] = ctx.trainer_strategy
            if ctx.model_family:
                training['model_family'] = ctx.model_family
            if ctx.hyperparameters:
                training['hyperparameters'] = ctx.hyperparameters
            if training:
                process['training'] = training

        # Environment (tracked but not outcome-influencing)
        environment = {}
        if ctx.python_version:
            environment['python_version'] = ctx.python_version
        if ctx.library_versions:
            environment['library_versions'] = ctx.library_versions
        if ctx.cuda_version:
            environment['cuda_version'] = ctx.cuda_version
        if environment:
            process['environment'] = environment

        return process

    def _normalize_process(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize process section."""
        process = {}

        # Split integrity
        if additional_data:
            process['split'] = {
                'min_cs': additional_data.get('min_cs'),
                'max_cs_samples': additional_data.get('max_cs_samples')
            }

        # Training regime
        if additional_data and 'training' in additional_data:
            training_data = additional_data['training']
            training = {
                'strategy': training_data.get('strategy'),
                'model_family': training_data.get('model_family')
            }
            # Add hyperparameters (exclude strategy, model_family, seeds - handled separately)
            excluded_keys = {'strategy', 'model_family', 'split_seed', 'train_seed', 'seed'}
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            hyperparameters = {k: v for k, v in sorted_items(training_data)
                              if k not in excluded_keys and v is not None}
            if hyperparameters:
                training['hyperparameters'] = hyperparameters
            # Remove None values
            training = {k: v for k, v in sorted_items(training) if v is not None}
            if training:
                process['training'] = training
        elif additional_data:
            # Fallback: try to extract from top-level additional_data
            training = {
                'strategy': additional_data.get('strategy'),
                'model_family': additional_data.get('model_family')
            }
            # Remove None values
            training = {k: v for k, v in training.items() if v is not None}
            if training:
                process['training'] = training

        # Compute environment (if available)
        process['environment'] = {
            'python_version': None,  # Could extract from sys.version
            'library_versions': {}  # Could extract from package versions
        }

        return process

    def _normalize_outputs(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize outputs section.

        Extracts all numeric metrics from:
        1. run_data['metrics'] (if available)
        2. resolved_metadata['metrics'] (if available)
        3. metrics.json/parquet files in cohort_dir (most common for TARGET_RANKING/FEATURE_SELECTION)

        This ensures we capture all metrics for proper delta computation.
        """
        outputs = {}
        metrics_data = {}

        # Try run_data first
        if run_data.get('metrics'):
            metrics_data = run_data['metrics']

        # Fallback: Reconstruct metrics from top-level keys (for backward compatibility)
        # This handles cases where metrics were spread to top level instead of nested under 'metrics'
        if not metrics_data:
            known_metric_keys = {
                'schema', 'scope', 'primary_metric', 'coverage', 'features',
                'y_stats', 'label_stats', 'models', 'score', 'fold_timestamps',
                'leakage', 'mismatch_telemetry', 'metrics_schema_version',
                'scoring_schema_version', 'n_effective', 'metric_name'
            }
            # Check if any known metric keys exist at top level in run_data
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            top_level_metrics = {k: v for k, v in sorted_items(run_data) if k in known_metric_keys}
            if top_level_metrics:
                metrics_data = top_level_metrics
                logger.debug(f"Reconstructed metrics from top-level keys in run_data: {list(top_level_metrics.keys())}")

        # Check resolved_metadata if run_data doesn't have metrics
        if not metrics_data and resolved_metadata:
            metrics_data = resolved_metadata.get('metrics', {})

        # SST Architecture: Read metrics from canonical location (cohort_dir) first
        # This is the most common case for TARGET_RANKING/FEATURE_SELECTION stages
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
                        # Extract all numeric metrics (exclude metadata fields, but preserve schema versions)
                        metrics_data = {
                            k: v for k, v in metrics_dict.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                            and (isinstance(v, (int, float, str)) or (isinstance(v, (list, dict)) and v))
                            # Preserve schema versions (strings)
                            or k in ['metrics_schema_version', 'scoring_schema_version']
                        }
                        logger.debug(f"✅ Loaded metrics from canonical parquet: {metrics_parquet}")
                except Exception as e:
                    logger.debug(f"Failed to read metrics.parquet from {cohort_dir}: {e}")

            # 2. Fall back to metrics.json in cohort directory (debug export)
            if not metrics_data:
                metrics_json_file = cohort_path / "metrics.json"
                if metrics_json_file.exists():
                    try:
                        with open(metrics_json_file, 'r') as f:
                            metrics_json = json.load(f)
                        # Extract all numeric metrics (exclude metadata fields, but preserve schema versions)
                        metrics_data = {
                            k: v for k, v in metrics_json.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                            and (isinstance(v, (int, float, str)) or (isinstance(v, (list, dict)) and v))
                            # Preserve schema versions (strings)
                            or k in ['metrics_schema_version', 'scoring_schema_version']
                        }
                        logger.debug(f"✅ Loaded metrics from JSON: {metrics_json_file}")
                    except Exception as e:
                        logger.debug(f"Failed to read metrics.json from {cohort_dir}: {e}")

            # Fallback: Try target-first structure if metrics not found in cohort_dir
            if not metrics_data:
                try:
                    # Try to extract target from cohort_dir path or resolved_metadata
                    target = None
                    if resolved_metadata:
                        target = resolved_metadata.get('target')

                    # If we can't get target from metadata, try to extract from path
                    if not target:
                        # Walk up from cohort_dir to find target
                        current = cohort_path
                        for _ in range(10):
                            if current.name.startswith('cohort='):
                                # Parent should be target directory
                                if current.parent.name not in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'FEATURE_SELECTION', 'TARGET_RANKING']:
                                    target = current.parent.name
                                    break
                            if not current.parent.exists():
                                break
                            current = current.parent

                    if target:
                        # Find run directory using SST helper
                        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                        run_dir = get_run_root(cohort_path)

                        # Try target-first metrics location
                        target_metrics_dir = run_dir / "targets" / target / "metrics"
                        if target_metrics_dir.exists():
                            # Check for view-organized metrics
                            # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                            for item in iterdir_sorted(target_metrics_dir):
                                if item.is_dir() and item.name.startswith("view="):
                                    metrics_file = item / "metrics.json"
                                    if metrics_file.exists():
                                        try:
                                            with open(metrics_file, 'r') as f:
                                                metrics_json = json.load(f)
                                            metrics_data = {
                                                k: v for k, v in metrics_json.items()
                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                           'composite_version', 'leakage', 'leakage_flag']
                                                and (isinstance(v, (int, float, str)) or (isinstance(v, (list, dict)) and v))
                                                # Preserve schema versions (strings)
                                                or k in ['metrics_schema_version', 'scoring_schema_version']
                                            }
                                            if metrics_data:
                                                break
                                        except Exception:
                                            pass
                                elif item.name == "metrics.json":
                                    try:
                                        with open(item, 'r') as f:
                                            metrics_json = json.load(f)
                                        metrics_data = {
                                            k: v for k, v in metrics_json.items()
                                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                       'composite_version', 'leakage', 'leakage_flag']
                                            and (isinstance(v, (int, float, str)) or (isinstance(v, (list, dict)) and v))
                                            # Preserve schema versions (strings)
                                            or k in ['metrics_schema_version', 'scoring_schema_version']
                                        }
                                        if metrics_data:
                                            break
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.debug(f"Failed to read metrics from target-first structure: {e}")

        # Store all extracted metrics
        if metrics_data:
            outputs['metrics'] = metrics_data

        # Stability metrics (if available)
        if run_data.get('additional_data'):
            if 'stability' in run_data['additional_data']:
                outputs['stability'] = run_data['additional_data']['stability']

        return outputs
