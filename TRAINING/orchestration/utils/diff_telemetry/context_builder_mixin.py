# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Context Builder Mixin for DiffTelemetry.

Contains methods for building and validating ResolvedRunContext objects
which resolve outcome-influencing metadata from various sources.

Extracted from diff_telemetry.py for maintainability.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import types
from .types import ResolvedRunContext

# Import Stage enum for stage-aware logic
from TRAINING.orchestration.utils.scope_resolution import Stage

logger = logging.getLogger(__name__)


class ContextBuilderMixin:
    """
    Mixin class providing context building methods for DiffTelemetry.

    This mixin contains methods related to:
    - Building ResolvedRunContext from available data sources
    - Getting required fields for each stage
    - Validating stage schema requirements

    Methods in this mixin are standalone and don't depend on other mixins.
    """

    def _build_resolved_context(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None
    ) -> ResolvedRunContext:
        """Build resolved run context from available data sources.

        CRITICAL: This resolves all outcome-influencing metadata from the actual sources
        to ensure no nulls for required fields.

        Priority order (for SST consistency):
        1. resolved_metadata (in-memory, most authoritative - same data that will be written)
        2. metadata.json from filesystem (only if resolved_metadata not provided, and verify run_id matches)
        3. cohort_metadata (fallback)
        4. additional_data (fallback)

        Args:
            stage: Pipeline stage
            run_data: Run data dict
            cohort_metadata: Cohort metadata (fallback)
            additional_data: Additional data dict
            cohort_dir: Cohort directory (fallback - only used if resolved_metadata not provided)
            resolved_metadata: In-memory metadata dict (SST - preferred source)

        Returns:
            ResolvedRunContext with all resolved values
        """
        ctx = ResolvedRunContext()

        # Priority 1: Use resolved_metadata if provided (SST consistency)
        # CRITICAL: Use shallow copy to prevent mutation of caller's dict
        metadata = {}
        if resolved_metadata:
            metadata = dict(resolved_metadata)  # Shallow copy to prevent mutation
            logger.debug("Using resolved_metadata for SST consistency (in-memory dict, shallow copy)")
        else:
            # Priority 2: Try to load metadata.json from filesystem (only if resolved_metadata not provided)
            if cohort_dir:
                metadata_file = Path(cohort_dir) / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            file_metadata = json.load(f)
                        # CRITICAL: Verify run_id AND stage match to avoid stale file hazard and cross-stage weirdness
                        current_run_id = run_data.get('run_id') or run_data.get('timestamp')
                        file_run_id = file_metadata.get('run_id')
                        file_stage = file_metadata.get('stage')
                        current_stage = stage

                        if file_run_id == current_run_id and file_stage == current_stage:
                            metadata = file_metadata
                            logger.debug(f"Using metadata.json from filesystem (run_id={current_run_id}, stage={current_stage} match)")
                        else:
                            mismatch_reasons = []
                            if file_run_id != current_run_id:
                                mismatch_reasons.append(f"run_id mismatch: file={file_run_id}, current={current_run_id}")
                            if file_stage != current_stage:
                                mismatch_reasons.append(f"stage mismatch: file={file_stage}, current={current_stage}")
                            logger.debug(f"Skipping stale metadata.json ({', '.join(mismatch_reasons)})")
                    except Exception as e:
                        logger.debug(f"Could not load metadata.json from {metadata_file}: {e}")

        # Data provenance (from metadata.json or cohort_metadata)
        ctx.n_symbols = (
            metadata.get('n_symbols') or
            cohort_metadata.get('n_symbols') if cohort_metadata else None
        )
        ctx.symbols = (
            metadata.get('symbols') or
            cohort_metadata.get('symbols') if cohort_metadata else None
        )
        # Date range - use SST accessor
        from TRAINING.orchestration.utils.reproducibility.utils import extract_date_range
        ctx.date_start, ctx.date_end = extract_date_range(metadata, cohort_metadata)
        ctx.n_effective = (
            metadata.get('n_effective') or
            cohort_metadata.get('n_effective_cs') if cohort_metadata else None
        )
        ctx.data_dir = (
            metadata.get('data_dir') or
            cohort_metadata.get('data_dir') if cohort_metadata else None or
            (additional_data.get('data_dir') if additional_data else None) or
            (resolved_metadata.get('data_dir') if resolved_metadata else None)
        )
        ctx.min_cs = (
            metadata.get('min_cs') or
            cohort_metadata.get('min_cs') if cohort_metadata else None
        )
        ctx.max_cs_samples = (
            metadata.get('max_cs_samples') or
            cohort_metadata.get('max_cs_samples') if cohort_metadata else None
        )

        # Task provenance - extract target from multiple sources
        # CRITICAL: Check resolved_metadata first (SST consistency), then fallback to other sources
        ctx.target = (
            (resolved_metadata.get('target') if resolved_metadata else None) or
            (resolved_metadata.get('target') if resolved_metadata else None) or
            metadata.get('target') or  # Primary: metadata.json uses 'target'
            metadata.get('target') or  # Fallback: some sources use 'target'
            (additional_data.get('target') if additional_data else None) or
            (additional_data.get('target') if additional_data else None) or
            (run_data.get('target')) or
            (run_data.get('target')) or
            (run_data.get('target'))  # Fallback: target often contains target
        )

        # If still None, try to parse from cohort_dir path as last resort
        if not ctx.target and cohort_dir:
            try:
                # Path format: .../TARGET_RANKING/CROSS_SECTIONAL/{target}/cohort=...
                parts = str(cohort_dir).split('/')
                for i, part in enumerate(parts):
                    if part in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO'] and i + 1 < len(parts):
                        ctx.target = parts[i + 1]
                        break
            except Exception:
                pass
        cv_details = metadata.get('cv_details', {})
        ctx.horizon_minutes = (
            cv_details.get('horizon_minutes') or
            additional_data.get('horizon_minutes') if additional_data else None
        )
        ctx.labeling_impl_hash = (
            cv_details.get('label_definition_hash') or
            additional_data.get('labeling_hash') if additional_data else None
        )

        # Split provenance
        ctx.cv_method = cv_details.get('cv_method')
        ctx.purge_minutes = cv_details.get('purge_minutes')
        ctx.embargo_minutes = cv_details.get('embargo_minutes')
        ctx.leakage_filter_version = (
            metadata.get('leakage_filter_version') or
            additional_data.get('leakage_filter_version') if additional_data else None
        )
        ctx.split_seed = (
            metadata.get('seed') or
            additional_data.get('split_seed') if additional_data else None
        )
        ctx.fold_assignment_hash = (
            additional_data.get('fold_assignment_hash') if additional_data else None
        )

        # Feature provenance
        # Check multiple sources in priority order:
        # 1. resolved_metadata['evaluation']['n_features'] (nested - where it's actually stored)
        # 2. resolved_metadata['n_features'] (top-level fallback)
        # 3. metadata['evaluation']['n_features'] (from filesystem, nested)
        # 4. metadata['n_features'] (from filesystem, top-level)
        # 5. additional_data['n_features'] (direct pass-through)
        # Also check n_features_selected as alternative key name
        ctx.n_features = (
            (resolved_metadata.get('evaluation', {}).get('n_features') if resolved_metadata else None) or
            (resolved_metadata.get('n_features') if resolved_metadata else None) or
            (resolved_metadata.get('evaluation', {}).get('n_features_selected') if resolved_metadata else None) or
            (resolved_metadata.get('n_features_selected') if resolved_metadata else None) or
            metadata.get('evaluation', {}).get('n_features') or
            metadata.get('n_features') or
            metadata.get('evaluation', {}).get('n_features_selected') or
            metadata.get('n_features_selected') or
            (additional_data.get('n_features') if additional_data else None) or
            (additional_data.get('n_features_selected') if additional_data else None)
        )
        ctx.feature_names = (
            additional_data.get('feature_names') if additional_data else None
        )

        # Stage strategy
        ctx.view = (
            metadata.get('view') or
            additional_data.get('view') if additional_data else None
        )
        ctx.model_family = (
            additional_data.get('model_family') if additional_data else None
        )
        ctx.model_families = (
            additional_data.get('model_families') if additional_data else None
        )
        ctx.feature_selection = (
            additional_data.get('feature_selection') if additional_data else None
        )
        ctx.trainer_strategy = (
            additional_data.get('strategy') if additional_data else None
        )

        # Hyperparameters (extract from training data if available)
        hyperparameters = None
        if additional_data and 'training' in additional_data:
            training_data = additional_data['training']
            # Extract hyperparameters (exclude strategy, model_family, seeds - handled separately)
            excluded_keys = {'strategy', 'model_family', 'split_seed', 'train_seed', 'seed'}
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            from TRAINING.common.utils.determinism_ordering import sorted_items
            hyperparameters = {k: v for k, v in sorted_items(training_data)
                              if k not in excluded_keys and v is not None}
        elif run_data.get('training'):
            training_data = run_data['training']
            excluded_keys = {'strategy', 'model_family', 'split_seed', 'train_seed', 'seed'}
            # DETERMINISM: Use sorted_items() for deterministic iteration in artifact-shaping code
            from TRAINING.common.utils.determinism_ordering import sorted_items
            hyperparameters = {k: v for k, v in sorted_items(training_data)
                              if k not in excluded_keys and v is not None}
        ctx.hyperparameters = hyperparameters if hyperparameters else None

        # Environment
        ctx.python_version = (
            additional_data.get('python_version') if additional_data else None
        )
        ctx.library_versions = (
            additional_data.get('library_versions') if additional_data else None
        )

        # Experiment tracking
        ctx.experiment_id = (
            metadata.get('experiment_id') or
            additional_data.get('experiment_id') if additional_data else None
        )

        return ctx

    def _get_required_fields_for_stage(self, stage: str) -> List[str]:
        """
        Get list of required fields for a given stage.

        These fields must be present and non-null in resolved_metadata before finalize_run().

        Args:
            stage: Pipeline stage

        Returns:
            List of required field names
        """
        base_required = ['stage', 'run_id', 'cohort_id']

        # SST: Use Stage enum for comparison
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        if stage_enum == Stage.TARGET_RANKING:
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'min_cs', 'max_cs_samples'
            ]
        elif stage_enum == Stage.FEATURE_SELECTION:
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'min_cs', 'max_cs_samples'
            ]
        elif stage_enum == Stage.TRAINING:
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'model_family', 'min_cs', 'max_cs_samples'
            ]
        else:
            # Unknown stage - require base fields only
            return base_required

    def _validate_stage_schema(
        self,
        stage: str,
        ctx: ResolvedRunContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate that required fields for stage are present and non-null.

        Returns:
            (is_valid, reason) - if not valid, reason explains what's missing
        """
        missing = []

        # All stages require:
        if ctx.n_effective is None:
            missing.append("n_effective")
        if ctx.target is None:
            missing.append("target")
        if ctx.view is None:
            missing.append("view")
        if ctx.date_start is None:
            missing.append("date_start")
        if ctx.date_end is None:
            missing.append("date_end")

        # Stage-specific requirements
        # SST: Use Stage enum for comparison
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        if stage_enum == Stage.TARGET_RANKING:
            # TARGET_RANKING does NOT require model_family or feature_signature
            pass
        elif stage_enum == Stage.FEATURE_SELECTION:
            # FEATURE_SELECTION requires feature pipeline info
            if ctx.n_features is None:
                missing.append("n_features")
        elif stage_enum == Stage.TRAINING:
            # TRAINING requires model_family and feature info
            if ctx.model_family is None:
                missing.append("model_family")
            if ctx.n_features is None:
                missing.append("n_features")

        if missing:
            return False, f"Missing required fields for {stage}: {', '.join(missing)}"
        return True, None
