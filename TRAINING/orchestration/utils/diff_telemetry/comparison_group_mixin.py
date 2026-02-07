# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Comparison Group Mixin for DiffTelemetry.

Contains methods for building ComparisonGroup objects that define
which runs can be compared (same data, config, features, etc.).

Extracted from diff_telemetry.py for maintainability.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

# Import types
from .types import ResolvedRunContext, ComparisonGroup

# Import Stage enum for stage-aware logic
from TRAINING.orchestration.utils.scope_resolution import Stage

logger = logging.getLogger(__name__)


class ComparisonGroupMixin:
    """
    Mixin class providing comparison group building methods for DiffTelemetry.

    This mixin contains methods related to:
    - Building ComparisonGroup from resolved context (stage-aware)
    - Building ComparisonGroup from raw run data

    Methods in this mixin expect the following:
    - self._repro_base_dir: Optional[Path] - for view lookup
    - self._normalize_value_for_hash(val): Method from FingerprintMixin
    - self._compute_feature_fingerprint(run_data, additional_data): Method from FingerprintMixin

    IMPORTANT: This mixin must be listed AFTER FingerprintMixin in the class inheritance
    to ensure _normalize_value_for_hash and _compute_feature_fingerprint are available.
    """

    def _build_comparison_group_from_context(
        self,
        stage: str,
        ctx: ResolvedRunContext,
        config_fp: Optional[str],
        data_fp: Optional[str],
        task_fp: Optional[str],
        hyperparameters_signature: Optional[str] = None,
        train_seed: Optional[int] = None,
        library_versions_signature: Optional[str] = None,
        symbol: Optional[str] = None,
        universe_sig: Optional[str] = None,
        run_identity: Optional[Any] = None,  # RunIdentity SST object
    ) -> ComparisonGroup:
        """Build comparison group from resolved context (stage-aware).

        CRITICAL: Only includes fields that are relevant for the stage.
        - TARGET_RANKING: Does NOT include model_family (not applicable) BUT DOES include feature_signature
        - FEATURE_SELECTION: Includes feature_signature but NOT model_family
        - TRAINING: Includes both model_family and feature_signature

        CRITICAL: symbol and universe_sig are required for proper comparison scoping:
        - symbol: For SS runs, ensures AAPL only compares to AAPL (not AVGO)
        - universe_sig: For CS runs, ensures same symbol set comparisons

        NEW: If run_identity is provided (RunIdentity SST object), use its signatures
        instead of the fallback values from ResolvedRunContext.

        This prevents storing null placeholders for fields that aren't stage-relevant.
        """
        # Routing signature from view (SST) - use view if available, fallback to view
        routing_signature = None
        view_for_fingerprint = None
        try:
            from TRAINING.orchestration.utils.run_context import get_view
            if hasattr(self, '_repro_base_dir'):
                view_for_fingerprint = get_view(self._repro_base_dir)
        except Exception:
            pass

        # Use view (SST) if available, otherwise fallback to view
        mode_for_signature = view_for_fingerprint if view_for_fingerprint else ctx.view
        if mode_for_signature:
            routing_signature = hashlib.sha256(mode_for_signature.encode()).hexdigest()[:16]

        # Stage-specific fields
        model_family = None
        feature_signature = None

        # Stage-specific fields
        hp_sig = None
        seed = None
        lib_sig = None  # Initialize for all stages

        # NEW: If run_identity is provided, use its signatures (SST source of truth)
        if run_identity is not None:
            # Extract signatures from RunIdentity if available
            if hasattr(run_identity, 'feature_signature') and run_identity.feature_signature:
                feature_signature = run_identity.feature_signature
            if hasattr(run_identity, 'hparams_signature') and run_identity.hparams_signature:
                hp_sig = run_identity.hparams_signature
            if hasattr(run_identity, 'train_seed') and run_identity.train_seed is not None:
                seed = run_identity.train_seed
            if hasattr(run_identity, 'dataset_signature') and run_identity.dataset_signature:
                # Override data_fp with the authoritative dataset_signature
                data_fp = run_identity.dataset_signature
            if hasattr(run_identity, 'target_signature') and run_identity.target_signature:
                task_fp = run_identity.target_signature
            if hasattr(run_identity, 'routing_signature') and run_identity.routing_signature:
                routing_signature = run_identity.routing_signature
            # Model family from context (not in RunIdentity)
            # SST: Use Stage enum for comparison
            stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
            if stage_enum == Stage.TRAINING:
                model_family = ctx.model_family

        # SST fallback: Always populate train_seed for traceability (all stages)
        # Even if not required for comparison in TARGET_RANKING, it's useful for auditing
        if seed is None:
            if train_seed is not None:
                seed = train_seed
            else:
                try:
                    from CONFIG.config_loader import get_cfg
                    seed = get_cfg("pipeline.determinism.base_seed", default=42)
                except Exception:
                    seed = 42  # FALLBACK_DEFAULT_OK
                logger.debug(f"Using SST fallback train_seed={seed} for {stage}")

        if run_identity is None:
            # Fallback to old logic when run_identity not provided
            if stage == "TARGET_RANKING":
                # TARGET_RANKING: model_family, hyperparameters, train_seed, and library_versions are NOT applicable
                # BUT feature_signature IS required - different features = different results
                feature_signature = ctx.feature_fingerprint
                # NOTE: If feature_fingerprint is not available, will be computed below from feature_names
            elif stage == "FEATURE_SELECTION":
                # FEATURE_SELECTION: feature_signature, hyperparameters, train_seed, and library_versions are required
                # CRITICAL: Different hyperparameters/seeds/versions in feature selection models = different features selected
                feature_signature = ctx.feature_fingerprint
                hp_sig = hyperparameters_signature
                seed = train_seed
                lib_sig = library_versions_signature
            elif stage == "TRAINING":
                # TRAINING: model_family, feature_signature, hyperparameters, train_seed, and library_versions are all required
                model_family = ctx.model_family
                feature_signature = ctx.feature_fingerprint
                hp_sig = hyperparameters_signature
                seed = train_seed
                lib_sig = library_versions_signature

        # Extract split_signature from run_identity or context
        # CRITICAL: split_signature is required for TARGET_RANKING, so compute it if missing
        split_sig = None
        if run_identity is not None and hasattr(run_identity, 'split_signature') and run_identity.split_signature:
            split_sig = run_identity.split_signature
        elif ctx.split_protocol_fingerprint:
            split_sig = ctx.split_protocol_fingerprint
        elif ctx.fold_assignment_hash:
            # Fallback: use fold_assignment_hash as split signature
            split_sig = ctx.fold_assignment_hash
        else:
            # CRITICAL: Compute split_signature from context fields if not available
            # This ensures TARGET_RANKING has required split_signature
            split_parts = []
            if ctx.cv_method:
                split_parts.append(f"cv_method={ctx.cv_method}")
            if ctx.folds is not None:
                split_parts.append(f"folds={ctx.folds}")
            if ctx.purge_minutes is not None:
                split_parts.append(f"purge_minutes={ctx.purge_minutes}")
            if ctx.embargo_minutes is not None:
                normalized_embargo = self._normalize_value_for_hash(ctx.embargo_minutes)
                split_parts.append(f"embargo_minutes={normalized_embargo}")
            if ctx.leakage_filter_version:
                split_parts.append(f"leakage_filter_version={ctx.leakage_filter_version}")
            if ctx.horizon_minutes is not None:
                split_parts.append(f"horizon_minutes={ctx.horizon_minutes}")
            if ctx.split_seed is not None:
                split_parts.append(f"split_seed={ctx.split_seed}")
            if ctx.fold_assignment_hash:
                split_parts.append(f"fold_assignment_hash={ctx.fold_assignment_hash}")

            if split_parts:
                split_str = "|".join(sorted(split_parts))
                split_sig = hashlib.sha256(split_str.encode()).hexdigest()
            else:
                # Last resort: use a default signature if no split info available
                # This should rarely happen, but prevents validation failure
                logger.warning(f"No split information available for {stage}, using default split signature")
                split_sig = hashlib.sha256(b"default_split").hexdigest()

        # CRITICAL: Compute feature_signature from feature_names if not available
        # This ensures TARGET_RANKING has the required feature_signature
        if feature_signature is None and ctx.feature_names:
            feature_list_str = "|".join(sorted(ctx.feature_names))
            feature_signature = hashlib.sha256(feature_list_str.encode()).hexdigest()
            logger.debug(f"Computed feature_signature from {len(ctx.feature_names)} feature names")

        comparison_group = ComparisonGroup(
            experiment_id=ctx.experiment_id,  # Can be None if not tracked
            dataset_signature=data_fp,
            task_signature=task_fp,
            routing_signature=routing_signature,
            split_signature=split_sig,  # CRITICAL: CV split identity (required for all stages)
            n_effective=ctx.n_effective,  # CRITICAL: Exact sample size (required, non-null)
            model_family=model_family,  # Stage-specific: None for TARGET_RANKING/FEATURE_SELECTION
            feature_signature=feature_signature,  # CRITICAL: Required for all stages (different features = different results)
            hyperparameters_signature=hp_sig,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            train_seed=seed,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            library_versions_signature=lib_sig,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            universe_sig=universe_sig,  # CRITICAL: For CS, ensures same symbol set comparisons
            symbol=symbol  # CRITICAL: For SS, ensures AAPL only compares to AAPL
        )

        # Validate in strict mode
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            comparison_group.validate(stage, strict=True)  # Will raise if invalid

        return comparison_group

    def _build_comparison_group(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]],
        config_fp: Optional[str],
        data_fp: Optional[str],
        task_fp: Optional[str]
    ) -> ComparisonGroup:
        """Build comparison group.

        CRITICAL: Includes ALL outcome-influencing metadata to ensure only runs with
        EXACTLY the same configuration are compared and stored together.

        Outcome-influencing metadata includes:
        - Exact n_effective (sample size) - 5k vs 5k, not 5k vs 10k
        - Dataset (universe, date range, min_cs, max_cs_samples)
        - Task (target, horizon, objective)
        - Routing/view configuration
        - Model family (different families = different outcomes)
        - Feature set (different features = different outcomes)
        - Hyperparameters (different HPs = different outcomes)
        - Train seed (different seeds = different outcomes)
        """
        # Extract experiment_id if available
        experiment_id = None
        if additional_data and 'experiment_id' in additional_data:
            experiment_id = additional_data['experiment_id']
        elif run_data.get('additional_data') and 'experiment_id' in run_data['additional_data']:
            experiment_id = run_data['additional_data']['experiment_id']

        # Routing signature from view
        routing_signature = None
        if additional_data and 'view' in additional_data:
            routing_signature = hashlib.sha256(additional_data['view'].encode()).hexdigest()[:16]

        # Extract exact n_effective (CRITICAL: must match exactly for comparison)
        n_effective = None
        if cohort_metadata and 'n_effective_cs' in cohort_metadata:
            n_effective = int(cohort_metadata['n_effective_cs'])
        elif additional_data and 'n_effective_cs' in additional_data:
            n_effective = int(additional_data['n_effective_cs'])
        elif run_data.get('metrics') and 'n_effective_cs' in run_data['metrics']:
            n_effective = int(run_data['metrics']['n_effective_cs'])
        elif run_data.get('additional_data') and 'n_effective_cs' in run_data['additional_data']:
            n_effective = int(run_data['additional_data']['n_effective_cs'])

        # Extract model_family (CRITICAL: different families = different outcomes)
        model_family = None
        if additional_data and 'model_family' in additional_data:
            model_family = additional_data['model_family']
        elif run_data.get('additional_data') and 'model_family' in run_data['additional_data']:
            model_family = run_data['additional_data']['model_family']
        elif run_data.get('target'):
            # Extract from target (format: "target:family" or "target:symbol:family")
            parts = run_data['target'].split(':')
            if len(parts) >= 2:
                model_family = parts[-1]  # Last part is usually the family

        # Extract feature signature (CRITICAL: different features = different outcomes)
        feature_signature = None
        if additional_data and 'n_features' in additional_data:
            # Use feature fingerprint if available, otherwise hash n_features
            feature_fp = self._compute_feature_fingerprint(run_data, additional_data)
            if feature_fp:
                feature_signature = feature_fp
        elif run_data.get('additional_data') and 'n_features' in run_data['additional_data']:
            feature_fp = self._compute_feature_fingerprint(run_data, run_data.get('additional_data'))
            if feature_fp:
                feature_signature = feature_fp

        # Extract hyperparameters and train_seed (CRITICAL: different HPs/seeds = different outcomes)
        hyperparameters_signature = None
        train_seed = None

        # Try to get hyperparameters from additional_data or run_data
        training_data = {}
        if additional_data and 'training' in additional_data:
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
        library_versions = None
        if additional_data and 'library_versions' in additional_data:
            library_versions = additional_data['library_versions']
        elif run_data.get('additional_data') and 'library_versions' in run_data.get('additional_data', {}):
            library_versions = run_data['additional_data']['library_versions']
        elif run_data.get('library_versions'):
            library_versions = run_data['library_versions']

        if library_versions and isinstance(library_versions, dict):
            # Sort keys for stable hash, include python_version if available
            lib_parts = []
            python_version = (
                (additional_data.get('python_version') if additional_data else None) or
                (run_data.get('python_version'))
            )
            if python_version:
                lib_parts.append(f"python={python_version}")
            # Sort library versions for stable hash
            for key in sorted(library_versions.keys()):
                lib_parts.append(f"{key}={library_versions[key]}")
            if lib_parts:
                lib_str = "|".join(lib_parts)
                library_versions_signature = hashlib.sha256(lib_str.encode()).hexdigest()[:16]

        return ComparisonGroup(
            experiment_id=experiment_id,
            dataset_signature=data_fp,
            task_signature=task_fp,
            routing_signature=routing_signature,
            n_effective=n_effective,  # CRITICAL: Exact sample size
            model_family=model_family,  # CRITICAL: Model family
            feature_signature=feature_signature,  # CRITICAL: Feature set
            hyperparameters_signature=hyperparameters_signature,  # CRITICAL: Different HPs = different outcomes
            train_seed=train_seed,  # CRITICAL: Different seeds = different outcomes
            library_versions_signature=library_versions_signature  # CRITICAL: Different versions = different outcomes
        )
