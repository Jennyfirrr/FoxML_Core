# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Fingerprint Computation Mixin for DiffTelemetry.

Contains methods for computing fingerprints (hashes) for various run components:
- Config fingerprints
- Data fingerprints
- Feature fingerprints
- Target fingerprints

Extracted from diff_telemetry.py for maintainability.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Import types (ResolvedRunContext is used by *_from_context methods)
from .types import ResolvedRunContext

logger = logging.getLogger(__name__)


class FingerprintMixin:
    """
    Mixin class providing fingerprint computation methods for DiffTelemetry.

    This mixin contains methods related to:
    - Computing config fingerprints from context or raw data
    - Computing data fingerprints
    - Computing feature fingerprints
    - Computing target fingerprints
    - Normalizing values for stable hashing

    Methods in this mixin expect the following attributes on self:
    - output_dir: Optional[Path] - Output directory for loading resolved config
    """

    def _normalize_value_for_hash(self, val: Any) -> str:
        """Normalize value for stable hashing.

        Ensures:
        - Dicts are sorted by key
        - Lists preserve order (only sort when explicitly unordered, e.g., feature sets)
        - Floats use repr() for exact representation (avoids 1e-7 vs 0.0 collapse)
        - NaN/inf/-0.0 handled reproducibly
        - None/null are handled consistently
        """
        if val is None:
            return "None"
        elif isinstance(val, (int, str, bool)):
            return str(val)
        elif isinstance(val, float):
            # Use repr() for exact representation (avoids precision loss)
            # Handle special cases reproducibly
            if np.isnan(val):
                return "nan"
            elif np.isinf(val):
                return "inf" if val > 0 else "-inf"
            elif val == 0.0:
                # Distinguish -0.0 from 0.0
                return "0.0" if str(val) == "0.0" else "-0.0"
            else:
                # Use repr() to preserve exact value (e.g., 1e-7 stays 1e-7, not 0.0)
                return repr(val)
        elif isinstance(val, (list, tuple)):
            # CRITICAL: Preserve order by default (order may be semantic)
            # Only sort when explicitly marked as unordered (e.g., feature sets)
            # For now, preserve order - caller should sort feature lists before hashing
            normalized = [self._normalize_value_for_hash(v) for v in val]
            return "[" + ",".join(normalized) + "]"
        elif isinstance(val, dict):
            # Sort by key (dicts are unordered by definition)
            # DETERMINISM: Use sorted_items() helper for consistency
            from TRAINING.common.utils.determinism_ordering import sorted_items
            normalized = {k: self._normalize_value_for_hash(v) for k, v in sorted_items(val)}
            sorted_items_list = sorted(normalized.items())
            return "{" + ",".join(f"{k}:{v}" for k, v in sorted_items_list) + "}"
        else:
            return str(val)

    def _compute_config_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext,
        additional_data: Optional[Dict[str, Any]],
        resolved_metadata: Optional[Dict[str, Any]] = None,
        cohort_dir: Optional[Path] = None
    ) -> Optional[Union[str, Dict[str, str]]]:
        """Compute config fingerprint from resolved context.

        Returns:
            If both fingerprints available: dict with 'config_fingerprint' and 'deterministic_config_fingerprint'
            Otherwise: single fingerprint string (for backward compatibility)

        Priority:
        1. deterministic_config_fingerprint from resolved_metadata (if available)
        2. config_fingerprint from resolved_metadata (if available)
        3. Load from config.resolved.json (if cohort_dir or output_dir available)
        4. Compute from context fields (fallback)
        """
        # Try to extract both fingerprints from resolved_metadata first
        if resolved_metadata:
            deterministic_fp = resolved_metadata.get('deterministic_config_fingerprint')
            full_fp = resolved_metadata.get('config_fingerprint')

            # If both available, return dict
            if deterministic_fp and full_fp:
                return {
                    'config_fingerprint': full_fp,
                    'deterministic_config_fingerprint': deterministic_fp
                }

            # If only deterministic available, return it
            if deterministic_fp:
                return deterministic_fp

            # If only full available, return it
            if full_fp:
                return full_fp

        # Fallback: Try to load from config.resolved.json if available
        # This ensures we get deterministic fingerprint even if resolved_metadata doesn't have it
        output_dir = getattr(self, 'output_dir', None)
        if cohort_dir is None and output_dir:
            # Try to find globals/config.resolved.json from output_dir
            resolved_config_path = output_dir / "globals" / "config.resolved.json"
            if resolved_config_path.exists():
                try:
                    with open(resolved_config_path, 'r') as f:
                        resolved_config = json.load(f)
                    deterministic_fp = resolved_config.get('deterministic_config_fingerprint')
                    full_fp = resolved_config.get('config_fingerprint')

                    if deterministic_fp and full_fp:
                        return {
                            'config_fingerprint': full_fp,
                            'deterministic_config_fingerprint': deterministic_fp
                        }
                    if deterministic_fp:
                        return deterministic_fp
                    if full_fp:
                        return full_fp
                except Exception as e:
                    logger.debug(f"Could not load deterministic fingerprint from {resolved_config_path}: {e}")

        # Fallback: compute from context fields (legacy behavior)
        config_parts = []

        # Strategy and model_family (if applicable)
        if ctx.trainer_strategy:
            config_parts.append(f"strategy={ctx.trainer_strategy}")
        if ctx.model_family:
            config_parts.append(f"model_family={ctx.model_family}")
        if ctx.n_features is not None:
            config_parts.append(f"n_features={ctx.n_features}")
        if ctx.min_cs is not None:
            config_parts.append(f"min_cs={ctx.min_cs}")
        if ctx.max_cs_samples is not None:
            config_parts.append(f"max_cs_samples={ctx.max_cs_samples}")

        # Split protocol signature
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
            # SST: Use sha256_short for consistent config hashing
            from TRAINING.common.utils.config_hashing import sha256_short
            config_parts.append(f"split={sha256_short(split_str, 8)}")

        if config_parts:
            config_str = "|".join(sorted(config_parts))
            # SST: Use sha256_short for consistent config hashing
            from TRAINING.common.utils.config_hashing import sha256_short
            return sha256_short(config_str, 16)
        return None

    def _compute_config_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute config fingerprint.

        Includes:
        - Strategy, model_family, n_features, min_cs, max_cs_samples
        - Split protocol signature (CV scheme, folds, purge/embargo, leakage guards)
        """
        config_parts = []

        # Extract config-relevant fields
        if additional_data:
            for key in ['strategy', 'model_family', 'n_features', 'min_cs', 'max_cs_samples']:
                if key in additional_data:
                    config_parts.append(f"{key}={additional_data[key]}")

            # Split protocol signature (CV scheme, folds, purge/embargo, leakage guards)
            # CRITICAL: Include split_seed (if fold assignment depends on seed) but NOT train_seed
            # CRITICAL: Include fold_assignment_hash (from actual fold IDs, not just seed)
            #           This ensures fold logic changes break comparability even if seed stays same
            split_parts = []
            for key in ['cv_method', 'folds', 'purge_minutes', 'embargo_minutes',
                       'leakage_filter_version', 'horizon_minutes', 'split_seed', 'fold_assignment_hash']:
                if key in additional_data:
                    val = additional_data[key]
                    if val is not None:
                        # Normalize value for stable hashing
                        normalized_val = self._normalize_value_for_hash(val)
                        split_parts.append(f"{key}={normalized_val}")

            if split_parts:
                split_str = "|".join(sorted(split_parts))
                # SST: Use sha256_short for consistent config hashing
                from TRAINING.common.utils.config_hashing import sha256_short
                config_parts.append(f"split={sha256_short(split_str, 8)}")

        if run_data.get('additional_data'):
            for key in ['strategy', 'model_family']:
                if key in run_data['additional_data']:
                    config_parts.append(f"{key}={run_data['additional_data'][key]}")

        if config_parts:
            # Canonicalize: sorted keys, normalized values
            config_str = "|".join(sorted(config_parts))
            # SST: Use sha256_short for consistent config hashing
            from TRAINING.common.utils.config_hashing import sha256_short
            return sha256_short(config_str, 16)
        return None

    def _compute_data_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute data fingerprint from resolved context."""
        data_parts = []

        # Data parameters (all required, so should be non-null)
        if ctx.n_symbols is not None:
            data_parts.append(f"n_symbols={ctx.n_symbols}")
        if ctx.date_start:
            data_parts.append(f"date_start={ctx.date_start}")
        if ctx.date_end:
            data_parts.append(f"date_end={ctx.date_end}")
        if ctx.min_cs is not None:
            data_parts.append(f"min_cs={ctx.min_cs}")
        if ctx.max_cs_samples is not None:
            data_parts.append(f"max_cs_samples={ctx.max_cs_samples}")

        # Data identity (if available)
        if ctx.data_fingerprint:
            data_parts.append(f"data_id={ctx.data_fingerprint}")

        if data_parts:
            data_str = "|".join(sorted(data_parts))
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return None

    def _compute_data_fingerprint(
        self,
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute data fingerprint.

        Includes:
        - Data parameters (n_symbols, date_range, min_cs, max_cs_samples)
        - Data identity (if available: row IDs hash, file manifest, parquet metadata)
        """
        data_parts = []

        if cohort_metadata:
            # Extract data-relevant fields
            for key in ['n_symbols', 'date_start', 'date_end', 'min_cs', 'max_cs_samples']:
                if key in cohort_metadata:
                    val = cohort_metadata[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")

            # Data identity (actual data fingerprint if available)
            if 'data_fingerprint' in cohort_metadata:
                data_parts.append(f"data_id={cohort_metadata['data_fingerprint']}")
            elif 'data_hash' in cohort_metadata:
                data_parts.append(f"data_id={cohort_metadata['data_hash']}")

        if additional_data:
            for key in ['n_symbols', 'date_range']:
                if key in additional_data:
                    val = additional_data[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")

            # Data identity from additional_data
            if 'data_fingerprint' in additional_data:
                data_parts.append(f"data_id={additional_data['data_fingerprint']}")

        if data_parts:
            # Canonicalize: sorted keys, normalized values
            data_str = "|".join(sorted(data_parts))
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return None

    def _compute_feature_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute feature fingerprint from resolved context."""
        feature_parts = []

        # Feature count (if available)
        if ctx.n_features is not None:
            feature_parts.append(f"n_features={ctx.n_features}")

        # Feature names (if available)
        if ctx.feature_names:
            # Sort feature names for stable hash (features are unordered set)
            feature_list_str = "|".join(sorted(ctx.feature_names))
            feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")

        # Feature pipeline signature
        if ctx.feature_pipeline_signature:
            feature_parts.append(f"pipeline={ctx.feature_pipeline_signature}")

        if feature_parts:
            features_str = "|".join(sorted(feature_parts))
            return hashlib.sha256(features_str.encode()).hexdigest()[:16]
        return None

    def _compute_feature_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute feature fingerprint.

        Includes:
        - Feature count and names (if available)
        - Feature pipeline signature (transforms, lookbacks, normalization, winsorization, missing-value policy)
        """
        feature_parts = []

        # Feature count and names
        if additional_data:
            if 'n_features' in additional_data:
                feature_parts.append(f"count={additional_data['n_features']}")
            if 'feature_names' in additional_data:
                # Hash actual feature list for precise matching
                feature_list = additional_data['feature_names']
                if isinstance(feature_list, list):
                    feature_list_str = "|".join(sorted(feature_list))
                    feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")
            # Feature pipeline signature
            if 'feature_pipeline_hash' in additional_data:
                feature_parts.append(f"pipeline={additional_data['feature_pipeline_hash']}")
            elif 'feature_transforms' in additional_data:
                # Hash transform config (normalization, winsorization, missing-value policy, lookbacks)
                transforms = additional_data['feature_transforms']
                if isinstance(transforms, dict):
                    transforms_str = "|".join(f"{k}={v}" for k, v in sorted(transforms.items()))
                    feature_parts.append(f"transforms={hashlib.sha256(transforms_str.encode()).hexdigest()[:8]}")

        if run_data.get('additional_data'):
            if 'n_features' in run_data['additional_data']:
                feature_parts.append(f"count={run_data['additional_data']['n_features']}")
            if 'feature_names' in run_data['additional_data']:
                feature_list = run_data['additional_data']['feature_names']
                if isinstance(feature_list, list):
                    feature_list_str = "|".join(sorted(feature_list))
                    feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")

        if feature_parts:
            # Canonicalize: sorted keys, normalized values
            features_str = "|".join(sorted(feature_parts))
            return hashlib.sha256(features_str.encode()).hexdigest()[:16]
        return None

    def _compute_target_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute target fingerprint from resolved context."""
        target_parts = []

        # Target name (from resolved context)
        if ctx.target:
            target_parts.append(f"target={ctx.target}")

        # Labeling implementation signature (from resolved context)
        if ctx.labeling_impl_hash:
            target_parts.append(f"labeling={ctx.labeling_impl_hash}")

        # Horizon and objective (from resolved context)
        if ctx.horizon_minutes is not None:
            target_parts.append(f"horizon={ctx.horizon_minutes}")
        if ctx.objective:
            target_parts.append(f"objective={ctx.objective}")

        if target_parts:
            # Canonicalize: sorted keys, normalized values
            target_str = "|".join(sorted(target_parts))
            return hashlib.sha256(target_str.encode()).hexdigest()[:16]
        return None

    def _compute_target_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute target fingerprint.

        Includes:
        - Target name
        - Labeling implementation signature (labeling code/config hash, not just target name)
        """
        target_parts = []

        # Target name
        target = None
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('target'):
            # Extract target from target (format: "target:family" or "target:symbol:family")
            parts = run_data['target'].split(':')
            target = parts[0] if parts else None

        if target:
            target_parts.append(f"target={target}")

        # Labeling implementation signature (code/config hash)
        if additional_data:
            if 'labeling_hash' in additional_data:
                target_parts.append(f"labeling={additional_data['labeling_hash']}")
            elif 'labeling_config' in additional_data:
                # Hash labeling config (horizon, labeling rules, etc.)
                labeling_config = additional_data['labeling_config']
                if isinstance(labeling_config, dict):
                    labeling_str = "|".join(f"{k}={v}" for k, v in sorted(labeling_config.items()))
                    target_parts.append(f"labeling={hashlib.sha256(labeling_str.encode()).hexdigest()[:8]}")
            # Horizon and objective
            if 'horizon_minutes' in additional_data:
                target_parts.append(f"horizon={additional_data['horizon_minutes']}")
            if 'objective' in additional_data:
                target_parts.append(f"objective={additional_data['objective']}")

        if target_parts:
            # Canonicalize: sorted keys, normalized values
            target_str = "|".join(sorted(target_parts))
            return hashlib.sha256(target_str.encode()).hexdigest()[:16]
        return None
