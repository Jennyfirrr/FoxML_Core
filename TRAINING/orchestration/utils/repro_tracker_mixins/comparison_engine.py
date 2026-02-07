# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Comparison Engine Mixin for ReproducibilityTracker.

Contains methods for finding comparable runs and computing drift.
Extracted from reproducibility_tracker.py for maintainability.

SST COMPLIANCE:
- Uses extract_n_effective() helper for sample size extraction
- Uses target_first_paths for path construction
- Uses Stage/View enums for consistent handling

DETERMINISM:
- Sort-based comparisons for index queries
- Enum comparisons instead of string comparisons
- Sample-adjusted statistics for robust comparisons
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# SST: Import Stage and View enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

# SST: Import extraction helpers
from TRAINING.orchestration.utils.reproducibility.utils import extract_n_effective

# Schema version for reproducibility files
from TRAINING.orchestration.utils.reproducibility_tracker import REPRODUCIBILITY_SCHEMA_VERSION

logger = logging.getLogger(__name__)


class ComparisonEngineMixin:
    """
    Mixin class providing comparison methods for ReproducibilityTracker.

    This mixin contains methods related to:
    - Finding matching cohorts from index
    - Comparing runs within cohorts
    - Computing drift between runs
    - Sample-adjusted statistical comparisons

    Methods in this mixin expect the following attributes on self:
    - _repro_base_dir: Path - Base directory for reproducibility artifacts
    - n_ratio_threshold: float - Threshold for N ratio comparisons
    - thresholds: Dict[str, Dict[str, float]] - Classification thresholds

    Methods in this mixin call the following methods on self:
    - _classify_diff: Method for classifying differences
    - _compute_cohort_id: Method for computing cohort IDs (from CohortManagerMixin)
    """

    # Type hints for expected attributes (set by the main class)
    _repro_base_dir: Path
    n_ratio_threshold: float
    thresholds: Dict[str, Dict[str, float]]

    def _find_matching_cohort(
        self,
        stage: str,
        target: str,
        cohort_metadata: Dict[str, Any],
        view: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None
    ) -> Optional[str]:
        """Find matching cohort ID from index.parquet."""
        # Read index from globals/ first, then fall back to legacy REPRODUCIBILITY/
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        globals_dir = get_globals_dir(self._repro_base_dir)
        index_file = globals_dir / "index.parquet"
        if not index_file.exists():
            # Fallback to legacy
            repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
            index_file = repro_dir / "index.parquet"

        if not index_file.exists():
            return None

        try:
            df = pd.read_parquet(index_file)

            # Normalize stage to enum, then to string
            stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
            phase = str(stage_enum)  # Stage enum's __str__ returns .value

            # Filter for same phase, target, mode, symbol, model_family
            mask = (df['phase'] == phase) & (df['target'] == target)

            if view:
                mask &= (df['mode'] == view.upper())
            if symbol:
                mask &= (df['symbol'] == symbol)
            if model_family:
                mask &= (df['model_family'] == model_family)

            candidates = df[mask]

            if len(candidates) == 0:
                return None

            # Try exact match first (same cohort_id)
            # Derive view from view for cohort_id computation
            view_for_cohort = "CROSS_SECTIONAL"
            if view:
                rt_upper = view.upper()
                if rt_upper == "SYMBOL_SPECIFIC":
                    view_for_cohort = "SYMBOL_SPECIFIC"
            target_id = self._compute_cohort_id(cohort_metadata, view=view_for_cohort)
            exact_match = candidates[candidates['cohort_id'] == target_id]
            if len(exact_match) > 0:
                return target_id

            # Try close match (similar N, same config)
            n_target = cohort_metadata.get('n_effective_cs', 0)
            n_ratio_threshold = self.n_ratio_threshold

            # Need repro_dir for path resolution
            repro_dir = self._repro_base_dir / "REPRODUCIBILITY"

            for _, row in candidates.iterrows():
                n_existing = row.get('n_effective', 0)
                if n_existing == 0:
                    continue

                n_ratio = min(n_target, n_existing) / max(n_target, n_existing)
                if n_ratio >= n_ratio_threshold:
                    # Load metadata to check config match
                    try:
                        prev_path = repro_dir / row['path']
                        metadata_file = prev_path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                prev_meta = json.load(f)

                            # Check config match
                            config_match = (
                                prev_meta.get('universe_sig') == cohort_metadata.get('cs_config', {}).get('universe_sig') and
                                prev_meta.get('min_cs') == cohort_metadata.get('cs_config', {}).get('min_cs') and
                                prev_meta.get('leakage_filter_version') == cohort_metadata.get('cs_config', {}).get('leakage_filter_version', 'v1')
                            )
                            if config_match:
                                return row['cohort_id']
                    except Exception as e:
                        logger.debug(f"Failed to check config match for cohort {row['cohort_id']}: {e}")
                        continue

            return None
        except Exception as e:
            logger.debug(f"Failed to find matching cohort: {e}")
            return None

    def _compare_within_cohort(
        self,
        prev_run: Dict[str, Any],
        curr_run: Dict[str, Any],
        metric_type: str = 'roc_auc'
    ) -> Tuple[str, float, float, Optional[float], Dict[str, Any]]:
        """
        Compare runs within the same cohort using sample-adjusted statistics.

        Returns:
            (classification, abs_diff, rel_diff, z_score, stats_dict)
        """
        prev_value = float(prev_run.get('auc', 0.0))
        curr_value = float(curr_run.get('auc', 0.0))

        prev_std = float(prev_run.get('std_score', 0.0)) if prev_run.get('std_score') else None
        curr_std = float(curr_run.get('std_score', 0.0)) if curr_run.get('std_score') else None

        # Get sample sizes - use SST accessor
        prev_n = extract_n_effective(prev_run)
        curr_n = extract_n_effective(curr_run)

        if prev_n is None or curr_n is None:
            # Fallback to non-sample-adjusted comparison
            class_result = self._classify_diff(prev_value, curr_value, prev_std, metric_type)
            return class_result + ({'sample_adjusted': False},)

        prev_n = int(prev_n)
        curr_n = int(curr_n)

        # Sample-adjusted variance estimation
        # For AUC: var ≈ AUC * (1 - AUC) / N
        if prev_value > 0 and prev_value < 1:
            var_prev = prev_value * (1 - prev_value) / prev_n
        else:
            var_prev = (prev_std ** 2) / prev_n if prev_std and prev_std > 0 else None

        if curr_value > 0 and curr_value < 1:
            var_curr = curr_value * (1 - curr_value) / curr_n
        else:
            var_curr = (curr_std ** 2) / curr_n if curr_std and curr_std > 0 else None

        # Compute z-score
        delta = curr_value - prev_value
        abs_diff = abs(delta)
        rel_diff = (abs_diff / max(abs(prev_value), 1e-8)) * 100 if prev_value != 0 else 0.0

        z_score = None
        if var_prev is not None and var_curr is not None:
            sigma = math.sqrt(var_prev + var_curr)
            if sigma > 0:
                z_score = abs_diff / sigma

        # Classification using z-score
        thresholds = self.thresholds.get(metric_type, self.thresholds.get('roc_auc'))
        z_thr = thresholds.get('z_score', 1.0)

        if z_score is not None:
            if z_score < 1.0:
                classification = 'STABLE'
            elif z_score < 2.0:
                classification = 'DRIFTING'
            else:
                classification = 'DIVERGED'
        else:
            # Fallback to non-sample-adjusted
            classification, _, _, _ = self._classify_diff(prev_value, curr_value, prev_std, metric_type)

        stats = {
            'prev_n': prev_n,
            'curr_n': curr_n,
            'var_prev': var_prev,
            'var_curr': var_curr,
            'z_score': z_score,
            'sample_adjusted': True
        }

        return classification, abs_diff, rel_diff, z_score, stats

    def get_last_comparable_run(
        self,
        stage: str,
        target: str,
        view: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        cohort_id: Optional[str] = None,
        current_N: Optional[int] = None,
        n_ratio_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the last comparable run from index.parquet.

        Args:
            stage: Pipeline stage
            target: Target/item name
            view: Route type (CROSS_SECTIONAL/INDIVIDUAL)
            symbol: Symbol name (for INDIVIDUAL mode)
            model_family: Model family (for TRAINING)
            cohort_id: Cohort ID (if already computed)
            current_N: Current n_effective (for N ratio check)
            n_ratio_threshold: Override default N ratio threshold

        Returns:
            Previous run metrics dict or None if no comparable run found
        """
        # Read index from globals/ first, then fall back to legacy REPRODUCIBILITY/
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        globals_dir = get_globals_dir(self._repro_base_dir)
        index_file = globals_dir / "index.parquet"
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        if not index_file.exists():
            # Fallback to legacy
            index_file = repro_dir / "index.parquet"

        if not index_file.exists():
            return None

        try:
            df = pd.read_parquet(index_file)

            # Normalize stage to enum, then to string
            stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
            phase = str(stage_enum)  # Stage enum's __str__ returns .value

            # Filter for matching stage, target, mode, symbol, model_family
            mask = (df['phase'] == phase) & (df['target'] == target)

            # FIX: Handle null mode/symbol for backward compatibility
            # For FEATURE_SELECTION, require mode non-null (new runs must have mode)
            # For other stages, allow nulls (backward compatibility)
            if view:
                route_upper = view.upper()
                # Normalize stage to enum for comparison
                stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
                if stage_enum == Stage.FEATURE_SELECTION:
                    # For FEATURE_SELECTION, require mode non-null (new runs must have mode)
                    mask &= (df['mode'].notna()) & (df['mode'] == route_upper)
                else:
                    # For other stages, allow nulls (backward compatibility)
                    mask &= ((df['mode'].isna()) | (df['mode'] == route_upper))

            # FIX: Handle null symbol for backward compatibility
            # For INDIVIDUAL mode, require symbol non-null
            # For CROSS_SECTIONAL, allow nulls (backward compatibility)
            if symbol:
                # Normalize view to enum for comparison
                view_enum = View.from_string(view) if isinstance(view, str) else view
                if view_enum == View.SYMBOL_SPECIFIC:
                    # For SYMBOL_SPECIFIC mode, require symbol non-null
                    mask &= (df['symbol'].notna()) & (df['symbol'] == symbol)
                else:
                    # For CROSS_SECTIONAL, allow nulls (backward compatibility)
                    mask &= ((df['symbol'].isna()) | (df['symbol'] == symbol))
            elif not symbol:
                # If no symbol specified, normalize view to enum for comparison
                view_enum = View.from_string(view) if isinstance(view, str) else view
                if view_enum == View.CROSS_SECTIONAL:
                    # For CROSS_SECTIONAL, require symbol is null (prevent history forking)
                    mask &= (df['symbol'].isna())

            if model_family:
                mask &= (df['model_family'] == model_family)

            # FIX: Always filter by cohort_id or data_fingerprint (don't compare across cohorts)
            # This prevents noisy comparisons when underlying dataset changes
            if cohort_id:
                mask &= (df['cohort_id'] == cohort_id)
            else:
                # Try to compute cohort_id from current run metadata if available
                # Or filter by data_fingerprint if available (stronger than cohort_id)
                # For now, log warning and allow comparison (may be noisy)
                logger.debug("No cohort_id provided to get_last_comparable_run, comparisons may be noisy")

            # Sort by run_started_at (monotonic, more reliable than date/timestamp)
            # Fallback to date if run_started_at not available (backward compatibility)
            if 'run_started_at' in df.columns:
                # Prefer monotonic run_started_at for correct ordering (handles clock skew, resumed runs)
                candidates = df[mask].sort_values('run_started_at', ascending=False, na_position='last')
                logger.debug(f"Using run_started_at for prev run selection (monotonic, more reliable)")
            else:
                # Fallback to date for backward compatibility with old index files
                candidates = df[mask].sort_values('date', ascending=False)
                logger.debug(f"Using date for prev run selection (run_started_at not available, backward compatibility)")

            if len(candidates) == 0:
                return None

            # Apply N ratio filter if current_N provided
            threshold = n_ratio_threshold or self.n_ratio_threshold
            if current_N is not None:
                for _, row in candidates.iterrows():
                    prev_n = row.get('n_effective', 0)
                    if prev_n == 0:
                        continue

                    n_ratio = min(current_N, prev_n) / max(current_N, prev_n)
                    if n_ratio >= threshold:
                        # Load metrics from path
                        try:
                            prev_path = repro_dir / row['path']
                            metrics_file = prev_path / "metrics.json"
                            if metrics_file.exists():
                                # Use atomic read pattern (though reads don't need atomicity, be consistent)
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                # Also load metadata for cohort_id
                                metadata_file = prev_path / "metadata.json"
                                if metadata_file.exists():
                                    # Use atomic read pattern (though reads don't need atomicity, be consistent)
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    metrics['cohort_id'] = metadata.get('cohort_id')
                                    metrics['n_effective'] = metadata.get('n_effective')
                                return metrics
                        except Exception as e:
                            logger.debug(f"Failed to load previous run from {row['path']}: {e}")
                            continue

                # No run passed N ratio filter
                return None
            else:
                # No N filter - just return latest
                latest = candidates.iloc[0]
                try:
                    prev_path = repro_dir / latest['path']
                    metrics_file = prev_path / "metrics.json"
                    if metrics_file.exists():
                        # Use atomic read pattern (though reads don't need atomicity, be consistent)
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        metadata_file = prev_path / "metadata.json"
                        if metadata_file.exists():
                            # Use atomic read pattern (though reads don't need atomicity, be consistent)
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            metrics['cohort_id'] = metadata.get('cohort_id')
                            metrics['n_effective'] = metadata.get('n_effective')
                        return metrics
                except Exception as e:
                    logger.debug(f"Failed to load previous run from {latest['path']}: {e}")
                    return None
        except Exception as e:
            logger.debug(f"Failed to query index for previous run: {e}")
            return None

    def _compute_drift(
        self,
        prev_run: Dict[str, Any],
        curr_run: Dict[str, Any],
        cohort_metadata: Dict[str, Any],
        stage: Union[str, Stage],
        target: str,
        view: Optional[str],
        symbol: Optional[str],
        model_family: Optional[str],
        cohort_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Compute drift comparison and return drift.json data.

        Explicitly links both runs (current + previous) for self-contained drift.json.
        """
        # Normalize stage to Stage enum
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage

        # Use SST accessor for sample size
        prev_n = extract_n_effective(prev_run) or 0
        curr_n = cohort_metadata.get('n_effective_cs', 0)

        n_ratio = min(prev_n, curr_n) / max(prev_n, curr_n) if max(prev_n, curr_n) > 0 else 0.0

        # Extract previous run metadata
        prev_run_id = prev_run.get('run_id') or prev_run.get('timestamp', 'unknown')
        prev_cohort_id = prev_run.get('cohort_id') or prev_run.get('cohort_metadata', {}).get('cohort_id', 'unknown')
        prev_auc = float(prev_run.get('auc', 0.0))

        # Current run metadata
        curr_auc = float(curr_run.get('auc', 0.0))

        if n_ratio < self.n_ratio_threshold:
            # Defensive: handle both string and Enum for view/stage
            view_str = None
            if view:
                if isinstance(view, str):
                    view_str = view.upper()
                elif hasattr(view, 'value'):
                    view_str = view.value.upper() if isinstance(view.value, str) else str(view.value).upper()
                else:
                    view_str = str(view).upper()

            stage_str = stage
            if isinstance(stage, str):
                # Normalize stage to enum, then to string
                stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
                stage_str = str(stage_enum)  # Stage enum's __str__ returns .value
            elif hasattr(stage, 'value'):
                stage_str = stage.value.upper().replace("MODEL_TRAINING", "TRAINING") if isinstance(stage.value, str) else str(stage.value).upper()
            else:
                stage_str = str(stage).upper().replace("MODEL_TRAINING", "TRAINING")

            return {
                "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
                "stage": stage_str,
                "view": view_str,
                "target": target,
                "symbol": symbol,
                "model_family": model_family,
                "current": {
                    "run_id": run_id,
                    "cohort_id": cohort_id,
                    "n_effective": curr_n,
                    "auc": curr_auc
                },
                "previous": {
                    "run_id": prev_run_id,
                    "cohort_id": prev_cohort_id,
                    "n_effective": prev_n,
                    "auc": prev_auc
                },
                "status": "INCOMPARABLE",
                "reason": f"n_effective ratio={n_ratio:.3f} ({prev_n} vs {curr_n}) < {self.n_ratio_threshold}",
                "n_ratio": n_ratio,
                "threshold": self.n_ratio_threshold,
                "created_at": datetime.now().isoformat()
            }

        # Sample-adjusted comparison
        prev_std = float(prev_run.get('std_score', 0.0)) if prev_run.get('std_score') else None

        classification, abs_diff, rel_diff, z_score, stats = self._compare_within_cohort(
            prev_run, curr_run, 'roc_auc'
        )

        # Build status label
        if stats.get('sample_adjusted', False):
            status_label = f"{classification}_SAMPLE_ADJUSTED"
        else:
            status_label = classification

        # Build reason string
        if z_score is not None:
            reason = f"n_ratio={n_ratio:.3f}, |z|={abs(z_score):.2f}"
            if z_score < 1.0:
                reason += " → stable"
            elif z_score < 2.0:
                reason += " → drifting"
            else:
                reason += " → diverged"
        else:
            reason = f"n_ratio={n_ratio:.3f}, abs_diff={abs_diff:.4f}"

        # Defensive: handle both string and Enum for view
        view_str = None
        if view:
            if isinstance(view, str):
                view_str = view.upper()
            elif hasattr(view, 'value'):
                view_str = view.value.upper() if isinstance(view.value, str) else str(view.value).upper()
            else:
                view_str = str(view).upper()

        # Use normalized stage_enum for string conversion
        stage_str = str(stage_enum).replace("MODEL_TRAINING", "TRAINING")

        return {
            "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
            "stage": stage_str,
            "view": view_str,
            "target": target,
            "symbol": symbol,
            "model_family": model_family,
            "current": {
                "run_id": run_id,
                "cohort_id": cohort_id,
                "n_effective": curr_n,
                "auc": curr_auc
            },
            "previous": {
                "run_id": prev_run_id,
                "cohort_id": prev_cohort_id,
                "n_effective": prev_n,
                "auc": prev_auc
            },
            "delta_auc": curr_auc - prev_auc,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "z_score": z_score,
            "status": status_label,
            "reason": reason,
            "n_ratio": n_ratio,
            "sample_adjusted": stats.get('sample_adjusted', False),
            "created_at": datetime.now().isoformat()
        }
