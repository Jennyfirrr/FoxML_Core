# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Diff Engine Mixin for DiffTelemetry.

Contains methods for computing diffs, checking comparability, and classifying severity.
Extracted from diff_telemetry.py for maintainability.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import (
    ChangeSeverity,
    NormalizedSnapshot,
    DiffResult,
)
from .run_hash import _can_runs_be_compared

# SST: Use deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import iterdir_sorted

logger = logging.getLogger(__name__)


class DiffEngineMixin:
    """
    Mixin class providing diff computation methods for DiffTelemetry.

    This mixin contains all methods related to:
    - Computing diffs between snapshots
    - Checking comparability of snapshots
    - Computing metric deltas
    - Classifying severity of changes
    - Trend analysis

    Methods in this mixin expect the following attributes on self:
    - run_dir: Path to the run directory
    - logger: Logger instance (optional, falls back to module logger)
    """

    def compute_diff(
        self,
        current_snapshot: NormalizedSnapshot,
        prev_snapshot: NormalizedSnapshot,
        prev_cohort_dir: Optional[Path] = None,
        curr_cohort_dir: Optional[Path] = None
    ) -> DiffResult:
        """
        Compute diff between two snapshots.

        Args:
            current_snapshot: Current run snapshot
            prev_snapshot: Previous run snapshot
            prev_cohort_dir: Optional path to previous cohort directory (for trend loading)
            curr_cohort_dir: Optional path to current cohort directory (for trend loading)

        Returns:
            DiffResult with comparability status, changes, and severity
        """
        # Check comparability
        comparable, reason = self._check_comparability(current_snapshot, prev_snapshot)

        if not comparable:
            # CRITICAL: If not comparable, severity must be CRITICAL with reason
            return DiffResult(
                prev_run_id=prev_snapshot.run_id,
                current_run_id=current_snapshot.run_id,
                comparable=False,
                comparability_reason=reason,
                severity=ChangeSeverity.CRITICAL,
                severity_reason=f"Runs are not comparable: {reason}",
                excluded_factors_changed={},  # Empty but present (stable shape)
                summary={
                    'total_changes': 0,
                    'input_changes': 0,
                    'process_changes': 0,
                    'output_changes': 0,
                    'metric_deltas_count': 0,
                    'excluded_factors_changed': False,
                    'excluded_factors_summary': None
                }
            )

        # Compute changes
        changed_keys = []
        patch = []

        # Diff inputs
        input_changes = self._diff_dict(
            prev_snapshot.inputs,
            current_snapshot.inputs,
            prefix="inputs"
        )
        changed_keys.extend(input_changes['keys'])
        patch.extend(input_changes['patch'])

        # Diff process
        process_changes = self._diff_dict(
            prev_snapshot.process,
            current_snapshot.process,
            prefix="process"
        )
        changed_keys.extend(process_changes['keys'])
        patch.extend(process_changes['patch'])

        # Diff outputs (metrics)
        output_changes = self._diff_dict(
            prev_snapshot.outputs,
            current_snapshot.outputs,
            prefix="outputs"
        )
        changed_keys.extend(output_changes['keys'])
        patch.extend(output_changes['patch'])

        # Compute metric deltas
        metric_deltas, metric_deltas_total = self._compute_metric_deltas(
            prev_snapshot.outputs,
            current_snapshot.outputs
        )

        # Compute trend deltas (if cohort directories are provided)
        trend_deltas = {}
        if prev_cohort_dir and curr_cohort_dir:
            prev_metadata_path = Path(prev_cohort_dir) / "metadata.json"
            curr_metadata_path = Path(curr_cohort_dir) / "metadata.json"

            prev_trend = self._load_trend_from_metadata(prev_metadata_path)
            curr_trend = self._load_trend_from_metadata(curr_metadata_path)

            if prev_trend and curr_trend:
                trend_deltas = self._compute_trend_deltas(prev_trend, curr_trend)

        # CRITICAL: Check output digests for artifact/metric reproducibility
        output_digest_changes = []
        if prev_snapshot.metrics_sha256 != current_snapshot.metrics_sha256:
            output_digest_changes.append("metrics_sha256")
        if prev_snapshot.artifacts_manifest_sha256 != current_snapshot.artifacts_manifest_sha256:
            output_digest_changes.append("artifacts_manifest_sha256")
        if prev_snapshot.predictions_sha256 != current_snapshot.predictions_sha256:
            output_digest_changes.append("predictions_sha256")

        # Extract excluded factors (hyperparameters, seeds, versions) for reporting
        excluded_factors_changed = self._extract_excluded_factor_changes(
            current_snapshot, prev_snapshot
        )

        # Build summary with readable excluded factors summary
        excluded_summary = self._format_excluded_factors_summary(excluded_factors_changed)

        # Compute impact classification from metric deltas
        impact_label, top_regressions, top_improvements = self._classify_metric_impact(metric_deltas)

        # Count significant deltas (all entries in metric_deltas are significant by design)
        metric_deltas_significant = len(metric_deltas)

        # Compute trend direction change (if slope_per_day is available)
        trend_direction_change = None
        if trend_deltas and 'slope_per_day' in trend_deltas:
            prev_slope = trend_deltas['slope_per_day'].get('prev')
            curr_slope = trend_deltas['slope_per_day'].get('curr')
            if prev_slope is not None and curr_slope is not None:
                prev_improving = prev_slope > 0
                curr_improving = curr_slope > 0
                if prev_improving != curr_improving:
                    if prev_improving:
                        trend_direction_change = "improving→declining"
                    else:
                        trend_direction_change = "declining→improving"

        summary = {
            'total_changes': len(changed_keys),
            'input_changes': len(input_changes['keys']),
            'process_changes': len(process_changes['keys']),
            'output_changes': len(output_changes['keys']),
            'metric_deltas_total': metric_deltas_total,
            'metric_deltas_count': metric_deltas_significant,
            'metric_deltas_significant': metric_deltas_significant,
            'excluded_factors_changed': bool(excluded_factors_changed),
            'excluded_factors_summary': excluded_summary,
            'output_digest_changes': output_digest_changes,
            'impact_label': impact_label,
            'top_regressions': top_regressions,
            'top_improvements': top_improvements,
            'trend_deltas_count': len(trend_deltas),
            'trend_direction_change': trend_direction_change
        }

        # CRITICAL: Determine severity purely from the report (SST-style)
        severity, severity_reason = self._determine_severity(
            changed_keys=changed_keys,
            input_changes=input_changes,
            process_changes=process_changes,
            output_changes=output_changes,
            metric_deltas=metric_deltas,
            excluded_factors_changed=excluded_factors_changed,
            excluded_factors_summary=excluded_summary,
            summary=summary
        )

        return DiffResult(
            prev_run_id=prev_snapshot.run_id,
            current_run_id=current_snapshot.run_id,
            comparable=True,
            prev_timestamp=prev_snapshot.timestamp,
            prev_snapshot_seq=prev_snapshot.snapshot_seq,
            prev_stage=prev_snapshot.stage,
            prev_view=prev_snapshot.view,
            comparison_source=getattr(prev_snapshot, '_comparison_source', None),
            changed_keys=changed_keys,
            severity=severity,
            severity_reason=severity_reason,
            summary=summary,
            excluded_factors_changed=excluded_factors_changed,
            patch=patch,
            metric_deltas=metric_deltas,
            trend_deltas=trend_deltas
        )

    def _check_comparability(
        self,
        current: NormalizedSnapshot,
        prev: NormalizedSnapshot
    ) -> Tuple[bool, Optional[str]]:
        """Check if two snapshots are comparable.

        Checks both manifest flags (authoritative) and legacy heuristics.
        Refuses unstable-vs-stable comparisons.
        """
        # Check manifest comparability flags first (authoritative)
        current_run_dir = None
        prev_run_dir = None

        if hasattr(current, '_run_dir') and current._run_dir:
            current_run_dir = Path(current._run_dir)
        if hasattr(prev, '_run_dir') and prev._run_dir:
            prev_run_dir = Path(prev._run_dir)

        # If we have both run directories, check manifest flags
        if current_run_dir and prev_run_dir:
            can_compare, reason = _can_runs_be_compared(
                current_run_dir, prev_run_dir,
                run1_id=current.run_id, run2_id=prev.run_id
            )
            if not can_compare:
                return False, reason or "manifest flags indicate not comparable"

        # CRITICAL: Check fingerprint schema version compatibility
        if current.fingerprint_schema_version != prev.fingerprint_schema_version:
            return False, (
                f"Different fingerprint schema versions: "
                f"{current.fingerprint_schema_version} vs {prev.fingerprint_schema_version}. "
                f"Fingerprint computation changed - runs are not comparable."
            )

        # Must be same stage
        if current.stage != prev.stage:
            return False, f"Different stages: {current.stage} vs {prev.stage}"

        # Must be same view
        if current.view != prev.view:
            return False, f"Different views: {current.view} vs {prev.view}"

        # Must be same target (if specified)
        if current.target and prev.target and current.target != prev.target:
            return False, f"Different targets: {current.target} vs {prev.target}"

        # CRITICAL: For SYMBOL_SPECIFIC view, must be same symbol
        if current.view == "SYMBOL_SPECIFIC":
            if current.symbol != prev.symbol:
                return False, f"Different symbols: {current.symbol} vs {prev.symbol}"

        # CRITICAL: Must have identical comparison groups
        if current.comparison_group and prev.comparison_group:
            # Validate both comparison groups before comparing
            try:
                is_valid, missing = current.comparison_group.validate(current.stage, strict=False)
                if not is_valid:
                    return False, f"Current snapshot missing required fields: {missing}"
            except Exception as e:
                return False, f"Current snapshot validation failed: {e}"

            try:
                is_valid, missing = prev.comparison_group.validate(prev.stage, strict=False)
                if not is_valid:
                    return False, f"Previous snapshot missing required fields: {missing}"
            except Exception as e:
                return False, f"Previous snapshot validation failed: {e}"

            # Compare keys
            cg_curr = current.comparison_group.to_key(current.stage, strict=False)
            cg_prev = prev.comparison_group.to_key(prev.stage, strict=False)

            if cg_curr is None or cg_prev is None:
                return False, "One or both comparison groups are invalid (missing required fields)"

            if cg_curr != cg_prev:
                return False, f"Different comparison groups: {cg_curr} vs {cg_prev}"
        elif current.comparison_group or prev.comparison_group:
            return False, "One snapshot missing comparison group"

        return True, None

    def _extract_excluded_factor_changes(
        self,
        current: NormalizedSnapshot,
        prev: NormalizedSnapshot
    ) -> Dict[str, Any]:
        """Extract changes in excluded factors.

        NOTE: All outcome-influencing factors are now part of the comparison group.
        Returns empty dict (no excluded factors remain).
        """
        return {}

    def _format_excluded_factors_summary(self, excluded: Dict[str, Any]) -> Optional[str]:
        """Format excluded factors changes into readable summary.

        NOTE: All outcome-influencing factors are now in comparison group.
        Returns None (no excluded factors).
        """
        return None

    def _count_excluded_factors_changed(self, excluded: Dict[str, Any]) -> int:
        """Count number of excluded factors that changed.

        Returns 0 (no excluded factors remain).
        """
        return 0

    def _diff_dict(
        self,
        prev: Dict[str, Any],
        current: Dict[str, Any],
        prefix: str = ""
    ) -> Dict[str, List]:
        """Diff two dictionaries, return changed keys and patch operations."""
        changed_keys = []
        patch = []

        all_keys = set(prev.keys()) | set(current.keys())

        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key

            prev_val = prev.get(key)
            current_val = current.get(key)

            if prev_val != current_val:
                changed_keys.append(path)

                if key not in prev:
                    patch.append({
                        "op": "add",
                        "path": f"/{path}",
                        "value": self._normalize_value(current_val)
                    })
                elif key not in current:
                    patch.append({
                        "op": "remove",
                        "path": f"/{path}"
                    })
                else:
                    patch.append({
                        "op": "replace",
                        "path": f"/{path}",
                        "value": self._normalize_value(current_val),
                        "old_value": self._normalize_value(prev_val)
                    })

        return {'keys': changed_keys, 'patch': patch}

    def _normalize_value(self, val: Any) -> Any:
        """Normalize value for diffing (round floats, sort lists, etc.)."""
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 6)
        elif isinstance(val, (list, tuple)):
            try:
                return sorted([self._normalize_value(v) for v in val])
            except TypeError:
                return [self._normalize_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: self._normalize_value(v) for k, v in sorted(val.items())}
        else:
            return val

    def _flatten_metrics_dict(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Recursively flatten nested metrics dict, preserving dot-notation keys."""
        flattened = {}
        if not isinstance(metrics, dict):
            if prefix:
                flattened[prefix] = metrics
            return flattened

        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics_dict(value, full_key))
            elif isinstance(value, (int, float, str, bool, type(None))):
                flattened[full_key] = value
            elif isinstance(value, (list, tuple)):
                flattened[full_key] = list(value) if isinstance(value, tuple) else value
            elif isinstance(value, set):
                flattened[full_key] = sorted(value)
            else:
                flattened[full_key] = str(value)
        return flattened

    def _load_trend_from_metadata(self, metadata_path: Path) -> Optional[Dict[str, Any]]:
        """Load trend metadata from file."""
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get('trend', {})
        except Exception as e:
            logger.debug(f"Failed to load trend from {metadata_path}: {e}")
            return None

    def _compute_trend_deltas(
        self,
        prev_trend: Dict[str, Any],
        curr_trend: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute trend changes between previous and current."""
        deltas = {}

        trend_fields = ['slope_per_day', 'r_squared', 'direction', 'volatility']

        for field in trend_fields:
            prev_val = prev_trend.get(field)
            curr_val = curr_trend.get(field)

            if prev_val is not None or curr_val is not None:
                deltas[field] = {
                    'prev': prev_val,
                    'curr': curr_val,
                }

                # Compute delta for numeric fields
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    deltas[field]['delta'] = curr_val - prev_val
                    if prev_val != 0:
                        deltas[field]['delta_pct'] = (curr_val - prev_val) / abs(prev_val) * 100

        return deltas

    def _classify_metric_impact(
        self,
        metric_deltas: Dict[str, Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Classify overall impact from metric deltas and extract top regressions/improvements."""
        if not metric_deltas:
            return 'none', [], []

        # Per-metric polarity: True = higher is better
        higher_is_better = {
            'auc': True,
            'composite_score': True,
            'mean_importance': True,
            'std_score': False,
            'n_models': True,
            'pos_rate': True,
            'n_effective_cs': True,
        }

        impact_hierarchy = {'none': 0, 'noise': 1, 'minor': 2, 'major': 3}

        worst_impact = 'none'
        regressions = []
        improvements = []

        for metric_name, delta_info in metric_deltas.items():
            impact = delta_info.get('impact_label', 'none')
            if impact_hierarchy.get(impact, 0) > impact_hierarchy.get(worst_impact, 0):
                worst_impact = impact

            delta_abs = delta_info.get('delta_abs', 0)
            is_higher_better = higher_is_better.get(metric_name, True)
            signed_delta = delta_abs if is_higher_better else -delta_abs

            if signed_delta < 0:
                regressions.append({
                    'metric': metric_name,
                    'delta_abs': delta_abs,
                    'signed_delta': signed_delta,
                    'delta_pct': delta_info.get('delta_pct', 0),
                    'z_score': delta_info.get('z_score'),
                    'impact': impact
                })
            elif signed_delta > 0:
                improvements.append({
                    'metric': metric_name,
                    'delta_abs': delta_abs,
                    'signed_delta': signed_delta,
                    'delta_pct': delta_info.get('delta_pct', 0),
                    'z_score': delta_info.get('z_score'),
                    'impact': impact
                })

        regressions.sort(key=lambda x: x['signed_delta'])
        improvements.sort(key=lambda x: x['signed_delta'], reverse=True)

        return worst_impact, regressions[:top_k], improvements[:top_k]

    def _determine_severity(
        self,
        changed_keys: List[str],
        input_changes: Dict,
        process_changes: Dict,
        output_changes: Dict,
        metric_deltas: Dict[str, Dict[str, float]],
        excluded_factors_changed: Dict[str, Any],
        excluded_factors_summary: Optional[str],
        summary: Dict[str, Any]
    ) -> Tuple[ChangeSeverity, str]:
        """Determine severity of changes (SST-style: purely derived from report)."""
        total_changes = summary.get('total_changes', len(changed_keys))
        output_changes_count = summary.get('output_changes', len(output_changes.get('keys', [])))
        metric_deltas_count = summary.get('metric_deltas_significant', summary.get('metric_deltas_count', len(metric_deltas)))
        input_changes_count = summary.get('input_changes', len(input_changes.get('keys', [])))
        process_changes_count = summary.get('process_changes', len(process_changes.get('keys', [])))
        has_excluded_factors = summary.get('excluded_factors_changed', bool(excluded_factors_changed))
        output_digest_changes = summary.get('output_digest_changes', [])

        impact_label = summary.get('impact_label', 'none')
        has_reproducibility_variance = bool(output_digest_changes)

        # If no changes at all, severity must be NONE
        if total_changes == 0 and metric_deltas_count == 0:
            if has_excluded_factors and excluded_factors_summary:
                return ChangeSeverity.MINOR, f"Only excluded factors changed: {excluded_factors_summary}"
            else:
                return ChangeSeverity.NONE, "No changes detected"

        # Critical: hard invariants
        critical_paths = [
            'inputs.data', 'inputs.target', 'inputs.features.feature_fingerprint',
            'process.split', 'process.leakage'
        ]

        for key in changed_keys:
            for critical in critical_paths:
                if key.startswith(critical):
                    return ChangeSeverity.CRITICAL, f"Critical change detected in {key}"

        # Handle reproducibility variance
        if has_reproducibility_variance:
            if impact_label in ['none', 'noise']:
                return ChangeSeverity.MAJOR, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (z-score analysis indicates noise-level changes only)."
                )
            elif impact_label == 'minor':
                return ChangeSeverity.MAJOR, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (minor changes detected)."
                )
            else:
                return ChangeSeverity.CRITICAL, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (significant changes detected)."
                )

        # Major paths
        major_paths = ['inputs.config', 'process.training', 'process.environment']

        if output_changes_count > 0 or metric_deltas_count > 0:
            if metric_deltas_count > 0:
                if impact_label == 'major':
                    return ChangeSeverity.MAJOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label}."
                    )
                elif impact_label == 'minor':
                    return ChangeSeverity.MINOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label}."
                    )
                else:
                    return ChangeSeverity.MINOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label} (noise-level)."
                    )
            else:
                return ChangeSeverity.MAJOR, f"Output changes detected: {output_changes_count} output changes"

        for key in changed_keys:
            for major in major_paths:
                if key.startswith(major):
                    return ChangeSeverity.MAJOR, f"Major config change detected in {key}"

        if has_excluded_factors and excluded_factors_summary and total_changes == 0:
            return ChangeSeverity.MINOR, f"Only excluded factors changed: {excluded_factors_summary}"

        if changed_keys and all(key.startswith('outputs.metrics') for key in changed_keys) and metric_deltas_count == 0:
            return ChangeSeverity.MINOR, f"Only metric metadata changed (no actual metric deltas): {len(changed_keys)} keys"

        if (input_changes_count > 0 or process_changes_count > 0) and output_changes_count == 0 and metric_deltas_count == 0:
            return ChangeSeverity.MINOR, f"Only input/process changes: {input_changes_count} input, {process_changes_count} process"

        if changed_keys:
            return ChangeSeverity.MAJOR, f"Mixed changes detected: {total_changes} total changes across inputs/process/outputs"

        return ChangeSeverity.NONE, "No changes detected (fallback)"
