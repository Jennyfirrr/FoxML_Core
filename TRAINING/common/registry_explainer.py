# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry Patch Explanation System

Provides query/explanation capabilities for registry patches.
Aggregates evidence from patches and audit logs to explain why features were excluded.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json

from TRAINING.common.registry_patch_naming import find_patch_file

logger = logging.getLogger(__name__)


def explain_feature_exclusion(
    feature_name: str,
    target: str,
    horizon_bars: int,
    registry_overlay_dir: Optional[Path] = None,
    persistent_override_dir: Optional[Path] = None,
    persistent_unblock_dir: Optional[Path] = None,
    audit_log_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Explain why a feature is excluded for a target/horizon.
    
    Returns the first layer that causes exclusion under precedence, plus full layer stack.
    
    Args:
        feature_name: Name of the feature
        target: Target column name
        horizon_bars: Target horizon in bars
        registry_overlay_dir: Optional directory containing run patches
        persistent_override_dir: Optional directory containing persistent overrides
        persistent_unblock_dir: Optional directory containing unblock patches
        audit_log_path: Optional path to audit log for evidence
    
    Returns:
        Dict with:
        - excluded: bool
        - reason: str (first layer that excludes)
        - layer_stack: List[Dict] (full precedence chain)
        - evidence: Optional[Dict] (from audit log if available)
    """
    from TRAINING.common.feature_registry import FeatureRegistry
    
    # Load registry with all patches
    repo_root = Path(__file__).resolve().parents[2]
    if persistent_override_dir is None:
        persistent_override_dir = repo_root / "CONFIG" / "data" / "feature_registry_per_target"
    if persistent_unblock_dir is None:
        persistent_unblock_dir = persistent_override_dir
    
    # Get bar_minutes from patches (for compatibility)
    bar_minutes = None
    if registry_overlay_dir:
        patch_file = find_patch_file(registry_overlay_dir, target)
        if patch_file and patch_file.exists():
            with open(patch_file, 'r') as f:
                patch_data = yaml.safe_load(f) or {}
                bar_minutes = patch_data.get('bar_minutes')
    
    registry = FeatureRegistry(
        target_column=target,
        registry_overlay_dir=registry_overlay_dir,
        current_bar_minutes=bar_minutes
    )
    
    # Check if allowed (uses two-phase check)
    is_allowed_result = registry.is_allowed(feature_name, horizon_bars, target_column=target)
    
    if is_allowed_result:
        return {
            'excluded': False,
            'reason': 'Feature is allowed',
            'layer_stack': [],
            'evidence': None
        }
    
    # Build layer stack (in precedence order)
    layer_stack = []
    
    # Phase A: Base eligibility
    if feature_name in registry.features:
        if registry.features[feature_name].get('rejected', False):
            layer_stack.append({
                'layer': 'base_registry',
                'type': 'global_rejected',
                'reason': 'Feature marked as rejected in base registry (structural leak)'
            })
            return {
                'excluded': True,
                'reason': 'Global rejected in base registry',
                'layer_stack': layer_stack,
                'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
            }
        
        allowed_horizons = registry.features[feature_name].get('allowed_horizons', [])
        if not allowed_horizons or horizon_bars not in allowed_horizons:
            layer_stack.append({
                'layer': 'base_registry',
                'type': 'not_in_allowed_horizons',
                'reason': f'Feature not in allowed_horizons for horizon {horizon_bars}',
                'allowed_horizons': allowed_horizons
            })
            return {
                'excluded': True,
                'reason': 'Not in base registry allowed_horizons',
                'layer_stack': layer_stack,
                'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
            }
    
    # Phase B: Overlays
    # Check unblocks first (highest priority allow)
    if registry.per_target_unblocks:
        unblock_features = registry.per_target_unblocks.get('features', {})
        if feature_name in unblock_features:
            unblocked = unblock_features[feature_name].get('unblocked_horizons_bars', [])
            if horizon_bars in unblocked:
                layer_stack.append({
                    'layer': 'unblock',
                    'type': 'unblocked',
                    'reason': f'Feature unblocked for horizon {horizon_bars}'
                })
                return {
                    'excluded': False,
                    'reason': 'Unblocked (cancels overlay denies)',
                    'layer_stack': layer_stack,
                    'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
                }
    
    # Check run patch excludes
    if registry.per_target_patches:
        patch_features = registry.per_target_patches.get('features', {})
        if feature_name in patch_features:
            excluded = patch_features[feature_name].get('excluded_horizons_bars', [])
            if horizon_bars in excluded:
                layer_stack.append({
                    'layer': 'run_patch',
                    'type': 'excluded',
                    'reason': f'Excluded by run patch for horizon {horizon_bars}'
                })
                return {
                    'excluded': True,
                    'reason': 'Excluded by run patch (highest priority deny)',
                    'layer_stack': layer_stack,
                    'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
                }
    
    # Check persistent override excludes
    if registry.per_target_overrides:
        override_features = registry.per_target_overrides.get('features', {})
        if feature_name in override_features:
            excluded = override_features[feature_name].get('excluded_horizons_bars', [])
            if horizon_bars in excluded:
                layer_stack.append({
                    'layer': 'persistent_override',
                    'type': 'excluded',
                    'reason': f'Excluded by persistent override for horizon {horizon_bars}'
                })
                return {
                    'excluded': True,
                    'reason': 'Excluded by persistent override (medium priority deny)',
                    'layer_stack': layer_stack,
                    'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
                }
    
    # Should not reach here if is_allowed() returned False
    return {
        'excluded': True,
        'reason': 'Unknown exclusion reason',
        'layer_stack': layer_stack,
        'evidence': _get_audit_evidence(feature_name, target, horizon_bars, audit_log_path)
    }


def _get_audit_evidence(
    feature_name: str,
    target: str,
    horizon_bars: int,
    audit_log_path: Optional[Path]
) -> Optional[Dict[str, Any]]:
    """
    Get evidence from audit log for a feature exclusion.
    
    Args:
        feature_name: Feature name
        target: Target column name
        horizon_bars: Horizon in bars
        audit_log_path: Optional path to audit log
    
    Returns:
        Most recent audit entry matching criteria, or None
    """
    if audit_log_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        audit_log_path = repo_root / "RESULTS" / "audit" / "registry_patch_ops.jsonl"
    
    if not audit_log_path.exists():
        return None
    
    # Read audit log (reverse order to get most recent first)
    matching_entries = []
    try:
        with open(audit_log_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):  # Most recent first
                try:
                    entry = json.loads(line.strip())
                    if (entry.get('target') == target and
                        entry.get('action') == 'exclude' and
                        entry.get('feature') == feature_name and
                        entry.get('horizon_bars') == horizon_bars):
                        matching_entries.append(entry)
                except Exception:
                    continue
        
        if matching_entries:
            return matching_entries[0]  # Most recent
    except Exception as e:
        logger.debug(f"Failed to read audit log: {e}")
    
    return None


def query_excluded_features(
    target: str,
    registry_overlay_dir: Optional[Path] = None,
    persistent_override_dir: Optional[Path] = None,
    audit_log_path: Optional[Path] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Query all excluded features for a target.
    
    Args:
        target: Target column name
        registry_overlay_dir: Optional directory containing run patches
        persistent_override_dir: Optional directory containing persistent overrides
        audit_log_path: Optional path to audit log
    
    Returns:
        Dict mapping feature_name -> List of exclusion explanations (one per horizon)
    """
    from TRAINING.common.feature_registry import FeatureRegistry
    
    repo_root = Path(__file__).resolve().parents[2]
    if persistent_override_dir is None:
        persistent_override_dir = repo_root / "CONFIG" / "data" / "feature_registry_per_target"
    
    # Get bar_minutes and horizon_bars from patches
    bar_minutes = None
    horizon_bars = None
    
    if registry_overlay_dir:
        patch_file = find_patch_file(registry_overlay_dir, target)
        if patch_file and patch_file.exists():
            with open(patch_file, 'r') as f:
                patch_data = yaml.safe_load(f) or {}
                bar_minutes = patch_data.get('bar_minutes')
                horizon_bars = patch_data.get('horizon_bars')
    
    registry = FeatureRegistry(
        target_column=target,
        registry_overlay_dir=registry_overlay_dir,
        current_bar_minutes=bar_minutes
    )
    
    # Collect all excluded features and horizons
    excluded_features = {}
    
    # Check run patches
    if registry.per_target_patches:
        patch_features = registry.per_target_patches.get('features', {})
        for feat_name, feat_data in patch_features.items():
            excluded_horizons = feat_data.get('excluded_horizons_bars', [])
            if excluded_horizons:
                if feat_name not in excluded_features:
                    excluded_features[feat_name] = []
                for h in excluded_horizons:
                    excluded_features[feat_name].append(h)
    
    # Check persistent overrides
    if registry.per_target_overrides:
        override_features = registry.per_target_overrides.get('features', {})
        for feat_name, feat_data in override_features.items():
            excluded_horizons = feat_data.get('excluded_horizons_bars', [])
            if excluded_horizons:
                if feat_name not in excluded_features:
                    excluded_features[feat_name] = []
                for h in excluded_horizons:
                    if h not in excluded_features[feat_name]:
                        excluded_features[feat_name].append(h)
    
    # Generate explanations for each feature/horizon
    results = {}
    for feat_name, horizons in excluded_features.items():
        results[feat_name] = []
        for h in sorted(set(horizons)):
            explanation = explain_feature_exclusion(
                feat_name, target, h,
                registry_overlay_dir=registry_overlay_dir,
                persistent_override_dir=persistent_override_dir,
                audit_log_path=audit_log_path
            )
            results[feat_name].append({
                'horizon_bars': h,
                **explanation
            })
    
    return results
