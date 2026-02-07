# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry AutoPatch System

Deterministic, auditable patch suggestion system for registry metadata fixes.
Generates patch overlay files without mutating canonical YAML.

Key Properties:
- Single-writer (orchestrator only)
- Atomic writes (tmp + os.replace)
- Deterministic ordering (sorted feature keys, sorted fields)
- SST-compliant (single canonical patch per run)
"""

import os
import yaml
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict

from TRAINING.common.utils.determinism_ordering import sorted_items, sorted_keys

logger = logging.getLogger(__name__)

# Tool version for provenance tracking (keep stable, single source of truth)
# This version should only change when the patch format or provenance schema changes
TOOL_VERSION = "registry_autopatch_v1"


class RegistryAutopatch:
    """
    Collects and writes deterministic patch suggestions for registry metadata.
    
    Patch suggestions are aggregated across all targets/stages/views and written
    once at end-of-run by orchestrator.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        write: bool = True,
        apply: bool = False,
        allow_overwrite: bool = False
    ):
        """
        Initialize autopatch collector.
        
        Args:
            enabled: Master switch (default: False)
            write: Generate patch suggestions in run dir (default: True)
            apply: Apply workspace overlay on this run (default: False)
            allow_overwrite: Allow overwriting explicit non-null values (default: False)
        """
        self.enabled = enabled
        self.write = write
        self.apply = apply
        self.allow_overwrite = allow_overwrite
        
        # Collect suggestions: feature_name -> {field -> (value, reason, source)}
        self._suggestions: Dict[str, Dict[str, Tuple[Any, str, str]]] = defaultdict(dict)
        
        # Conflict tracking: list of conflict dicts with full provenance
        self._conflicts: List[Dict[str, Any]] = []
        
        # Tombstone set: (feature_name, field) tuples that are conflicted (no further suggestions accepted)
        self._conflicted_fields: Set[Tuple[str, str]] = set()
    
    def suggest_patch(
        self,
        feature_name: str,
        field: str,
        value: Any,
        reason: str,
        source: str
    ) -> None:
        """
        Suggest a patch for a feature field.
        
        Args:
            feature_name: Feature name
            field: Field to patch (e.g., 'allowed_horizons', 'lag_bars')
            value: Suggested value
            reason: Reason for patch (stable string, no timestamps)
            source: Source of suggestion (e.g., 'family_match', 'pattern_parse')
        
        Policy:
        - Only patches inheritable fields (null/missing)
        - Never overwrites explicit non-null values unless allow_overwrite=True
        - Conflicting values for same (feature, field) → field omitted from patch, conflict recorded
        - Same value, different evidence → merge evidence deterministically (lexicographic smallest)
        """
        if not self.enabled:
            return
        
        # Check tombstone first: if field already conflicted, ignore all further suggestions
        if (feature_name, field) in self._conflicted_fields:
            logger.debug(
                f"Registry autopatch: ignoring suggestion for {feature_name}.{field} "
                f"(already conflicted, tombstoned)"
            )
            return
        
        # Check if field already has a suggestion
        if feature_name in self._suggestions and field in self._suggestions[feature_name]:
            existing_value, existing_reason, existing_source = self._suggestions[feature_name][field]
            
            # Compare values using canonical representation
            canon_existing = self._canonicalize_value(existing_value)
            canon_new = self._canonicalize_value(value)
            
            if canon_existing != canon_new:
                # Conflicting values → record conflict, tombstone field, remove from suggestions
                conflict = {
                    'feature': feature_name,
                    'field': field,
                    'existing_value': existing_value,
                    'existing_value_repr': self._canonicalize_value_to_str(existing_value),
                    'existing_reason': existing_reason,
                    'existing_source': existing_source,
                    'conflicting_value': value,
                    'conflicting_value_repr': self._canonicalize_value_to_str(value),
                    'conflicting_reason': reason,
                    'conflicting_source': source
                }
                self._conflicts.append(conflict)
                self._conflicted_fields.add((feature_name, field))
                
                # Remove field from suggestions (no patch emitted)
                del self._suggestions[feature_name][field]
                
                logger.warning(
                    f"Registry autopatch conflict: {feature_name}.{field} has conflicting values. "
                    f"Existing: {existing_value} from {existing_source} ({existing_reason}). "
                    f"Conflicting: {value} from {source} ({reason}). "
                    f"Field omitted from patch."
                )
                return
            
            # Same value (canonical) but different evidence → merge deterministically
            # Keep lexicographically smallest (source, reason) tuple
            existing_evidence = (existing_source, existing_reason)
            new_evidence = (source, reason)
            
            if new_evidence < existing_evidence:
                # New evidence is lexicographically smaller → replace
                self._suggestions[feature_name][field] = (value, reason, source)
                logger.debug(
                    f"Registry autopatch: updated evidence for {feature_name}.{field} "
                    f"(kept lexicographically smallest: {new_evidence})"
                )
            else:
                # Existing evidence is smaller or equal → keep existing
                logger.debug(
                    f"Registry autopatch: keeping existing evidence for {feature_name}.{field} "
                    f"(lexicographically smallest: {existing_evidence})"
                )
            return
        
        # Field doesn't exist → store suggestion
        self._suggestions[feature_name][field] = (value, reason, source)
        logger.debug(
            f"Registry autopatch suggestion: {feature_name}.{field} = {value} "
            f"(reason: {reason}, source: {source})"
        )
    
    def merge_suggestions(
        self,
        suggestions_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple suggestion dictionaries deterministically.
        
        Args:
            suggestions_list: List of suggestion dicts (each with feature_name -> {field -> (value, reason, source)})
        
        Returns:
            Dict with:
                - 'suggestions': Merged suggestions dict with sorted feature keys and sorted fields
                - 'conflicts': List of conflict dicts (sorted deterministically)
        
        Determinism:
        - Features sorted alphabetically
        - Fields within feature sorted alphabetically
        - Conflicting values for same (feature, field) → field omitted, conflict recorded
        - Same value, different evidence → merge deterministically (lexicographic smallest)
        """
        merged = defaultdict(dict)
        conflicts: List[Dict[str, Any]] = []
        conflicted_fields: Set[Tuple[str, str]] = set()
        
        # Merge all suggestions with conflict detection
        for suggestions in suggestions_list:
            # Use sorted_items for deterministic iteration order
            for feature_name, fields in sorted_items(suggestions):
                # Use sorted_keys for deterministic field iteration
                for field in sorted_keys(fields):
                    value, reason, source = fields[field]
                    # Check tombstone
                    if (feature_name, field) in conflicted_fields:
                        continue
                    
                    # Check if field already exists
                    if feature_name in merged and field in merged[feature_name]:
                        existing_value, existing_reason, existing_source = merged[feature_name][field]
                        
                        # Compare values using canonical representation
                        canon_existing = self._canonicalize_value(existing_value)
                        canon_new = self._canonicalize_value(value)
                        
                        if canon_existing != canon_new:
                            # Conflicting values → record conflict, tombstone field, omit from output
                            conflict = {
                                'feature': feature_name,
                                'field': field,
                                'existing_value_repr': self._canonicalize_value_to_str(existing_value),
                                'existing_reason': existing_reason,
                                'existing_source': existing_source,
                                'conflicting_value_repr': self._canonicalize_value_to_str(value),
                                'conflicting_reason': reason,
                                'conflicting_source': source
                            }
                            conflicts.append(conflict)
                            conflicted_fields.add((feature_name, field))
                            # Remove from merged (omit field)
                            del merged[feature_name][field]
                            continue
                        
                        # Same value (canonical) but different evidence → merge deterministically
                        existing_evidence = (existing_source, existing_reason)
                        new_evidence = (source, reason)
                        
                        if new_evidence < existing_evidence:
                            merged[feature_name][field] = (value, reason, source)
                        # else: keep existing (already in merged)
                    else:
                        # Field doesn't exist → store
                        merged[feature_name][field] = (value, reason, source)
        
        # Sort features and fields for deterministic output
        sorted_merged = {}
        for feature_name in sorted(merged.keys()):
            sorted_fields = {}
            for field in sorted(merged[feature_name].keys()):
                sorted_fields[field] = merged[feature_name][field]
            sorted_merged[feature_name] = sorted_fields
        
        # Sort conflicts deterministically for return
        sorted_conflicts = sorted(
            conflicts,
            key=lambda c: (
                c['feature'],
                c['field'],
                c['existing_value_repr'],
                c['conflicting_value_repr'],
                c['existing_source'],
                c['conflicting_source'],
                c['existing_reason'],
                c['conflicting_reason']
            )
        )
        
        # Return merged suggestions and conflicts
        return {
            'suggestions': sorted_merged,
            'conflicts': sorted_conflicts
        }
    
    def write_patch_file(
        self,
        run_root: Path,
        suggestions: Optional[Dict[str, Dict[str, Tuple[Any, str, str]]]] = None,  # NEW: explicit suggestions
        target_column: Optional[str] = None,  # NEW: for per-target patches
        allow_overwrite: Optional[bool] = None  # Keep for backward compatibility
    ) -> Optional[Path]:
        """
        Write patch suggestion file to run directory.
        
        Args:
            run_root: Run root directory (writes to {run_root}/globals/registry_autopatch/)
            suggestions: Explicit suggestions dict (if None, uses self._suggestions for backward compatibility)
            target_column: Optional target column name for per-target patches
            allow_overwrite: Override instance allow_overwrite flag (if None, uses self.allow_overwrite)
        
        Returns:
            Path to written patch file, or None if no suggestions or disabled
        
        Determinism:
        - Features sorted alphabetically
        - Fields within feature sorted alphabetically
        - Stable YAML emission (no timestamps)
        - Atomic write (tmp + os.replace)
        """
        if not self.enabled or not self.write:
            return None
        
        # Use provided suggestions or fall back to internal (for backward compatibility)
        suggestions_to_write = suggestions if suggestions is not None else self._suggestions
        
        if not suggestions_to_write:
            logger.debug("No patch suggestions to write")
            return None
        
        allow_overwrite_flag = allow_overwrite if allow_overwrite is not None else self.allow_overwrite
        
        # Create output directory
        output_dir = run_root / "globals" / "registry_autopatch"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output path using stem-only naming
        from TRAINING.common.registry_patch_naming import safe_target_filename_stem
        if target_column:
            target_stem = safe_target_filename_stem(target_column)
            patch_file = output_dir / f"feature_registry_overrides.{target_stem}.suggested.yaml"
        else:
            patch_file = output_dir / "feature_registry_overrides.suggested.yaml"
        
        # Build patch structure with deterministic ordering
        from TRAINING.common.utils.determinism_ordering import sorted_items, sorted_keys
        feature_overrides = {}
        for feature_name, fields in sorted_items(suggestions_to_write):
            feature_patch = {}
            
            for field in sorted_keys(fields):
                value, reason, source = fields[field]
                # Skip provenance fields in output (they're metadata only)
                if field.startswith('_'):
                    continue
                feature_patch[field] = value
                # Add provenance fields
                feature_patch['_autofix_reason'] = reason
                feature_patch['_autofix_source'] = source
            
            feature_overrides[feature_name] = feature_patch
        
        # Sort conflicts deterministically for output
        sorted_conflicts = sorted(
            self._conflicts,
            key=lambda c: (
                c['feature'],
                c['field'],
                c['existing_value_repr'],
                c['conflicting_value_repr'],
                c['existing_source'],
                c['conflicting_source'],
                c['existing_reason'],
                c['conflicting_reason']
            )
        )
        
        # Build full patch document
        patch_data = {
            'feature_overrides': feature_overrides,
            '_metadata': {
                'generated_by': 'registry_autopatch.py',
                'note': 'DO NOT EDIT MANUALLY - regenerated on each run when enabled',
                'n_conflicts': len(sorted_conflicts),
                'conflicts': sorted_conflicts if sorted_conflicts else None
            }
        }
        
        # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
        # Note: YAML determinism policy - if patch YAML is hashed/diffed, consider switching to JSON
        # or locking down YAML dump options + pinning library version
        try:
            from TRAINING.common.utils.file_utils import write_atomic_yaml
            write_atomic_yaml(patch_file, patch_data)
            
            logger.info(
                f"Registry autopatch: Wrote {len(feature_overrides)} feature patches to {patch_file}"
            )
            
            # Also write summary JSON
            # CRITICAL: Use atomic write for crash safety (fsync + dir fsync)
            # Note: If summary.json is hashed/diffed for determinism, must use canonical_json()
            from TRAINING.common.utils.file_utils import write_atomic_json
            summary_file = output_dir / "summary.json"
            summary = {
                'n_features': len(feature_overrides),
                'n_patches': sum(len(feature_overrides[feature_name]) for feature_name in sorted_keys(feature_overrides)),
                'n_conflicts': len(sorted_conflicts),
                'top_reasons': self._get_top_reasons(),
                'conflicts_sample': sorted_conflicts[:10] if sorted_conflicts else []
            }
            
            write_atomic_json(summary_file, summary)
            
            # Write decision log (applied/rejected/skipped) with hashes
            # Note: "applied" here means "written to suggestion file", not "promoted to overlay"
            # Full decision tracking (with overlay comparison) happens in promote_patch()
            decisions = {
                "applied": {},  # Suggestions written to patch file
                "rejected": {},  # Conflicts (from self._conflicts)
                "skipped": {}  # Empty for now (would need registry comparison to determine)
            }
            
            # Build applied decisions from suggestions
            for feature_name, fields in sorted_items(suggestions_to_write):
                for field in sorted_keys(fields):
                    value, reason, source = fields[field]
                    if field.startswith('_'):
                        continue
                    if feature_name not in decisions["applied"]:
                        decisions["applied"][feature_name] = {}
                    decisions["applied"][feature_name][field] = {
                        "value": value,
                        "reason": reason,
                        "source": source
                    }
            
            # Build rejected decisions from conflicts
            for conflict in sorted_conflicts:
                feature_name = conflict['feature']
                field = conflict['field']
                if feature_name not in decisions["rejected"]:
                    decisions["rejected"][feature_name] = {}
                decisions["rejected"][feature_name][field] = {
                    "suggested": conflict.get('conflicting_value'),
                    "existing": conflict.get('existing_value'),
                    "reason": "conflict",
                    "conflict_with": conflict.get('existing_source', 'unknown')
                }
            
            # Compute decision hashes if there are any decisions
            if decisions.get("applied") or decisions.get("rejected") or decisions.get("skipped"):
                from TRAINING.common.utils.registry_overlay_fingerprint import hash_decisions
                decision_hashes = {
                    "applied_hash": hash_decisions({"applied": decisions.get("applied", {})}) if decisions.get("applied") else "",
                    "rejected_hash": hash_decisions({"rejected": decisions.get("rejected", {})}) if decisions.get("rejected") else "",
                    "skipped_hash": hash_decisions({"skipped": decisions.get("skipped", {})}) if decisions.get("skipped") else ""
                }
                
                decision_log = {
                    **decisions,
                    "decision_hashes": decision_hashes
                }
                
                # Write decision log
                from TRAINING.common.registry_patch_naming import safe_target_filename_stem
                target_stem = safe_target_filename_stem(target_column) if target_column else "global"
                decision_file = output_dir / f"decisions.{target_stem}.json"
                write_atomic_json(decision_file, decision_log)
                logger.info(f"Wrote decision log to {decision_file}")
            
            # If apply=True, automatically promote to workspace overlay
            if self.apply:
                try:
                    # Get registry instance for promote_patch (needed for hash computation)
                    from TRAINING.common.feature_registry import get_registry
                    registry = get_registry(target_column=target_column) if target_column else get_registry()
                    self.promote_patch(run_root, target_column=target_column, registry=registry)
                    logger.info(f"Auto-applied registry patches (target={target_column or 'global'})")
                except Exception as e:
                    logger.error(f"Failed to auto-apply registry patches: {e}")
                    # Don't fail the run - patches are still written for review
            
            return patch_file
        except Exception as e:
            logger.error(f"Failed to write patch file {patch_file}: {e}")
            # Cleanup: atomic write helpers handle temp file cleanup, but be defensive
            temp_file = patch_file.with_suffix('.yaml.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            return None
    
    def _get_top_reasons(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top reasons for patches (for summary)."""
        reason_counts = defaultdict(int)
        # Use sorted_keys for deterministic iteration order
        for feature_name in sorted_keys(self._suggestions):
            fields = self._suggestions[feature_name]
            for field in sorted_keys(fields):
                _, reason, _ = fields[field]
                reason_counts[reason] += 1
        
        # Sort by count (descending), then by reason (alphabetical) for ties
        sorted_reasons = sorted(
            reason_counts.items(),
            key=lambda x: (-x[1], x[0])
        )[:limit]
        
        return [{'reason': reason, 'count': count} for reason, count in sorted_reasons]
    
    def promote_patch(
        self,
        from_run_root: Path,
        to_overlay_path: Optional[Path] = None,  # Made optional
        target_column: Optional[str] = None,  # NEW: for per-target promotion
        registry: Optional[Any] = None  # NEW: for computing hashes at promotion time
    ) -> bool:
        """
        Promote patch suggestion to workspace overlay.
        
        Merges with existing overlay content (deterministic merge) to avoid overwriting
        patches from previous runs or manual edits.
        
        If target_column is provided, promotes to per-target overlay.
        Otherwise promotes to global overlay (existing behavior).
        
        Args:
            from_run_root: Run root directory containing suggestion file
            to_overlay_path: Optional path to workspace overlay file (if None, auto-determines)
            target_column: Optional target column name for per-target promotion
            registry: Optional FeatureRegistry instance for computing hashes at promotion time
        
        Returns:
            True if promotion successful, False otherwise
        """
        from TRAINING.common.registry_patch_naming import safe_target_filename_stem
        from TRAINING.common.utils.registry_overlay_fingerprint import normalize_overrides, hash_overrides
        
        # Determine suggestion file path (using stem-only naming)
        # Initialize target_stem for decision log path
        target_stem = None
        if target_column:
            target_stem = safe_target_filename_stem(target_column)
            suggestion_file = from_run_root / "globals" / "registry_autopatch" / f"feature_registry_overrides.{target_stem}.suggested.yaml"
        else:
            suggestion_file = from_run_root / "globals" / "registry_autopatch" / "feature_registry_overrides.suggested.yaml"
        
        if not suggestion_file.exists():
            logger.warning(f"Cannot promote: suggestion file not found: {suggestion_file}")
            return False
        
        # Determine overlay path (using stem-only naming, with interval-specific support)
        if to_overlay_path is None:
            from TRAINING.common.utils.horizon_conversion import is_effectively_integer
            
            repo_root = Path(__file__).resolve().parents[3]  # Go up to repo root
            overlay_dir = repo_root / "CONFIG" / "data" / "overrides"
            
            # Get current_bar_minutes from registry or config
            # CRITICAL: Normalize to float|int|None before is_effectively_integer() (fail-closed for non-numeric)
            current_bar_minutes = None
            if registry:
                current_bar_minutes = getattr(registry, 'current_bar_minutes', None)
            if current_bar_minutes is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    raw_interval = get_cfg('data.bar_interval', default=None)
                    if raw_interval is not None:
                        try:
                            current_bar_minutes = float(raw_interval)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid bar_interval config: {raw_interval}. Treating as None.")
                            current_bar_minutes = None
                except Exception:
                    pass
            
            # Normalize if not already numeric (defensive, should already be normalized from registry)
            if current_bar_minutes is not None:
                try:
                    current_bar_minutes = float(current_bar_minutes)
                except (ValueError, TypeError):
                    current_bar_minutes = None
            
            interval_int = is_effectively_integer(current_bar_minutes)
            
            if target_column:
                if target_stem is None:  # Redundant but defensive
                    target_stem = safe_target_filename_stem(target_column)
                # Include interval in path if effectively integer
                if interval_int is not None:
                    to_overlay_path = overlay_dir / f"feature_registry_overrides.auto.{target_stem}.{interval_int}m.yaml"
                else:
                    to_overlay_path = overlay_dir / f"feature_registry_overrides.auto.{target_stem}.yaml"
            else:
                # Global overlay
                if interval_int is not None:
                    to_overlay_path = overlay_dir / f"feature_registry_overrides.auto.{interval_int}m.yaml"
                else:
                    to_overlay_path = overlay_dir / "feature_registry_overrides.auto.yaml"
        
        try:
            # Load suggestion file
            with open(suggestion_file, 'r') as f:
                suggestion_data = yaml.safe_load(f) or {}
            
            new_overrides = suggestion_data.get('feature_overrides', {})
            
            # Load existing overlay if it exists (merge, don't overwrite)
            existing_overrides = {}
            if to_overlay_path.exists():
                try:
                    with open(to_overlay_path, 'r') as f:
                        existing_data = yaml.safe_load(f) or {}
                    existing_overrides = existing_data.get('feature_overrides', {})
                    logger.debug(f"Loaded {len(existing_overrides)} existing feature overrides from {to_overlay_path}")
                except Exception as e:
                    logger.warning(f"Failed to load existing overlay for merge: {e}. Will overwrite.")
            
            # Merge deterministically (new suggestions take precedence for same feature+field)
            # DETERMINISM: Use sorted iteration for deterministic merge order
            # Track decisions: applied, rejected, skipped
            decisions = {
                "applied": {},
                "rejected": {},
                "skipped": {}
            }
            
            merged_overrides = dict(existing_overrides)
            for feature_name, feature_patch in sorted_items(new_overrides):
                if feature_name not in merged_overrides:
                    merged_overrides[feature_name] = {}
                
                # Extract provenance fields before filtering (needed for decision tracking)
                reason = feature_patch.get('_autofix_reason', 'unknown')
                source = feature_patch.get('_autofix_source', 'unknown')
                
                # Merge fields (new takes precedence, but preserve existing fields not in new patch)
                # DETERMINISM: Use sorted iteration for deterministic merge order
                for field, value in sorted_items(feature_patch):
                    # Skip provenance fields in overlay (they're metadata only)
                    if field.startswith('_autofix_') or field.startswith('_inference_') or field.startswith('_registry_') or field.startswith('_tool_') or field.startswith('_run_'):
                        continue
                    
                    # Check if field already exists in existing overlay
                    existing_value = None
                    if feature_name in existing_overrides and field in existing_overrides[feature_name]:
                        existing_value = existing_overrides[feature_name][field]
                    
                    # Determine decision category
                    if existing_value is not None:
                        # Compare values using canonical representation
                        from TRAINING.common.utils.config_hashing import canonical_json
                        canon_existing = canonical_json(existing_value)
                        canon_new = canonical_json(value)
                        
                        if canon_existing == canon_new:
                            # Already satisfied (no-op)
                            if feature_name not in decisions["skipped"]:
                                decisions["skipped"][feature_name] = {}
                            decisions["skipped"][feature_name][field] = {
                                "suggested": value,
                                "existing": existing_value,
                                "reason": "already_satisfied"
                            }
                            # Don't overwrite (already correct)
                            continue
                        else:
                            # Different value - check if we should overwrite
                            if not self.allow_overwrite:
                                # Rejected: would overwrite but allow_overwrite=False
                                if feature_name not in decisions["rejected"]:
                                    decisions["rejected"][feature_name] = {}
                                decisions["rejected"][feature_name][field] = {
                                    "suggested": value,
                                    "existing": existing_value,
                                    "reason": "would_overwrite_existing",
                                    "conflict_with": "existing_overlay"
                                }
                                # Don't overwrite
                                continue
                            else:
                                # Applied: overwriting with allow_overwrite=True
                                if feature_name not in decisions["applied"]:
                                    decisions["applied"][feature_name] = {}
                                decisions["applied"][feature_name][field] = {
                                    "value": value,
                                    "reason": reason,
                                    "source": source
                                }
                    else:
                        # Applied: new field (not in existing overlay)
                        if feature_name not in decisions["applied"]:
                            decisions["applied"][feature_name] = {}
                        decisions["applied"][feature_name][field] = {
                            "value": value,
                            "reason": reason,
                            "source": source
                        }
                    
                    # New suggestion takes precedence (deterministic: last write wins)
                    merged_overrides[feature_name][field] = value
            
            # CRITICAL: Normalize merged overrides before writing (semantic content only)
            normalized_merged = normalize_overrides(merged_overrides)
            normalized_new = normalize_overrides(new_overrides)
            
            # Compute hashes of normalized content (post-merge, post-filter)
            effective_hash = hash_overrides(normalized_merged)
            applied_hash = hash_overrides(normalized_new)
            
            # Compute registry/inference config hashes at promotion time (truth, not from suggestion)
            promotion_registry_hash = self._compute_registry_version_hash(registry) if registry else ''
            promotion_inference_hash = self._compute_inference_config_hash()
            
            # Also get from suggestion file (provenance, for audit)
            suggestion_metadata = suggestion_data.get('_metadata', {}) or suggestion_data.get('_provenance', {})
            suggestion_registry_hash = suggestion_metadata.get('registry_version_hash', '')
            suggestion_inference_hash = suggestion_metadata.get('inference_config_hash', '')
            
            # Build merged overlay structure with normalized content
            merged_data = {
                'feature_overrides': normalized_merged,  # Normalized semantic content only
                '_metadata': {
                    'generated_by': 'registry_autopatch.py',
                    'note': 'DO NOT EDIT MANUALLY - auto-generated overlay (merged from multiple runs)',
                    'overlay_content_hash': effective_hash,  # Hash of what's actually applied
                    'applied_suggestions_hash': applied_hash,  # Hash of what was added
                    'promotion_registry_version_hash': promotion_registry_hash,  # Computed at promotion (truth)
                    'promotion_inference_config_hash': promotion_inference_hash,  # Computed at promotion (truth)
                    'suggestion_registry_version_hash': suggestion_registry_hash,  # From suggestion (provenance)
                    'suggestion_inference_config_hash': suggestion_inference_hash,  # From suggestion (provenance)
                    'last_updated_run': str(from_run_root.name) if from_run_root else None,
                    'target': target_column if target_column else 'global',
                    'n_features': len(normalized_merged),
                    'n_new_features': len(normalized_new),
                    'n_existing_features': len(existing_overrides)
                }
            }
            
            # Create overlay directory
            to_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write overlay file atomically (use existing atomic write helpers for crash safety)
            from TRAINING.common.utils.file_utils import write_atomic_yaml
            write_atomic_yaml(to_overlay_path, merged_data)
            
            # Write decision log with hashes
            if decisions and (decisions.get("applied") or decisions.get("rejected") or decisions.get("skipped")):
                from TRAINING.common.utils.file_utils import write_atomic_json
                from TRAINING.common.utils.registry_overlay_fingerprint import hash_decisions
                
                # Compute decision hashes (using separate normalization for decisions)
                decision_hashes = {
                    "applied_hash": hash_decisions({"applied": decisions.get("applied", {})}) if decisions.get("applied") else "",
                    "rejected_hash": hash_decisions({"rejected": decisions.get("rejected", {})}) if decisions.get("rejected") else "",
                    "skipped_hash": hash_decisions({"skipped": decisions.get("skipped", {})}) if decisions.get("skipped") else ""
                }
                
                decision_log = {
                    **decisions,
                    "decision_hashes": decision_hashes
                }
                
                # Write decision log to run directory (not overlay directory)
                # Use target_stem from earlier in function (defined at line 517 or 531)
                decision_stem = target_stem if target_column else "global"
                decision_file = from_run_root / "globals" / "registry_autopatch" / f"decisions.{decision_stem}.json"
                decision_file.parent.mkdir(parents=True, exist_ok=True)
                write_atomic_json(decision_file, decision_log)
                logger.info(f"Wrote decision log to {decision_file}")
            
            n_new = len(normalized_new)
            n_existing = len(existing_overrides)
            n_merged = len(normalized_merged)
            logger.info(
                f"Promoted patch from {suggestion_file} to {to_overlay_path} "
                f"(merged: {n_existing} existing + {n_new} new = {n_merged} total features, "
                f"effective_hash={effective_hash[:16]}...)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to promote patch: {e}")
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
            return False
    
    def _canonicalize_value(self, v: Any) -> Any:
        """
        Canonicalize a value for deterministic comparison (recursive).
        
        Rules:
        - Lists of ints → sorted ints
        - Lists of strings → sorted strings
        - Lists of mixed/nested → recursively canonicalize each element, then sort
        - Dicts → recursively canonicalize values, then canonical JSON string
        - Scalars → as-is
        
        Args:
            v: Value to canonicalize
            
        Returns:
            Canonical representation of value
        """
        if isinstance(v, list):
            if not v:
                return []
            # Check if all ints
            if all(isinstance(x, int) for x in v):
                return sorted(v)
            # Check if all strings
            if all(isinstance(x, str) for x in v):
                return sorted(v)
            # Mixed/nested → recursively canonicalize each element, then sort
            canon_elements = [self._canonicalize_value(x) for x in v]
            # Sort by canonical string representation for deterministic order
            return sorted(canon_elements, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        elif isinstance(v, dict):
            # Recursively canonicalize values, then return canonical JSON string
            canon_dict = {k: self._canonicalize_value(val) for k, val in sorted(v.items())}
            return json.dumps(canon_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        else:
            # Scalars → as-is
            return v
    
    def _canonicalize_value_to_str(self, v: Any) -> str:
        """
        Convert value to canonical string representation for conflict records.
        
        Args:
            v: Value to convert
            
        Returns:
            Canonical string representation
        """
        canon = self._canonicalize_value(v)
        if isinstance(canon, str):
            return canon
        else:
            # For non-string canonical values (e.g., sorted lists), convert to JSON
            return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    
    def _compute_inference_config_hash(self) -> str:
        """
        Compute SHA256 hash of feature_inference.yaml config.
        
        Returns:
            64-character SHA256 hex digest
        """
        try:
            from TRAINING.common.feature_registry import _load_feature_inference_config
            from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
            inference_config = _load_feature_inference_config()
            if inference_config:
                # Use canonical JSON for deterministic hashing
                config_str = canonical_json(inference_config)
                return sha256_full(config_str)
            # Fallback: hash of "fallback" string if config not available
            return sha256_full('{"fallback": true}')
        except Exception as e:
            logger.debug(f"Failed to compute inference config hash: {e}")
            from TRAINING.common.utils.config_hashing import sha256_full
            return sha256_full('{"error": "config_unavailable"}')
    
    def _compute_registry_version_hash(self, registry: Any) -> str:
        """
        Compute SHA256 hash of feature_registry.yaml content.
        
        Args:
            registry: FeatureRegistry instance
            
        Returns:
            64-character SHA256 hex digest
        """
        try:
            if hasattr(registry, 'config_path') and registry.config_path and registry.config_path.exists():
                content = registry.config_path.read_bytes()
                return hashlib.sha256(content).hexdigest()
            # Fallback: hash of empty string if registry path not available
            return hashlib.sha256(b'').hexdigest()
        except Exception as e:
            logger.debug(f"Failed to compute registry version hash: {e}")
            return hashlib.sha256(b'{"error": "registry_unavailable"}').hexdigest()
    
    def _infer_metadata_with_pattern_id(self, registry: Any, feature_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Call auto_infer_metadata and track which pattern matched (for provenance).
        
        Args:
            registry: FeatureRegistry instance
            feature_name: Feature name to infer
            
        Returns:
            Tuple of (metadata_dict, pattern_id) where pattern_id is None if inference failed
        """
        try:
            # Try to load inference config and match patterns manually to get pattern ID
            from TRAINING.common.feature_registry import _load_feature_inference_config, _build_pattern_list_with_precedence
            import re
            
            inference_config = _load_feature_inference_config()
            if inference_config:
                patterns_list = _build_pattern_list_with_precedence(inference_config)
                
                # Match in precedence order (first match wins - same as auto_infer_metadata)
                for priority, pattern_str, pattern_config in patterns_list:
                    match = re.match(pattern_str, feature_name, re.I)
                    if match:
                        # Build metadata from pattern config (same logic as _build_metadata_from_pattern)
                        metadata = registry._build_metadata_from_pattern(feature_name, match, pattern_config)
                        if metadata:
                            # Construct pattern ID from group name and pattern
                            # Try to get group name from pattern_config
                            group_name = pattern_config.get('description', 'unknown')
                            pattern_id = f"pattern:{group_name}:{pattern_str}"
                            return metadata, pattern_id
            
            # Fallback to auto_infer_metadata (for hardcoded patterns)
            metadata = registry.auto_infer_metadata(feature_name)
            if metadata:
                # For fallback patterns, use generic pattern ID
                pattern_id = "pattern:fallback:hardcoded"
                return metadata, pattern_id
            
            return None, None
        except Exception as e:
            logger.debug(f"Failed to infer metadata with pattern ID for {feature_name}: {e}")
            return None, None
    
    def suggest_missing_features_from_breakdown(
        self,
        breakdown: Any,  # CoverageBreakdown
        registry: Any,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest patches for missing features that can be auto-inferred.
        
        Adds suggestions to internal _suggestions dict (via suggest_patch) so they're
        included in write_patch_file() output and can be applied via existing apply mechanism.
        
        Args:
            breakdown: CoverageBreakdown instance
            registry: FeatureRegistry instance
            run_id: Optional run ID for provenance
            
        Returns:
            Dict with keys:
                - suggestions: Dict[feature_name, metadata_dict] (sorted) - for reference/audit
                - stats: Dict with inferable_count, uninferable_count, etc.
        """
        if not self.enabled:
            return {"suggestions": {}, "stats": {}}
        
        suggestions = {}
        inferable_ids = []
        uninferable_ids = []
        
        # Compute provenance hashes (for metadata, not for suggest_patch calls)
        inference_config_hash = self._compute_inference_config_hash()
        registry_version_hash = self._compute_registry_version_hash(registry)
        
        # Defensive: Filter out non-feature columns before suggesting registry adds
        # This prevents autopatch from suggesting "register y_*" even if they slip through filtering
        from TRAINING.ranking.utils.column_exclusion import exclude_non_feature_columns
        missing_features_filtered, excluded = exclude_non_feature_columns(
            breakdown.missing_feature_ids_full,
            reason="autopatch-input-filter"
        )
        
        if excluded:
            logger.warning(
                f"Autopatch defensive filter: excluded {len(excluded)} non-feature columns from suggestions. "
                f"These should never be registered as features: {excluded[:10]}"
            )
        
        # Process each missing feature (use filtered list)
        for feature_name in missing_features_filtered:
            metadata, pattern_id = self._infer_metadata_with_pattern_id(registry, feature_name)
            
            if metadata:
                # Check if rejected (don't suggest rejected features)
                if metadata.get('rejected', False):
                    uninferable_ids.append(feature_name)
                    continue
                
                # Feature is inferable
                inferable_ids.append(feature_name)
                
                # Add each field to internal suggestions via suggest_patch (integrates with existing system)
                # This ensures suggestions are collected in _suggestions and will be written by write_patch_file()
                # DETERMINISM: Sort fields for deterministic suggestion order
                for field, value in sorted_items(metadata):
                    # Skip provenance fields (they're metadata, not registry fields)
                    if field.startswith('_'):
                        continue
                    
                    # Use suggest_patch to add to internal collection
                    reason = f"auto_inferred_from_pattern:{pattern_id or 'unknown'}"
                    source = "coverage_automation"
                    self.suggest_patch(feature_name, field, value, reason, source)
                
                # Build full suggestion dict for return (includes provenance for audit)
                suggestion = {
                    **metadata,
                    "_autofix_reason": "auto_inferred",
                    "_autofix_source": "pattern_match",
                    "_inference_pattern_id": pattern_id or "pattern:unknown",
                    "_inference_config_hash": inference_config_hash,
                    "_registry_version_hash": registry_version_hash,
                    "_tool_version": "registry_v1.0"
                }
                if run_id:
                    suggestion["_run_id"] = run_id
                
                suggestions[feature_name] = suggestion
            else:
                # Feature cannot be inferred
                uninferable_ids.append(feature_name)
        
        # Sort suggestions by feature name (deterministic)
        suggestions_sorted = {k: suggestions[k] for k in sorted_keys(suggestions)}
        
        stats = {
            "inferable_count": len(inferable_ids),
            "uninferable_count": len(uninferable_ids),
            "total_missing": len(breakdown.missing_feature_ids_full),
            "filtered_non_features": len(excluded) if excluded else 0
        }
        
        return {
            "suggestions": suggestions_sorted,
            "stats": stats
        }
    
    def suggest_horizon_compatibility_from_breakdown(
        self,
        breakdown: Any,  # CoverageBreakdown
        target: str,
        target_horizon: int,
        registry: Any,
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest patches for horizon compatibility issues.
        
        Adds suggestions to internal _suggestions dict (via suggest_patch) so they're
        included in write_patch_file() output and can be applied via existing apply mechanism.
        
        Args:
            breakdown: CoverageBreakdown instance
            target: Target name (e.g., "fwd_ret_10m")
            target_horizon: Target horizon in bars
            registry: FeatureRegistry instance
            run_id: Optional run ID for provenance
            
        Returns:
            Dict with keys:
                - suggestions: Dict[feature_name, metadata_dict] (sorted) - for reference/audit
                - stats: Dict with counts per reason
        """
        if not self.enabled:
            return {"suggestions": {}, "stats": {}}
        
        suggestions = {}
        stats = {
            "raw_allowed_horizons_none_count": len(breakdown.blocked_feature_ids_by_reason.get("raw_allowed_horizons_none", [])),
            "effective_horizon_missing_count": len(breakdown.blocked_feature_ids_by_reason.get("effective_horizon_missing", [])),
            "raw_explicit_disabled_count": len(breakdown.blocked_feature_ids_by_reason.get("raw_explicit_disabled", []))
        }
        
        # Compute provenance hashes (for metadata, not for suggest_patch calls)
        inference_config_hash = self._compute_inference_config_hash()
        registry_version_hash = self._compute_registry_version_hash(registry)
        
        # Handle raw_allowed_horizons_none: suggest family defaults
        for feature_name in breakdown.blocked_feature_ids_by_reason.get("raw_allowed_horizons_none", []):
            try:
                raw_metadata = registry.get_feature_metadata_raw(feature_name)
                # Check if family has defaults
                family_meta = registry._match_feature_family(feature_name) if hasattr(registry, '_match_feature_family') else None
                if family_meta and family_meta.get('default_allowed_horizons'):
                    default_horizons = family_meta['default_allowed_horizons']
                    family_name = family_meta.get('description', 'unknown')
                    
                    # Add to internal suggestions via suggest_patch
                    reason = f"inherit_family_default:{family_name}"
                    source = "coverage_automation"
                    self.suggest_patch(feature_name, "allowed_horizons", default_horizons, reason, source)
                    
                    # Build full suggestion dict for return (includes provenance for audit)
                    suggestion = {
                        "allowed_horizons": default_horizons,
                        "_autofix_reason": "inherit_family_default",
                        "_autofix_source": "coverage_breakdown",
                        "_inference_config_hash": inference_config_hash,
                        "_registry_version_hash": registry_version_hash,
                        "_tool_version": "registry_v1.0",
                        "_review_required": False,
                        "_family_name": family_name
                    }
                    if run_id:
                        suggestion["_run_id"] = run_id
                    
                    suggestions[feature_name] = suggestion
            except Exception as e:
                logger.debug(f"Failed to suggest family defaults for {feature_name}: {e}")
        
        # Handle effective_horizon_missing: suggest adding horizon
        for feature_name in breakdown.blocked_feature_ids_by_reason.get("effective_horizon_missing", []):
            try:
                effective_metadata = registry.get_feature_metadata_effective(feature_name, resolve_defaults=True)
                current_horizons = effective_metadata.get('allowed_horizons', [])
                
                if isinstance(current_horizons, list) and target_horizon not in current_horizons:
                    # Suggest adding target horizon
                    suggested_horizons = sorted(list(set(current_horizons + [target_horizon])))
                    
                    # Add to internal suggestions via suggest_patch
                    reason = f"horizon_missing:target={target}:horizon={target_horizon}"
                    source = "coverage_automation"
                    self.suggest_patch(feature_name, "allowed_horizons", suggested_horizons, reason, source)
                    
                    # Build full suggestion dict for return (includes provenance for audit)
                    suggestion = {
                        "allowed_horizons": suggested_horizons,
                        "_autofix_reason": "horizon_missing",
                        "_autofix_source": "coverage_breakdown",
                        "_inference_config_hash": inference_config_hash,
                        "_registry_version_hash": registry_version_hash,
                        "_tool_version": "registry_v1.0",
                        "_review_required": True,
                        "_current_allowed_horizons": current_horizons,
                        "_target_horizon": target_horizon
                    }
                    if run_id:
                        suggestion["_run_id"] = run_id
                    
                    suggestions[feature_name] = suggestion
            except Exception as e:
                logger.debug(f"Failed to suggest horizon addition for {feature_name}: {e}")
        
        # Note: raw_explicit_disabled features are NOT suggested (policy decision)
        
        # Sort suggestions by feature name (deterministic)
        suggestions_sorted = {k: suggestions[k] for k in sorted_keys(suggestions)}
        
        return {
            "suggestions": suggestions_sorted,
            "stats": stats
        }
    
    def write_coverage_patches(
        self,
        run_root: Path,
        missing_features_result: Dict[str, Any],
        horizon_results_by_target: Dict[str, Dict[str, Any]],
        provenance: Optional[Dict[str, Any]] = None,  # NEW: precomputed provenance from SST
        target_horizon_by_target: Optional[Dict[str, Optional[int]]] = None,  # NEW: target horizons (None = extract from suggestions)
        run_id: Optional[str] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Write coverage automation patch files (missing features, horizon suggestions, SUMMARY).
        
        Args:
            run_root: Run root directory
            missing_features_result: Result from suggest_missing_features_from_breakdown()
            horizon_results_by_target: Dict[target, result] from suggest_horizon_compatibility_from_breakdown()
            run_id: Optional run ID for provenance
            
        Returns:
            Dict with keys: "missing_features", "horizon_<target>", "summary" -> Path or None
        """
        if not self.enabled or not self.write:
            return {}
        
        output_dir = run_root / "registry_autopatch"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        written_files = {}
        
        # Write missing_features.suggested.yaml (only if suggestions exist)
        missing_suggestions = missing_features_result.get("suggestions", {})
        if missing_suggestions:
            try:
                from TRAINING.common.utils.file_utils import write_atomic_yaml
                
                # DETERMINISM: Use precomputed provenance from SST (passed from caller)
                # Fallback to empty dict if not provided (backward compatibility)
                patch_provenance = provenance.copy() if provenance else {
                    "inference_config_hash": "",
                    "registry_version_hash": "",
                    "tool_version": TOOL_VERSION
                }
                
                # Validation: Check if suggestions have consistent provenance (debug-only, cheap)
                if logger.isEnabledFor(logging.DEBUG):
                    # Sample first 3 sorted keys (cheap, deterministic)
                    sample_keys = list(sorted_keys(missing_suggestions))[:3]
                    for feature_name in sample_keys:
                        suggestion = missing_suggestions[feature_name]
                        # Only validate if fields exist (avoid log spam for older suggestions)
                        if "_inference_config_hash" in suggestion:
                            if suggestion.get("_inference_config_hash") != patch_provenance.get("inference_config_hash"):
                                logger.debug(
                                    f"Provenance mismatch in missing_features: {feature_name} has different "
                                    f"inference_config_hash (expected: {patch_provenance.get('inference_config_hash', '')[:16]}...)"
                                )
                        if "_registry_version_hash" in suggestion:
                            if suggestion.get("_registry_version_hash") != patch_provenance.get("registry_version_hash"):
                                logger.debug(
                                    f"Provenance mismatch in missing_features: {feature_name} has different "
                                    f"registry_version_hash (expected: {patch_provenance.get('registry_version_hash', '')[:16]}...)"
                                )
                
                # Use patch_provenance (from SST) instead of extracting from suggestions
                provenance_dict = patch_provenance.copy()
                if run_id:
                    provenance_dict["run_id"] = run_id
                
                # NEW: Add overlay paths and interval if available from registry
                if registry:
                    overlay_paths = getattr(registry, '_overlay_paths_selected', {})
                    if overlay_paths:
                        provenance_dict["overlay_path_global"] = overlay_paths.get("global")
                        provenance_dict["overlay_path_per_target"] = overlay_paths.get("per_target")
                        provenance_dict["interval_int"] = overlay_paths.get("interval_int")
                    
                    # Also add registry path
                    if hasattr(registry, 'config_path'):
                        provenance_dict["registry_path_selected"] = str(registry.config_path)
                
                patch_data = {
                    "_provenance": provenance_dict,
                    "features": missing_suggestions
                }
                
                patch_file = output_dir / "missing_features.suggested.yaml"
                write_atomic_yaml(patch_file, patch_data)
                written_files["missing_features"] = patch_file
                logger.info(f"Wrote {len(missing_suggestions)} missing feature suggestions to {patch_file}")
            except Exception as e:
                logger.error(f"Failed to write missing_features patch: {e}")
                written_files["missing_features"] = None
        
        # Write per-target horizon suggestion files (only if suggestions exist)
        # DETERMINISM: Sort targets for deterministic file write order
        for target, horizon_result in sorted_items(horizon_results_by_target):
            horizon_suggestions = horizon_result.get("suggestions", {})
            if horizon_suggestions:
                try:
                    from TRAINING.common.utils.file_utils import write_atomic_yaml
                    from TRAINING.common.registry_patch_naming import safe_target_filename
                    
                    # DETERMINISM: Use precomputed provenance from SST (passed from caller)
                    # Prefer target_horizon from SST (target_horizon_by_target), fallback to suggestion extraction
                    patch_provenance = provenance.copy() if provenance else {
                        "inference_config_hash": "",
                        "registry_version_hash": "",
                        "tool_version": TOOL_VERSION
                    }
                    
                    # Add target-specific fields
                    patch_provenance["target"] = target
                    
                    # Prefer target_horizon from SST (target_horizon_by_target), fallback to suggestion extraction
                    target_horizon = None
                    if target_horizon_by_target and target in target_horizon_by_target:
                        target_horizon = target_horizon_by_target[target]
                    
                    # If not available from SST, extract deterministically from first suggestion (sorted)
                    if target_horizon is None and horizon_suggestions:
                        # sorted_keys() returns an iterator, convert to list once (consistent with sampling code)
                        sorted_feature_names = list(sorted_keys(horizon_suggestions))
                        if sorted_feature_names:
                            first_feature_name = sorted_feature_names[0]
                            first_suggestion = horizon_suggestions[first_feature_name]
                            target_horizon = first_suggestion.get("_target_horizon")
                            if target_horizon is not None:
                                logger.debug(
                                    f"Extracted target_horizon={target_horizon} from suggestion (not available from SST)"
                                )
                    
                    patch_provenance["target_horizon"] = target_horizon if target_horizon is not None else 0
                    
                    # Validation: Check if suggestions have consistent provenance (debug-only, cheap)
                    if logger.isEnabledFor(logging.DEBUG):
                        # Sample first 3 sorted keys (cheap, deterministic)
                        sample_keys = list(sorted_keys(horizon_suggestions))[:3]
                        for feature_name in sample_keys:
                            suggestion = horizon_suggestions[feature_name]
                            # Only validate if fields exist (avoid log spam)
                            if "_inference_config_hash" in suggestion:
                                if suggestion.get("_inference_config_hash") != patch_provenance.get("inference_config_hash"):
                                    logger.debug(
                                        f"Provenance mismatch in horizon_suggestions: {feature_name} has different "
                                        f"inference_config_hash (expected: {patch_provenance.get('inference_config_hash', '')[:16]}...)"
                                    )
                            if "_registry_version_hash" in suggestion:
                                if suggestion.get("_registry_version_hash") != patch_provenance.get("registry_version_hash"):
                                    logger.debug(
                                        f"Provenance mismatch in horizon_suggestions: {feature_name} has different "
                                        f"registry_version_hash (expected: {patch_provenance.get('registry_version_hash', '')[:16]}...)"
                                    )
                            # target_horizon should be consistent across suggestions for same target
                            if "_target_horizon" in suggestion:
                                suggestion_horizon = suggestion.get("_target_horizon")
                                if suggestion_horizon != patch_provenance.get("target_horizon"):
                                    logger.debug(
                                        f"target_horizon mismatch in horizon_suggestions: {feature_name} has "
                                        f"{suggestion_horizon} (expected: {patch_provenance.get('target_horizon')})"
                                    )
                    
                    # Use patch_provenance (from SST) instead of extracting from suggestions
                    provenance_dict = patch_provenance.copy()
                    if run_id:
                        provenance_dict["run_id"] = run_id
                    
                    # NEW: Add overlay paths and interval if available from registry
                    if registry:
                        overlay_paths = getattr(registry, '_overlay_paths_selected', {})
                        if overlay_paths:
                            provenance_dict["overlay_path_global"] = overlay_paths.get("global")
                            provenance_dict["overlay_path_per_target"] = overlay_paths.get("per_target")
                            provenance_dict["interval_int"] = overlay_paths.get("interval_int")
                        
                        # Also add registry path
                        if hasattr(registry, 'config_path'):
                            provenance_dict["registry_path_selected"] = str(registry.config_path)
                    
                    patch_data = {
                        "_provenance": provenance_dict,
                        "features": horizon_suggestions
                    }
                    
                    safe_target = safe_target_filename(target)
                    patch_file = output_dir / f"allowed_horizons.{safe_target}.suggested.yaml"
                    write_atomic_yaml(patch_file, patch_data)
                    written_files[f"horizon_{target}"] = patch_file
                    logger.info(f"Wrote {len(horizon_suggestions)} horizon suggestions for {target} to {patch_file}")
                except Exception as e:
                    logger.error(f"Failed to write horizon patch for {target}: {e}")
                    written_files[f"horizon_{target}"] = None
        
        # Write SUMMARY.json (always write, even if no suggestions - deterministic audit artifact)
        try:
            from TRAINING.common.utils.file_utils import write_atomic_json
            from TRAINING.common.utils.config_hashing import canonical_json
            
            summary = {
                "missing_features": {
                    "inferable_count": missing_features_result.get("stats", {}).get("inferable_count", 0),
                    "uninferable_count": missing_features_result.get("stats", {}).get("uninferable_count", 0),
                    "total_missing": missing_features_result.get("stats", {}).get("total_missing", 0),
                    "suggestions_written": len(missing_suggestions) if missing_suggestions else 0
                },
                "horizon_compatibility": {}
            }
            
            # Add per-target horizon stats
            # DETERMINISM: Sort targets for deterministic summary order
            for target, horizon_result in sorted_items(horizon_results_by_target):
                stats = horizon_result.get("stats", {})
                summary["horizon_compatibility"][target] = {
                    "raw_allowed_horizons_none_count": stats.get("raw_allowed_horizons_none_count", 0),
                    "effective_horizon_missing_count": stats.get("effective_horizon_missing_count", 0),
                    "raw_explicit_disabled_count": stats.get("raw_explicit_disabled_count", 0),
                    "suggestions_written": len(horizon_result.get("suggestions", {}))
                }
            
            if run_id:
                summary["run_id"] = run_id
            
            summary_file = output_dir / "SUMMARY.json"
            write_atomic_json(summary_file, summary)
            written_files["summary"] = summary_file
            logger.info(f"Wrote coverage automation summary to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to write SUMMARY.json: {e}")
            written_files["summary"] = None
        
        return written_files
    
    def clear(self) -> None:
        """Clear all suggestions (useful for testing)."""
        self._suggestions.clear()
        self._conflicts.clear()
        self._conflicted_fields.clear()


# Global singleton instance (optional - can also create per-run instances)
_global_autopatch: Optional[RegistryAutopatch] = None


def get_autopatch(
    enabled: Optional[bool] = None,
    write: Optional[bool] = None,
    apply: Optional[bool] = None,
    allow_overwrite: Optional[bool] = None
) -> RegistryAutopatch:
    """
    Get global autopatch instance (singleton pattern).
    
    Args:
        enabled: Override enabled flag (if None, reads from config)
        write: Override write flag (if None, reads from config)
        apply: Override apply flag (if None, reads from config)
        allow_overwrite: Override allow_overwrite flag (if None, reads from config)
    
    Returns:
        RegistryAutopatch instance
    """
    global _global_autopatch
    
    # Read from config if flags not provided
    if enabled is None or write is None or apply is None or allow_overwrite is None:
        try:
            from config_loader import get_cfg
            autopatch_config = get_cfg('registry_autopatch', default={})
            enabled = enabled if enabled is not None else autopatch_config.get('enabled', False)
            write = write if write is not None else autopatch_config.get('write', True)
            apply = apply if apply is not None else autopatch_config.get('apply', False)
            allow_overwrite = allow_overwrite if allow_overwrite is not None else autopatch_config.get('allow_overwrite', False)
        except Exception:
            # Fallback defaults
            enabled = enabled if enabled is not None else False
            write = write if write is not None else True
            apply = apply if apply is not None else False
            allow_overwrite = allow_overwrite if allow_overwrite is not None else False
    
    if _global_autopatch is None:
        _global_autopatch = RegistryAutopatch(
            enabled=enabled,
            write=write,
            apply=apply,
            allow_overwrite=allow_overwrite
        )
    else:
        # Update flags if provided
        if enabled is not None:
            _global_autopatch.enabled = enabled
        if write is not None:
            _global_autopatch.write = write
        if apply is not None:
            _global_autopatch.apply = apply
        if allow_overwrite is not None:
            _global_autopatch.allow_overwrite = allow_overwrite
    
    return _global_autopatch


def aggregate_and_write_coverage_patches(
    run_root: Path,
    coverage_breakdowns_by_target: Dict[str, Any],  # Dict[target, CoverageBreakdown]
    registry: Any,
    run_id: Optional[str] = None
) -> Dict[str, Optional[Path]]:
    """
    Aggregate coverage breakdowns across targets and write patch files.
    
    This is the orchestrator-level integration point for registry coverage automation.
    
    Args:
        run_root: Run root directory
        coverage_breakdowns_by_target: Dict mapping target names to CoverageBreakdown instances
        registry: FeatureRegistry instance
        run_id: Optional run ID for provenance
        
    Returns:
        Dict with keys: "missing_features", "horizon_<target>", "summary" -> Path or None
    """
    try:
        from TRAINING.common.utils.registry_autopatch import get_autopatch
        autopatch = get_autopatch()
        
        if not autopatch.enabled:
            return {}
        
        # Aggregate missing features across all targets (deterministic merge)
        all_missing_features = set()
        # DETERMINISM: Sort targets for deterministic processing order (even though we only use values)
        for breakdown in [v for k, v in sorted_items(coverage_breakdowns_by_target)]:
            if hasattr(breakdown, 'missing_feature_ids_full'):
                all_missing_features.update(breakdown.missing_feature_ids_full)
        
        # Create aggregated breakdown for missing features
        from TRAINING.ranking.utils.registry_coverage import CoverageBreakdown
        aggregated_breakdown = CoverageBreakdown(
            n_in_registry=0,
            n_total=len(all_missing_features),
            n_in_registry_horizon_ok=0,
            coverage_in_registry=0.0,
            coverage_total=None,
            coverage_mode="membership_only",
            missing_ids_sample=[],
            missing_feature_ids_full=sorted(all_missing_features),
            blocked_feature_ids_by_reason={}
        )
        
        # Get missing features suggestions (merged across targets)
        missing_features_result = autopatch.suggest_missing_features_from_breakdown(
            aggregated_breakdown,
            registry,
            run_id=run_id
        )
        
        # Get horizon suggestions per target
        horizon_results_by_target = {}
        # DETERMINISM: Sort targets for deterministic processing order
        for target, breakdown in sorted_items(coverage_breakdowns_by_target):
            # DETERMINISM: Use horizon_bars from CoverageBreakdown (already computed correctly)
            # This avoids recomputation and ensures consistency with coverage computation
            target_horizon_bars = breakdown.horizon_bars if hasattr(breakdown, 'horizon_bars') else None
            
            # CRITICAL: If horizon_bars is None, record skip reason (fail-closed, not fail-silent)
            # None means conversion was not exact or not applicable (membership_only/unknown mode)
            if target_horizon_bars is None:
                reason = "horizon_not_convertible"
                if hasattr(breakdown, 'coverage_mode'):
                    if breakdown.coverage_mode != "horizon_ok":
                        reason = f"coverage_mode_{breakdown.coverage_mode}"
                
                logger.debug(
                    f"Target '{target}': horizon_bars is None (reason: {reason}). "
                    f"Skipping horizon suggestions for this target."
                )
                # Store explicit skip result (not fail-silent)
                horizon_results_by_target[target] = {
                    "suggestions": {},
                    "stats": {},
                    "_skipped": True,
                    "_skip_reason": reason
                }
                continue
            
            # Use horizon_bars from breakdown (already validated and computed correctly)
            horizon_result = autopatch.suggest_horizon_compatibility_from_breakdown(
                breakdown,
                target,
                target_horizon_bars,  # Use from breakdown, not recomputed
                registry,
                run_id=run_id
            )
            horizon_results_by_target[target] = horizon_result
        
        # DETERMINISM: Compute provenance from SST once (not from suggestions)
        # This ensures consistency and independence from suggestion insertion order
        # Extract interval_minutes from first breakdown (all should have same interval)
        interval_minutes = None
        if coverage_breakdowns_by_target:
            # DETERMINISM: Use sorted_keys to get first target deterministically
            from TRAINING.common.utils.determinism_ordering import sorted_keys
            first_target = sorted_keys(coverage_breakdowns_by_target)[0]
            first_breakdown = coverage_breakdowns_by_target[first_target]
            if hasattr(first_breakdown, 'interval_minutes'):
                interval_minutes = first_breakdown.interval_minutes
        
        provenance_sst = {
            "inference_config_hash": autopatch._compute_inference_config_hash(),
            "registry_version_hash": autopatch._compute_registry_version_hash(registry) if registry else "",
            "tool_version": TOOL_VERSION,
            "interval_minutes": int(interval_minutes) if interval_minutes is not None else None  # NEW: canonicalized as int
        }
        
        # Also compute target_horizon map from SST (use horizon_bars from breakdown)
        # This ensures we use the exact conversion computed in compute_registry_coverage()
        target_horizon_by_target = {}
        for target in sorted_keys(coverage_breakdowns_by_target):
            breakdown = coverage_breakdowns_by_target[target]
            # Use horizon_bars from breakdown (already computed correctly with exact conversion)
            target_horizon_by_target[target] = breakdown.horizon_bars if hasattr(breakdown, 'horizon_bars') else None
        
        # Write coverage-specific audit files (separate from main patch file for review)
        # Pass precomputed provenance down to avoid registry threading and ensure determinism
        coverage_files = autopatch.write_coverage_patches(
            run_root,
            missing_features_result,
            horizon_results_by_target,
            provenance=provenance_sst,  # NEW: pass precomputed provenance from SST
            target_horizon_by_target=target_horizon_by_target,  # NEW: pass target horizons (None = extract from suggestions)
            run_id=run_id
        )
        
        # Collect per-target suggestions (no mutation of autopatch._suggestions)
        per_target_suggestions = {}  # Dict[target, suggestions_dict]

        # DETERMINISM: Sort targets for deterministic processing order
        for target, breakdown in sorted_items(coverage_breakdowns_by_target):
            # Get horizon suggestions for this target
            horizon_result = horizon_results_by_target.get(target, {})
            target_horizon_suggestions = horizon_result.get("suggestions", {})
            
            # Convert to internal suggestions format: feature_name -> {field -> (value, reason, source)}
            target_suggestions = {}
            # DETERMINISM: Sort features for deterministic suggestion order
            for feature_name, suggestion_dict in sorted_items(target_horizon_suggestions):
                target_suggestions[feature_name] = {}
                # DETERMINISM: Sort fields for deterministic suggestion order
                for field, value in sorted_items(suggestion_dict):
                    if field.startswith('_'):
                        continue  # Skip provenance
                    reason = suggestion_dict.get('_autofix_reason', 'unknown')
                    source = suggestion_dict.get('_autofix_source', 'coverage_automation')
                    target_suggestions[feature_name][field] = (value, reason, source)
            
            if target_suggestions:
                per_target_suggestions[target] = target_suggestions
        
        # Write per-target patch files (no mutation)
        # DETERMINISM: Sort targets for deterministic file write order
        for target, target_suggestions in sorted_items(per_target_suggestions):
            patch_file = autopatch.write_patch_file(
                run_root,
                suggestions=target_suggestions,  # Explicit parameter, no mutation
                target_column=target
            )
            if patch_file:
                coverage_files[f"main_patch_{target}"] = patch_file
                
                # Promote per-target patch if apply=True
                if autopatch.apply:
                    autopatch.promote_patch(run_root, target_column=target)
        
        # Also write global patch (for non-target-specific suggestions from missing features)
        # Use existing autopatch._suggestions for global (backward compatible)
        global_patch_file = autopatch.write_patch_file(run_root, suggestions=None, target_column=None)
        if global_patch_file:
            coverage_files["main_patch"] = global_patch_file
            if autopatch.apply:
                autopatch.promote_patch(run_root, target_column=None)
        
        return coverage_files
    except Exception as e:
        logger.error(f"Failed to aggregate and write coverage patches: {e}", exc_info=True)
        return {}
