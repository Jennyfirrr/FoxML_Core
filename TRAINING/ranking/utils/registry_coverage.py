# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Canonical Registry Coverage Computation

Single Source of Truth for computing registry coverage metrics.
Returns detailed breakdown to prevent "coverage is lying" issues.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Set, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CoverageBreakdown:
    """
    Detailed breakdown of registry coverage computation.
    
    Prevents "coverage is lying" by explicitly tracking:
    - Membership-only vs full coverage (horizon-validated)
    - Missing features (not in registry)
    - Coverage mode (horizon_ok, membership_only, unknown)
    - Reason-coded blocked features (raw vs effective eligibility)
    """
    n_in_registry: int  # Features present in registry (membership check)
    n_total: int  # Total features in universe
    n_in_registry_horizon_ok: int  # Features in registry AND horizon-compatible
    coverage_in_registry: float  # n_in_registry / n_total (membership-only metric)
    coverage_total: Optional[float]  # n_in_registry_horizon_ok / n_total (full metric, None if mode != "horizon_ok")
    coverage_mode: Literal["horizon_ok", "membership_only", "unknown"]  # Coverage computation mode
    missing_ids_sample: List[str]  # Sample of missing feature IDs (up to 50) - DO NOT use for automation
    view: Optional[str] = None  # View parameter (informational only, for diagnostics)
    error_summary: Optional[dict] = None  # Stable error info (exception_type, exception_message) - traceback logged separately, not in artifact
    
    # NEW: Reason-coded full lists (sorted, deterministic)
    missing_feature_ids_full: List[str] = field(default_factory=list)  # All missing features (sorted)
    blocked_feature_ids_by_reason: Dict[str, List[str]] = field(default_factory=dict)  # Reason -> sorted IDs
    # Keys: "effective_rejected", "raw_explicit_disabled", "effective_horizon_missing", "raw_allowed_horizons_none"
    # Note: Split raw vs effective for clarity (raw = before inheritance, effective = after inheritance)
    # Note: Inferable/uninferable classification happens in autopatch layer, NOT in coverage


def compute_registry_coverage(
    feature_names: List[str],
    target: str,
    interval_minutes: float,
    horizon_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    experiment_config: Optional[Any] = None,
    view: Optional[str] = None,  # Informational only (for diagnostics), does not affect registry namespace or feature-id mapping
    error_policy: Literal["strict", "best_effort"] = "best_effort"  # Policy for handling errors: strict raises, best_effort returns with error_summary
) -> CoverageBreakdown:
    """
    Compute canonical registry coverage for a target.
    
    This is the SINGLE SOURCE OF TRUTH for registry coverage computation.
    All eligibility gates must use this function.
    
    CRITICAL ORDER:
    1. Check `if feat_id in registry.features` FIRST (pure membership check)
    2. ONLY THEN call get_feature_metadata() for rejected/allowed_horizons
    3. For missing features: record as missing WITHOUT calling metadata
       (prevents registry from fabricating defaults that blur "missing" vs "rejected" vs "wrong horizon")
    
    Args:
        feature_names: List of feature names to check
        target: Target column name (for horizon extraction)
        interval_minutes: Data bar interval in minutes
        horizon_minutes: Optional target horizon in minutes (if None, extracted from target name)
        registry: Optional FeatureRegistry instance (if None, will get default)
        experiment_config: Optional experiment config (for horizon extraction)
        view: Optional view parameter (informational only, for diagnostics, does not affect registry namespace or feature-id mapping)
    
    Returns:
        CoverageBreakdown with detailed metrics and mode
    
    Raises:
        ValueError: If interval_minutes is not a valid near-integer
    """
    # TEMPORARY TEST - remove after verification
    import os
    if os.getenv("FOXML_COVERAGE_TRACE_TEST") == "1":
        raise RuntimeError("COVERAGE_TEST - verify traceback appears in logs")
    
    # CRITICAL: Validate interval_minutes BEFORE any computation
    interval_rounded = round(interval_minutes)
    if abs(interval_minutes - interval_rounded) >= 1e-3:
        error_msg = (
            f"Invalid interval_minutes={interval_minutes} (not near-integer). "
            f"Difference from rounded: {abs(interval_minutes - interval_rounded)}."
        )
        logger.error("%s", error_msg)
        if error_policy == "strict":
            # Contract violation - invalid config
            raise ValueError(error_msg)
            return CoverageBreakdown(
                n_in_registry=0,
                n_total=len(feature_names),
                n_in_registry_horizon_ok=0,
                coverage_in_registry=0.0,
                coverage_total=None,
                coverage_mode="unknown",
                missing_ids_sample=[],
                view=view,
                missing_feature_ids_full=sorted(feature_names),  # All features missing in unknown mode
                blocked_feature_ids_by_reason={}
            )
    
    interval_minutes_int = int(interval_rounded)
    
    # FIX ISSUE-010: Guard against zero interval before any division/modulo
    if interval_minutes_int == 0:
        error_msg = f"Invalid interval_minutes={interval_minutes} (zero)."
        logger.error("%s", error_msg)
        if error_policy == "strict":
            # Contract violation - invalid config
            raise ValueError(error_msg)
        # Defensive: handle None feature_names
        n_total = len(feature_names) if feature_names is not None else 0
        return CoverageBreakdown(
            n_in_registry=0,
            n_total=n_total,
            n_in_registry_horizon_ok=0,
            coverage_in_registry=0.0,
            coverage_total=None,
            coverage_mode="unknown",
            missing_ids_sample=[],
            view=view,
            missing_feature_ids_full=sorted(feature_names) if feature_names else [],
            blocked_feature_ids_by_reason={}
        )
    
    # Get registry if not provided
    if registry is None:
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception as e:
            # Log traceback to logs (not in artifact)
            logger.exception("Failed to get registry: %s", e)
            if error_policy == "strict":
                # Cannot compute meaningful coverage without registry
                raise
            # best_effort: return unknown with stable error summary
            error_summary = {"exception_type": type(e).__name__, "exception_message": str(e)}
            return CoverageBreakdown(
                n_in_registry=0,
                n_total=len(feature_names),
                n_in_registry_horizon_ok=0,
                coverage_in_registry=0.0,
                coverage_total=None,
                coverage_mode="unknown",
                missing_ids_sample=[],
                view=view,
                error_summary=error_summary,
                missing_feature_ids_full=sorted(feature_names),
                blocked_feature_ids_by_reason={}
            )
    
    # Resolve horizon_minutes using SST function
    if horizon_minutes is None:
        try:
            from TRAINING.common.utils.sst_contract import resolve_target_horizon_minutes
            horizon_minutes = resolve_target_horizon_minutes(target, experiment_config)
        except Exception as e:
            logger.exception("Failed to resolve horizon from target '%s': %s", target, e)
            if error_policy == "strict":
                # Contract violation - invalid config
                raise
            horizon_minutes = None  # Continue with None (may be acceptable in best_effort)
    
    # CRITICAL: Validate horizon_minutes is integer (for naming scheme)
    if horizon_minutes is not None:
        horizon_rounded = round(horizon_minutes)
        if abs(horizon_minutes - horizon_rounded) >= 1e-3:
            logger.warning(
                f"Invalid horizon_minutes={horizon_minutes} (not near-integer). "
                f"Treating as unknown horizon. Returning coverage_mode='unknown'."
            )
            return CoverageBreakdown(
                n_in_registry=0,
                n_total=len(feature_names),
                n_in_registry_horizon_ok=0,
                coverage_in_registry=0.0,
                coverage_total=None,
                coverage_mode="unknown",
                missing_ids_sample=[],
                view=view,
                missing_feature_ids_full=sorted(feature_names),  # All features missing in unknown mode
                blocked_feature_ids_by_reason={}
            )
        horizon_minutes_int = int(horizon_rounded)
    else:
        # Cannot compute horizon-validated coverage without horizon
        logger.warning(f"horizon_minutes is None for target '{target}'. Returning membership_only coverage.")
        return _compute_membership_only_coverage(feature_names, registry, view)
    
    # Convert horizon to bars using SST function (returns None if not exactly divisible)
    from TRAINING.common.utils.horizon_conversion import horizon_minutes_to_bars
    horizon_bars = horizon_minutes_to_bars(horizon_minutes_int, interval_minutes_int)
    if horizon_bars is None:
        logger.warning(
            f"Target '{target}': horizon_minutes={horizon_minutes_int} not exactly divisible by "
            f"interval_minutes={interval_minutes_int}. Cannot compute horizon-validated coverage. "
            f"Returning membership_only coverage."
        )
        return _compute_membership_only_coverage(feature_names, registry, view)
    
    # DIAGNOSTIC: Log horizon being used for validation
    logger.info(
        f"[COVERAGE_DIAG] Horizon validation: target={target}, horizon_minutes={horizon_minutes_int}, "
        f"interval_minutes={interval_minutes_int}, horizon_bars={horizon_bars}"
    )
    
    # Defensive: Exclude non-feature columns before counting
    # This prevents target columns from inflating n_total even if they slip through upstream filtering
    from TRAINING.ranking.utils.column_exclusion import exclude_non_feature_columns
    from collections import Counter
    
    # DIAGNOSTIC: Log input feature count to trace coverage denominator discrepancy
    logger.info(
        f"[COVERAGE_DIAG] compute_registry_coverage input: len(feature_names)={len(feature_names)}, "
        f"target={target}, view={view}"
    )
    
    feature_names_filtered, excluded_non_features = exclude_non_feature_columns(
        feature_names,
        reason="coverage-defensive-filter"
    )
    
    # DIAGNOSTIC: Log filtered feature count
    logger.info(
        f"[COVERAGE_DIAG] After defensive filter: len(feature_names_filtered)={len(feature_names_filtered)}, "
        f"excluded_non_features={len(excluded_non_features)}"
    )
    
    # Log prefix counts for diagnostics
    if excluded_non_features:
        excluded_prefixes = Counter()
        for col in excluded_non_features:
            for prefix in ['y_', 'fwd_ret_', 'p_', 'barrier_', 'ts', 'timestamp', 'symbol']:
                if col.startswith(prefix):
                    excluded_prefixes[prefix] += 1
                    break
        
        logger.warning(
            f"Coverage defensive filter: excluded {len(excluded_non_features)} non-feature columns. "
            f"Prefix breakdown: {dict(excluded_prefixes)}. "
            f"Sample: {excluded_non_features[:5]}"
        )
    
    # Count features (use filtered list)
    n_total = len(feature_names_filtered)
    
    # DIAGNOSTIC: Log n_total to trace coverage denominator
    logger.info(
        f"[COVERAGE_DIAG] Coverage calculation: n_total={n_total}, "
        f"will compute coverage_total = n_in_registry_horizon_ok / {n_total}"
    )
    
    n_in_registry = 0
    n_in_registry_horizon_ok = 0
    missing_ids = []
    
    # NEW: Track blocked features by reason (raw vs effective split)
    blocked_by_reason: Dict[str, List[str]] = {
        "effective_rejected": [],
        "raw_explicit_disabled": [],
        "effective_horizon_missing": [],
        "raw_allowed_horizons_none": []
    }
    
    # Track metadata errors for throttled logging (prevent log spam)
    metadata_errors = 0
    metadata_error_samples = []  # Store first 3 exceptions for traceback logging
    MAX_ERROR_SAMPLES = 3
    
    # CRITICAL ORDER: Check membership FIRST, then metadata
    # Use filtered feature names (non-feature columns already excluded)
    for feat_name in feature_names_filtered:
        # Step 1: Pure membership check (do NOT call get_feature_metadata yet)
        if feat_name not in registry.features:
            missing_ids.append(feat_name)
            continue  # Missing feature - record without calling metadata
        
        # Step 2: Feature is in registry - now check metadata
        n_in_registry += 1
        
        try:
            # Get both raw and effective metadata for reason classification
            raw_metadata = registry.get_feature_metadata_raw(feat_name)
            effective_metadata = registry.get_feature_metadata_effective(feat_name, resolve_defaults=True)
            
            # Check if rejected (effective - post-inheritance)
            if effective_metadata.get('rejected', False):
                blocked_by_reason["effective_rejected"].append(feat_name)
                continue  # In registry but rejected - doesn't count for horizon-ok
            
            # Use explicit checks to distinguish None (unknown) from [] (disallowed)
            # Use MISSING sentinel to detect missing key vs None value
            MISSING = object()
            raw_allowed_horizons = raw_metadata.get('allowed_horizons', MISSING)
            effective_allowed_horizons = effective_metadata.get('allowed_horizons', MISSING)
            
            if effective_allowed_horizons is None:
                # unknown/inherit not resolved or intentionally unset (effective)
                # Check raw to see if it was None before inheritance
                if raw_allowed_horizons is None or raw_allowed_horizons is MISSING:
                    blocked_by_reason["raw_allowed_horizons_none"].append(feat_name)
                # Suggest patch if family has defaults
                _suggest_allowed_horizons_patch(feat_name, registry)
                continue
            elif effective_allowed_horizons == []:
                # explicitly disallowed (effective)
                # Check raw to see if it was [] before inheritance
                if raw_allowed_horizons == []:
                    blocked_by_reason["raw_explicit_disabled"].append(feat_name)
                continue
            elif effective_allowed_horizons is MISSING:
                # Key missing entirely (shouldn't happen after merge, but defensive)
                if raw_allowed_horizons is None or raw_allowed_horizons is MISSING:
                    blocked_by_reason["raw_allowed_horizons_none"].append(feat_name)
                # Suggest patch if family has defaults
                _suggest_allowed_horizons_patch(feat_name, registry)
                continue
            else:
                # FIX ISSUE-011: Normalize allowed_horizons to ints before comparison (handle float/string types)
                allowed_horizons_ints = []
                for h in effective_allowed_horizons:
                    try:
                        allowed_horizons_ints.append(int(h))
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping invalid horizon value in allowed_horizons: {h} (type: {type(h)})")
                        continue
                
                # Check if horizon is in allowed list (normalized to ints)
                if horizon_bars in allowed_horizons_ints:
                    n_in_registry_horizon_ok += 1
                else:
                    # Horizon missing from effective allowed_horizons list
                    blocked_by_reason["effective_horizon_missing"].append(feat_name)
                    # DIAGNOSTIC: Log first few blocked features to understand why
                    if len(blocked_by_reason["effective_horizon_missing"]) <= 5:
                        logger.debug(
                            f"[COVERAGE_DIAG] Feature '{feat_name}' blocked: horizon_bars={horizon_bars} not in "
                            f"allowed_horizons={allowed_horizons_ints}"
                        )
        except Exception as e:
            metadata_errors += 1
            # Throttled traceback logging: log first N with exc_info=True, then just message
            if len(metadata_error_samples) < MAX_ERROR_SAMPLES:
                metadata_error_samples.append((feat_name, e))
                logger.debug(f"Error checking metadata for feature '{feat_name}': {e}", exc_info=True)
            else:
                logger.debug(f"Error checking metadata for feature '{feat_name}': {e}")
            # Feature is in registry but metadata check failed - count as in_registry but not horizon_ok
    
    # Log summary if metadata errors occurred (after loop to prevent log spam)
    if metadata_errors > 0:
        logger.warning(
            f"Metadata check failed for {metadata_errors} features. "
            f"First {len(metadata_error_samples)} errors logged with traceback above."
        )
    
    # Compute coverage metrics
    coverage_in_registry = n_in_registry / n_total if n_total > 0 else 0.0
    coverage_total = n_in_registry_horizon_ok / n_total if n_total > 0 else 0.0
    
    # DIAGNOSTIC: Log actual coverage calculation values
    logger.info(
        f"[COVERAGE_DIAG] Coverage result: n_in_registry={n_in_registry}, n_in_registry_horizon_ok={n_in_registry_horizon_ok}, "
        f"n_total={n_total}, coverage_in_registry={coverage_in_registry:.4f}, coverage_total={coverage_total:.4f}"
    )
    
    # DIAGNOSTIC: Log breakdown of blocked features by reason
    total_blocked = sum(len(v) for v in blocked_by_reason.values())
    if total_blocked > 0:
        logger.info(
            f"[COVERAGE_DIAG] Blocked features breakdown: total_blocked={total_blocked}, "
            f"effective_rejected={len(blocked_by_reason['effective_rejected'])}, "
            f"raw_explicit_disabled={len(blocked_by_reason['raw_explicit_disabled'])}, "
            f"effective_horizon_missing={len(blocked_by_reason['effective_horizon_missing'])}, "
            f"raw_allowed_horizons_none={len(blocked_by_reason['raw_allowed_horizons_none'])}"
        )
    
    # Store sample of missing IDs (up to 50) - for logs only, NOT for automation
    # FIX ISSUE-024: missing_ids_sample preserves input order (diagnostics only; do not rely on ordering in tests)
    # If deterministic sample order is required for testing, sort before slicing: sorted(missing_ids)[:50]
    missing_ids_sample = missing_ids[:50]
    
    # NEW: Sort full lists deterministically (for automation)
    missing_feature_ids_full = sorted(missing_ids)
    
    # Diagnostic: Log missing features by prefix (after defensive filtering)
    if missing_ids:
        missing_prefixes = Counter()
        suspicious_prefixes = ['y_', 'fwd_ret_', 'p_', 'barrier_', 'ts', 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in missing_ids:
            for prefix in suspicious_prefixes:
                if col.startswith(prefix):
                    missing_prefixes[prefix] += 1
                    break
        
        logger.debug(
            f"Missing features after filtering: total={len(missing_ids)}, "
            f"by prefix: {dict(missing_prefixes)}"
        )
        
        # If missing is dominated by raw columns, that's a different issue
        raw_cols = missing_prefixes.get('open', 0) + missing_prefixes.get('high', 0) + missing_prefixes.get('low', 0) + missing_prefixes.get('close', 0) + missing_prefixes.get('volume', 0)
        if raw_cols > len(missing_ids) * 0.5:
            logger.warning(
                f"Missing features are mostly raw OHLCV columns ({raw_cols}/{len(missing_ids)}). "
                f"This suggests upstream feature candidate list is too permissive."
            )
    
    # Sort blocked lists deterministically
    blocked_feature_ids_by_reason = {
        reason: sorted(feature_ids) 
        for reason, feature_ids in blocked_by_reason.items() 
        if feature_ids  # Only include non-empty lists
    }
    
    result = CoverageBreakdown(
        n_in_registry=n_in_registry,
        n_total=n_total,
        n_in_registry_horizon_ok=n_in_registry_horizon_ok,
        coverage_in_registry=coverage_in_registry,
        coverage_total=coverage_total,
        coverage_mode="horizon_ok",
        missing_ids_sample=missing_ids_sample,
        view=view,
        missing_feature_ids_full=missing_feature_ids_full,
        blocked_feature_ids_by_reason=blocked_feature_ids_by_reason
    )
    
    # DIAGNOSTIC: Log final result to verify it matches calculation
    logger.info(
        f"[COVERAGE_DIAG] Returning CoverageBreakdown: n_total={result.n_total}, "
        f"n_in_registry_horizon_ok={result.n_in_registry_horizon_ok}, "
        f"coverage_total={result.coverage_total:.4f}, id={id(result)}"
    )
    
    return result


def summarize_coverage_breakdown(cb: CoverageBreakdown) -> Dict[str, Any]:
    """
    Summarize coverage breakdown for logging and diagnostics (deterministic).
    
    All ID lists are sorted before sampling to ensure deterministic output.
    
    Args:
        cb: CoverageBreakdown instance
    
    Returns:
        Dict with summary fields:
        - n_total, n_in_registry, coverage_in_registry
        - mode (horizon_ok, membership_only, unknown)
        - missing_count, missing_sample_sorted (first 10, sorted)
        - blocked_counts_by_reason (sorted keys)
        - blocked_sample_by_reason (first 3 per reason, sorted IDs)
    """
    # Deterministic sorting: use sorted() for all ID lists before sampling
    missing_sample_sorted = sorted(cb.missing_feature_ids_full)[:10] if cb.missing_feature_ids_full else []
    
    # Sort blocked by reason (keys are already sorted in CoverageBreakdown, but ensure)
    blocked_counts_by_reason = {}
    blocked_sample_by_reason = {}
    for reason in sorted(cb.blocked_feature_ids_by_reason.keys()):
        feature_ids = cb.blocked_feature_ids_by_reason[reason]
        blocked_counts_by_reason[reason] = len(feature_ids)
        blocked_sample_by_reason[reason] = sorted(feature_ids)[:3] if feature_ids else []
    
    return {
        'n_total': cb.n_total,
        'n_in_registry': cb.n_in_registry,
        'coverage_in_registry': cb.coverage_in_registry,
        'mode': cb.coverage_mode,
        'missing_count': len(cb.missing_feature_ids_full),
        'missing_sample_sorted': missing_sample_sorted,
        'blocked_counts_by_reason': blocked_counts_by_reason,
        'blocked_sample_by_reason': blocked_sample_by_reason
    }


def _suggest_allowed_horizons_patch(feature_name: str, registry: Any) -> None:
    """
    Suggest allowed_horizons patch if feature should inherit from family defaults.
    
    Only suggests if:
    - Feature has allowed_horizons=None or missing (not [])
    - Family has default_allowed_horizons defined
    """
    try:
        from TRAINING.common.utils.registry_autopatch import get_autopatch
        autopatch = get_autopatch()
        
        if not autopatch.enabled:
            return
        
        # Get raw metadata to check if allowed_horizons is None/missing (not [])
        raw_metadata = registry.get_feature_metadata_raw(feature_name)
        allowed_horizons = raw_metadata.get('allowed_horizons')
        
        # Only suggest if None/missing (not empty list)
        if allowed_horizons is None or 'allowed_horizons' not in raw_metadata:
            # Check if family has defaults
            family_meta = registry._match_feature_family(feature_name) if hasattr(registry, '_match_feature_family') else None
            if family_meta and family_meta.get('default_allowed_horizons'):
                default_horizons = family_meta['default_allowed_horizons']
                family_name = family_meta.get('description', 'unknown')
                
                autopatch.suggest_patch(
                    feature_name=feature_name,
                    field='allowed_horizons',
                    value=default_horizons,
                    reason=f"inherit_family_default {family_name}",
                    source='family_match'
                )
    except Exception as e:
        logger.debug(f"Failed to suggest allowed_horizons patch for {feature_name}: {e}")


def _compute_membership_only_coverage(
    feature_names: List[str],
    registry: Any,
    view: Optional[str] = None
) -> CoverageBreakdown:
    """
    Compute membership-only coverage (when horizon validation is not possible).
    
    This is a different metric than full coverage and must NOT feed the same eligibility gate.
    """
    # Defensive: Exclude non-feature columns before counting (same as full coverage)
    from TRAINING.ranking.utils.column_exclusion import exclude_non_feature_columns
    feature_names_filtered, excluded_non_features = exclude_non_feature_columns(
        feature_names,
        reason="membership-only-coverage-defensive-filter"
    )
    
    if excluded_non_features:
        logger.debug(
            f"Membership-only coverage defensive filter: excluded {len(excluded_non_features)} non-feature columns"
        )
    
    n_total = len(feature_names_filtered)
    n_in_registry = 0
    missing_ids = []
    
    # CRITICAL ORDER: Check membership FIRST, do NOT call metadata for missing features
    # Use filtered feature names (non-feature columns already excluded)
    for feat_name in feature_names_filtered:
        if feat_name not in registry.features:
            missing_ids.append(feat_name)
            continue  # Missing - record without calling metadata
        
        n_in_registry += 1
    
    coverage_in_registry = n_in_registry / n_total if n_total > 0 else 0.0
    
    # Store sample of missing IDs (up to 50) - for logs only, NOT for automation
    # FIX ISSUE-024: missing_ids_sample preserves input order (diagnostics only; do not rely on ordering in tests)
    # If deterministic sample order is required for testing, sort before slicing: sorted(missing_ids)[:50]
    missing_ids_sample = missing_ids[:50]
    
    # NEW: Sort full lists deterministically (for automation)
    missing_feature_ids_full = sorted(missing_ids)
    # No blocked features in membership_only mode (no horizon check)
    blocked_feature_ids_by_reason = {}
    
    return CoverageBreakdown(
        n_in_registry=n_in_registry,
        n_total=n_total,
        n_in_registry_horizon_ok=0,  # Not computed for membership_only mode
        coverage_in_registry=coverage_in_registry,
        coverage_total=None,  # Not computed for membership_only mode
        coverage_mode="membership_only",
        missing_ids_sample=missing_ids_sample,
        view=view,
        missing_feature_ids_full=missing_feature_ids_full,
        blocked_feature_ids_by_reason=blocked_feature_ids_by_reason
    )
