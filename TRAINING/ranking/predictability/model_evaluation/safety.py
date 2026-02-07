# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Safety Gate Validation

Final gatekeeper that enforces safety at the last possible moment before
data touches the model. Drops features that violate policy caps.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def enforce_final_safety_gate(
    X: np.ndarray,
    feature_names: List[str],
    *,
    policy_cap_minutes: float,  # REQUIRED - no None, no fallback
    interval_minutes: Optional[float],
    feature_time_meta_map: Optional[Dict[str, Any]] = None,
    base_interval_minutes: Optional[float] = None,
    logger: logging.Logger = logger,
    dropped_tracker: Optional[Any] = None  # Optional DroppedFeaturesTracker
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Final Gatekeeper: Enforce safety at the last possible moment.

    DUMB FUNCTION: Given cap + lookbacks → drop features + report.
    Does NOT load config. Does NOT compute policy cap.

    This runs AFTER all loading/merging/sanitization is done.
    It physically drops features that violate the policy cap from the dataframe.
    This is the "worry-free" auto-corrector that handles race conditions.

    Why this is needed:
    - Schema loader might add features after sanitization
    - Registry might allow features that violate policy cap
    - Ghost features might slip through multiple layers
    - This is the absolute last check before data touches the model

    Args:
        X: Feature matrix (numpy array)
        feature_names: List of feature names
        policy_cap_minutes: Policy cap (REQUIRED, must be computed by caller)
        interval_minutes: Data interval in minutes
        feature_time_meta_map: Optional map of feature_name -> FeatureTimeMeta
        base_interval_minutes: Optional base training grid interval
        logger: Logger instance
        dropped_tracker: Optional DroppedFeaturesTracker

    Returns:
        (filtered_X, filtered_feature_names, gate_report) tuple
        gate_report contains: {"enforced_feature_set": EnforcedFeatureSet, ...}
    """
    if X is None or len(feature_names) == 0:
        return X, feature_names, {"enforced_feature_set": None}

    # Use policy_cap_minutes directly as safe_lookback_max (no fallback, no config loading)
    safe_lookback_max = policy_cap_minutes
    safe_lookback_max_source = "policy_cap"

    # Add dev_mode indicator
    dev_mode_indicator = ""
    try:
        from CONFIG.dev_mode import get_dev_mode
        if get_dev_mode():
            dev_mode_indicator = " [DEV_MODE]"
    except Exception:
        pass
    logger.info(f"🛡️ Gatekeeper threshold: {safe_lookback_max:.1f}m (source: policy_cap){dev_mode_indicator}")

    # Load over_budget_action from config (for violation handling)
    over_budget_action = "drop"  # Default: drop (for backward compatibility)
    try:
        from CONFIG.config_loader import get_cfg
        over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
    except Exception:
        pass

    # Get feature registry for lookback calculation
    registry = None
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        pass

    # CRITICAL: Use apply_lookback_cap() to follow the same structure as all other phases
    # This ensures consistency: same canonical map, same quarantine logic, same invariants
    # Gatekeeper has extra logic (X matrix manipulation, daily pattern heuristics, dropped_tracker),
    # so we use apply_lookback_cap() for the core structure and preserve the extra logic
    from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
    from CONFIG.config_loader import get_cfg

    # Load policy and log_mode from config
    policy = "drop"  # Gatekeeper uses "drop" by default (over_budget_action controls behavior)
    try:
        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
    except Exception:
        pass

    log_mode = "summary"
    try:
        log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
    except Exception:
        pass

    # feature_time_meta_map and base_interval_minutes are now passed as parameters

    # Use apply_lookback_cap() - follows standard 6-step structure
    # This ensures gatekeeper uses the same canonical map and quarantine logic as all other phases
    cap_result = apply_lookback_cap(
        features=feature_names,
        interval_minutes=interval_minutes,
        cap_minutes=safe_lookback_max,
        policy=policy,
        stage="GATEKEEPER",
        registry=registry,
        feature_time_meta_map=feature_time_meta_map,
        base_interval_minutes=base_interval_minutes,
        log_mode=log_mode
    )

    # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
    # This is the authoritative feature set - downstream code must use this, not raw lists
    enforced = cap_result.to_enforced_set(stage="GATEKEEPER", cap_minutes=safe_lookback_max)

    # Build gate_report for return
    gate_report = {
        "enforced_feature_set": enforced,
        "cap_minutes": safe_lookback_max,
        "source": safe_lookback_max_source
    }

    # Extract results (for backward compatibility with existing gatekeeper logic)
    safe_features = enforced.features  # Use enforced.features (the truth)
    quarantined_features = list(enforced.quarantined.keys()) + enforced.unknown  # All quarantined

    # DIAGNOSTIC: Count features with _d suffix (day-based patterns)
    day_suffix_features = [f for f in feature_names if re.search(r'_\d+d$', f, re.I)]
    logger.info(f"🔍 GATEKEEPER DIAGNOSTIC: Found {len(day_suffix_features)} features with _Xd suffix pattern")
    if day_suffix_features:
        logger.info(f"   Sample _Xd features: {day_suffix_features[:5]}")

    # Build lookup dict from canonical map for per-feature iteration (needed for X matrix manipulation)
    # Use the canonical map from enforced result (SST)
    from TRAINING.ranking.utils.leakage_budget import _feat_key
    feature_lookback_dict = {}
    for feat_name in feature_names:
        feat_key = _feat_key(feat_name)
        lookback = enforced.canonical_map.get(feat_key)
        if lookback is None:
            lookback = float("inf")  # Unknown = unsafe
        feature_lookback_dict[feat_name] = lookback

    # GATEKEEPER-SPECIFIC: Additional logic for "daily/24h naming pattern" heuristic
    # This is gatekeeper-specific and not part of apply_lookback_cap()
    # We need to check this for features that passed apply_lookback_cap but might still violate purge
    # due to the "daily/24h naming pattern" heuristic

    # Build dropped_features and dropped_indices from quarantined_features
    # Also check for "daily/24h naming pattern" heuristic (gatekeeper-specific)
    dropped_features = []
    dropped_indices = []
    violating_features = []  # Track violations for hard_stop/warn modes

    # First, add all quarantined features from apply_lookback_cap()
    for idx, feature_name in enumerate(feature_names):
        if feature_name in quarantined_features:
            lookback_minutes = feature_lookback_dict.get(feature_name, float("inf"))
            if lookback_minutes == float("inf"):
                reason = "unknown lookback (cannot infer - treated as unsafe)"
            else:
                reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m)"
            dropped_features.append((feature_name, reason))
            dropped_indices.append(idx)
            violating_features.append((feature_name, reason))
            continue

        # CRITICAL: Use the canonical map lookback directly (single source of truth)
        # The canonical map already includes all inference logic (patterns, heuristics, etc.)
        # So if a feature has lookback > cap in the canonical map, it should be dropped
        # The "daily/24h naming pattern" heuristic is redundant - canonical map already handles it
        lookback_minutes = feature_lookback_dict.get(feature_name)
        if lookback_minutes is None:
            lookback_minutes = float("inf")  # Unknown = unsafe

        if lookback_minutes == float("inf"):
            # Unknown lookback - already handled by apply_lookback_cap, but check again for safety
            continue

        # Calendar features have 0m lookback and should NEVER be dropped
        is_calendar_feature = (lookback_minutes == 0.0)

        # CRITICAL: If canonical map says lookback > cap, drop it (canonical map is the truth)
        # This ensures gatekeeper sees the same lookbacks as POST_PRUNE
        # The canonical map already includes all inference (patterns, heuristics, etc.)
        if lookback_minutes > safe_lookback_max and not is_calendar_feature:
            # This feature should have been quarantined by apply_lookback_cap
            # But if it wasn't (e.g., due to a bug), drop it here as a safety net
            reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m) [canonical map]"
            dropped_features.append((feature_name, reason))
            dropped_indices.append(idx)
            violating_features.append((feature_name, reason))

    # Handle violations based on over_budget_action
    if violating_features:
        if over_budget_action == "hard_stop":
            # Hard-stop: fail the run if any violating feature exists
            violation_list = ", ".join([f"{name} ({reason})" for name, reason in violating_features[:10]])
            if len(violating_features) > 10:
                violation_list += f" ... and {len(violating_features) - 10} more"
            raise RuntimeError(
                f"🚨 OVER_BUDGET VIOLATION (policy: hard_stop - training blocked): "
                f"{len(violating_features)} features exceed policy cap (safe_lookback_max={safe_lookback_max:.1f}m). "
                f"Violating features: {violation_list}"
            )
        elif over_budget_action == "warn":
            # Warn: allow but log violations (NOT recommended for production)
            logger.warning(
                f"⚠️ OVER_BUDGET VIOLATION (policy: warn - allowing violating features): "
                f"{len(violating_features)} features exceed policy cap (safe_lookback_max={safe_lookback_max:.1f}m)"
            )
            logger.info(f"   Violating features ({len(violating_features)}):")
            for feat_name, feat_reason in violating_features[:10]:
                logger.warning(f"   ⚠️ {feat_name}: {feat_reason}")
            if len(violating_features) > 10:
                logger.warning(f"   ... and {len(violating_features) - 10} more")
            # Don't drop - just warn
        # else: over_budget_action == "drop" - handled below

    # Mutate the Dataframe (drop columns) - only if action is "drop"
    if dropped_features:
        # Log policy context with explicit source
        source_str = safe_lookback_max_source if safe_lookback_max_source else "unknown"
        logger.warning(
            f"🛡️ FINAL GATEKEEPER: Dropping {len(dropped_features)} features that violate lookback threshold "
            f"(threshold={safe_lookback_max:.1f}m, source={source_str})"
        )
        logger.info(f"   Policy: drop_features (auto-drop violating features)")
        logger.info(f"   Drop list ({len(dropped_features)} features):")
        for feat_name, feat_reason in dropped_features[:10]:  # Show first 10
            logger.warning(f"   🗑️ {feat_name}: {feat_reason}")
        if len(dropped_features) > 10:
            logger.warning(f"   ... and {len(dropped_features) - 10} more")

        # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
        # The enforced.features list IS the authoritative order - X columns must match it
        # Build indices for safe features (enforced.features)
        feature_indices = [i for i, name in enumerate(feature_names) if name in enforced.features]
        if len(feature_indices) == len(enforced.features):
            X = X[:, feature_indices]
            feature_names = enforced.features.copy()  # Use enforced.features (the truth)
        else:
            # Fallback: use dropped_indices (shouldn't happen, but safety net)
            logger.warning(
                f"   ⚠️ Gatekeeper: Index mismatch. Expected {len(enforced.features)} features, "
                f"got {len(feature_indices)} indices. Using dropped_indices fallback."
            )
            keep_indices = [i for i in range(X.shape[1]) if i not in dropped_indices]
            X = X[:, keep_indices]
            feature_names = [name for idx, name in enumerate(feature_names) if idx not in dropped_indices]

        logger.info(f"   ✅ After final gatekeeper: {X.shape[1]} features remaining")

        # Track dropped features for telemetry with structured reasons
        if dropped_tracker is not None:
            from TRAINING.ranking.utils.dropped_features_tracker import DropReason

            # Capture input/output for stage record
            input_features_before_gatekeeper = feature_names.copy() if 'feature_names' in locals() else []

            # Create structured reasons
            structured_reasons = {}
            for feat_name, reason_str in dropped_features:
                # Parse reason string to extract structured info
                reason_code = "LOOKBACK_CAP"
                measured_value = None
                threshold_value = safe_lookback_max

                # Try to extract lookback value from reason string
                lookback_match = re.search(r'lookback \(([\d.]+)m\)', reason_str)
                if lookback_match:
                    measured_value = float(lookback_match.group(1))

                # Get config provenance
                config_provenance = f"lookback_budget_minutes={safe_lookback_max:.1f}m (source={safe_lookback_max_source})"

                structured_reasons[feat_name] = DropReason(
                    reason_code=reason_code,
                    stage="gatekeeper",
                    human_reason=reason_str,
                    measured_value=measured_value,
                    threshold_value=threshold_value,
                    config_provenance=config_provenance
                )

            # Get config provenance dict
            config_provenance_dict = {
                "safe_lookback_max": safe_lookback_max,
                "safe_lookback_max_source": safe_lookback_max_source,
                "over_budget_action": over_budget_action
            }

            dropped_tracker.add_gatekeeper_drops(
                [name for name, _ in dropped_features],
                structured_reasons,
                input_features=input_features_before_gatekeeper,
                output_features=feature_names,  # After drop
                config_provenance=config_provenance_dict
            )

    return X, feature_names, gate_report


# Backward compatibility alias (internal name used in model_evaluation.py)
_enforce_final_safety_gate = enforce_final_safety_gate
