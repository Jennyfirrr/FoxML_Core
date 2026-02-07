# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Registry overlay fingerprinting utilities.

Provides deterministic normalization and hashing for overlay content and decision logs.
This module is neutral (no circular imports) - both FeatureRegistry and RegistryAutopatch import from here.
"""

from typing import Dict, Any, List, Optional, Union


def normalize_horizon_value(value: Any) -> int:
    """
    Strict horizon value parsing (no truncation, accept strings, log skips).
    
    Accepts:
    - int (direct)
    - float only if is_integer() (no truncation)
    - digit strings ("5")
    
    Returns:
        int value
    
    Raises:
        ValueError: If value cannot be parsed (caller should record skipped decision)
    """
    if isinstance(value, int):
        return value
    
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        # Non-integer float → invalid (don't truncate)
        raise ValueError(f"Horizon value {value} is non-integer float (truncation not allowed)")
    
    if isinstance(value, str):
        try:
            parsed = float(value)
            if parsed.is_integer():
                return int(parsed)
            raise ValueError(f"Horizon string '{value}' parses to non-integer float")
        except ValueError:
            raise ValueError(f"Horizon string '{value}' cannot be parsed as number")
    
    # numpy ints, decimals, etc. → try conversion but be explicit
    try:
        if hasattr(value, 'item'):  # numpy scalar
            item = value.item()
            if isinstance(item, int):
                return item
            if isinstance(item, float) and item.is_integer():
                return int(item)
        # Fallback: try direct int conversion (will raise if invalid)
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Horizon value {value} (type {type(value).__name__}) cannot be converted to int")


def normalize_overrides(overrides: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize overlay overrides for deterministic hashing.
    
    Applies same filtering/normalization as get_feature_metadata_raw() merge logic:
    - Removes private keys (starting with '_')
    - Removes protected fields ('rejected')
    - Removes None values
    - Normalizes allowed_horizons to sorted unique list (set semantics, strict parsing)
    
    This ensures hash matches what's actually applied to registry semantics.
    
    Args:
        overrides: Dict[feature_name, Dict[field, value]]
    
    Returns:
        Normalized overrides dict (ready for hashing or persistence)
    """
    from TRAINING.common.utils.determinism_ordering import sorted_items
    PROTECTED_FIELDS = {"rejected"}
    normalized = {}
    
    for feature_name, feature_patch in sorted_items(overrides):
        normalized_patch = {}
        for field, value in sorted_items(feature_patch):
            # Skip private keys
            if field.startswith('_'):
                continue
            
            # Skip protected fields
            if field in PROTECTED_FIELDS:
                continue
            
            # Skip None values
            if value is None:
                continue
            
            # Normalize allowed_horizons (set semantics - sorted, unique, strict parsing)
            if field == 'allowed_horizons' and isinstance(value, list):
                normalized_horizons = []
                for x in value:
                    if x is None:
                        continue
                    try:
                        horizon_int = normalize_horizon_value(x)
                        normalized_horizons.append(horizon_int)
                    except ValueError:
                        # Invalid horizon → skip (caller should record in decision log)
                        # Don't include in normalized output
                        continue
                value = sorted(set(normalized_horizons)) if normalized_horizons else None
                if value is None:
                    continue  # Skip field if all horizons invalid
            
            normalized_patch[field] = value
        
        if normalized_patch:  # Only include if has non-filtered fields
            normalized[feature_name] = normalized_patch
    
    return normalized


def normalize_decisions(decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize decision logs for deterministic hashing.
    
    Decision logs have different structure than overlays:
    - applied: {feature: {field: {value, reason, source}}}
    - rejected: {feature: {field: {suggested, existing, reason, conflict_with}}}
    - skipped: {feature: {field: {suggested, existing, reason}}}
    
    Normalization:
    - Stable ordering (sorted keys)
    - Exclude unstable fields (run IDs, file paths, timestamps from source)
    - Normalize horizon values (strict parsing)
    - Canonicalize reason codes
    
    Args:
        decisions: Dict[category, Dict[feature, Dict[field, decision_data]]]
    
    Returns:
        Normalized decisions dict (ready for hashing)
    """
    from TRAINING.common.utils.determinism_ordering import sorted_items, sorted_keys
    
    normalized = {}
    
    for category in sorted_keys(decisions):
        category_decisions = decisions[category]
        normalized_category = {}
        
        for feature_name in sorted_keys(category_decisions):
            feature_decisions = category_decisions[feature_name]
            normalized_feature = {}
            
            for field in sorted_keys(feature_decisions):
                decision_data = feature_decisions[field]
                
                # Normalize decision data (exclude unstable fields)
                normalized_decision = {}
                
                # Common fields
                if 'value' in decision_data:
                    # Normalize value (especially horizons)
                    value = decision_data['value']
                    if field == 'allowed_horizons' and isinstance(value, list):
                        normalized_horizons = []
                        for x in value:
                            try:
                                horizon_int = normalize_horizon_value(x)
                                normalized_horizons.append(horizon_int)
                            except ValueError:
                                continue
                        normalized_decision['value'] = sorted(set(normalized_horizons)) if normalized_horizons else []
                    else:
                        normalized_decision['value'] = value
                
                if 'suggested' in decision_data:
                    suggested = decision_data['suggested']
                    if field == 'allowed_horizons' and isinstance(suggested, list):
                        normalized_horizons = []
                        for x in suggested:
                            try:
                                horizon_int = normalize_horizon_value(x)
                                normalized_horizons.append(horizon_int)
                            except ValueError:
                                continue
                        normalized_decision['suggested'] = sorted(set(normalized_horizons)) if normalized_horizons else []
                    else:
                        normalized_decision['suggested'] = suggested
                
                if 'existing' in decision_data:
                    existing = decision_data['existing']
                    if field == 'allowed_horizons' and isinstance(existing, list):
                        normalized_horizons = []
                        for x in existing:
                            try:
                                horizon_int = normalize_horizon_value(x)
                                normalized_horizons.append(horizon_int)
                            except ValueError:
                                continue
                        normalized_decision['existing'] = sorted(set(normalized_horizons)) if normalized_horizons else []
                    else:
                        normalized_decision['existing'] = existing
                
                # Stable fields (always include)
                if 'reason' in decision_data:
                    normalized_decision['reason'] = decision_data['reason']
                
                if 'conflict_with' in decision_data:
                    normalized_decision['conflict_with'] = decision_data['conflict_with']
                
                # Exclude unstable fields (run IDs, file paths, timestamps)
                # source field may contain run IDs → exclude from hash (or normalize to stable identifier)
                # For now: exclude source from hash (it's in full decision log for audit)
                
                if normalized_decision:
                    normalized_feature[field] = normalized_decision
            
            if normalized_feature:
                normalized_category[feature_name] = normalized_feature
        
        if normalized_category:
            normalized[category] = normalized_category
    
    return normalized


def hash_overrides(overrides: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute deterministic hash of normalized overlay content.
    
    Uses normalize_overrides() to ensure hash matches what's actually applied.
    This is the SINGLE canonical hash function for overlays - use everywhere.
    
    Args:
        overrides: Dict[feature_name, Dict[field, value]] (raw overlay content)
    
    Returns:
        64-character SHA256 hex digest
    """
    from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
    normalized = normalize_overrides(overrides)
    content = canonical_json(normalized)
    return sha256_full(content)


def hash_decisions(decisions: Dict[str, Dict[str, Any]]) -> str:
    """
    Compute deterministic hash of normalized decision logs.
    
    Uses normalize_decisions() to ensure hash matches decision structure.
    This is the SINGLE canonical hash function for decisions - use everywhere.
    
    Args:
        decisions: Dict[category, Dict[feature, Dict[field, decision_data]]]
    
    Returns:
        64-character SHA256 hex digest
    """
    from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
    normalized = normalize_decisions(decisions)
    content = canonical_json(normalized)
    return sha256_full(content)
