# 2026-01-16: Unified Policy Cap System and Circular Dependency Fix

**Date**: 2026-01-16  
**Type**: Architecture Improvement + Bug Fix  
**Impact**: High - Fixes circular dependency, enables auto mode, ensures consistency  
**Breaking**: No - Backward compatible (old config format still works)

## Summary

Implemented a unified policy cap computation system that fixes a circular dependency in auto lookback and ensures feature selection and target ranking use the same core logic. The system decouples policy cap (intent) from safety windows (mechanics), enabling configurable prepass enforcement with control knobs, while feature selection maintains a separate heavier pass for model-based importance computation.

## Problem: Circular Dependency in Auto Lookback

**Root Cause**: The gatekeeper threshold depended on `purge`, which in turn depended on `feature lookback`. This created a circular dependency:
- Gatekeeper needed purge to set threshold
- Purge needed feature lookback to compute
- Feature lookback needed features to be selected
- But gatekeeper runs before feature selection completes

**Impact**: 
- Auto mode couldn't work correctly
- Policy cap couldn't be computed independently
- Feature selection and target ranking had different logic

## Solution: Decouple Policy Cap from Purge

### Architecture Change

**Before**: Policy cap derived from purge → purge derived from feature lookback → circular dependency

**After**: Policy cap computed independently from target horizon → gatekeeper uses policy cap → purge computed after feature selection

### Key Components

1. **`LookbackBudgetSpec`** - Structured config for policy cap computation
2. **`PolicyCapResult`** - Result with diagnostics and source tracking
3. **`load_lookback_budget_spec()`** - Config loader with experiment override support
4. **`compute_policy_cap_minutes()`** - Pure function that always returns float (never None)

### Policy Cap Computation

**Auto Mode**: `cap = k * horizon` (with min/max bounds)
- `k`: Multiplier (default: 10.0)
- `horizon`: Target horizon in minutes
- `min_minutes`: Floor (default: 240.0 = 4 hours)
- `max_minutes`: Optional ceiling (default: 28800.0 = 20 days)

**Fixed Mode**: `cap = fixed_minutes` (explicit value)

**Fallback**: If horizon missing, uses `min_minutes`

## Unified System: Feature Selection + Target Ranking

### Core Philosophy

Both feature selection and target ranking now use the **same policy cap computation logic**:
- **Configurable prepass**: Lightweight enforcement with control knobs (policy cap, policy mode, log mode)
- **Shared function**: `apply_lookback_cap()` used by both stages
- **Consistent behavior**: Same canonical map, same quarantine logic, same validation

### Feature Selection Stages

**FS_PRE (Pre-Selection)**:
- **Purpose**: Lightweight prepass to prevent selector from seeing unsafe features
- **When**: Before importance producers run
- **Logic**: Uses policy cap to quarantine features exceeding cap
- **Benefit**: Faster (selector doesn't process unsafe features) + safer (early enforcement)

**FS_POST (Post-Selection)**:
- **Purpose**: Final validation after model-based importance computation
- **When**: After `_aggregate_multi_model_importance()`, before returning
- **Logic**: Uses same policy cap to validate selected features
- **Benefit**: Catches long-lookback features that selection surfaced

**Heavier Pass**: Feature selection still runs full model-based importance computation (LightGBM, XGBoost, etc.) - this is the "heavier pass" that happens after the prepass.

### Target Ranking Stages

**Gatekeeper**:
- **Purpose**: Final safety gate before model training
- **When**: After all loading/merging/sanitization, before data touches model
- **Logic**: Uses same policy cap computation
- **Benefit**: Consistent enforcement with feature selection

### Rationale: Why Same Core Logic?

1. **Single Source of Truth**: Same policy cap computation everywhere
2. **Consistency**: Feature selection and target ranking enforce same rules
3. **Maintainability**: One place to update policy cap logic
4. **Determinism**: Same inputs → same policy cap → same enforcement
5. **Configurability**: Experiment configs can override policy cap settings

### Why Feature Selection Has Separate Heavier Pass?

Feature selection needs to:
1. **Prepass (FS_PRE)**: Quickly filter unsafe features using policy cap
2. **Heavier Pass**: Run expensive model-based importance computation (LightGBM, XGBoost, CatBoost, etc.) on safe features
3. **Postpass (FS_POST)**: Validate final selection against policy cap

The heavier pass is necessary because:
- Model-based importance is expensive (requires training multiple models)
- We want to avoid training models on unsafe features (waste of compute)
- But we still need the full importance computation to rank features
- Postpass ensures no long-lookback features slip through

## Configuration Changes

### New Structured Format

**Old Format** (still supported for backward compatibility):
```yaml
safety:
  leakage_detection:
    lookback_budget_minutes: "auto"  # or fixed number
```

**New Format** (recommended):
```yaml
safety:
  leakage_detection:
    lookback_budget:
      mode: auto  # "auto" | "fixed"
      auto_rule: k_times_horizon  # Only rule for now
      k: 10.0  # Multiplier for k_times_horizon (cap = k * horizon)
      min_minutes: 240.0  # Floor for auto rules (4 hours)
      max_minutes: 28800.0  # Optional maximum cap (20 days, null to disable)
```

**Fixed Mode**:
```yaml
safety:
  leakage_detection:
    lookback_budget:
      mode: fixed
      fixed_minutes: 7200.0  # Fixed 5-day cap
```

### Experiment Config Overrides

Experiment configs can now override policy cap settings:

```yaml
# CONFIG/experiments/my_experiment.yaml
safety:
  leakage_detection:
    lookback_budget:
      mode: auto
      k: 5.0  # Override: use 5x horizon instead of 10x
      min_minutes: 120.0  # Override: 2 hour floor
```

## Files Changed

### Core System

1. **`TRAINING/ranking/utils/leakage_budget.py`**
   - Added `LookbackBudgetSpec` dataclass
   - Added `PolicyCapResult` dataclass
   - Added `parse_lookback_budget_dict()` - pure function, no config dependency
   - Added `load_lookback_budget_spec()` - thin wrapper with experiment config support
   - Added `compute_policy_cap_minutes()` - total function (always returns float)

2. **`CONFIG/pipeline/training/safety.yaml`**
   - Updated to use new structured `lookback_budget` format
   - Maintains backward compatibility with old format

3. **`CONFIG/experiments/_template.yaml`**
   - Added documentation for overriding `lookback_budget` settings

### Target Ranking

4. **`TRAINING/ranking/predictability/model_evaluation.py`**
   - Updated `_enforce_final_safety_gate()` signature (removed `resolved_config`, added `policy_cap_minutes`)
   - Updated `evaluate_target_predictability()` to compute policy cap before gatekeeper
   - Updated `train_and_evaluate_models()` to compute policy cap before gatekeeper

5. **`TRAINING/ranking/shared_ranking_harness.py`**
   - Updated `apply_cleaning_and_audit_checks()` to compute policy cap before gatekeeper

### Feature Selection

6. **`TRAINING/ranking/feature_selector.py`**
   - Updated FS_PRE (SYMBOL_SPECIFIC) to use new policy cap computation
   - Updated FS_PRE (CROSS_SECTIONAL) to use new policy cap computation
   - Updated FS_POST to use new policy cap computation
   - All three locations now use `load_lookback_budget_spec()` + `compute_policy_cap_minutes()`

### Testing

7. **`CONFIG/experiments/determinism_test.yaml`**
   - Added explicit `lookback_budget` config for determinism testing

8. **`CONFIG/experiments/determinism_test_large.yaml`**
   - Added explicit `lookback_budget` config for determinism testing

## Benefits

### 1. Circular Dependency Fixed
- Policy cap computed independently from purge
- Gatekeeper uses policy cap directly
- Purge computed after feature selection (single-pass flow)

### 2. Auto Mode Enabled
- Policy cap = k * horizon (configurable multiplier)
- Works with any target horizon
- Fallback to min_minutes if horizon missing

### 3. Consistency
- Feature selection and target ranking use same policy cap logic
- Same canonical map, same quarantine logic, same validation
- No split-brain between stages

### 4. Configurability
- Experiment configs can override policy cap settings
- Supports both auto and fixed modes
- Backward compatible with old format

### 5. Better Logging
- Clear source labels (policy_cap_auto, policy_cap_fixed, policy_cap_fallback_min)
- Diagnostics for horizon missing, clamping, etc.
- Consistent logging across all stages

### 6. Determinism
- Policy cap computation is deterministic (same inputs → same output)
- Explicit config values in determinism tests
- Reproducible enforcement behavior

## Migration Guide

### For Users

**No action required** - old config format still works:
```yaml
safety:
  leakage_detection:
    lookback_budget_minutes: "auto"  # Still works
```

**Recommended**: Migrate to new format for better control:
```yaml
safety:
  leakage_detection:
    lookback_budget:
      mode: auto
      k: 10.0
      min_minutes: 240.0
      max_minutes: 28800.0
```

### For Developers

**Old code** (deprecated):
```python
cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto")
if cap_raw != "auto" and isinstance(cap_raw, (int, float)):
    lookback_cap = float(cap_raw)
```

**New code** (recommended):
```python
from TRAINING.ranking.utils.leakage_budget import load_lookback_budget_spec, compute_policy_cap_minutes

spec, warnings = load_lookback_budget_spec("safety_config", experiment_config=experiment_config)
for warning in warnings:
    logger.warning(f"Config validation: {warning}")

policy_cap_result = compute_policy_cap_minutes(spec, target_horizon_minutes, interval_minutes)
lookback_cap = policy_cap_result.cap_minutes  # Always a float, never None
```

## Testing

### Config Parsing Tests
- ✅ Old format parsing (backward compatibility)
- ✅ New format parsing (auto and fixed modes)
- ✅ Invalid config handling (graceful degradation)
- ✅ Both formats present (warns and uses new)

### Policy Cap Computation Tests
- ✅ Auto mode with horizon (cap = k * horizon)
- ✅ Auto mode without horizon (fallback to min_minutes)
- ✅ Fixed mode (cap = fixed_minutes)
- ✅ Max clamp (cap clamped to max_minutes)
- ✅ Min floor (cap floored to min_minutes)

### Integration Tests
- ✅ Feature selection FS_PRE uses policy cap
- ✅ Feature selection FS_POST uses policy cap
- ✅ Target ranking gatekeeper uses policy cap
- ✅ Experiment config overrides work
- ✅ Determinism tests pass

## Related Documentation

- [Feature Selection Lookback Cap Integration](../../03_technical/implementation/training_utils/FEATURE_SELECTION_LOOKBACK_CAP_INTEGRATION.md) - Original integration guide
- [Unified Lookback Cap Structure](../../03_technical/implementation/UNIFIED_LOOKBACK_CAP_STRUCTURE.md) - Standard structure documentation
- [Safety & Leakage Detection Configuration](../configuration/SAFETY_LEAKAGE_CONFIGS.md) - Configuration guide

## Impact

- **Circular Dependency**: Fixed - policy cap computed independently
- **Auto Mode**: Enabled - supports k * horizon computation
- **Consistency**: Achieved - feature selection and target ranking use same logic
- **Configurability**: Enhanced - experiment configs can override settings
- **Determinism**: Improved - explicit config values in determinism tests
- **Backward Compatibility**: Maintained - old config format still works

All changes maintain backward compatibility and follow SST (Single Source of Truth) principles.
