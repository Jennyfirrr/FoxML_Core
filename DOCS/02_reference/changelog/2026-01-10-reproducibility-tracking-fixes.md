---
Type: Bug Fix
Impact: Critical
Stage: TARGET_RANKING, All Stages
---

# 2026-01-10: Reproducibility Tracking Fixes

## Summary

Fixed five critical issues affecting reproducibility tracking, leakage auto-fixer, and target evaluation correctness. All fixes maintain determinism and use existing SST solutions.

## Issues Fixed

### 1. Missing horizon_minutes Causing COHORT_AWARE Downgrade

**Problem**: `horizon_minutes` was set to `None` if `target_horizon_minutes` was not in locals, causing COHORT_AWARE mode to downgrade to NON_COHORT.

**Root Cause**:** No fallback to extract horizon from target column name when `target_horizon_minutes` is missing.

**Fix**: Added fallback extraction using SST function `resolve_target_horizon_minutes()` in `model_evaluation.py` line ~8173:
```python
# FIX 1: Extract horizon_minutes with fallback using SST function
horizon_minutes_for_ctx = None
if 'target_horizon_minutes' in locals() and target_horizon_minutes is not None:
    horizon_minutes_for_ctx = target_horizon_minutes
elif 'target_column' in locals() and target_column:
    # Fallback: extract from target column name using SST function
    try:
        from TRAINING.common.utils.sst_contract import resolve_target_horizon_minutes
        horizon_minutes_for_ctx = resolve_target_horizon_minutes(target_column)
    except Exception:
        # Last resort: try _extract_horizon
        try:
            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
            leakage_config = _load_leakage_config()
            horizon_minutes_for_ctx = _extract_horizon(target_column, leakage_config)
        except Exception:
            pass
```

**Impact**: Targets like `y_will_peak_mfe_10m_0.002` now have `horizon_minutes` populated correctly, enabling COHORT_AWARE mode.

**Determinism**: Uses SST function `resolve_target_horizon_minutes()` - deterministic regex extraction.

---

### 2. Array Truth Value Errors in Two Locations

**Problem**: `TypeError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()` in two locations:
- Line 4355: When checking if `ctx.X`, `ctx.y`, or `ctx.time_vals` are None
- Line 4417: When checking `if ctx.universe_sig or ctx.symbols:` where `ctx.symbols` is a numpy array

**Root Cause**: Direct boolean checks on numpy arrays trigger ambiguity errors.

**Fix**: Updated `reproducibility_tracker.py` at two locations:

**Location 1 (line ~4355)**: Safely check for None/empty arrays:
```python
# FIX 2: Safely check for None without triggering numpy array boolean ambiguity
import numpy as np
is_fallback = (
    (ctx.X is None or (isinstance(ctx.X, (np.ndarray, list)) and len(ctx.X) == 0)) and
    (ctx.y is None or (isinstance(ctx.y, (np.ndarray, list)) and len(ctx.y) == 0)) and
    (ctx.time_vals is None or (isinstance(ctx.time_vals, (np.ndarray, list)) and len(ctx.time_vals) == 0))
)
```

**Location 2 (line ~4417)**: Safely check `ctx.symbols`:
```python
# FIX 2 (Location 2): Safely check ctx.symbols (may be numpy array)
has_universe_sig = ctx.universe_sig is not None
has_symbols = ctx.symbols is not None and (
    (isinstance(ctx.symbols, (np.ndarray, list)) and len(ctx.symbols) > 0) or
    (not isinstance(ctx.symbols, (np.ndarray, list)) and bool(ctx.symbols))
)
```

**Impact**: Eliminates "ambiguous truth value" errors in reproducibility tracking at both locations.

**Determinism**: Inline None/empty checks - deterministic (no iteration, no filesystem).

---

### 3. ExperimentConfig Type Error in resolve_target_horizon_minutes

**Problem**: `TypeError: argument of type 'ExperimentConfig' is not iterable` when leakage auto-fixer calls `resolve_target_horizon_minutes(target, experiment_config)`.

**Root Cause**: Function expects `Optional[Dict[str, Any]]` but receives `ExperimentConfig` object. Tries to check `'horizon_extraction' in config` which fails because ExperimentConfig is not iterable.

**Fix**: Updated `sst_contract.py` line ~399 to check if config is a dict before using `in` operator:
```python
# FIX 3: Only check config if it's a dict (ExperimentConfig objects don't have horizon_extraction anyway)
if config and isinstance(config, dict) and 'horizon_extraction' in config:
    patterns = config['horizon_extraction'].get('patterns', patterns)
```

**Impact**: Leakage auto-fixer now works correctly. When ExperimentConfig is passed, function uses default patterns (which is correct since ExperimentConfig doesn't have `horizon_extraction`).

**Determinism**: Type check - deterministic.

---

### 4. LOW_N_CS Check Using Wrong Metric and Not View-Aware

**Problem**: 
- `n_cs_valid` was calculated as number of models (9) instead of number of timestamps (~1998)
- LOW_N_CS check was applied to SYMBOL_SPECIFIC view where it doesn't make sense
- Error message "LOW_N_CS (n=9 < 20)" was confusing (shows models, not timestamps)

**Root Cause**: 
- Line 7389: `n_cs_valid = len([m for m in all_scores_by_model if len(all_scores_by_model[m]) > 0])` counts models, not timestamps
- `calculate_composite_score_tstat` doesn't receive view information, so it can't skip LOW_N_CS check for SYMBOL_SPECIFIC view

**Fix**: 

**File 1**: `model_evaluation.py` line ~7389 - Calculate `n_cs_valid` from timestamps:
```python
# FIX 4: n_cs_valid should be number of valid timestamps, not number of models
# Use time_vals if available (deterministic - comes from data loading)
if 'time_vals' in locals() and time_vals is not None and len(time_vals) > 0:
    # Count unique timestamps (deterministic - np.unique returns sorted array)
    n_cs_valid = len(np.unique(time_vals[~np.isnan(time_vals)])) if np.any(~np.isnan(time_vals)) else len(time_vals)
    n_cs_total = n_cs_valid  # Total = valid (after filtering)
else:
    # Fallback: use X length (deterministic - same data always gives same length)
    n_cs_valid = len(X) if 'X' in locals() and X is not None else 0
    n_cs_total = n_cs_valid
```

**File 2**: `composite_score.py` line ~386 - Make LOW_N_CS check view-aware:
```python
# Gate 3: Sample size (hard gate) - only for CROSS_SECTIONAL view
# FIX 4: Make LOW_N_CS check view-aware (only applies to CROSS_SECTIONAL)
view_str = view.value if hasattr(view, 'value') else (view if isinstance(view, str) else None)
if view_str != "SYMBOL_SPECIFIC" and n_slices_valid < min_n_for_ranking:
    invalid_reasons.append(f"LOW_N_CS (n_timestamps={n_slices_valid} < {min_n_for_ranking})")
elif view_str == "SYMBOL_SPECIFIC" and n_slices_valid < min_n_for_ranking:
    # For SYMBOL_SPECIFIC, use different error message (not "CS")
    invalid_reasons.append(f"LOW_N_SAMPLES (n_timestamps={n_slices_valid} < {min_n_for_ranking})")
```

**File 3**: `model_evaluation.py` line ~7537 - Pass view to composite score function:
```python
composite, composite_def, composite_ver, components, scoring_signature, eligibility = calculate_composite_score_tstat(
    # ... other params ...
    view=(view_for_writes.value if hasattr(view_for_writes, 'value') else view_for_writes) if 'view_for_writes' in locals() else (
        (view.value if hasattr(view, 'value') else view) if 'view' in locals() else "CROSS_SECTIONAL"
    ),  # FIX 4: Pass view for view-aware checks
)
```

**Impact**: 
- LOW_N_CS check now uses correct metric (n_timestamps, not n_models)
- LOW_N_CS check only applies to CROSS_SECTIONAL view (not SYMBOL_SPECIFIC)
- Error messages clearly indicate what "n" represents (n_timestamps vs n_symbols)

**Determinism**: 
- `np.unique()` is deterministic (returns sorted unique values)
- `time_vals` is sorted in `prepare_cross_sectional_data_for_ranking()` (lines 871-880)
- `len(X)` is deterministic (same data = same length)

---

### 5. Feature Importances Saved for Invalid Targets

**Problem**: Feature importances were saved unconditionally after every model evaluation, including:
- After CROSS_SECTIONAL evaluation (even if target is invalid)
- After SYMBOL_SPECIFIC evaluation (duplicate save if both views run)
- When target is marked `invalid_for_ranking`

**Root Cause**: `_save_feature_importances()` was called unconditionally at line 6934, before `valid_for_ranking` is determined (line 7527).

**Fix**: Moved feature importances save to after `valid_for_ranking` check in `model_evaluation.py`:

**Line ~6934**: Store reference for later conditional save:
```python
# FIX 5: Feature importances will be saved after valid_for_ranking is determined (see below)
# Store reference for later conditional save
_feature_importances_to_save = {
    'target_column': target_column,
    'symbol': symbol_for_importances,
    'importances': feature_importances,
    'output_dir': output_dir,
    'view': view_for_importances,
    'universe_sig': universe_sig_for_importances,
    'run_identity': identity_for_save,
    'model_metrics': model_metrics,
    'attempt_id': attempt_id if attempt_id is not None else 0,
}
```

**Line ~7549**: Conditional save after validation:
```python
# FIX 5: Save feature importances only if target is valid_for_ranking
if '_feature_importances_to_save' in locals() and _feature_importances_to_save and valid_for_ranking:
    _save_feature_importances(
        _feature_importances_to_save['target_column'],
        _feature_importances_to_save['symbol'],
        _feature_importances_to_save['importances'],
        _feature_importances_to_save['output_dir'],
        view=_feature_importances_to_save['view'],
        universe_sig=_feature_importances_to_save['universe_sig'],
        run_identity=_feature_importances_to_save['run_identity'],
        model_metrics=_feature_importances_to_save['model_metrics'],
        attempt_id=_feature_importances_to_save['attempt_id'],
    )
elif '_feature_importances_to_save' in locals() and _feature_importances_to_save:
    if not valid_for_ranking:
        logger.debug(f"Skipping feature importances save for {target}: invalid_for_ranking")
    elif not _feature_importances_to_save.get('importances'):
        logger.debug(f"Skipping feature importances save for {target}: no importances computed")
```

**Impact**: 
- Feature importances only saved for valid targets
- Reduces clutter in output directories
- Works for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views

**Determinism**: Conditional save based on boolean `valid_for_ranking` - deterministic.

---

## Determinism Analysis

All fixes maintain determinism:

1. **Fix 1**: Uses SST function `resolve_target_horizon_minutes()` - deterministic regex extraction
2. **Fix 2**: Inline None/empty checks - no iteration, no filesystem access
3. **Fix 3**: Type check `isinstance(config, dict)` - deterministic
4. **Fix 4**: Uses `np.unique(time_vals)` and `len(X)` - both deterministic (sorted unique, fixed length)
5. **Fix 5**: Conditional save based on boolean - deterministic

**No new non-deterministic sources introduced.**

## Code Reuse Analysis

All fixes update existing code rather than adding new functions:

1. **Fix 1**: Updates existing `RunContext` creation (adds fallback)
2. **Fix 2**: Updates existing `is_fallback` check (inline, no new function)
3. **Fix 3**: Updates existing `resolve_target_horizon_minutes()` (adds type check)
4. **Fix 4**: Updates existing `n_cs_valid` calculation (uses existing `time_vals`/`X`)
5. **Fix 5**: Moves existing feature importances save (after validation)

**No unnecessary bloat - all changes are minimal updates to existing code paths.**

## Files Changed

- `TRAINING/ranking/predictability/model_evaluation.py` (lines ~8173, ~7389, ~6934, ~7537, ~7549)
- `TRAINING/orchestration/utils/reproducibility_tracker.py` (line ~4355)
- `TRAINING/common/utils/sst_contract.py` (line ~399)
- `TRAINING/ranking/predictability/composite_score.py` (line ~232, ~386)

## Testing

After fix, verify:
- `y_will_peak_mfe_10m_0.002` and similar targets have `horizon_minutes` populated
- No "truth value of an array" errors
- COHORT_AWARE mode works correctly when horizon_minutes is extracted from target name
- No `TypeError: argument of type 'ExperimentConfig' is not iterable` when auto-fix runs
- Leakage auto-fixer can extract horizon from target names without crashing
- LOW_N_CS check uses correct metric (n_timestamps, not n_models)
- LOW_N_CS check only applies to CROSS_SECTIONAL view (not SYMBOL_SPECIFIC)
- Error messages clearly indicate what "n" represents (n_timestamps vs n_symbols)
- Feature importances are saved only once per target (not for every CS/SS evaluation)
- Feature importances are NOT saved for invalid targets
- Works for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
