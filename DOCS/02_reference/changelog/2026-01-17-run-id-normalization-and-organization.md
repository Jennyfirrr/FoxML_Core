# Run ID Normalization and Run Organization Improvements

**Date**: 2026-01-17  
**Category**: Reproducibility, Architecture, Enhancement  
**Impact**: High (normalizes run_id generation across all stages, improves run organization, fixes architectural sharp edges)

## Summary

Implemented comprehensive normalization of `run_id` generation across all pipeline stages and improved run organization to group runs by universe/config signatures instead of just sample count. This ensures deterministic, hash-based run IDs and better organization of comparable runs.

## Problem

### Run ID Inconsistency

Different pipeline stages were generating `run_id` values using different methods:
- **TARGET_RANKING**: Timestamp-based (`"2026-01-17T01:45:14.745352"`)
- **FEATURE_SELECTION**: UUID-based (`"438f828a-5322-48c6-b03a-be4393557eab"`)
- **TRAINING**: Directory name-based (`"intelligent_output_20260117_014407"`)

This inconsistency caused:
- Run matching failures in diff telemetry
- Non-deterministic run IDs (same inputs → different IDs)
- Difficulty tracking runs across stages

### Run Organization Issues

Runs were organized by `n_effective` and `model_family` in directory names:
- Format: `cg-{hash}_n-{sample_size}_fam-{model_family}`
- Problem: Runs with same config but different sample sizes were split into separate directories
- Result: Fragmented organization, harder to find comparable runs

### Architectural Sharp Edges

1. **Boolean Comparability Flags**: No explicit `is_comparable` flag in manifests
2. **Config Signature Inconsistency**: No canonical definition of `config_sig`
3. **Canonicalization Contract**: Needed better handling of floats, sets, unordered lists
4. **Truncation Length Safety**: Short prefixes risked collisions
5. **Directory Parsing**: Hardcoded `startswith()` checks instead of proper parser
6. **Dataset Fingerprint**: Needed to include actual data snapshot identity

## Solution

### Phase 1: Core Run ID Normalization

**New Functions**:
- `derive_run_id_from_identity(run_identity)`: Pure function that derives deterministic `run_id` from `RunIdentity` object
  - Format: `ridv1_{sha256(strict_key + ":" + replicate_key)[:20]}`
  - Hash-based to prevent prefix collisions
  - Raises `ValueError` if `run_identity` not finalized (pure function contract)

- `derive_unstable_run_id(run_instance_id)`: Generates unstable run_id when identity unavailable
  - Format: `rid_unstable_{run_instance_id}`
  - Used as fallback when `run_identity` not available

- `assess_comparability(run_identity, dataset_snapshot_hash, mode)`: Assesses if run is comparable
  - Returns: `(is_comparable: bool, run_id_kind: "stable" | "unstable")`
  - Separates ID derivation from comparability assessment (prevents drift)

- `generate_run_instance_id()`: Generates unique directory name
  - Format: `intelligent_output_YYYYMMDD_HHMMSS_{uuid4()[:8]}`
  - Includes UUID suffix for uniqueness

- `parse_run_instance_dirname(dirname)`: Parses directory names with tolerant handling
  - Handles: old underscore, old dash, new suffix forms
  - Returns `RunInstanceParts` dataclass or `None` for invalid formats

**Updated `create_manifest()`**:
- Now accepts `run_identity`, `dataset_snapshot_hash`, `run_instance_id`, `mode` parameters
- Uses `assess_comparability()` to set `is_comparable` and `run_id_kind` flags
- Stores flags in `manifest.json` for authoritative comparability checking
- Derives `run_id` from `run_identity` if available, otherwise uses `derive_unstable_run_id()`

### Phase 2: Stage-Specific Run ID Fixes

Updated all stages to use normalized run_id generation:
- **FEATURE_SELECTION**: `TRAINING/stability/feature_importance/schema.py`
- **TRAINING**: `TRAINING/training_strategies/reproducibility/schema.py`
- **TARGET_RANKING**: `TRAINING/ranking/target_ranker.py`
- **Training Plan Generator**: `TRAINING/orchestration/training_plan_generator.py`
- **Reproducibility Tracker**: `TRAINING/orchestration/utils/reproducibility_tracker.py`

All stages now:
- Try to derive `run_id` from `run_identity` if available
- Fall back to `derive_unstable_run_id()` if identity missing
- Use consistent error handling pattern

### Phase 3: Run Organization Improvements

**New Directory Format**:
- Old: `cg-{hash}_n-{sample_size}_fam-{model_family}`
- New: `cg-{cg_hash}_u-{universe_sig[:8]}_c-{config_sig[:8]}`
- `cg_hash` = `sha256("u="+universe_sig+";c="+config_sig)[:12]` (derived from u+c, prevents drift)
- `n_effective` moved to run leaf metadata (not in directory name)
- Allows runs with different sample sizes but same config to be grouped together

**New Helper Function**:
- `compute_config_signature(...)`: Canonical definition of config signature
  - Includes: dataset_signature, task_signature, routing_signature, split_signature, **feature_signature** (CRITICAL: different features = different outcomes), hyperparameters_signature, registry_overlay_signature, leakage_filter_version, model_family
  - Excludes: universe_sig (separate), n_effective (outcome, not identity), train_seed (in replicate_key)
  - Always returns 64-char SHA256 hash (even with all None inputs = empty config signature)
  - **sig_version**: Bumped from 1 to 2 when `feature_signature` was added (2026-01-17)

**Updated `ComparisonGroup.to_dir_name()`**:
- Uses new format with `universe_sig` and `config_sig`
- Derives `cg_hash` from u+c to prevent drift
- Uses `compute_config_signature()` for canonical config signature

### Phase 4: Matching Logic Updates

**Updated `diff_telemetry.py`**:
- Replaced `startswith("intelligent_output_")` with `parse_run_instance_dirname()`
- Added `_load_manifest_comparability_flags()`: Loads `is_comparable` and `run_id_kind` from manifest
- Added `_can_runs_be_compared()`: Encapsulates comparability logic
  - Checks manifest flags first (authoritative)
  - Falls back to legacy normalization if flags missing
  - Refuses unstable-vs-stable comparisons
- Updated `_check_comparability()`: Prioritizes manifest flags over legacy heuristics
- Updated `compute_run_hash_with_changes()`: Uses comparability flags for matching

**Updated `intelligent_trainer.py`**:
- Uses `generate_run_instance_id()` for output directory names
- Uses `parse_run_instance_dirname()` for finding previous runs
- Passes `run_identity` to `create_manifest()`

## Changes

### Files Modified

1. **`TRAINING/orchestration/utils/manifest.py`**:
   - Added `derive_run_id_from_identity()` (pure function)
   - Added `derive_unstable_run_id()` (fallback)
   - Added `assess_comparability()` (comparability assessment)
   - Added `generate_run_instance_id()` (directory name generation)
   - Added `parse_run_instance_dirname()` (directory name parser)
   - Updated `create_manifest()` to use comparability assessment and store flags

2. **`TRAINING/orchestration/utils/diff_telemetry/types.py`**:
   - Added `compute_config_signature()` (canonical config signature)
   - Updated `ComparisonGroup.to_dir_name()` to new format

3. **`TRAINING/orchestration/utils/diff_telemetry.py`**:
   - Added `_load_manifest_comparability_flags()` helper
   - Added `_can_runs_be_compared()` helper
   - Updated `_organize_run_by_comparison_group()` to use parser
   - Updated `_check_comparability()` to prioritize manifest flags
   - Updated `compute_run_hash_with_changes()` to use comparability flags

4. **`TRAINING/orchestration/intelligent_trainer.py`**:
   - Updated to use `generate_run_instance_id()` for directory names
   - Updated to use `parse_run_instance_dirname()` for finding previous runs
   - Updated `create_manifest()` calls to pass `run_identity`

5. **`TRAINING/stability/feature_importance/schema.py`**:
   - Updated `from_dict_series()` to use normalized run_id generation

6. **`TRAINING/training_strategies/reproducibility/schema.py`**:
   - Updated `from_training_result()` to use normalized run_id generation

7. **`TRAINING/ranking/target_ranker.py`**:
   - Updated `rank_targets()` to use normalized run_id generation

8. **`TRAINING/orchestration/utils/reproducibility_tracker.py`**:
   - Updated `_save_to_cohort()` to use normalized run_id generation

9. **`TRAINING/orchestration/training_plan_generator.py`**:
   - Updated `create_resolved_config()` to use normalized run_id generation

## Backward Compatibility

✅ **Old runs remain accessible**: Directory parser handles old formats (underscore, dash)  
✅ **Legacy normalization still works**: Falls back to old comparison logic if manifest flags missing  
✅ **Old directory structures preserved**: Not moved or renamed  
✅ **Both structures searched**: Code searches both old (`sample_*` bins) and new (`cg-*` directories)  
✅ **Graceful degradation**: If `run_identity` unavailable, uses unstable run_id (still unique)  
✅ **Function signatures**: All new parameters are optional with defaults

## Edge Cases Handled

- ✅ Missing `run_identity` → Falls back to unstable run_id
- ✅ Unfinalized `run_identity` → Raises `ValueError` (pure function contract)
- ✅ Old directory formats → Parser handles all formats gracefully
- ✅ Missing manifest flags → Falls back to legacy normalization
- ✅ Unstable-vs-stable comparisons → Explicitly refused
- ✅ Missing `registry_overlay_signature` → Uses `getattr()` safely
- ✅ Empty config signature → Returns hash of empty config (deterministic)

## Testing

All files compile successfully:
- ✅ Syntax validation passed
- ✅ Import validation passed
- ✅ Function execution tests passed
- ✅ No linter errors

## Future Work

**Phase 2.4**: Update `ReproducibilityTracker` to check `is_comparable` flag (can be done later if needed)

## Verification

## Bug Fixes (2026-01-17)

### Missing `feature_signature` in Config Signature

**Issue**: `compute_config_signature()` was missing `feature_signature` parameter, even though features are outcome-influencing metadata.

**Impact**: Runs with different feature sets would have the same config signature, causing incorrect grouping in comparison group directories.

**Fix**:
- Added `feature_signature` parameter to `compute_config_signature()`
- Updated `ComparisonGroup.to_dir_name()` to pass `feature_signature` to `compute_config_signature()`
- Bumped `sig_version` from 1 to 2 to reflect the change
- Updated documentation to clarify that features are included

**Files Modified**:
- `TRAINING/orchestration/utils/diff_telemetry/types.py`

## Verification

Before implementing, verified:
- ✅ All call sites updated to use new functions
- ✅ Backward compatibility maintained
- ✅ Error handling in place
- ✅ No breaking changes to existing behavior
- ✅ Syntax checks pass
- ✅ Imports successfully
