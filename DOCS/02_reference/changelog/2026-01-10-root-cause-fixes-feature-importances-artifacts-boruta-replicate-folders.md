---
Type: Bug Fix
Impact: Critical
Stage: TARGET_RANKING, FEATURE_SELECTION
---

# 2026-01-10: Root Cause Fixes - Feature Importances, Artifacts Manifest, Boruta Config, and Replicate Folders

## Summary

Fixed four critical root cause issues: feature importances not being saved (unreliable `locals()` check), artifacts_manifest_sha256 being null (manual path traversal), Boruta not using ranking configs, and replicate folders in wrong location. All fixes use existing SST functions and maintain determinism.

## Issues Fixed

### 1. Feature Importances Not Being Saved (ROOT CAUSE)

**Problem**: Feature importances CSV files were not being saved because the check `'universe_sig_for_writes' in locals() and universe_sig_for_writes` at line 6939 was unreliable. Even though `universe_sig_for_writes` was set at line 5766, the `locals()` check could fail, causing `universe_sig_for_save` to be `None`.

**Root Cause**: The `locals()` check is unreliable in Python, especially in complex function scopes. The variable `universe_sig_for_writes` exists but the check `'universe_sig_for_writes' in locals()` might fail, or the variable might be `None` even when `resolved_data_config.get("universe_sig")` has a value.

**Fix**: 
- Removed unreliable `locals()` check at line 6939
- Use variable directly with try/except: `try: universe_sig_for_save = universe_sig_for_writes except NameError: pass`
- Added explicit fallback to `resolved_data_config.get('universe_sig')` if `universe_sig_for_save` is still `None`
- Improved fallback at line 7564-7566 to also check `resolved_data_config.get('universe_sig')` if `universe_sig_for_writes` is None

**Files**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 6939, 7564-7566)

**Impact**: Feature importances are now correctly saved with proper `universe_sig` resolution.

### 2. Artifacts Manifest SHA256 Null

**Problem**: `artifacts_manifest_sha256` was null in snapshots because `_compute_artifacts_manifest_digest()` used manual path traversal that failed to find `feature_importances_dir` in the new `batch_*/attempt_*/feature_importances/` structure.

**Root Cause**: Manual path traversal (lines 2478-2554) was fragile and didn't handle all path structures correctly. The code tried to manually construct paths using `cohort_dir.parent.parent.parent` which could fail if the structure changed.

**Fix**: 
- Replaced manual path traversal with existing SST function `get_scoped_artifact_dir()`
- Uses `parse_reproducibility_path()` to extract path components (target, view, symbol, universe_sig)
- Uses `parse_attempt_id_from_cohort_dir()` to extract attempt_id
- Uses `run_root()` to get base output directory
- Uses `normalize_target_name()` for consistent target name handling

**Files**: `TRAINING/orchestration/utils/diff_telemetry.py` (lines 2478-2554)

**Impact**: `artifacts_manifest_sha256` is now correctly computed using SST path resolution, ensuring consistency and maintainability.

### 3. Boruta Not Using Ranking Configs

**Problem**: Boruta time limits and thresholds were loaded from `preprocessing_config` instead of `multi_model_config` (ranking configs), ignoring user updates to `CONFIG/ranking/targets/multi_model.yaml` or `CONFIG/ranking/features/multi_model.yaml`.

**Root Cause**: Lines 4681 and 4752 only checked `preprocessing.multi_model_feature_selection.boruta` config path, ignoring the `multi_model_config` parameter that contains updated values from ranking configs.

**Fix**:
- Updated threshold loading (line 4679-4687) to use `get_model_config('boruta', multi_model_config)` first, then fallback to `preprocessing_config`
- Updated time limit loading (line 4750-4756) to use `boruta_config` from `multi_model_config` first, then fallback to `preprocessing_config`
- Uses existing `get_model_config()` SST function for consistency

**Files**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 4679-4687, 4750-4756)

**Impact**: TARGET_RANKING now uses ranking configs (20 minutes for targets, 45 minutes for features) instead of preprocessing config defaults (10 minutes).

### 4. Replicate Folders Missing/Wrong Location

**Problem**: Replicate folders (containing stability snapshots) were always created in `attempt_0/` even when the actual attempt was different, because `get_snapshot_base_dir()` hardcoded `attempt_id=0`.

**Root Cause**: `attempt_id` was available in `save_feature_importances()` but not passed through the chain to `get_snapshot_base_dir()`, causing replicate folders to be created in the wrong location.

**Fix**:
- Added `attempt_id` parameter to `get_snapshot_base_dir()` (line 276 in `io.py`)
- Added `attempt_id` parameter to `save_snapshot_hook()` (line 28 in `hooks.py`)
- Pass `attempt_id` from `save_feature_importances()` → `save_snapshot_hook()` → `get_snapshot_base_dir()`
- Use `attempt_id` when calling `ensure_scoped_artifact_dir()` and `get_scoped_artifact_dir()` (existing SST functions)

**Files**: 
- `TRAINING/stability/feature_importance/io.py` (lines 276, 337-340, 344-347)
- `TRAINING/stability/feature_importance/hooks.py` (lines 28, 205-209)
- `TRAINING/ranking/predictability/model_evaluation/reporting.py` (line 297)

**Impact**: Replicate folders are now created in correct `attempt_{id}/` directories, preserving attempt-specific stability snapshots.

## Config Updates

### Boruta Time Limit for TARGET_RANKING

Updated `CONFIG/ranking/targets/multi_model.yaml` to set Boruta `max_time_minutes: 20` for TARGET_RANKING stage:
- `max_time_minutes: 20` - Time budget for Boruta fit in TARGET_RANKING
- `max_features_threshold: 200` - Skip if too many features
- `max_samples_threshold: 200000` - Skip if too many samples

**Files**: `CONFIG/ranking/targets/multi_model.yaml` (lines 193-196)

## Determinism Analysis

All fixes maintain determinism:
- Use existing SST functions (`get_model_config`, `get_scoped_artifact_dir`, `parse_reproducibility_path`, etc.)
- No new non-deterministic sources introduced
- Path resolution uses SST functions for consistency
- Config loading prioritizes passed-in configs (deterministic) over global configs

## Testing

After fixes:
1. Verify feature importances CSV files are saved (check for "SCOPE BUG" errors in logs)
2. Verify `universe_sig` is not None when `save_feature_importances()` is called
3. Verify `artifacts_manifest_sha256` is not null in snapshots (check snapshot.json files)
4. Verify Boruta uses `max_time_minutes: 20` from `CONFIG/ranking/targets/multi_model.yaml` for TARGET_RANKING
5. Verify Boruta uses `max_time_minutes: 45` from `CONFIG/ranking/features/multi_model.yaml` for FEATURE_SELECTION
6. Verify replicate folders are created in correct `attempt_{id}/` directories
7. Verify replicate folders contain snapshots with `replicate_key/strict_key.json` structure
