# Output Structure Cleanup and Backward Compatibility Fixes

**Date**: 2026-01-10  
**Type**: Infrastructure, Backward Compatibility, Bug Fix  
**Impact**: High - Cleaner output structure, preserved artifact history, comprehensive backward compatibility, scoring schema version consistency

## Summary

Comprehensive cleanup of output directory structure with human-readable batch IDs, per-attempt artifact storage, and extensive backward compatibility fixes. All readers now correctly handle nested `batch_*/attempt_*/cohort=*` structures.

## Key Changes

### 1. Human-Readable Batch IDs

**Change**: Replaced `universe={long_hash}` with `batch_{short_hash[:12]}` for CROSS_SECTIONAL directories.

**Files**: 
- `TRAINING/orchestration/utils/target_first_paths.py` - `build_target_cohort_dir()`, `target_repro_dir()`, `parse_reproducibility_path()`
- `TRAINING/orchestration/utils/output_layout.py` - `repro_dir()` property
- `TRAINING/orchestration/utils/reproducibility_tracker.py` - `_get_cohort_dir_v2()` (unused, but updated for consistency)

**Impact**: Cleaner, more readable directory names while maintaining determinism. Example: `batch_f517a23ce02c` instead of `universe=f517a23ce02cdcad4887b95107f165cc69f15796ccfd07c3b8e1466fbd2102f5`.

**Backward Compatibility**: All path parsing functions check both `universe=*` (legacy) and `batch_*` (new) formats, preferring legacy for consistency.

### 2. Per-Attempt Artifacts

**Change**: Moved `feature_importances`, `featureset_artifacts`, and `feature_exclusions` into `attempt_{id}/` subdirectories.

**Rationale**: These artifacts can change during auto-fix reruns. Storing them per-attempt preserves full history across leakage detection and auto-fix cycles.

**Files Modified**:
- `TRAINING/orchestration/utils/target_first_paths.py`:
  - `get_scoped_artifact_dir()` - Added `attempt_id` parameter, includes `attempt_{id}/` in path
  - `ensure_scoped_artifact_dir()` - Added `attempt_id` parameter, passes to `get_scoped_artifact_dir()`
- `TRAINING/orchestration/utils/output_layout.py`:
  - `OutputLayout.__init__()` - Added `attempt_id` parameter (defaults to 0)
  - `OutputLayout.feature_importance_dir()` - Returns `repro_dir() / attempt_{id} / feature_importances/`
- `TRAINING/ranking/predictability/model_evaluation.py`:
  - All `ensure_scoped_artifact_dir()` calls for `feature_exclusions` and `featureset_artifacts` now pass `attempt_id`
  - `_save_feature_importances()` call passes `attempt_id`
- `TRAINING/ranking/predictability/model_evaluation/reporting.py`:
  - `save_feature_importances()` - Added `attempt_id` parameter, passes to `OutputLayout`
- `TRAINING/ranking/predictability/leakage_detection/reporting.py`:
  - `save_feature_importances()` - Added `attempt_id` parameter, passes to `OutputLayout`
- `TRAINING/ranking/predictability/leakage_detection.py`:
  - `_save_feature_importances()` - Added `attempt_id` parameter, passes to `OutputLayout`
- `TRAINING/ranking/feature_selection_reporting.py`:
  - `save_feature_importances_for_reproducibility()` - Added `attempt_id` parameter, passes to `OutputLayout`
- `TRAINING/ranking/shared_ranking_harness.py`:
  - `ensure_scoped_artifact_dir()` call for `feature_exclusions` passes `attempt_id=0` (default, attempt_id not available in this context)
- `TRAINING/stability/feature_importance/io.py`:
  - Both `ensure_scoped_artifact_dir()` and `get_scoped_artifact_dir()` calls pass `attempt_id=0` (default, attempt_id not available in this context)

**Impact**: Artifacts no longer overwritten on reruns. Full history preserved across auto-fix cycles. Structure: `batch_{sig}/attempt_{id}/feature_importances/` instead of `batch_{sig}/feature_importances/`.

### 3. Fixed `find_cohort_dir_by_id()` for CROSS_SECTIONAL

**Change**: Updated to search in `batch_*/attempt_*/cohort={id}/` structure instead of directly in `view_dir`.

**File**: `TRAINING/orchestration/utils/target_first_paths.py`

**Details**:
- Checks both `universe=*` (legacy) and `batch_*` (new) formats
- Searches within each `batch_/universe=` directory for `attempt_*` subdirectories
- Numerically sorts attempt directories (not lexicographic: `attempt_2` before `attempt_10`)
- Falls back to legacy structure (direct `cohort=` under `batch_/universe=`) if no `attempt_*` level found

**Impact**: Correctly locates cohorts in new nested structure. Backward compatible with legacy structures.

### 4. Backward Compatibility: Fixed All `iterdir()` Usages

**Change**: Replaced direct `iterdir()` scans with `rglob("cohort=*")` to handle nested structures.

**Files Fixed**:
- `TRAINING/common/utils/trend_analyzer.py` (2 fixes):
  - Line 236: SYMBOL_SPECIFIC view cohort scanning
  - Line 304: CROSS_SECTIONAL view cohort scanning
- `TRAINING/common/utils/metrics.py` (1 fix):
  - Line 1050: SYMBOL_SPECIFIC view cohort scanning
- `TRAINING/orchestration/metrics_aggregator.py` (3 fixes):
  - Line 260: CROSS_SECTIONAL fallback path
  - Line 557: SYMBOL_SPECIFIC fallback path
  - Line 637: CROSS_SECTIONAL fallback path
- `TRAINING/training_strategies/reproducibility/io.py` (3 fixes):
  - Line 720: CROSS_SECTIONAL view cohort scanning
  - Line 738: SYMBOL_SPECIFIC aggregated snapshots
  - Line 752: SYMBOL_SPECIFIC per-symbol snapshots
- `TRAINING/orchestration/intelligent_trainer.py` (3 fixes):
  - Line 687: CROSS_SECTIONAL view cohort scanning
  - Line 739: Legacy structure cohort scanning
  - Line 822: CROSS_SECTIONAL view cohort scanning

**Impact**: All readers now correctly find cohorts in both new (`batch_*/attempt_*/cohort=*`) and legacy (direct `cohort=*`) structures. No breaking changes.

### 5. Fixed `drift.json` Path Construction

**Change**: Updated to use `build_target_cohort_dir()` instead of direct path construction.

**File**: `TRAINING/orchestration/utils/reproducibility_tracker.py`

**Details**:
- Extracts `attempt_id` from `additional_data` or defaults to 0
- Extracts `universe_sig` from `additional_data`, `cohort_metadata`, or `cs_config`
- Uses `build_target_cohort_dir()` for canonical path construction (includes `batch_` and `attempt_` levels)

**Impact**: Drift files written to correct nested structure. Maintains consistency with cohort directory structure.

### 6. Fixed `manifest.py` Cohort Scanning

**Change**: Updated to use `rglob("cohort=*")` instead of separate `iterdir()` loops for SYMBOL_SPECIFIC and CROSS_SECTIONAL.

**File**: `TRAINING/orchestration/utils/manifest.py`

**Details**:
- Single unified loop using `rglob("cohort=*")` for both views
- Extracts symbol from path parts if present
- Handles nested `batch_*/attempt_*/cohort=*` structure automatically

**Impact**: Manifest correctly discovers all cohorts in nested structure. Simpler, more maintainable code.

### 7. Fixed `artifacts_manifest_sha256` Lookup

**Change**: Updated to find artifacts in `attempt_*/feature_importances/` structure.

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Details**:
- Extracts `attempt_id` from `cohort_dir` path using `parse_attempt_id_from_cohort_dir()`
- For SYMBOL_SPECIFIC: Looks in `symbol=*/attempt_{id}/feature_importances/`
- For CROSS_SECTIONAL: Looks in `batch_*/attempt_{id}/feature_importances/` or `universe=*/attempt_{id}/feature_importances/`
- Falls back to highest `attempt_*` directory (numeric sort) or legacy structure if specific attempt not found

**Impact**: Artifact manifest hash correctly computed for new structure. Maintains backward compatibility.

### 8. Fixed `early_universe_sig` Fallback for Feature Exclusions

**Change**: Added fallback to compute `universe_sig` from `symbols` if `run_identity.dataset_signature` is not available.

**File**: `TRAINING/ranking/predictability/model_evaluation.py`

**Details**:
- If `early_universe_sig` is None, computes it from `symbols` using `compute_universe_signature()`
- Ensures `batch_` directory is created even when `run_identity` is not available

**Impact**: Feature exclusions now correctly saved to `batch_*/attempt_*/feature_exclusions/` even when `run_identity` is missing.

### 9. Fixed `_sanitize_for_json` Import Error

**Change**: Added `_sanitize_for_json` to `diff_telemetry/__init__.py` exports.

**File**: `TRAINING/orchestration/utils/diff_telemetry/__init__.py`

**Details**:
- Exported `_sanitize_for_json` from parent `diff_telemetry.py` module
- Added to `__all__` list for proper module exports

**Impact**: Resolves "cannot import name '_sanitize_for_json'" error in `run_context.py` when saving stage transitions.

### 10. Fixed Duplicate Cohort Folders and Missing artifacts_manifest_sha256

**Change**: Fixed duplicate cohort folder creation and null `artifacts_manifest_sha256` in snapshots.

**Files**: 
- `TRAINING/orchestration/utils/reproducibility_tracker.py` - Fixed `log_run()` and `save_snapshot()` path construction
- `TRAINING/orchestration/utils/diff_telemetry.py` - Fixed path resolution and feature_importances lookup
- `TRAINING/common/utils/metrics.py` - Added `attempt_id` extraction

**Details**:
- **Root Cause**: `log_run()` created cohort directories manually (line 4672-4680) without using `build_target_cohort_dir()`, missing `batch_` and `attempt_` levels. This created duplicate cohorts:
  - Wrong: `CROSS_SECTIONAL/cohort=...` (missing batch_ and attempt_)
  - Wrong: `SYMBOL_SPECIFIC/symbol={sym}/cohort=...` (missing attempt_)
  - Correct: `CROSS_SECTIONAL/batch_{sig}/attempt_{id}/cohort=...`
  - Correct: `SYMBOL_SPECIFIC/symbol={sym}/attempt_{id}/cohort=...`

- **Path Resolution Bug**: `_compute_artifacts_manifest_digest()` at line 2455 incorrectly set `view_dir = cohort_dir.parent.parent` when `attempt_` present:
  - For CROSS_SECTIONAL: `cohort_dir.parent.parent` = `batch_{sig}/` (NOT CROSS_SECTIONAL!)
  - Should be: `cohort_dir.parent.parent.parent` = `CROSS_SECTIONAL/`
  - This caused `artifacts_manifest_sha256` to be null because artifacts couldn't be found

- **Feature Importances Lookup Bug**: Line 2477 checked `view_dir.name.startswith('symbol=')` but `view_dir` is `CROSS_SECTIONAL/` or `SYMBOL_SPECIFIC/`, not `symbol=.../`. Should check `batch_or_symbol_dir` instead.

**Fixes**:
1. **reproducibility_tracker.py**:
   - `log_run()` (line 4669-4691): Replaced manual path construction with `build_target_cohort_dir()` using extracted `universe_sig` and `attempt_id`
   - `save_snapshot()` (line 2976-3007): Added canonical path builder with `universe_sig` and `attempt_id` extraction

2. **diff_telemetry.py**:
   - `_compute_artifacts_manifest_digest()` (line 2454-2469): Fixed path resolution to correctly identify `view_dir` (3 levels up) and `batch_or_symbol_dir` (2 levels up)
   - Feature importances lookup (line 2488-2513): Changed to use `batch_or_symbol_dir` instead of `view_dir` for directory checks

3. **metrics.py**:
   - Added `attempt_id` extraction from cohort_dir path (line 398-402) for better traceability

**Impact**: 
- Eliminates duplicate cohort folders in both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- All cohorts now correctly stored in `batch_*/attempt_*/cohort=...` or `symbol=*/attempt_*/cohort=...` structure
- `artifacts_manifest_sha256` now correctly computed and no longer null in snapshots
- Eliminates "Metadata files missing" and "required files are missing" warnings caused by path mismatches
- Determinism maintained (all path operations use deterministic string checks and sorted filesystem iteration)

### 11. Fixed Universe Signature Consistency for Batch Folders

**Change**: Ensured all artifacts use canonical `universe_sig_for_writes` from `resolved_data_config`, preventing multiple `batch_` folders for the same universe.

**Files**: 
- `TRAINING/ranking/predictability/model_evaluation.py` - Updated universe_sig consistency logic
- `TRAINING/orchestration/utils/target_first_paths.py` - Fixed `build_target_cohort_dir()` to always require batch_ level

**Details**:
- **Root Cause**: `early_universe_sig` was computed from requested `symbols` parameter, while `universe_sig_for_writes` comes from `resolved_data_config` (based on actually loaded symbols). If some symbols failed to load, these differed, causing artifacts to be written to different `batch_` folders.
- **Fix**: Updated `early_universe_sig` and `train_universe_sig` to match canonical `universe_sig_for_writes` after `resolved_data_config` is available (line 5770-5776).
- **File Movement**: If `feature_exclusions` were initially saved to wrong `batch_` directory (due to `universe_sig` mismatch), the file is now moved to the correct canonical directory (line 5778-5790).
- **Artifact Consistency**: All artifacts (`feature_exclusions`, `featureset_artifacts`, `feature_importances`) now use `universe_sig_for_writes` when available.
- **CROSS_SECTIONAL Batch Requirement**: Fixed `build_target_cohort_dir()` to always add `batch_` level for CROSS_SECTIONAL, extracting `universe_sig` from `cohort_id` if not provided (line 1198-1207).

**Impact**: Eliminates duplicate `batch_` folders for the same universe. All artifacts for the same loaded symbols are now consistently stored in the same `batch_{universe_sig[:12]}` folder, regardless of when they're saved during the pipeline.

### 12. Fixed OutputLayout attempt_id Attribute Error

**Change**: Fixed `'OutputLayout' object has no attribute 'attempt_id'` error when saving feature importances.

**Files**: 
- `TRAINING/orchestration/utils/output_layout.py` - Added `self.attempt_id` storage in `__init__()`
- `TRAINING/ranking/multi_model_feature_selection.py` - Added `attempt_id=0` to OutputLayout instantiations

**Details**:
- `OutputLayout.__init__()` accepted `attempt_id` parameter but didn't store it as `self.attempt_id`
- `feature_importance_dir()` method accessed `self.attempt_id` causing AttributeError
- Fixed by storing `self.attempt_id = attempt_id if attempt_id is not None else 0` in `__init__()`
- Added missing `attempt_id=0` to FEATURE_SELECTION OutputLayout instantiations in `multi_model_feature_selection.py`

**Impact**: Feature importances now save correctly for all stages (TARGET_RANKING, FEATURE_SELECTION) and views (CROSS_SECTIONAL, SYMBOL_SPECIFIC). Per-attempt artifact paths are now consistent across the codebase.

### 13. Fixed Scoring Schema Version Inconsistency

**Change**: Fixed inconsistency where `snapshot.json` had `scoring_schema_version: "1.1"` but `metrics.json` had `schema.scoring: "1.2"`.

**Root Cause**: `normalize_snapshot()` in `diff_telemetry.py` was looking for `outputs['metrics'].get('scoring_schema_version')`, but the actual structure stores it at `outputs['metrics']['schema']['scoring']` (as created by `build_clean_metrics_dict()` in `metrics_schema.py`).

**Files**: 
- `TRAINING/orchestration/utils/diff_telemetry.py` - Updated `normalize_snapshot()` (lines 1013-1060)

**Details**:
- Updated default `scoring_schema_version` from `"1.1"` to `"1.2"` to match `get_scoring_schema_version()` (current version)
- Primary extraction now checks nested `schema.scoring` path first: `outputs['metrics']['schema']['scoring']`
- Added fallback to top-level `scoring_schema_version` for backward compatibility with legacy data
- Applied same fix to `run_data` and `additional_data` fallback paths
- Final fallback updated to `"1.2"` instead of `"1.1"`

**Impact**: `snapshot.json` top-level `scoring_schema_version` now correctly matches `outputs.metrics.schema.scoring` and `metrics.json` `schema.scoring`. All three will consistently show `"1.2"` (or whatever version is in config).

## Directory Structure

### New Structure (CROSS_SECTIONAL)

```
targets/{target}/reproducibility/stage=TARGET_RANKING/CROSS_SECTIONAL/
└── batch_{universe_sig[:12]}/              ✅ Human-readable batch ID
    ├── attempt_0/
    │   ├── cohort=cs_2025Q3_ef91e9db233a_min_cs3_max2000_v1_3d940389/
    │   │   ├── metadata.json
    │   │   ├── metrics.json
    │   │   ├── snapshot.json
    │   │   └── diff_prev.json
    │   ├── feature_importances/            ✅ Per-attempt artifacts
    │   │   ├── lightgbm_importances.csv
    │   │   └── ...
    │   ├── featureset_artifacts/           ✅ Per-attempt artifacts
    │   │   └── featureset_post_prune.json
    │   └── feature_exclusions/             ✅ Per-attempt artifacts
    │       └── fwd_ret_10m_exclusions.yaml
    └── attempt_1/                          ✅ Preserves history across reruns
        └── ...
```

### New Structure (SYMBOL_SPECIFIC)

```
targets/{target}/reproducibility/stage=TARGET_RANKING/SYMBOL_SPECIFIC/
└── symbol=AAPL/
    ├── attempt_0/
    │   ├── cohort=sy_2025Q3_23692b3418d3_min_cs1_max2000_v1_d4d72908/
    │   │   ├── metadata.json
    │   │   ├── metrics.json
    │   │   └── snapshot.json
    │   ├── feature_importances/            ✅ Per-attempt artifacts
    │   │   └── ...
    │   └── feature_exclusions/             ✅ Per-attempt artifacts
    │       └── fwd_ret_10m_exclusions.yaml
    └── attempt_1/                          ✅ Preserves history across reruns
        └── ...
```

## Backward Compatibility

All changes maintain full backward compatibility:

1. **Path Parsing**: All parsers check both `universe=*` (legacy) and `batch_*` (new) formats
2. **Cohort Discovery**: All readers use `rglob("cohort=*")` which finds cohorts in both nested and flat structures
3. **Artifact Lookup**: All artifact lookups have fallback logic for legacy structures
4. **Attempt ID**: Defaults to `attempt_0` when not provided, ensuring existing code continues to work

## Determinism Maintained

All changes preserve determinism:

- `batch_` prefix uses deterministic slice: `universe_sig[:12]`
- All sorting uses semantic keys (attempt_id numeric, cohort_id, stable paths)
- No `mtime`-based selection
- All `rglob()` calls are deterministic (filesystem order is stable)

## Testing Recommendations

1. **Backward Compatibility**: Verify existing runs with legacy `universe=*` structure still work
2. **New Structure**: Verify new runs create `batch_*` directories correctly
3. **Per-Attempt Artifacts**: Verify artifacts are preserved across auto-fix reruns
4. **Cohort Discovery**: Verify all readers find cohorts in nested structure
5. **Feature Exclusions**: Verify `fwd_ret_10m_exclusions.yaml` is created in `batch_*/attempt_*/feature_exclusions/`

## Related Changes

- Builds on: `2026-01-10-complete-non-determinism-elimination.md` (determinism fixes)
- Completes: Output directory structure cleanup and organization
- Enables: Clean artifact history preservation across auto-fix cycles
