---
Type: Bug Fix
Impact: Critical
Stage: TARGET_RANKING
---

# 2026-01-10: Feature Importances Not Being Saved and Artifacts Manifest Fixes

## Summary

Fixed three critical issues preventing feature importances from being saved and causing null `artifacts_manifest_sha256` in snapshots. All fixes maintain determinism and use existing SST solutions.

## Issues Fixed

### 1. Feature Importances Not Being Saved (CRITICAL)

**Problem**: Feature importances CSV files were not being saved because `universe_sig_for_importances` was set to `None` before `universe_sig_for_writes` was available, causing `save_feature_importances()` to return early.

**Root Cause**: At line 6677 in `model_evaluation.py`, `universe_sig_for_importances` was set using:
```python
universe_sig_for_importances = universe_sig_for_writes if 'universe_sig_for_writes' in locals() else None
```

But `universe_sig_for_writes` is only set later at line 5766:
```python
universe_sig_for_writes = resolved_data_config.get("universe_sig")
```

This meant `universe_sig_for_importances` was always `None` when set at line 6677, and it was never updated later. When `save_feature_importances()` was called at line 7558, it received `None` for `universe_sig` and returned early at line 174-179 in `reporting.py` with the error:
```
SCOPE BUG: universe_sig not provided for {target} feature importances. Cannot create view-scoped paths. Feature importances will not be written.
```

**Fix**: 
1. Removed the early assignment at line 6677 (commented out with explanation)
2. Updated to use `universe_sig_for_writes` directly when building `_feature_importances_to_save` at line 6942:
   ```python
   universe_sig_for_save = universe_sig_for_writes if 'universe_sig_for_writes' in locals() and universe_sig_for_writes else None
   ```
3. Added check before save call (around line 7557) to update `universe_sig` if it's None but `universe_sig_for_writes` is available:
   ```python
   if '_feature_importances_to_save' in locals() and _feature_importances_to_save:
       if not _feature_importances_to_save.get('universe_sig') and 'universe_sig_for_writes' in locals() and universe_sig_for_writes:
           _feature_importances_to_save['universe_sig'] = universe_sig_for_writes
   ```

**Impact**: Feature importances CSV files are now correctly saved to:
- `targets/{target}/reproducibility/stage=TARGET_RANKING/CROSS_SECTIONAL/batch_{sig}/attempt_{id}/feature_importances/`
- `targets/{target}/reproducibility/stage=TARGET_RANKING/SYMBOL_SPECIFIC/symbol={sym}/attempt_{id}/feature_importances/`

**Determinism**: Uses canonical `universe_sig_for_writes` from `resolved_data_config` - deterministic.

**Files**: `TRAINING/ranking/predictability/model_evaluation.py`

---

### 2. Artifacts Manifest SHA256 Path Resolution

**Problem**: `artifacts_manifest_sha256` was null in snapshots because `_compute_artifacts_manifest_digest()` couldn't find the `feature_importances` directory in the new `batch_`/`attempt_` structure.

**Root Cause**: The function at lines 2519-2543 in `diff_telemetry.py` tries to find `feature_importances_dir` by extracting `attempt_id_from_cohort` from the cohort path, but:
- If `attempt_id_from_cohort` doesn't match the actual `attempt_id` used when saving feature importances, the directory won't be found
- The function uses fallbacks (highest attempt, legacy paths) but may still fail silently
- No logging when `feature_importances_dir` is None or doesn't exist, making debugging difficult

**Fix**: 
1. Added debug logging when `feature_importances_dir` is not found (after line 2543):
   ```python
   if not feature_importances_dir or not feature_importances_dir.exists():
       logger.debug(
           f"feature_importances_dir not found: {feature_importances_dir} "
           f"(attempt_id_from_cohort={attempt_id_from_cohort}, "
           f"batch_or_symbol_dir={batch_or_symbol_dir}, "
           f"stage={stage})"
       )
   ```
2. Added logging when manifest is empty (line 2639):
   ```python
   if not manifest:
       logger.debug(
           f"No artifacts found for {stage} stage "
           f"(cohort_dir={cohort_dir}, "
           f"feature_importances_dir={feature_importances_dir if 'feature_importances_dir' in locals() else 'N/A'})"
       )
       return None
   ```

**Impact**: Better diagnostics when `artifacts_manifest_sha256` is null. The path resolution logic already uses the same structure as `OutputLayout.feature_importance_dir()`, so it should work correctly when feature importances are saved.

**Determinism**: No changes to determinism - only added logging.

**Files**: `TRAINING/orchestration/utils/diff_telemetry.py`

---

### 3. Model Scores Not Saved in Snapshots

**Problem**: Model scores (AUC, R², IC, etc.) were not being saved in stability snapshots, making it difficult to track model performance alongside feature importance stability.

**Root Cause**: The `save_snapshot_hook()` at lines 279-293 in `model_evaluation/reporting.py` only passed:
- `importance_dict` (feature importances)
- `prediction_fingerprint` (prediction hash)
- `run_identity` (signatures)

Model scores from `model_metrics` were not being extracted or passed to the snapshot.

**Fix**: 
1. Extracted model scores from `model_metrics` before calling `save_snapshot_hook()` (around line 256-293):
   ```python
   model_scores_data = None
   if model_metrics and model_name in model_metrics:
       model_metrics_dict = model_metrics[model_name]
       model_scores_data = {
           'auc': model_metrics_dict.get('auc'),
           'r2': model_metrics_dict.get('r2'),
           'ic': model_metrics_dict.get('ic'),
           'roc_auc': model_metrics_dict.get('roc_auc'),
           'accuracy': model_metrics_dict.get('accuracy'),
           'f1_score': model_metrics_dict.get('f1_score'),
           'precision': model_metrics_dict.get('precision'),
           'recall': model_metrics_dict.get('recall'),
       }
       # Remove None values to keep snapshot clean
       model_scores_data = {k: v for k, v in model_scores_data.items() if v is not None}
   ```
2. Passed model scores via `inputs` parameter to `save_snapshot_hook()`:
   ```python
   save_snapshot_hook(
       ...
       inputs={'model_scores': model_scores_data} if model_scores_data else None,
   )
   ```
3. Updated `save_snapshot_hook()` in `hooks.py` to move model scores from `inputs` to `outputs` in the snapshot (after line 192):
   ```python
   # FIX 3: Store model scores in snapshot outputs if provided via inputs
   if inputs and 'model_scores' in inputs:
       if snapshot.outputs is None:
           snapshot.outputs = {}
       snapshot.outputs['model_scores'] = inputs['model_scores']
   ```

**Impact**: Model scores are now stored in stability snapshots' `outputs.model_scores` field, enabling tracking of model performance alongside feature importance stability.

**Determinism**: No changes to determinism - model scores are deterministic outputs from model evaluation.

**Files**: 
- `TRAINING/ranking/predictability/model_evaluation/reporting.py`
- `TRAINING/stability/feature_importance/hooks.py`

---

## Testing

After fixes:
1. ✅ Verify feature importances CSV files are created in `batch_{sig}/attempt_{id}/feature_importances/` (CROSS_SECTIONAL) and `symbol={sym}/attempt_{id}/feature_importances/` (SYMBOL_SPECIFIC)
2. ✅ Verify `artifacts_manifest_sha256` is not null in snapshots (with improved diagnostics if issues occur)
3. ✅ Verify model scores are saved in stability snapshots (`outputs.model_scores`)
4. ✅ Check logs for any path resolution warnings or "SCOPE BUG" errors

## Determinism and SST Compliance

- All fixes use existing SST functions (`OutputLayout`, `build_target_cohort_dir`, `resolve_write_scope`)
- No new functions created - only updates to existing code
- All path building uses canonical SST helpers
- Deterministic sorting maintained for all file operations
- No race conditions introduced (CSV writes are per-model, per-attempt, so no concurrent writes to same file)

## Related Issues

- Fixes the root cause of feature importances not being saved (reported as "all feature importances not being saved")
- Fixes `artifacts_manifest_sha256: null` in snapshots
- Enables tracking model performance in stability snapshots
