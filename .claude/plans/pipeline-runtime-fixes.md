# Pipeline Runtime Fixes (Demo Run Audit)

**Branch:** `fix/pipeline-nonetype-and-catboost`
**Date:** 2026-02-13
**Source:** End-to-end demo run audit (`--experiment-config demo`)

## Issues Found

### 1. NoneType `.get()` in feature_selector.py:3286
- **Status:** [x] FIXED
- **Severity:** MEDIUM (non-critical logging, wrapped in try/except)
- **Error:** `'NoneType' object has no attribute 'get'`
- **Location:** `TRAINING/ranking/feature_selector.py:3286`
- **Cause:** `cs_stability_results` can be None, then `.get('status')` is called on it
- **Fix:** Guard with `_cs_results = cs_stability_results or {}`

### 2. NoneType `.get()` in training.py:2006-2007
- **Status:** [x] FIXED
- **Severity:** MEDIUM (non-critical logging)
- **Error:** `'NoneType' object has no attribute 'get'`
- **Location:** `TRAINING/training_strategies/execution/training.py:2006-2007`
- **Cause:** `training_result['best_metrics']` can be None
- **Fix:** Guard with `(training_result.get('best_metrics') or {})`

### 3. NoneType `.get()` in ranking.py → log_comparison
- **Status:** [x] FIXED (multiple locations)
- **Severity:** MEDIUM (repro tracking, fail-open)
- **Error:** `Reproducibility tracking failed for fwd_ret_60: 'NoneType' object has no attribute 'get'`
- **Cause:** Multiple `.get('key', {}).get(...)` chains fail when nested value is explicitly None
- **Fixes applied:**
  - `ranking.py:4329` — guard `metrics.get('prediction_fingerprint')` (was `'prediction_fingerprint' in metrics`)
  - `reproducibility_tracker.py:1095,1115-1121` — `(x.get('cs_config') or {})` / `(x.get('date_range') or {})`
  - `logging_api.py:418,1490-1491` — same `or {}` pattern for `date_range` chains

### 4. CatBoost "Verbose period should be nonnegative"
- **Status:** [x] FIXED
- **Severity:** HIGH (all CatBoost training fails)
- **Error:** `_catboost.CatBoostError: Verbose period should be nonnegative`
- **Location:** `TRAINING/ranking/predictability/model_evaluation/training.py:3285-3297`
- **Cause:** `verbose=0` is being passed but CatBoost treats it as verbose_period=0 internally; newer CatBoost versions reject this
- **Fix:** Replace `verbose=0` with `logging_level='Silent'` and remove `verbose` key

### 5. `horizon_minutes is None` warnings throughout ranking
- **Status:** [x] FIXED
- **Severity:** LOW (warning only, falls back to defaults)
- **Error:** `horizon_minutes is None for target 'fwd_ret_15'. Returning membership_only coverage.`
- **Location:** `TRAINING/ranking/utils/registry_coverage.py`
- **Cause:** Horizon extraction fails for targets like `fwd_ret_10`, `fwd_ret_15` — no suffix pattern matches
- **Fix:** Added `fwd_ret_N` → N minutes fallback in `sst_contract.py:resolve_target_horizon_minutes()`

### 6. Manifest path mismatch warning
- **Status:** [x] FIXED
- **Severity:** LOW (warning only)
- **Error:** `Manifest creation reported success but manifest.json not found at .../globals/manifest.json`
- **Location:** `TRAINING/orchestration/intelligent_trainer.py:1343`
- **Cause:** Manifest is saved at `{output}/manifest.json` but the check looked at `{output}/globals/manifest.json`
- **Fix:** Changed check path to `self.output_dir / "manifest.json"`

### 7. PosixPath.startswith error in run hash computation
- **Status:** [x] FIXED
- **Severity:** MEDIUM (run hash computation fails)
- **Error:** `Failed to compute run hash: 'PosixPath' object has no attribute 'startswith'`
- **Location:** `TRAINING/orchestration/utils/diff_telemetry.py:262`
- **Cause:** `iterdir_sorted()` returns Path objects, but `.startswith()` was called on Path instead of `.name.startswith()`
- **Fix:** Changed `d.startswith("sample_")` to `d.name.startswith("sample_")`

## Completed Fixes (on main, prior commits)

- [x] `ret_1` lookback hard-fail (`bars >= 2` → `bars >= 1`) — commit addbdc2
- [x] Demo config data path (`data_labeled_v3` → `data_labeled_v2`) — commit addbdc2
- [x] `get_input_mode` ExperimentConfig crash — commit 791bb06
- [x] Cascading xgboost import failure in model_fun — commit 85e136b
- [x] `release_data` None guard in unified_loader — commit 85e136b
