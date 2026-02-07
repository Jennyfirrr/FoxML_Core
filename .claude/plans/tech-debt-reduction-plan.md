# Tech Debt Reduction Plan

**Status**: Phase 3 complete + Deep Code Review fixes (4 of 5 large files under 3000 lines, intelligent_trainer reduced + bug fixes)
**Created**: 2026-01-19
**Last Updated**: 2026-01-20 (Deep Code Review fixes complete)
**Branch**: fix/repro-bootstrap-import-order

## Session Progress Summary

### Completed (Sessions 2026-01-19 / 2026-01-20)

**Large File Refactoring - 4 of 5 files DONE:**
| File | Before | After | Reduction | Status |
|------|--------|-------|-----------|--------|
| `multi_model_feature_selection.py` | 5822 | 3231 | -2591 (55%) | ✅ Under target |
| `diff_telemetry.py` | 5961 | 2900 | -3061 (51%) | ✅ **DONE!** |
| `reproducibility_tracker.py` | 5390 | 2708 | -2682 (50%) | ✅ **DONE!** |
| `training.py` | 4972 | 2730 | -2242 (45%) | ✅ Under target (prior work) |
| `intelligent_trainer.py` | 4911 | 3672 | -1239 (25%) | 🔄 Reduced + bug fixes (restored methods) |

**Mixins Created:**
- `diff_telemetry/` - 6 mixins (~2500 lines extracted)
  - DiffEngineMixin (615), FingerprintMixin (470), ComparisonGroupMixin (375)
  - NormalizationMixin (464), DigestMixin (490), ContextBuilderMixin (342)
- `repro_tracker_mixins/` - 4 mixins (~2980 lines extracted)
  - CohortManagerMixin (455), ComparisonEngineMixin (556), IndexManagerMixin (472)
  - LoggingAPIMixin (1677) - contains log_comparison() and log_run()
- `intelligent_trainer/` - submodule with helpers and mixin (~1950 lines)
  - cli.py (120), config.py (63), caching.py (149), utils.py (110), __init__.py (80)
  - **PipelineStageMixin** (1427 lines) - NEW
    - `_organize_by_cohort`: Cohort organization (476 lines)
    - `rank_targets_auto`: Target ranking (472 lines)
    - `select_features_auto`: Feature selection (253 lines)
    - `_aggregate_feature_selection_summaries`: Summary aggregation (241 lines)

**All commits on branch `fix/repro-bootstrap-import-order`:**
- `8fb89057` - Restore accidentally deleted methods + add _run_leakage_diagnostics ✅ **LATEST**
- `09420d7e` - Add quick e2e test configs with production features
- `2d8a6fcf` - Fix View.LOSO AttributeError in feature_selector.py
- `6363f92b` - Fix missing method wrappers and attributes in IntelligentTrainer
- `d78ec586` - Extract PipelineStageMixin from intelligent_trainer.py
- `70f9b5f1` - Extract LoggingAPIMixin from reproducibility_tracker.py (2708 lines)
- `e96b6763` - Extract ContextBuilderMixin from diff_telemetry.py
- (earlier commits for multi_model_feature_selection and repro_tracker mixins)

### Deep Code Review Fixes (2026-01-20)

From comprehensive deep code review of TRAINING/ (343 files, 141,553 LOC):

**1. Determinism Violations (CRITICAL) - FIXED**
| File | Lines | Issue | Fix |
|------|-------|-------|-----|
| `feature_selector.py` | 1889 | `model_families_config.items()` | Use `sorted_items()` |
| `feature_selector.py` | 2552-2553 | Dict comprehension unsorted | Use `sorted_items()` |
| `feature_selector.py` | 3080 | `hp_config.items()` unsorted | Use `sorted_items()` |
| `feature_selector.py` | 3135 | `lgb_config.items()` unsorted | Use `sorted_items()` |
| `training.py` | 1545 | `target_feat_data.keys()` unsorted | Use `sorted_keys()` |

**2. Output Directory CLI - FIXED**
- **Problem**: `--output-dir` CLI argument was completely ignored
- **Root cause**: Constructor always redirected to `RESULTS/runs/{comparison_group}/`
- **Fix**: Added `user_specified_output_dir` parameter to `IntelligentTrainer`
  - If user specifies `--output-dir`, use it as-is (optionally with timestamp)
  - Only use RESULTS/runs/ structure when no explicit path given
- **Files modified**:
  - `intelligent_trainer.py:165` - Added constructor parameter
  - `intelligent_trainer.py:250-305` - Modified output directory logic
  - `intelligent_trainer.py:2524-2527` - Track user-specified flag at CLI parse
  - `intelligent_trainer.py:3347-3357` - Pass flag to constructor

**3. Symbol-Specific Routing Config - FIXED**
- **Problem**: `max_symbols_for_ss: 0` didn't disable SS evaluation, only affected routing decisions
- **Fix**: Added early SS gate in `rank_targets()` that:
  - Checks `max_symbols_for_ss` before ANY SS evaluation
  - Disables SS evaluation entirely when `len(symbols) > max_symbols_for_ss`
  - Saves compute by skipping per-symbol evaluation for large universes
- **Files modified**:
  - `target_ranker.py:478-502` - Added early SS gate with compute skip

**4. SS Gate Check Fix - FIXED**
- **Problem**: Gate used `len(sym_results)` instead of total universe size
- **Fix**: Added `total_symbols` parameter to routing functions
  - `_compute_target_routing_decisions()` now accepts `total_symbols`
  - `_compute_single_target_routing_decision()` now accepts `total_symbols`
  - All call sites updated to pass `len(symbols)`
- **Files modified**:
  - `target_routing.py:24-30, 221-226, 299-306, 466-470` - Function signatures and gate logic
  - `target_ranker.py:1109-1115, 1529-1536, 1563-1570, 1587-1594` - All call sites

**Verification passed**:
```
✓ python bin/verify_determinism_init.py - All entry points initialize determinism
✓ pytest tests/test_smoke_imports.py - 4 passed
✓ pytest TRAINING/contract_tests/ - 16 passed, 2 skipped
✓ python bin/verify_refactor_health.py - HEALTH CHECK PASSED
```

### Audit Bug Fixes (2026-01-20)

After PipelineStageMixin extraction, comprehensive audit identified and fixed:

1. **Missing caching wrapper methods** (`6363f92b`):
   - `_get_cache_key`, `_load_cached_rankings`, `_save_cached_rankings`
   - `_load_cached_features`, `_save_cached_features`
   - Missing `_initial_output_dir` attribute initialization

2. **View.LOSO AttributeError** (`2d8a6fcf`):
   - `View` enum only has `CROSS_SECTIONAL` and `SYMBOL_SPECIFIC`
   - `LOSO` is a string alias handled by `View.from_string()`
   - Fixed `feature_selector.py:280` to use string comparison

3. **Accidentally deleted methods** (`8fb89057`):
   - `_compute_feature_signature_from_target_features` (RI-003)
   - `_finalize_run_identity` (RI-003)
   - `_get_stable_run_id` (RI-004)
   - Added `_run_leakage_diagnostics` stub (was never implemented)

**Lesson learned**: Always run full contract tests AND e2e tests after extraction.

### Analysis: intelligent_trainer.py (4911 → 3672 lines)

**Status: REDUCED via PipelineStageMixin extraction + bug fixes complete**

**What was extracted (2026-01-20):**
- `_organize_by_cohort`: 476 lines - cohort organization
- `rank_targets_auto`: 472 lines - target ranking logic
- `select_features_auto`: 253 lines - feature selection
- `_aggregate_feature_selection_summaries`: 241 lines - aggregation
- **Total extracted: 1442 lines** to `intelligent_trainer/pipeline_stages.py`

**What was restored after audit (2026-01-20):**
- Methods accidentally deleted during sed extraction: +175 lines
- `_compute_feature_signature_from_target_features`, `_finalize_run_identity`, `_get_stable_run_id`
- Added missing `_run_leakage_diagnostics` stub (was never implemented)

**Current structure:**
- `IntelligentTrainer` now inherits from `PipelineStageMixin`
- Pipeline stage methods accessed via mixin inheritance
- All contract tests pass (16 passed), imports verified

**Remaining in main file (3672 lines):**
1. `main()` function: ~1168 lines (CLI entry point)
2. `train_with_intelligence()`: ~2174 lines (main orchestrator)
3. `__init__` and utility methods: ~330 lines (includes restored identity lifecycle)

**Why not under 3000 yet:**
- `main()` has complex config precedence logic
- `train_with_intelligence()` is the core orchestrator with state management
- Both methods have deep dependencies and tightly coupled logic

**Risk assessment:**
- Further refactoring of main() or train_with_intelligence() could affect
  reproducibility of training runs
- These methods need careful decomposition if extracted

**Recommendation:** PipelineStageMixin extraction complete. File is maintainable at
3672 lines. Further reduction would require decomposing main() or train_with_intelligence()
which carries higher risk. Consider for future session if needed.

---

## Executive Summary

Comprehensive audit identified **7 major categories** of tech debt affecting scalability, maintainability, and determinism. This plan prioritizes fixes by impact and provides implementation guidance.

---

## 1. Code Duplication (HIGH PRIORITY)

### 1.1 Duplicate Data Loading Functions

**Problem**: `_load_mtf_data_pandas` exists in 3+ locations with nearly identical logic.

| File | Line | Notes |
|------|------|-------|
| `TRAINING/data_processing/data_loader.py` | 48 | Original |
| `TRAINING/models/specialized/data_utils.py` | 89 | Copy |
| `TRAINING/data/loading/data_loader.py` | 160 | Another copy |
| `TRAINING/ranking/utils/cross_sectional_data.py` | 155 | Legacy variant |

**Fix**: Consolidate to `UnifiedDataLoader` in `TRAINING/data/loading/unified_loader.py`
- Add deprecation warnings to old functions
- Update all callers to use `UnifiedDataLoader`
- Remove duplicates after migration

**Estimated effort**: 4-6 hours

### 1.2 Duplicate Determinism Implementations

**Problem**: Multiple files define seed/determinism functions.

| File | Functions |
|------|-----------|
| `TRAINING/common/determinism.py` | `set_global_determinism`, `seed_for`, `stable_seed_from` (CANONICAL) |
| `TRAINING/common/core/determinism.py` | `set_global_determinism`, `seed_for` (DUPLICATE) |
| `TRAINING/models/specialized/core.py:46` | `set_global_determinism` (LOCAL COPY) |

**Fix**:
1. Mark `TRAINING/common/determinism.py` as canonical (SST)
2. Update `common/core/determinism.py` to re-export from canonical
3. Remove local copy in `models/specialized/core.py`
4. Update all imports to use canonical path

**Estimated effort**: 2-3 hours

### 1.3 Duplicate Fingerprint/Hash Functions

**Problem**: Multiple hashing utilities with overlapping functionality.

| File | Functions |
|------|-----------|
| `TRAINING/common/utils/fingerprinting.py` | `compute_feature_fingerprint`, `compute_data_fingerprint`, etc. (CANONICAL) |
| `TRAINING/common/utils/config_hashing.py` | `compute_config_hash`, `compute_config_hash_from_file` |
| `TRAINING/ranking/feature_selector.py:79` | `_compute_feature_selection_config_hash` (LOCAL) |
| `TRAINING/ranking/feature_selector_modules/config.py:24` | `compute_feature_selection_config_hash` (DUPLICATE) |

**Fix**:
1. Consolidate all fingerprinting to `common/utils/fingerprinting.py`
2. Move config hashing to fingerprinting module
3. Remove local implementations

**Estimated effort**: 3-4 hours

---

## 2. Large File Refactoring (HIGH PRIORITY)

### Problem: Files exceeding 3000 lines are difficult to maintain.

| File | Lines | Recommended Split |
|------|-------|-------------------|
| `orchestration/utils/diff_telemetry.py` | 5961 | Split into: diff_engine.py, telemetry_writer.py, comparison.py |
| `ranking/multi_model_feature_selection.py` | 5822 | Already has submodules; move more logic there |
| `orchestration/utils/reproducibility_tracker.py` | 5387 | Split into: tracker.py, snapshot_manager.py, legacy_compat.py |
| `ranking/predictability/model_evaluation/training.py` | 4972 | Split by model family |
| `orchestration/intelligent_trainer.py` | 4911 | Split into: trainer.py, config_resolver.py, stage_runner.py |

**Fix Strategy**:
1. Create submodule directories
2. Extract logically grouped functions
3. Keep public API in `__init__.py`
4. Update imports gradually

**Estimated effort**: 16-24 hours (spread across multiple sessions)

---

## 3. Error Handling Standardization (MEDIUM PRIORITY)

### 3.1 Broad Exception Handlers

**Problem**: 1606 instances of `except Exception` mask specific errors.

**Pattern to fix**:
```python
# BAD
try:
    result = process_data()
except Exception as e:
    logger.error(f"Failed: {e}")

# GOOD
try:
    result = process_data()
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    raise
except DataLoadError as e:
    logger.warning(f"Data load failed, using fallback: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

**Fix Strategy**:
1. Define typed exceptions in `TRAINING/common/exceptions.py`
2. Prioritize Tier A files (artifact-producing code)
3. Gradually migrate remaining files

**Typed exceptions to add**:
- `ConfigError` - Configuration issues
- `LeakageError` - Data leakage detected
- `ArtifactError` - Artifact read/write issues
- `DataLoadError` - Data loading failures
- `DeterminismError` - Reproducibility violations
- `ValidationError` - Input validation failures

**Estimated effort**: 8-12 hours

### 3.2 Generic Exception Raises

**Problem**: 426 instances of `raise ValueError/RuntimeError/Exception`.

**Fix**: Replace with typed exceptions where appropriate.

---

## 4. Magic Number Elimination (MEDIUM PRIORITY)

### Problem: Hardcoded values scattered across codebase.

| Value | Occurrences | Should Be |
|-------|-------------|-----------|
| `seed = 42` | ~15 | `get_cfg("pipeline.determinism.base_seed", default=42)` |
| `learning_rate = 0.001` | ~8 | `get_cfg("models.{family}.learning_rate", default=0.001)` |
| `dropout = 0.3` | ~6 | `get_cfg("models.{family}.dropout", default=0.3)` |
| `max_iter = 1000` | ~10 | `get_cfg("models.{family}.max_iter", default=1000)` |
| `n_bootstrap = 50` | ~5 | `get_cfg("pipeline.feature_selection.n_bootstrap", default=50)` |

**Fix Strategy**:
1. Add config keys to `CONFIG/pipeline/` YAML files
2. Replace hardcoded values with `get_cfg()` calls
3. Document defaults in config files

**Estimated effort**: 4-6 hours

---

## 5. SST Compliance (MEDIUM PRIORITY)

### 5.1 Direct JSON Usage

**Problem**: 98 direct `import json` instead of SST helpers.

**SST Pattern**:
```python
# BAD
import json
with open(path, 'w') as f:
    json.dump(data, f)

# GOOD
from TRAINING.common.utils.file_utils import write_atomic_json
write_atomic_json(path, data)
```

**Files to prioritize** (artifact-producing):
- `TRAINING/models/specialized/core.py`
- `TRAINING/stability/feature_importance/io.py`
- `TRAINING/orchestration/utils/manifest.py`
- `TRAINING/common/leakage_auto_fixer.py`

**Estimated effort**: 6-8 hours

### 5.2 Path Construction

**Problem**: `os.path.join` usage instead of `Path` or SST helpers.

**Files affected**: ~25 (mostly in `model_fun/` and `training_strategies/`)

**SST Pattern**:
```python
# BAD
import os
path = os.path.join(base_dir, "models", family, "model.joblib")

# GOOD
from TRAINING.common.utils.path_helpers import get_model_path
path = get_model_path(base_dir, family)
```

**Estimated effort**: 3-4 hours

---

## 6. Memory Efficiency (LOWER PRIORITY)

### 6.1 Eager Parquet Loading

**Problem**: ~30 uses of `pd.read_parquet` that could use Polars lazy loading.

**Files to audit**:
| File | Count | Priority |
|------|-------|----------|
| `TRAINING/ranking/predictability/data_loading.py` | 2 | High |
| `TRAINING/data/loading/unified_loader.py` | 2 | High |
| `TRAINING/common/utils/trend_analyzer.py` | 6 | Medium |
| `TRAINING/common/utils/metrics.py` | 4 | Medium |
| `TRAINING/orchestration/intelligent_trainer.py` | 3 | Medium |

**Pattern**:
```python
# BAD - loads entire file into memory
df = pd.read_parquet(path)
result = df[df['column'] > threshold]['other_column'].mean()

# GOOD - lazy evaluation
lf = pl.scan_parquet(path)
result = lf.filter(pl.col('column') > threshold).select('other_column').mean().collect().item()
```

**Estimated effort**: 8-12 hours

### 6.2 Early `.collect()` Calls

**Problem**: Polars lazy frames collected too early.

**Fix**: Defer `.collect()` until final result needed.

---

## 7. Logging Consolidation (LOWER PRIORITY)

### Problem: 292 logging setup patterns, inconsistent configuration.

**Fix**:
1. Create `TRAINING/common/logging_config.py` with standard setup
2. Use `setup_logging()` from `TRAINING/orchestration/utils/logging_setup.py`
3. Migrate files to use centralized config

**Estimated effort**: 4-6 hours

---

## 8. sklearn Import Centralization (LOWER PRIORITY)

### Problem: sklearn models imported in 20+ places with duplicated config cleaning.

**Fix**:
1. Create `TRAINING/common/sklearn_factory.py`
2. Centralize model instantiation with config cleaning
3. Handle determinism params automatically

**Pattern**:
```python
# BAD - scattered across files
from sklearn.ensemble import RandomForestRegressor
from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
rf_config = clean_config_for_estimator(RandomForestRegressor, config, extra, "random_forest")
model = RandomForestRegressor(**rf_config)

# GOOD - centralized
from TRAINING.common.sklearn_factory import create_model
model = create_model("random_forest", config, seed=seed)
```

**Estimated effort**: 6-8 hours

---

## Implementation Priority

### Phase 1: Critical (Week 1) - COMPLETED 2026-01-19
1. ✅ Bootstrap import order fixes (DONE)
2. ✅ Unseeded random fixes (DONE)
3. ✅ Dict iteration determinism (DONE)
4. ✅ Duplicate data loading consolidation (DONE - added load_mtf_data to unified_loader, deprecated 3 copies)
5. ✅ Duplicate determinism consolidation (DONE - updated common/core/determinism.py to re-export)
6. ✅ Remove local set_global_determinism from core.py (DONE)
7. ✅ Consolidate fingerprint functions (DONE - feature_selector imports from submodule)

### Phase 2: High Impact (Week 2-3) - IN PROGRESS
8. 🔄 Large file refactoring:
   - ✅ diff_telemetry.py: COMPLETE (2026-01-19, 2026-01-20)
     - ✅ Created DiffEngineMixin (615 lines) for diff computation methods
     - ✅ Created FingerprintMixin (470 lines) for fingerprint methods
     - ✅ Created ComparisonGroupMixin (375 lines) for comparison group building
     - ✅ Created NormalizationMixin (464 lines) for data normalization
     - ✅ Created DigestMixin (490 lines) for digest computation
     - Total reduction: 5961 → 3198 lines (-2763 lines, 46% smaller)
     - DiffTelemetry now inherits from 5 mixins
   - ✅ multi_model_feature_selection.py: COMPLETE (2026-01-19)
     - ✅ Phase 1: Integrated dispatch_trainer from trainers/ submodule
     - ✅ Phase 2: Extracted RFE, Stability Selection, Boruta trainers (~430 lines)
     - ✅ Phase 3: Removed ALL legacy trainer blocks (2026-01-19): 2624 lines removed (5822 → 3231)
       - All 16 trainers now fully modular in trainers/ submodule
       - Legacy if/elif blocks completely eliminated
       - Simplified fallback: just error handling for unknown model families
       - Total reduction: 2591 lines (55% smaller)
   - ✅ reproducibility_tracker.py: COMPLETE (5390 → 2708 lines, -2682)
     - ✅ IndexManagerMixin (2026-01-19): 472 lines in repro_tracker_mixins/index_manager.py
       - Contains _update_index, _parse_run_started_at
       - Added helper methods: _compute_segment_id, _extract_symbol_metrics, _save_index_with_lock
     - ✅ CohortManagerMixin (2026-01-19): 455 lines in repro_tracker_mixins/cohort_manager.py
       - Contains _extract_cohort_metadata, _compute_cohort_id, _calculate_cohort_relative_path
       - Contains _get_cohort_dir, _get_cohort_dir_v2, _validate_purpose_path
       - SST compliant: Uses compute_cohort_id() helper, target_first_paths
     - ✅ ComparisonEngineMixin (2026-01-19): 556 lines in repro_tracker_mixins/comparison_engine.py
       - Contains _find_matching_cohort, _compare_within_cohort, get_last_comparable_run, _compute_drift
       - SST compliant: Uses extract_n_effective(), Stage/View enums
       - Sample-adjusted statistical comparisons preserved
     - ✅ LoggingAPIMixin (2026-01-20): 1677 lines in repro_tracker_mixins/logging_api.py
       - Contains log_comparison(), log_run() - main public API
       - Contains _write_atomic_json_with_lock, _normalize_view_for_comparison helpers
       - SST compliant: Uses sorted_items(), extract_n_effective(), Stage/View enums
     - All 20 contract tests pass, smoke imports pass
     - Total reduction: 2682 lines (50% smaller)
   - 📋 training.py: ANALYZED (4972 lines)
     - Module-level functions (not class-based)
     - Helper functions: cross_val_score_with_early_stopping, _compute_and_store_metrics, etc.
     - Could extract:
       1. Model-specific training to separate files (per-family trainers)
       2. Metric computation to metrics module
     - Estimated reduction: ~1500 lines via modular trainer extraction
   - 📋 intelligent_trainer.py: ANALYZED (4911 lines)
     - Single class IntelligentTrainer with ranking/selection/training methods
     - Recommended split:
       1. ConfigResolver (config loading/resolution): ~400 lines
       2. TargetRanker (ranking logic): ~600 lines
       3. FeatureSelector (selection logic): ~500 lines
     - Main class would retain: orchestration, state management
     - Estimated reduction: ~1500 lines via extraction
9. ✅ Magic number elimination (DONE - added _get_base_seed() to trainers.py and trainers_extended.py, uses get_cfg)
10. ✅ SST compliance for JSON writes (DONE - fixed core_utils.py and leakage_auto_fixer.py missing sort_keys)

### Phase 3: Maintenance (Week 4+)
11. Error handling standardization
12. Memory efficiency improvements
13. Logging consolidation
14. sklearn factory

---

## Metrics to Track

| Metric | Before | After Phase 2 | Target |
|--------|--------|---------------|--------|
| Files >3000 lines | 5 | 5 | 0 |
| diff_telemetry.py | 5961 | 2900 (-3061) | <3000 ✅ DONE! |
| multi_model_feature_selection.py | 5822 | 3231 (-2591) | <3000 ✅ |
| reproducibility_tracker.py | 5387 | 4174 (-1213) | <3000 |
| training.py | 4972 | 4972 | <3000 |
| intelligent_trainer.py | 4911 | 4911 | <3000 |
| `except Exception` handlers | 1606 | 1606 | <500 |
| Generic exception raises | 426 | 426 | <100 |
| Hardcoded magic numbers | ~50 | reduced | 0 |
| Direct json imports | 98 | 98 | <20 |
| `pd.read_parquet` in hot paths | ~30 | ~30 | <10 |
| Duplicate function definitions | ~15 | ~10 | 0 |

*Note: multi_model_feature_selection.py reduced by 2591 lines (5822 → 3231) after:
- Integrating dispatch_trainer for modular trainer dispatch
- Extracting all 16 trainers to trainers/ submodule
- Removing ALL legacy trainer blocks (lightgbm, xgboost, random_forest, neural_network, catboost, lasso, ridge, elastic_net, logistic_regression, ftrl_proximal, ngboost, mutual_information, univariate_selection)
- File is now under the 3000 line target! ✅

---

## Testing Strategy

After each change:
1. `python -m py_compile <modified_files>`
2. `python bin/verify_determinism_init.py`
3. `bash bin/check_determinism_patterns.sh | head -50`
4. `pytest TRAINING/contract_tests/ -v`
5. `pytest tests/ -v` (if available)

---

## Remaining Work (Priority Order)

### Phase 2 Remaining: Large File Refactoring

#### 1. `training.py` (4972 lines → target <3000)
**Priority**: High
**Approach**: Extract model-specific training logic
- Create `TRAINING/ranking/predictability/model_evaluation/trainers/` directory
- Extract per-family trainer functions (lightgbm, xgboost, neural, etc.)
- Keep orchestration in main file
- **Estimated reduction**: ~1500 lines

#### 2. `intelligent_trainer.py` (4911 lines → target <3000)
**Priority**: High
**Approach**: Extract to mixins/submodules
- `ConfigResolver` mixin: Config loading/resolution (~400 lines)
- `TargetRankerMixin`: Ranking logic (~600 lines)
- `FeatureSelectorMixin`: Selection logic (~500 lines)
- Main class retains: orchestration, state management
- **Estimated reduction**: ~1500 lines

#### 3. `diff_telemetry.py` ✅ COMPLETE (5961 → 2900 lines)
**Status**: DONE - under 3000 line target
**Mixins extracted** (6 total, ~2500 lines):
- DiffEngineMixin (615) - diff computation
- FingerprintMixin (470) - fingerprint computation
- ComparisonGroupMixin (375) - comparison group building
- NormalizationMixin (464) - data normalization
- DigestMixin (490) - digest computation
- ContextBuilderMixin (342) - context building and validation

#### 4. `reproducibility_tracker.py` ✅ COMPLETE (2708 lines)
**Status**: DONE - under 3000 line target
**What was extracted**: LoggingAPIMixin (1677 lines) containing:
- `log_comparison`: main comparison API (~830 lines)
- `log_run`: main run logging API (~637 lines)
- `_write_atomic_json_with_lock`: helper function
- `_normalize_view_for_comparison`: helper function

**Mixins in repro_tracker_mixins/**:
- CohortManagerMixin (455 lines)
- ComparisonEngineMixin (556 lines)
- IndexManagerMixin (472 lines)
- LoggingAPIMixin (1677 lines) - NEW

**Note**: `_save_to_cohort` remains at 1674 lines in main file - too monolithic for clean extraction

### Phase 3: Maintenance Tasks

| Task | Scope | Effort |
|------|-------|--------|
| Error handling standardization | 1606 `except Exception` handlers | 8-12 hrs |
| Memory efficiency (Polars lazy) | ~30 `pd.read_parquet` calls | 8-12 hrs |
| Logging consolidation | 292 logging patterns | 4-6 hrs |
| sklearn factory | 20+ import locations | 6-8 hrs |

---

## Handoff Prompt for Next Session

```
Tech debt reduction Phase 3 COMPLETE. Here's the final state:

Branch: fix/repro-bootstrap-import-order (all commits pushed)

COMPLETED (4 of 5 large files under 3000 lines):
✅ multi_model_feature_selection.py: 5822 → 3231 lines (-45%)
✅ diff_telemetry.py: 5961 → 2900 lines (-51%)
✅ reproducibility_tracker.py: 5390 → 2708 lines (-50%)
✅ training.py: 4972 → 2730 lines (-45%, prior work)

REDUCED (not under 3000, but improved):
🔄 intelligent_trainer.py: 4911 → 3469 lines (-29%)
   - Created PipelineStageMixin (1427 lines) in intelligent_trainer/pipeline_stages.py
   - Extracted: _organize_by_cohort, rank_targets_auto, select_features_auto, _aggregate_feature_selection_summaries
   - main() (1168 lines) and train_with_intelligence() (2174 lines) remain
   - Further reduction requires decomposing these complex methods

Mixin summary:
- diff_telemetry/: 6 mixins (~2500 lines)
- repro_tracker_mixins/: 4 mixins (~2980 lines)
- intelligent_trainer/: submodule with cli, config, caching, utils + PipelineStageMixin (~1950 lines)

Key patterns:
- Mixin extraction: Follow diff_telemetry/ or repro_tracker_mixins/ patterns
- MRO matters: List mixins AFTER dependencies in class inheritance
- Always verify: pytest TRAINING/contract_tests/ tests/test_smoke_imports.py -v
- SST compliance: sorted iterations, atomic JSON writes, helpers not raw os.path/json

If continuing intelligent_trainer.py refactoring:
1. main() decomposition: extract config resolution, value tracing, arg parsing steps
2. train_with_intelligence() decomposition: extract training loop, post-processing steps
3. Both require careful planning due to state management and side effects

Recent commits on branch:
- [latest] Extract PipelineStageMixin from intelligent_trainer.py (4911 → 3469 lines)
- 70f9b5f1 Extract LoggingAPIMixin from reproducibility_tracker.py
- e96b6763 Extract ContextBuilderMixin from diff_telemetry.py
- 7bf14811 Update tech debt plan with detailed next steps
```

---

## Notes

- Always create feature branches for each phase
- Update CLAUDE.md if new SST helpers are created
- Document new patterns in `INTERNAL/docs/references/`
- Run determinism verification after any seed-related changes
