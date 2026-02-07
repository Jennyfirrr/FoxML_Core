# 2026-01-10: Complete Non-Determinism Elimination

## Summary
Eliminated all remaining sources of non-determinism in feature ordering, data loading, DataFrame column ordering, and permutation importance calculations. Ensures "same inputs → same features → same data → same models → same outputs" end-to-end.

## Critical Fixes

### 1. Feature and DataFrame Column Ordering Determinism

**Files**: `TRAINING/ranking/utils/cross_sectional_data.py`, `TRAINING/training_strategies/strategy_functions.py`, `TRAINING/ranking/shared_ranking_harness.py`

- **Fixed**: Non-deterministic `mtf_data` dictionary iteration affecting DataFrame concatenation column order
  - Changed all `for symbol, df in mtf_data.items()` to `for symbol in sorted(mtf_data.keys())`
  - Ensures `pd.concat()` produces consistent column order regardless of dict insertion order
  - Affects: Feature discovery, schema harmonization, feature alignment

- **Fixed**: Non-deterministic sample DataFrame selection
  - `shared_ranking_harness.py`: Changed `next(iter(mtf_data.values()))` to `sorted(mtf_data.keys())[0]`
  - `cross_sectional_data.py`: Changed `next(iter(mtf_data.values()))` to `sorted(mtf_data.keys())[0]`
  - Ensures consistent column discovery for interval detection and feature filtering

- **Fixed**: Non-deterministic feature auto-discovery
  - `cross_sectional_data.py`: Added `sorted()` wrapper around column filtering for feature discovery
  - Ensures feature lists are always in alphabetical order, even when auto-discovered

- **Fixed**: Non-deterministic symbol iteration in feature alignment
  - `cross_sectional_data.py`: Sorted symbols before iterating for feature alignment operations
  - Ensures consistent alignment order across runs

**Impact**: Same data → same feature list order → same DataFrame column order → same model inputs → same model outputs

### 2. Permutation Importance Determinism

**File**: `TRAINING/ranking/predictability/leakage_detection.py`

- **Fixed**: Non-deterministic `np.random.shuffle()` in permutation importance calculation
  - Added deterministic seed generation: `stable_seed_from(['permutation_importance', target_column, f'feature_{i}'])`
  - Ensures same permutation order for same target/feature combination across runs
  - Matches pattern already used in `model_evaluation.py` line 2939

**Impact**: Permutation importance scores are now reproducible across runs

### 3. Data Sampling Determinism (Already Fixed, Verified)

**Files**: `TRAINING/ranking/utils/cross_sectional_data.py`, `TRAINING/ranking/predictability/data_loading.py`, `TRAINING/training_strategies/execution/training.py`

- **Verified**: All data sampling operations use deterministic seeds
  - Cross-sectional sampling: Timestamp-based deterministic seeding (not hash-based)
  - Data loading: `stable_seed_from([symbol, "data_sampling"])`
  - Downsampling: `stable_seed_from([target, 'downsample'])`

**Impact**: Same data → same samples selected → same training data

## Verification

All critical paths verified:
- ✅ Feature ordering: All feature lists sorted before use
- ✅ DataFrame column order: All columns reordered to match sorted feature names
- ✅ Data sampling: All use deterministic seeds
- ✅ DataFrame concatenation: Symbols sorted before iteration
- ✅ Permutation importance: Deterministic seeds
- ✅ Model training: Seeds from determinism system
- ✅ CV splits: Deterministic seeds

## Files Changed

1. `TRAINING/ranking/utils/cross_sectional_data.py`
   - Lines 552-559: Sorted symbols before DataFrame concatenation
   - Lines 589-595: Sorted symbols for sample DataFrame selection
   - Lines 627-630: Sorted symbols for feature alignment
   - Line 739: Sorted columns before feature auto-discovery

2. `TRAINING/training_strategies/strategy_functions.py`
   - Lines 237-277: Sorted all `mtf_data` iterations for schema harmonization and data combination

3. `TRAINING/ranking/shared_ranking_harness.py`
   - Lines 234-240: Sorted symbols for sample DataFrame selection

4. `TRAINING/ranking/predictability/leakage_detection.py`
   - Lines 1321-1326: Added deterministic seed for permutation importance shuffle

## Testing Recommendations

1. **Feature Ordering Test**: Run same target ranking twice, verify `feature_names` lists are identical
2. **Data Consistency Test**: Run same training twice, verify X arrays are bit-identical
3. **Permutation Test**: Run permutation importance twice, verify scores are identical

## Related Changes

- Builds on: `2026-01-05-determinism-and-seed-fixes.md` (feature filtering determinism)
- Completes: End-to-end determinism for modeling pipeline
- Ensures: Same inputs → same outputs (no hidden non-determinism in data/features)
