# Training Pipeline Code Review - Phase 2

**Date**: 2026-01-18

## Overview

Phase 2 of the training pipeline code review focusing on testing gaps, performance issues, API design, and code duplication. This follows the previous PR that addressed determinism, exception handling, and imports.

## Changes Summary

### Unit Tests Added (P1-CRITICAL)

#### Target Routing Tests (`tests/test_target_routing.py`)
Comprehensive tests for routing decision logic:
- `TestComputeTargetRoutingDecisions` - Tests for batch routing:
  - CS route with good coverage
  - Symbol-specific route with weak CS
  - BOTH route with concentrated performance
  - BLOCKED route with suspicious scores (leakage detection)
  - High score not blocked when tstat is stable
  - Dev mode threshold relaxation
  - CS failed fallback to symbol-specific
  - BLOCKED when no viable route

- `TestComputeSingleTargetRoutingDecision` - Tests for single target routing
- `TestLoadRoutingDecisions` - Tests for loading routing decisions from files
- `TestSaveDualViewRankings` - Tests for saving routing decisions
- `TestRoutingDecisionSummary` - Tests for summary computation

#### Cross-Sectional Feature Ranker Tests (`tests/test_cross_sectional_feature_ranker.py`)
Tests for CS feature ranking functions:
- `TestNormalizeCrossSectionalPerDate` - zscore/rank normalization
- `TestTrainPanelModel` - LightGBM/XGBoost panel model training
- `TestComputeCrossSectionalImportance` - Importance computation
- `TestTagFeaturesByImportance` - CORE/SYMBOL_SPECIFIC/CS_SPECIFIC/WEAK tagging
- `TestComputeCrossSectionalStability` - Stability metrics

#### Model Factory Tests (`tests/test_model_factory.py`)
Tests for model factory:
- Singleton pattern verification
- `create_model()` for regression/classification
- `create_models_for_targets()` batch creation
- `_auto_select_model_type()` logic
- `validate_model_config()` validation
- Trainer class support

### Performance Fixes (P2-HIGH)

#### Fixed O(n²) Correlation Computation
**File**: `TRAINING/ranking/utils/feature_selection.py`
**Function**: `select_features_by_correlation()`

**Before** (O(n²) per-pair):
```python
for idx in sorted_indices:
    for selected_idx in selected_indices:
        corr = np.corrcoef(X[:, idx], X[:, selected_idx])[0, 1]
```

**After** (Vectorized):
```python
# Compute correlation matrix once (BLAS-accelerated)
corr_matrix = np.corrcoef(X.T)

# Vectorized lookup
selected_corrs = np.abs(corr_matrix[idx, selected_indices])
is_redundant = np.any(selected_corrs > correlation_threshold)
```

**Impact**: For 5000 features, ~25M pairwise operations → single matrix operation with BLAS acceleration.

#### Replaced .iterrows() with .itertuples()
**File**: `TRAINING/ranking/feature_selection_reporting.py`
**Lines**: 191-210, 245-260

**Before** (~100x slower):
```python
for i, (_, row) in enumerate(summary_df_sorted.iterrows()):
    row['feature']
```

**After**:
```python
for i, row in enumerate(summary_df_sorted.itertuples(index=False)):
    row_dict = row._asdict()
    row_dict.get('feature', '')
```

### Code Quality Improvements (P3-HIGH)

#### Centralized Config Helpers
**New File**: `TRAINING/common/utils/config_helpers.py`

Functions:
- `load_threshold(key, default, config_name)` - Generic threshold loading
- `load_routing_thresholds(config_name, config_key)` - All routing thresholds
- `apply_dev_mode_relaxation(thresholds)` - Dev mode threshold adjustment
- `load_feature_selection_thresholds(config_name)` - Feature selection thresholds
- `load_data_limits(config_name)` - Data limit thresholds

**Purpose**: Eliminates 5+ duplicate threshold loading patterns across codebase.

#### Centralized Routing Config
**New File**: `CONFIG/routing/thresholds.yaml`

```yaml
routing:
  thresholds:
    cs_skill01: 0.65
    symbol_skill01: 0.60
    frac_symbols_good: 0.5
    suspicious_cs_skill01: 0.90
    suspicious_symbol_skill01: 0.95
    concentrated_iqr: 0.15
    min_stable_tstat: 3.0

  dev_mode:
    enabled: false
    cs_relaxation: 0.25
    symbol_relaxation: 0.25
    frac_relaxation: 0.3
```

### API Design Fixes (P5-MEDIUM)

#### Parameter Validation in Feature Selector
**File**: `TRAINING/ranking/feature_selector.py`

Added validation:
1. Warns when multiple config sources provided
2. Raises error when SYMBOL_SPECIFIC view missing symbol parameter

```python
if config_sources_provided > 1:
    logger.warning(
        f"Multiple config sources provided: {', '.join(provided)}. "
        f"Using precedence: feature_selection_config > model_families_config > multi_model_config"
    )

if view_enum == View.SYMBOL_SPECIFIC and symbol is None:
    raise ValueError(
        "symbol parameter is required when view=SYMBOL_SPECIFIC."
    )
```

#### Debug Logging in Parallel Exec
**File**: `TRAINING/common/parallel_exec.py`

Added debug logging for config loading failures (previously silently swallowed):
```python
except Exception as e:
    logger.debug(f"Failed to load threading config for {task_type}: {e}")
```

### Logging Improvements (P6-MEDIUM)

#### Removed Emojis from Logs
**File**: `TRAINING/ranking/target_routing.py`

Replaced emojis with structured prefixes:
- `🔧` → `[DEV]`
- `⚠️` → `[WARN]`
- `✅` → `[OK]`
- `🚨` → `[ERROR]`

**Purpose**: Better log parsing/filtering in production environments.

## Files Modified

| File | Changes |
|------|---------|
| `TRAINING/ranking/utils/feature_selection.py` | Fixed O(n²) correlation |
| `TRAINING/ranking/feature_selection_reporting.py` | Replaced .iterrows() |
| `TRAINING/ranking/feature_selector.py` | Added parameter validation |
| `TRAINING/ranking/target_routing.py` | Fixed logging, removed emojis |
| `TRAINING/common/parallel_exec.py` | Added debug logging |

## Files Created

| File | Purpose |
|------|---------|
| `tests/test_target_routing.py` | Unit tests for routing |
| `tests/test_cross_sectional_feature_ranker.py` | Unit tests for CS ranker |
| `tests/test_model_factory.py` | Unit tests for model factory |
| `TRAINING/common/utils/config_helpers.py` | Centralized config helpers |
| `CONFIG/routing/thresholds.yaml` | Centralized routing thresholds |

## Documentation Updated

- `CHANGELOG.md` - Added 2026-01-18 entry
- `CONFIG/CHANGELOG.md` - Added routing thresholds entry
- `CONFIG/README.md` - Updated directory structure, recent changes
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - Added config helper functions
- `DOCS/02_reference/KNOWN_ISSUES.md` - Updated last modified date

## Testing

```bash
# Run new unit tests
pytest tests/test_target_routing.py -v
pytest tests/test_cross_sectional_feature_ranker.py -v
pytest tests/test_model_factory.py -v

# Run smoke tests
pytest tests/test_smoke_imports.py -v

# Lint check
ruff check TRAINING/
```

## Deferred Items (Out of Scope)

| Issue | Reason |
|-------|--------|
| Full parallelization refactor | Requires architectural changes |
| msgpack migration | Affects artifact compatibility |
| Streaming Polars conversion | Complex implementation |
| Memory optimization audit | Separate performance sprint |

## Related

- Previous PR: Determinism, exception handling, and imports fixes
- Related issue: Training pipeline code review
