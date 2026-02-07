# Interval-Agnostic Pipeline Implementation Plan

**Status**: Full implementation complete - 26 phases done (Phases 8, 9, 10 complete)
**Created**: 2026-01-18
**Last Updated**: 2026-01-19
**Revision**: 6.3 - Phases 8-10 complete (Multi-Horizon, Cross-Horizon Ensemble, Multi-Interval)

---
## 🚧 RESUME HERE - Context Window Handoff

### Quick Start Command
```bash
# In new context window, say:
"Read .claude/plans/interval_agnostic_pipeline.md - All core phases complete"
```

### What's Done (23 phases complete)
| Phase | Status | Summary |
|-------|--------|---------|
| 0 | ✅ | `interval.py` + `purge.py` - core types |
| 0.5 | ✅ | Feature flags + audit logging + v1/v2 tests |
| 1 | ✅ | Registry v2 schema (211 features have `lookback_minutes`) |
| 2A | ✅ | `leakage_budget.py` now uses `registry.get_lookback_minutes()` (v2 schema) |
| 2B | ✅ | Replaced 6 hardcoded `purge_overlap=17` with `get_purge_overlap_bars()` |
| 3 | ✅ | Floor divisions replaced with SST functions, `use_ceil_rounding=true` enabled |
| 4 | ✅ | Sequential models: `lookback_minutes=300` in config, derived `lookback_T` in family_router |
| 5 | ✅ | Hardcoded cleanup: replaced `max_lookback_bars=288` with time-based `1440.0` minutes |
| 6 | ✅ | Time-based CV: added `purge_overlap_minutes` parameter, updated 3 callers |
| 7 | ✅ | allowed_horizons: `get_allowed_horizon_minutes()`, `is_horizon_allowed()` methods added |
| 11 | ✅ | Feature Computation: `interval_minutes` in `ComprehensiveFeatureBuilder` + `FeatureBuilder` |
| 12 | ✅ | Target Computation: `hft_forward.py` tracks interval, `barrier.py` enforces it |
| 13 | ✅ | Cache Keys: `build_cache_key_with_symbol()` includes interval_minutes |
| 14 | ✅ | Alignment Validation: `validate_timestamp_alignment()` function added |
| 15 | ✅ | Gap Detection: `detect_data_gaps()`, `validate_interval_consistency()` functions |
| 16 | ✅ | Model Metadata: `interval_minutes` + `interval_source` in model metadata |
| 17 | ✅ | Inference Gate: `InferenceEngine._validate_interval()` rejects mismatched intervals |
| 18 | ✅ | Hyperparameter Scaling: `interval_scaling` config section with guidance |
| 19 | ✅ | Feature Aliases: `resolve_alias()`, `feature_aliases` section in registry |
| 20 | ✅ | Cross-Sectional Purge: Already using `get_purge_overlap_bars()` SST function |
| 21 | ✅ | Horizon Config: `get_configured_horizons()`, `discover_horizons_from_models()` |
| 22 | ✅ | Multi-Interval Test Fixtures: `tests/conftest.py`, `tests/test_multi_interval.py` |
| 23 | ✅ | Metric Scaling Documentation: `DOCS/02_reference/configuration/INTERVAL_METRIC_SCALING.md` |
| 24 | ✅ | Config Validation: `validate_interval_config()`, DataConfig interval conflict detection |
| 8 | ✅ | Multi-Horizon Bundle: `HorizonBundle`, `MultiHorizonTrainer`, `multi_horizon_orchestrator` |
| 9 | ✅ | Cross-Horizon Ensemble: `CrossHorizonEnsemble`, ridge weights, horizon decay |
| 10 | ✅ | Multi-Interval Experiments: `MultiIntervalExperiment`, cross-validation, feature transfer |

### What's Next
**All major phases complete!** The interval-agnostic pipeline is fully implemented:
- Phase 8: ✅ Multi-horizon bundle training (HorizonBundle, MultiHorizonTrainer, orchestrator)
- Phase 9: ✅ Cross-horizon ensemble (CrossHorizonEnsemble, ridge blending, horizon decay)
- Phase 10: ✅ Multi-interval experiments (MultiIntervalExperiment, cross-validation, comparison)

**All interval-agnostic infrastructure is now complete.**
**The pipeline can now run at any data interval (1m, 5m, 15m, 60m, etc.)**
**Multi-horizon bundle training is now available via `strategy: multi_horizon_bundle`.**

### Key SST Functions Available
```python
# Interval conversion (TRAINING/common/interval.py)
from TRAINING.common.interval import minutes_to_bars, bars_to_minutes
bars = minutes_to_bars(70, interval_minutes=5)  # 14 bars (ceil rounding)

# Purge calculation (TRAINING/ranking/utils/purge.py)
from TRAINING.ranking.utils.purge import get_purge_overlap_bars
purge = get_purge_overlap_bars(target_horizon_minutes=60, interval_minutes=5)  # 17 bars

# Feature lookback (TRAINING/common/feature_registry.py)
from TRAINING.common.feature_registry import get_registry
registry = get_registry()
lookback = registry.get_lookback_minutes("adx_14")  # 70.0 minutes (v2 schema)

# Phase 19: Feature aliases for interval-agnostic naming
canonical = registry.resolve_alias("adx_standard")  # "adx_14"
metadata = registry.get_feature_metadata("adx_standard")  # Returns adx_14 metadata

# Config validation (CONFIG/config_schemas.py)
from CONFIG.config_schemas import validate_interval_config
result = validate_interval_config(bar_interval="5m", base_interval_minutes=5.0)
# result['valid'] == True, result['resolved_minutes'] == 5.0

# Phase 8: Multi-horizon bundle training (TRAINING/common/horizon_bundle.py)
from TRAINING.common.horizon_bundle import (
    HorizonBundle,
    parse_horizon_from_target,
    create_bundles_from_targets,
    compute_bundle_diversity,
)
base, horizon = parse_horizon_from_target("fwd_ret_60m")  # ("fwd_ret", 60)
bundles = create_bundles_from_targets(["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m"])

# Multi-horizon trainer (TRAINING/model_fun/multi_horizon_trainer.py)
from TRAINING.model_fun.multi_horizon_trainer import MultiHorizonTrainer
trainer = MultiHorizonTrainer({"shared_layers": [256, 128], "epochs": 100})
result = trainer.train(X_tr, y_dict)  # y_dict = {"fwd_ret_5m": [...], "fwd_ret_15m": [...]}
predictions = trainer.predict(X_test)  # Returns dict: {target: predictions}

# Phase 9: Cross-horizon ensemble (TRAINING/model_fun/cross_horizon_ensemble.py)
from TRAINING.model_fun.cross_horizon_ensemble import (
    CrossHorizonEnsemble,
    calculate_horizon_weights,
    blend_horizon_predictions,
)
ensemble = CrossHorizonEnsemble({"ridge_lambda": 0.15, "horizon_decay_enabled": True})
result = ensemble.fit(horizon_predictions, y_true)  # Learn weights
blended = ensemble.blend(new_predictions)  # Blend using learned weights

# Phase 10: Multi-interval experiments (TRAINING/orchestration/multi_interval_experiment.py)
from TRAINING.orchestration.multi_interval_experiment import (
    MultiIntervalExperiment,
    CrossIntervalValidator,
    FeatureTransfer,
    IntervalComparator,
)
experiment = MultiIntervalExperiment({"intervals": [5, 15, 60]})
result = experiment.run(data_root="data/data_labeled_v2", output_dir="output", targets=["fwd_ret_5m"])
# result.best_interval, result.comparison_summary, result.cross_validation_results
```

### Test Commands
```bash
# Core interval tests (no lightgbm required)
conda activate trader && pytest tests/test_interval.py tests/test_interval_config_validation.py tests/test_feature_aliases.py -v
# Expected: 69 passed

# Full interval test suite (requires lightgbm)
conda activate trader && pytest tests/test_interval.py tests/test_purge.py tests/test_interval_v1_v2_equivalence.py -v
# Expected: 120 passed

# Multi-interval parameterized tests
conda activate trader && pytest tests/test_multi_interval.py -v
# Expected: 37 passed, 8 skipped (lightgbm tests)

# Multi-horizon bundle tests (requires tensorflow)
conda activate trader && pytest tests/test_horizon_bundle.py tests/test_multi_horizon_trainer.py tests/test_multi_horizon_orchestrator.py -v
# Expected: 49 passed

# Cross-horizon ensemble tests
conda activate trader && pytest tests/test_cross_horizon_ensemble.py -v
# Expected: 18 passed

# Multi-interval experiment tests
conda activate trader && pytest tests/test_multi_interval_experiment.py -v
# Expected: 29 passed

# All Phase 8-10 tests together
conda activate trader && pytest tests/test_horizon_bundle.py tests/test_multi_horizon_trainer.py tests/test_multi_horizon_orchestrator.py tests/test_cross_horizon_ensemble.py tests/test_multi_interval_experiment.py -v
# Expected: 96 passed
```

---

## Executive Summary

Make data interval a **first-class experiment dimension** instead of an implicit constant (5m). Currently, targets are time-based (`fwd_ret_30m`) but features are row-based (`lag_bars`), and the pipeline incorrectly converts at runtime. Fix: **store time, derive bars deterministically**.

### Core Principle
```
Targets, features, and leakage are all expressed in MINUTES.
Δt (interval) is detected once and stored.
Bars/windows are derived deterministically with ceil() rounding.
Artifacts store resolved windows so inference can't drift.
```

### Model Improvement Opportunities Unlocked
1. **Multi-horizon training** - Shared encoder + per-horizon heads
2. **Horizon bundle ranking** - Select complementary, non-redundant horizons
3. **Cross-horizon ensembles** - Ridge-blended predictions across time scales
4. **Regime-based horizon arbitration** - Volatility-adaptive horizon selection
5. **Feature transfer** - Warm-start from coarser intervals

---

## Determinism, SST, and DRY Compliance

### Critical Determinism Risks

#### Risk 1: Rounding Policy Flip (Phase 3) - BREAKING CHANGE
**Severity**: Critical | **Impact**: Historical reproducibility broken

Current code uses floor division:
```python
target_horizon_bars = horizon_minutes // interval_minutes  # Floor
```

Plan proposes ceil:
```python
target_horizon_bars = minutes_to_bars(horizon, interval, "ceil")  # Ceil
```

**Impact at non-aligned intervals**:
- 30m horizon / 7m interval: floor=4 bars, ceil=5 bars
- CV fold boundaries change
- Old runs (floor) can't be compared to new runs (ceil)

**Mitigation**:
1. Add `_ROUNDING_POLICY_VERSION: 2` to config fingerprints
2. Phase 3 requires explicit sign-off
3. Document metric delta before rollout
4. Old runs tagged with `rounding_policy: floor`, new with `ceil`

#### Risk 2: Config Fingerprint Missing Interval Component
**Severity**: High | **Impact**: Cache collisions across intervals

Current `compute_config_hash()` doesn't include interval:
```python
# Old: hash(pipeline_config, model_configs)
# Missing: interval_minutes, rounding_policy
```

**Result**: Same config hash at 5m and 1m → wrong cached features loaded silently

**Mitigation**:
1. Phase 0.5 must update fingerprinting BEFORE Phase 1
2. Add to `compute_config_hash()`:
   ```python
   extended = {**config, '_interval_minutes': interval, '_rounding_policy': 'ceil'}
   ```

#### Risk 3: DRY Violation - Two Conversion Functions
**Severity**: Medium | **Impact**: Inconsistent rounding

Existing function:
```python
# TRAINING/common/utils/horizon_conversion.py
def horizon_minutes_to_bars(...) -> Optional[int]:
    return round(ratio)  # Round, not ceil!
```

Plan adds:
```python
# TRAINING/common/interval.py
def minutes_to_bars(..., rounding="ceil") -> int:
    return ceil(...)  # Ceil!
```

**Problem**: Different rounding in different places

**Mitigation**:
1. Document clearly in Phase 0:
   - `minutes_to_bars(ceil)` → feature/purge windows (conservative)
   - `horizon_minutes_to_bars(round)` → target horizon validation (exact match)
2. Add to SST catalog with usage guidance

### SST (Single Source of Truth) Requirements

#### SST Requirement 1: Interval Provider
**Problem**: `interval_minutes` stored in 5+ places (config, registry, manifest, metadata, etc.)

**Solution**: Create SST helper in Phase 0:
```python
# TRAINING/common/interval_provider.py (NEW)
def get_interval_spec(context: PipelineContext) -> IntervalSpec:
    """Single source of truth for interval in current run.

    Precedence:
    1. Detected from data timestamps (highest priority)
    2. Explicit config override
    3. Default (5m)
    """
```

#### SST Requirement 2: PurgeSpec Everywhere
**Problem**: `purge_overlap = 17` hardcoded in 4+ places

**Solution**: Phase 2B must enforce:
- ALL purge values use `PurgeSpec`
- Remove ALL hardcoded `purge_overlap = 17`
- Fail closed if raw int passed

#### SST Requirement 3: Registry Stores Original Definition
**Problem**: Registry encodes interval assumption (lag_bars × 5m = lookback_minutes)

**Solution**: Registry stores **original feature definition**:
```yaml
features:
  adx_14:
    original_interval_minutes: 5  # When feature was defined
    lag_bars: 14                  # Original bar count
    lookback_minutes: 70          # Computed: 14 × 5
```

Runtime derives current bars: `minutes_to_bars(70, current_interval)`

### DRY (Don't Repeat Yourself) Requirements

#### DRY Requirement 1: Unified Conversion with Clear Purpose
Keep both functions but document clearly:
```python
# SST catalog entry:
# minutes_to_bars(minutes, interval, "ceil") → Feature windows, purge (conservative)
# horizon_minutes_to_bars(minutes, interval) → Target validation (exact, returns None if inexact)
```

#### DRY Requirement 2: Use Atomic Writes
ALL artifact writes must use:
- `write_atomic_json()` - JSON artifacts
- `write_atomic_yaml()` - YAML configs
- `canonical_json()` - For hashing/signatures

Never use raw `json.dump()` or `yaml.dump()`.

### Backward Compatibility Requirements

#### Requirement 1: v1 Feature Fallback
Phase 1 registry migration must support v1 features:
```python
def get_lookback_minutes(feature, interval_minutes=None):
    if 'lookback_minutes' in meta:
        return meta['lookback_minutes']  # v2
    if 'lag_bars' in meta:
        return meta['lag_bars'] * (interval_minutes or 5)  # v1 fallback
```

#### Requirement 2: Feature Aliases (Not Renames)
Phase 19 must NOT rename features. Instead:
```yaml
features:
  ret_short:  # New interval-agnostic name
    aliases: [ret_5m]  # Old name still works
```

#### Requirement 3: Old Manifest Fallback
Phase 7 must handle old manifests without `interval_minutes`:
```python
if 'interval_minutes' not in manifest:
    logger.warning("Old run, inferring interval from data")
    interval = infer_from_data_or_default(5)
```

### Critical Path Corrections

The following sequencing is REQUIRED for determinism:

```
Phase 0 (Core Types)
    ↓
Phase 0.5 (Fingerprinting) ← MUST update config hash with interval BEFORE Phase 1
    ↓
Phase 13 (Cache Keys) ← MUST add interval to cache BEFORE Phase 11
    ↓
Phase 1 (Registry v2) + Phase 11 (Feature Windows)  ← Now safe to run in parallel
    ↓
Phase 2B (Purge) + Phase 20 (Cross-Sectional Purge)
    ↓
Phase 3 (Rounding Policy) ← SIGN-OFF REQUIRED, breaks historical reproducibility
    ↓
[Remaining phases]
```

**Critical ordering**:
1. Fingerprinting before registry migration (cache invalidation)
2. Cache keys before feature windows (no stale cache)
3. Rounding policy as separate gate (explicit sign-off)

---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Existing Infrastructure](#existing-infrastructure)
3. [Gaps and Blind Spots](#gaps-and-blind-spots)
4. [Implementation Phases](#implementation-phases)
5. [Model Improvement Opportunities](#model-improvement-opportunities)
6. [File Change Inventory](#file-change-inventory)
7. [Code Specifications](#code-specifications)
8. [Testing Requirements](#testing-requirements)
9. [Rollout Strategy](#rollout-strategy)
10. [Checklist](#checklist)

---

## Current State Analysis

### The Problem
Features store `lag_bars` (row count), not time. The registry assumes 5m bars:

```yaml
# CONFIG/data/feature_registry.yaml
features:
  adx_14:
    lag_bars: 14  # Means 14 bars = 70 minutes (at 5m)
```

At runtime, code converts: `lookback_minutes = lag_bars * interval_minutes`

**If interval changes from 5m to 1m:**
- Code thinks `adx_14` has 14-minute lookback (wrong!)
- Should be 70 minutes regardless of data interval
- Result: **data leakage** - features appear "safe" when they're not

### Interval Dependency Map

| Component | Depends On | What Breaks |
|-----------|------------|-------------|
| Feature lookbacks | `lag_bars * interval` | 5x wrong at 1m |
| Purge/embargo | `buffer_bars * interval` | CV splits wrong |
| CV purge_overlap | `horizon / interval` | Leakage |
| Sequential lookback_T | Hardcoded 60 bars | Wrong history window |
| Ring buffer TTL | Hardcoded 300s | Stale data |
| allowed_horizons | Bar-based | Wrong filtering |
| max_sequence_gap | Hardcoded 300s | Sequence validation breaks |

### Critical Conversion Points (must update)

| File | Line | Current Code | Issue |
|------|------|--------------|-------|
| `TRAINING/ranking/utils/leakage_budget.py` | ~786 | `result = float(lag_bars * interval_minutes)` | v1 schema only |
| `TRAINING/ranking/utils/feature_time_meta.py` | 89 | `lookback = meta.lookback_bars * interval` | Prefers bars |
| `TRAINING/ranking/utils/resolved_config.py` | 100 | `purge_buffer_minutes = purge_buffer_bars * interval_minutes` | No abstraction |
| `TRAINING/model_fun/ensemble_trainer.py` | 172 | `purge_overlap = 17` | Hardcoded 5m assumption |
| `TRAINING/ranking/multi_model_feature_selection.py` | 1977-1978 | `target_horizon_bars = horizon // interval` | Floor division (BUG) |
| `TRAINING/orchestration/interfaces/unified_training_interface.py` | ~varies | `purge_overlap = 17` | Another hardcode |

### Hardcoded Values Requiring Update

| File | Value | Assumption |
|------|-------|------------|
| `ensemble_trainer.py:172` | `purge_overlap = 17` | 60m/5m + 5 buffer |
| `multi_model_feature_selection.py:1980` | `purge_overlap = 17` | Same fallback |
| `unified_training_interface.py` | `purge_overlap = 17` | Same fallback |
| `seq_ring_buffer.py:28` | `ttl_seconds = 300.0` | 5m interval |
| `sequential.yaml:10` | `lookback_T: 60` | 300min at 5m |
| `sequential.yaml:87` | `max_sequence_gap: 300` | 5m interval |
| `model_evaluation.py:1590` | `max_lookback_bars = 288` | 1 day at 5m |
| `LIVE_TRADING/models/inference.py` | `ttl_seconds=300.0` | 5m interval |

---

## Existing Infrastructure

### 1. Duration Class ✓
**Location**: `TRAINING/common/utils/duration_parser.py:123-193`

```python
@dataclass(frozen=True)
class Duration:
    microseconds: int  # Canonical representation

    @staticmethod
    def from_seconds(seconds: float) -> "Duration": ...
    def to_minutes(self) -> float: ...
    def to_seconds(self) -> float: ...
    # Full arithmetic: +, *, <, <=, >, >=, ==
```

Also provides:
- `parse_duration("85.0m")` - parse string to Duration
- `parse_duration("1h30m")` - compound durations supported
- `parse_duration_bars(20, "5m")` - explicit bar→time conversion
- `format_duration(d)` - Duration to string

### 2. FeatureTimeMeta ✓
**Location**: `TRAINING/ranking/utils/feature_time_meta.py:19-57`

```python
@dataclass(frozen=True)
class FeatureTimeMeta:
    name: str
    native_interval_minutes: Optional[float] = None
    embargo_minutes: float = 0.0
    lookback_bars: Optional[int] = None       # Bar-based (v1)
    lookback_minutes: Optional[float] = None  # Time-based (v2 PREFERRED)
    max_staleness_minutes: Optional[float] = None
    publish_offset_minutes: float = 0.0
```

**Already has `lookback_minutes` field** - just need to populate it.

### 3. PurgedTimeSeriesSplit ✓
**Location**: `TRAINING/ranking/utils/purged_time_series_split.py`

Already supports **time-based mode**:
```python
purge_overlap_time: Optional[pd.Timedelta]  # Time-based purge
purge_overlap: int  # Row-based (legacy)
```

**Key insight**: Plan should integrate with this, not replace it.

### 4. Horizon Extraction ✓
**Location**: `TRAINING/ranking/utils/leakage_filtering.py`

Functions `_extract_horizon()` already parse target suffixes (`_5m`, `_1h`, `_60d`).

### 5. Multi-Horizon Predictor ✓ (Live Trading)
**Location**: `LIVE_TRADING/prediction/predictor.py`

Production code already implements:
- `MultiHorizonPredictor` - coordinates 5m, 10m, 15m, 30m, 60m predictions
- Ridge-blended weights across horizons
- Confidence scoring per horizon

**This is the architectural ideal for training to match.**

### 6. Registry Interval Validation ✓
**Location**: `TRAINING/common/feature_registry.py:372-437`

Already validates `registry_bar_minutes` vs `current_bar_minutes` with strict/best_effort policy.

### 7. ResolvedConfig ✓
**Location**: `TRAINING/ranking/utils/resolved_config.py:33-160`

Already tracks `interval_minutes`, `horizon_minutes`, `purge_minutes`, `embargo_minutes`.

---

## Gaps and Blind Spots

### Gap 1: No `IntervalSpec` Type
Current code passes `interval_minutes: float` everywhere. Need a proper type with source tracking.

### Gap 2: No Centralized `minutes_to_bars()` with Rounding Policy
Some places use `//` (floor), need `ceil` to avoid under-lookback.

**Critical bug at line 1977**: `target_horizon_bars = horizon // interval` uses floor division.

### Gap 3: Registry Has `lag_bars`, Not `lookback_minutes`
800+ features need migration. Schema version needed.

### Gap 4: `allowed_horizons` is Ambiguous
```yaml
default_allowed_horizons: [1, 2, 3, 5, 12, 24, 60]
```
No documentation whether these are bars or minutes. Usage suggests bars.

### Gap 5: Sequential Config Hardcoded (Partial)
- `lookback_T: 60` is hardcoded bars
- `horizon_map` is ALREADY in minutes (plan was initially wrong here)
- `max_sequence_gap: 300` seconds hardcoded

### Gap 6: Ring Buffer TTL Hardcoded in 3 Places
1. `seq_ring_buffer.py:28` - `ttl_seconds: float = 300.0`
2. `sequential.yaml` - `ttl_seconds: 300`
3. `LIVE_TRADING/models/inference.py` - `ttl_seconds=300.0`

### Gap 7: `purge_overlap = 17` Hardcoded in 3 Places
1. `ensemble_trainer.py:172`
2. `multi_model_feature_selection.py:1980`
3. `unified_training_interface.py`

### Gap 8: No Multi-Horizon Training Strategy
Live trading has `MultiHorizonPredictor` but training produces single-horizon models.

### Gap 9: No Horizon Bundle Ranking
Targets ranked independently. No diversity/correlation analysis across horizons.

### Gap 10: Floor Division Bug
`multi_model_feature_selection.py:1977` uses `//` instead of `ceil()`.

### Gap 11: Model Metadata Missing Interval Provenance
**Location**: `LIVE_TRADING/models/loader.py:136-151`, model serialization

Models serialize to `model_meta.json` without `interval_minutes` or `interval_source`. Inference cannot validate data interval matches training interval.

### Gap 12: Feature Computation Layer - Bar-Based Windows
**Location**: `DATA_PROCESSING/features/comprehensive_builder.py:160-177`

Feature functions use hardcoded bar counts:
```python
rolling_mean(20)  # 20 bars, not 100 minutes
rolling_std(5)    # 5 bars, not 25 minutes
```
At 5m: `rolling_mean(20)` = 100m. At 1m: `rolling_mean(20)` = 20m (wrong!)

### Gap 13: Target Computation - Bar Shifts Without Time Tracking
**Location**: `DATA_PROCESSING/targets/hft_forward.py:50-65`

Targets defined with time semantics (`fwd_ret_30m`) but computed using `shift(-horizon_bars)` without storing which interval was used.

### Gap 14: Cache Keys Missing Interval Component
**Location**: `TRAINING/common/utils/cache_manager.py:38-58`

Cache keys built from `(target, config_hash, view)` but NO interval component. Running at 1m then 5m reuses cached results from wrong interval.

### Gap 15: Feature-Target Alignment - No Temporal Validation
**Location**: `DATA_PROCESSING/features/comprehensive_builder.py:382-392`

Features and targets loaded separately without timestamp verification. No check that feature rows align temporally with target rows.

### Gap 16: Data Gap Detection Missing
**Location**: Target/feature computation

No detection of market close gaps. `shift(-1)` assumes contiguous bars but bars may be missing during market closure.

### Gap 17: Live Inference - No Cross-Interval Validation Gate
**Location**: `LIVE_TRADING/models/inference.py:116-173`

`InferenceEngine.predict()` loads models without checking if data interval matches training interval. Mismatched intervals produce garbage silently.

### Gap 18: Manifest Missing Interval Recording
**Location**: `TRAINING/orchestration/utils/manifest.py:150+`

Manifest records git SHA and config hash but NOT `interval_minutes`. Cannot validate inference uses same interval as training.

### Gap 19: Model Hyperparameters - No Interval Scaling Guidance
**Location**: `CONFIG/models/*.yaml`

Model configs have no guidance on how hyperparameters should scale with interval. Tree depth, learning rate may need adjustment at 1m vs 5m.

### Gap 20: Feature Window Times Not in Registry
**Location**: `CONFIG/data/feature_registry.yaml`

Features have `lag_bars` but no `window_minutes` for rolling calculations. Registry doesn't capture feature computation windows.

### Gap 21: Feature Names Embed Interval Suffixes
**Location**: `CONFIG/excluded_features.yaml`, `CONFIG/pipeline/pipeline_config.yaml`, `DATA_PROCESSING/features/simple_features.py`

Features named with hardcoded suffixes: `ret_1m`, `vol_5m`, `ret_5m`, `fwd_ret_5m`. When switching intervals, feature registry lookup fails because names don't exist.

### Gap 22: Test Fixtures Only Support 5m
**Location**: `DOCS/02_reference/testing/TESTING_NOTICE.md`, test data paths

Explicit statement: "All testing and development is performed using 5-minute interval data". No test fixtures for 1m, 15m, or other intervals. Cross-interval validation tests don't exist.

### Gap 23: Cross-Sectional CV Purge Hardcoded
**Location**: `TRAINING/data/processing/cross_sectional.py:100-142`

```python
purge_overlap: int = 17) -> ...:
    """purge_overlap: ... (default: 17 = 60m target / 5m bars + 5 buffer)"""
```

Cross-sectional CV has same 5m assumption as time-series CV but wasn't identified.

### Gap 24: Live Trading Horizons Not Configurable
**Location**: `LIVE_TRADING/common/constants.py:20-36`

Hardcoded:
```python
HORIZONS: List[str] = ["5m", "10m", "15m", "30m", "60m", "1d"]
```

Live trading can't use models trained at non-standard intervals (1m, 2m, 3m).

### Gap 25: Volatility Scaling Assumes Annualized
**Location**: `LIVE_TRADING/sizing/vol_scaling.py:56-74`

Position sizing assumes volatility is annualized. If training at 1m computes intra-minute volatility (much higher raw values), z-score clipping will be wrong.

### Gap 26: Config Validation Missing Interval Consistency
**Location**: `CONFIG/config_schemas.py:16-88`

`DataConfig` has `bar_interval` and `base_interval_minutes` but no validation that they don't conflict. Both could be set differently with no error.

### Gap 27: Registry Schema Missing Window Definitions
**Location**: Registry schema design

Plan Phase 1 adds `lookback_minutes` but features like `rolling_mean(20)` also need `window_minutes`. Lookback != window size for rolling calculations.

### Gap 28: No Resampling/Interpolation Logic
**Location**: Feature computation layer

No code exists to handle multi-interval features (e.g., compute 1m feature from 5m data). Multi-interval experiments (Phase 10) impossible until resampling is defined.

### Gap 29: Timezone Handling Not Validated
**Location**: Multiple files (33+ handle timezones)

No validation that timezone doesn't affect interval detection. Market close gaps (15.5 hours overnight) exist regardless of timezone but could confuse interval auto-detection.

### Gap 30: Evaluation Metrics Not Interval-Aware
**Location**: Model evaluation/scoring

Time-dependent metrics (Sharpe ratio, annualized returns) need interval-aware scaling. Not currently addressed.

---

## Implementation Phases (Expanded)

### Phase 0: Core Types and Utilities
**Risk**: None (new code only)
**Validation**: Unit tests pass

**Deliverables:**
1. `TRAINING/common/interval.py` - `IntervalSpec`, `IntervalSource`, conversion utilities
2. `TRAINING/ranking/utils/purge.py` - `PurgeSpec`, `compute_purge_minutes()`

**No existing code changes** - just new modules.

---

### Phase 0.5: Validation Infrastructure
**Risk**: Low
**Validation**: Feature flags work, comparison harness runs

**Deliverables:**
1. Add `use_v2_lookback: bool` feature flag to `CONFIG/pipeline/pipeline_config.yaml`
2. Add `use_ceil_rounding: bool` feature flag (default=false for safety)
3. Add comparison test harness for v1 vs v2 results
4. Add logging for conversion auditing

**Purpose**: Enable gradual rollout with fallback.

---

### Phase 1: Registry Schema v2
**Risk**: Medium (800+ features)
**Validation**: All features have `lookback_minutes`, tests pass

**Deliverables:**
1. Update `CONFIG/data/feature_registry.yaml` - add `schema_version: 2`, `lookback_minutes`
2. Migration script `scripts/migrate_feature_registry_v1_to_v2.py`
3. Update `TRAINING/common/feature_registry.py` - support v2 schema with fallback
4. Add `get_lookback_minutes()` and `get_schema_version()` methods

**Backward compatible** - v1 features still work via `lag_bars * assumed_interval`.

---

### Phase 2A: Leakage/Feature Conversion
**Risk**: Medium (changes leakage detection)
**Validation**: Leakage decisions identical at 5m, correct at 1m/15m

**Deliverables:**
1. `TRAINING/ranking/utils/leakage_budget.py` - prefer `lookback_minutes`
2. `TRAINING/ranking/utils/feature_time_meta.py` - prefer `lookback_minutes`

**Gate**: Compare leakage decisions before/after on test dataset.

---

### Phase 2B: Purge Consolidation
**Risk**: Medium (affects CV splits)
**Validation**: CV splits identical at 5m

**Deliverables:**
1. Replace **all 3** hardcoded `purge_overlap=17` sites:
   - `ensemble_trainer.py:172`
   - `multi_model_feature_selection.py:1980`
   - `unified_training_interface.py`
2. Integrate with existing `PurgedTimeSeriesSplit.purge_overlap_time`
3. `TRAINING/ranking/utils/resolved_config.py` - use PurgeSpec

**Gate**: CV fold boundaries match historical at 5m.

---

### Phase 3: Rounding Policy Flip
**Risk**: HIGH (changes CV boundaries at non-aligned intervals)
**Validation**: Explicit sign-off required

**Deliverables:**
1. Enable `use_ceil_rounding=true` by default in conversion utilities
2. Update `multi_model_feature_selection.py:1977` to use `minutes_to_bars()`
3. Document historical reproducibility impact
4. Add deprecation warning for floor division

**Gate**: Full regression test suite passes. Document any metric changes.

---

### Phase 4: Sequential Model Support
**Risk**: Medium
**Validation**: Sequential models produce identical results at 5m

**Deliverables:**
1. Update `CONFIG/pipeline/training/sequential.yaml`:
   - Add `lookback_minutes: 300`
   - Derive `lookback_T` at runtime
   - Parameterize `max_sequence_gap`
2. Update `TRAINING/models/family_router.py` - derive `lookback_T`
3. Update TTL in 3 locations:
   - `TRAINING/common/live/seq_ring_buffer.py`
   - `CONFIG/pipeline/training/sequential.yaml`
   - `LIVE_TRADING/models/inference.py`

---

### Phase 5: Hardcoded Value Cleanup
**Risk**: Low
**Validation**: Tests pass, no 5m assumptions in codebase

**Deliverables:**
1. `TRAINING/ranking/predictability/model_evaluation.py` - `max_lookback_minutes`
2. `CONFIG/pipeline/training/preprocessing.yaml` - parameterize `max_sequence_gap`
3. Audit for remaining hardcoded `5`, `300`, `17`, `288` values

---

### Phase 6: `allowed_horizons` Migration
**Risk**: Low
**Validation**: Horizon filtering works correctly

**Deliverables:**
1. Add `allowed_horizon_minutes` to registry schema
2. Update horizon filtering logic
3. Migration for existing features
4. Deprecate bar-based `allowed_horizons`

---

### Phase 7: Artifact Versioning
**Risk**: Low
**Validation**: Manifests include interval, validation gates work

**Deliverables:**
1. Update manifest schema to include interval metadata
2. Add validation gates at stage boundaries
3. Inference-time compatibility checks
4. Add `interval_minutes` to run fingerprint

---

### Phase 8: Multi-Horizon Training (New Capability)
**Risk**: Medium (new feature)
**Validation**: Multi-horizon models train and produce valid predictions

**Deliverables:**
1. Add `HorizonBundle` type for grouping related targets
2. Update target ranking to score bundles by diversity
3. Add multi-task training strategy to `CONFIG/experiments/`
4. Wire `MultiTask` model family for shared encoder + per-horizon heads

**New experiment config:**
```yaml
intelligent_training:
  strategy: multi_horizon_bundle
  bundle_config:
    horizons: [5, 15, 60]  # minutes
    diversity_threshold: 0.7  # max correlation for inclusion
    blend_method: ridge_risk_parity
```

---

### Phase 9: Cross-Horizon Ensemble (New Capability)
**Risk**: Medium (new feature)
**Validation**: Ensemble produces blended predictions

**Deliverables:**
1. Add cross-horizon stacking to `ensemble_trainer.py`
2. Learn ridge weights across horizons (like live trading)
3. Add horizon decay function (shorter horizons weighted higher)
4. Update ensemble config:
```yaml
ensemble:
  cross_horizon:
    enabled: true
    base_horizons: [5, 15, 60]
    decay_function: exponential
    decay_half_life_minutes: 30
```

---

### Phase 10: Multi-Interval Experiments (Future)
**Risk**: High (significant new capability)
**Validation**: Models generalize across intervals

**Deliverables:**
1. Add `multi_intervals` to experiment config
2. Cross-interval validation (train 5m, validate 1m)
3. Feature transfer warm-start from coarser intervals
4. Regime-based interval selection

---

### Phase 11: Feature Computation Windows (NEW - Critical)
**Risk**: High (affects all feature calculations)
**Validation**: Features produce identical values at 5m, correct values at other intervals

**Deliverables:**
1. Add `window_minutes` to feature registry schema alongside `lag_bars`
2. Update `DATA_PROCESSING/features/comprehensive_builder.py` to use time-based windows
3. Convert all `rolling(N)` calls to `rolling(minutes_to_bars(window_minutes, interval))`
4. Add feature computation audit logging

**Files:**
- `CONFIG/data/feature_registry.yaml` - add `window_minutes` per feature
- `DATA_PROCESSING/features/comprehensive_builder.py` - parameterize windows
- `LIVE_TRADING/models/feature_builder.py` - same updates

---

### Phase 12: Target Computation Tracking (NEW - Critical)
**Risk**: Medium
**Validation**: Target metadata includes interval used at computation

**Deliverables:**
1. Record `interval_minutes` at target computation time
2. Store target metadata in parquet/artifacts
3. Add validation gate: reject if inference interval != training interval

**Files:**
- `DATA_PROCESSING/targets/hft_forward.py` - add interval recording
- `DATA_PROCESSING/targets/comprehensive_builder.py` - same

---

### Phase 13: Cache Key Interval Component (NEW - High Priority)
**Risk**: Low (additive change)
**Validation**: Cache invalidates correctly when interval changes

**Deliverables:**
1. Add `interval_minutes` to cache key construction
2. Update `build_cache_key_with_symbol()` signature
3. Invalidate existing caches on deployment

**Files:**
- `TRAINING/common/utils/cache_manager.py:38-58`
- `TRAINING/ranking/feature_selector.py:118-140`

---

### Phase 14: Feature-Target Alignment Validation (NEW)
**Risk**: Medium
**Validation**: Alignment errors caught before training

**Deliverables:**
1. Add timestamp alignment check before feature-target merge
2. Add row count validation
3. Log/fail on misalignment

**Files:**
- `DATA_PROCESSING/features/comprehensive_builder.py:382-392`
- Add new `validate_alignment()` utility

---

### Phase 15: Data Gap Detection (NEW)
**Risk**: Medium
**Validation**: Market close gaps detected and handled

**Deliverables:**
1. Detect gaps in timestamp sequence
2. Option to fill gaps or exclude gap-adjacent rows
3. Log gap statistics

**Example:**
```python
gaps = data.ts.diff() != expected_interval_ns
if gaps.any():
    logger.warning(f"Data gaps: {gaps.sum()} rows")
```

---

### Phase 16: Model Metadata Interval Provenance (NEW - Critical)
**Risk**: Medium
**Validation**: Models reject mismatched intervals at inference

**Deliverables:**
1. Add `interval_minutes` and `interval_source` to `model_meta.json`
2. Add `feature_lookbacks_minutes` dict to metadata
3. Add validation gate in `InferenceEngine.predict()`

**Files:**
- Model serialization in `TRAINING/models/specialized/core.py`
- `LIVE_TRADING/models/loader.py:136-151`
- `LIVE_TRADING/models/inference.py:116-173`

**New metadata schema:**
```json
{
  "interval_minutes": 5.0,
  "interval_source": "detected",
  "feature_lookbacks_minutes": {
    "adx_14": 70,
    "rsi_7": 35
  },
  "target_horizon_minutes": 30
}
```

---

### Phase 17: Live Inference Cross-Interval Gate (NEW - Critical)
**Risk**: Low (validation only)
**Validation**: Mismatched intervals raise errors

**Deliverables:**
1. Add interval check in `InferenceEngine.predict()`
2. Reject predictions when data interval != model training interval
3. Add clear error messages

**Example:**
```python
if training_interval != data_interval:
    raise InferenceError(
        f"Interval mismatch: model trained at {training_interval}m, "
        f"data is {data_interval}m"
    )
```

---

### Phase 18: Hyperparameter Scaling Guidance (NEW - Low Priority)
**Risk**: Low (documentation + optional config)
**Validation**: Guidelines documented, optional scaling in place

**Deliverables:**
1. Document how hyperparameters should scale with interval
2. Add `interval_scaling` section to model configs (optional)
3. Add automated scaling option for tree-based models

**Example config addition:**
```yaml
interval_scaling:
  reference_interval_minutes: 5
  num_leaves:
    1m: 15
    5m: 31
    15m: 63
```

---

### Phase 19: Feature Naming Convention (NEW - High Priority)
**Risk**: High (affects feature discovery)
**Validation**: Features discoverable regardless of data interval

**Deliverables:**
1. Rename features to be interval-agnostic (e.g., `ret_short` instead of `ret_5m`)
2. Store interval in feature metadata, not name
3. Update feature discovery logic
4. Add migration for existing named features

**Files:**
- `CONFIG/excluded_features.yaml`
- `CONFIG/pipeline/pipeline_config.yaml`
- `DATA_PROCESSING/features/simple_features.py`

---

### Phase 20: Cross-Sectional Purge Fix (NEW)
**Risk**: Medium
**Validation**: Cross-sectional CV uses dynamic purge

**Deliverables:**
1. Replace hardcoded `purge_overlap=17` in cross-sectional.py
2. Use same `PurgeSpec` abstraction as time-series CV

**Files:**
- `TRAINING/data/processing/cross_sectional.py:100-142`

---

### Phase 21: Live Trading Horizon Configuration (NEW)
**Risk**: Medium
**Validation**: Live trading horizons configurable

**Deliverables:**
1. Move `HORIZONS` from constants to config
2. Add horizon discovery from model metadata
3. Support non-standard intervals (1m, 2m, 3m)

**Files:**
- `LIVE_TRADING/common/constants.py:20-36`
- Add `CONFIG/live_trading/horizons.yaml`

---

### Phase 22: Multi-Interval Test Fixtures (NEW)
**Risk**: Low
**Validation**: Tests run at 1m, 5m, 15m intervals

**Deliverables:**
1. Generate test fixtures for 1m, 15m intervals
2. Add parameterized tests across intervals
3. Update testing documentation

**Files:**
- Test data generation scripts
- `DOCS/02_reference/testing/TESTING_NOTICE.md`

---

### Phase 23: Metric Scaling Documentation (NEW - Low Priority)
**Risk**: Low
**Validation**: Documentation complete

**Deliverables:**
1. Document volatility scaling by interval
2. Document Sharpe ratio annualization rules
3. Add interval-aware metric helpers if needed

---

### Phase 24: Config Validation Rules (NEW)
**Risk**: Low
**Validation**: Config conflicts raise errors

**Deliverables:**
1. Add validation that `bar_interval` and `base_interval_minutes` are consistent
2. Add interval conflict detection in config loading

**Files:**
- `CONFIG/config_schemas.py:16-88`
- `CONFIG/config_loader.py`

---

## Phase Summary Table

| Phase | Name | Risk | Priority | Est. Hours |
|-------|------|------|----------|------------|
| 0 | Core Types | None | P0 | 4h |
| 0.5 | Validation Infrastructure | Low | P0 | 4h |
| 1 | Registry Schema v2 | Medium | P0 | 8h |
| 2A | Leakage/Feature Conversion | Medium | P1 | 6h |
| 2B | Purge Consolidation | Medium | P1 | 4h |
| 3 | Rounding Policy Flip | **HIGH** | P1 | 4h |
| 4 | Sequential Model Support | Medium | P1 | 6h |
| 5 | Hardcoded Cleanup | Low | P2 | 4h |
| 6 | allowed_horizons Migration | Low | P2 | 4h |
| 7 | Artifact Versioning | Low | P1 | 4h |
| 8 | Multi-Horizon Training | Medium | P3 | 12h |
| 9 | Cross-Horizon Ensemble | Medium | P3 | 8h |
| 10 | Multi-Interval Experiments | High | Future | 20h |
| **11** | **Feature Computation Windows** | **High** | **P0** | **12h** |
| **12** | **Target Computation Tracking** | Medium | P1 | 4h |
| **13** | **Cache Key Interval** | Low | P1 | 2h |
| **14** | **Feature-Target Alignment** | Medium | P2 | 4h |
| **15** | **Data Gap Detection** | Medium | P2 | 4h |
| **16** | **Model Metadata Interval** | Medium | P1 | 6h |
| **17** | **Live Inference Gate** | Low | P1 | 3h |
| **18** | **Hyperparameter Scaling** | Low | P3 | 6h |
| **19** | **Feature Naming Convention** | **High** | **P0** | **8h** |
| **20** | **Cross-Sectional Purge Fix** | Medium | P1 | 4h |
| **21** | **Live Trading Horizons** | Medium | P2 | 4h |
| **22** | **Multi-Interval Test Fixtures** | Low | P2 | 6h |
| **23** | **Metric Scaling Documentation** | Low | P3 | 3h |
| **24** | **Config Validation Rules** | Low | P2 | 2h |

**Total estimated work:** ~156 hours (~4 weeks full-time)

---

## Critical Path (Blocking Order)

```
Phase 0 (Core Types)
    ↓
Phase 0.5 (Feature Flags)
    ↓
┌─────────────────────────────────────┐
│ PARALLEL TRACK A                    │
│ Phase 1 (Registry v2)               │
│     ↓                               │
│ Phase 11 (Feature Windows) ←CRITICAL│
│     ↓                               │
│ Phase 2A (Leakage Conversion)       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ PARALLEL TRACK B                    │
│ Phase 12 (Target Tracking)          │
│ Phase 13 (Cache Keys)               │
│ Phase 2B (Purge Consolidation)      │
└─────────────────────────────────────┘
    ↓
Phase 3 (Rounding Policy) ← SIGN-OFF REQUIRED
    ↓
Phase 4 (Sequential Models)
    ↓
┌─────────────────────────────────────┐
│ PARALLEL TRACK C                    │
│ Phase 7 (Artifact Versioning)       │
│ Phase 16 (Model Metadata)           │
│ Phase 17 (Inference Gate)           │
└─────────────────────────────────────┘
    ↓
Phase 5-6 (Cleanup)
    ↓
Phase 14-15 (Alignment, Gap Detection)
    ↓
Phase 8-9 (Multi-Horizon) ← NEW CAPABILITIES
    ↓
Phase 10, 18 (Future)
```

---

## Model Improvement Opportunities

### Opportunity 1: Multi-Horizon Training (+3-5% AUC)
**Current**: Each horizon trained independently
**Proposed**: Shared encoder + per-horizon heads

```
Stage 1: Feature extraction (shared)
  features → SharedEncoder → latent_repr

Stage 2: Per-horizon heads
  latent_repr → Head_5m  → pred_5m
  latent_repr → Head_15m → pred_15m
  latent_repr → Head_60m → pred_60m

Stage 3: Loss = w1*L(5m) + w2*L(15m) + w3*L(60m)
```

**Benefits:**
- 3x fewer models (shared features)
- Better generalization (multi-task regularization)
- Correlated signal diversity

---

### Opportunity 2: Horizon Bundle Ranking (+5-8% AUC)
**Current**: Targets ranked independently
**Proposed**: Rank by diversity benefit

```python
# Current: Top-5 targets by individual score
targets = rank_targets(all_targets)[:5]
# Result: [fwd_ret_5m, fwd_ret_10m, fwd_ret_15m, fwd_ret_30m, fwd_ret_60m]
# Problem: 5m and 10m are ~0.9 correlated = redundant

# Proposed: Top-3 bundles by diversity
bundles = rank_bundles_by_diversity(all_targets, max_correlation=0.7)
# Result: [Bundle(5m), Bundle(60m), Bundle(1d)]
# Benefit: Non-redundant signals for ensemble
```

---

### Opportunity 3: Cross-Horizon Ensemble (+3-5% Sharpe)
**Current**: Per-horizon blending only
**Proposed**: Inter-horizon stacking

```
Stage 1: Base models per horizon
  LightGBM(5m)  → pred_5m
  LightGBM(15m) → pred_15m
  LightGBM(60m) → pred_60m

Stage 2: Meta-learner (Ridge)
  [pred_5m, pred_15m, pred_60m] → final_signal

Stage 3: Learned weights capture:
  - Horizon decay (short-term stronger)
  - Correlation reduction (diversification)
```

---

### Opportunity 4: Horizon Alpha Decay
**Intuition**: Short-horizon predictions decay faster
**Implementation**:
```yaml
blend_config:
  decay_function: exponential
  decay_half_life_minutes: 30

# Weight calculation:
# w_h = base_weight * exp(-horizon_minutes / half_life)
# 5m:  w = 1.0 * exp(-5/30)  = 0.85
# 15m: w = 1.0 * exp(-15/30) = 0.61
# 60m: w = 1.0 * exp(-60/30) = 0.14
```

---

### Opportunity 5: Regime-Based Horizon Selection
**Intuition**: High volatility → prefer short horizons
**Implementation**:
```python
def select_horizon(volatility_regime: str) -> list[int]:
    if volatility_regime == "high":
        return [5, 10]  # Short horizons more reliable
    elif volatility_regime == "low":
        return [60, 240]  # Long horizons have signal
    else:
        return [15, 30]  # Balanced
```

---

## File Change Inventory

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `TRAINING/common/interval.py` | IntervalSpec, conversions | 0 |
| `TRAINING/ranking/utils/purge.py` | PurgeSpec, compute_purge | 0 |
| `scripts/migrate_feature_registry_v1_to_v2.py` | Registry migration | 1 |
| `tests/test_interval.py` | Interval unit tests | 0 |
| `tests/test_purge.py` | Purge unit tests | 0 |

### Modified Files (Critical Path)

| File | Changes | Phase |
|------|---------|-------|
| `CONFIG/data/feature_registry.yaml` | schema_version, lookback_minutes | 1 |
| `TRAINING/common/feature_registry.py` | v2 schema support | 1 |
| `TRAINING/ranking/utils/leakage_budget.py` | prefer lookback_minutes | 2A |
| `TRAINING/ranking/utils/feature_time_meta.py` | prefer lookback_minutes | 2A |
| `TRAINING/ranking/utils/resolved_config.py` | PurgeSpec | 2B |
| `TRAINING/model_fun/ensemble_trainer.py` | remove hardcoded purge | 2B |
| `TRAINING/ranking/multi_model_feature_selection.py` | ceil rounding | 3 |
| `TRAINING/orchestration/interfaces/unified_training_interface.py` | remove hardcoded purge | 2B |

### Modified Files (Secondary)

| File | Changes | Phase |
|------|---------|-------|
| `CONFIG/pipeline/training/sequential.yaml` | lookback_minutes, TTL | 4 |
| `TRAINING/models/family_router.py` | derive lookback_T | 4 |
| `TRAINING/common/live/seq_ring_buffer.py` | parameterize TTL | 4 |
| `TRAINING/ranking/predictability/model_evaluation.py` | max_lookback_minutes | 5 |
| `CONFIG/pipeline/training/preprocessing.yaml` | max_sequence_gap | 5 |
| `TRAINING/orchestration/utils/manifest.py` | interval in manifest | 7 |
| `LIVE_TRADING/models/inference.py` | parameterize TTL | 4 |

---

## Code Specifications

### 1. IntervalSpec and Utilities

**File**: `TRAINING/common/interval.py`

```python
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Interval Specification and Conversion Utilities

Single source of truth for interval handling. All bar↔minute conversions
must use these utilities to ensure consistent rounding policy.

Core invariant: Store time (minutes), derive bars with ceil() rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from TRAINING.common.utils.duration_parser import Duration, parse_duration


class IntervalSource(str, Enum):
    """How the interval was determined."""
    CONFIG = "config"          # Explicit in config file
    DETECTED = "detected"      # Auto-detected from timestamps
    EXPLICIT = "explicit"      # Passed as function argument
    UNKNOWN = "unknown"        # Legacy/untracked


@dataclass(frozen=True)
class IntervalSpec:
    """
    Canonical interval specification with provenance tracking.

    Attributes:
        duration: Interval as Duration (canonical representation)
        source: How the interval was determined
        confidence: Detection confidence (1.0 = certain, <1.0 = inferred)
        is_uniform: Whether timestamps are uniformly spaced
    """
    duration: Duration
    source: IntervalSource
    confidence: float = 1.0
    is_uniform: bool = True

    def __post_init__(self) -> None:
        if self.duration.to_minutes() <= 0:
            raise ValueError(f"IntervalSpec.duration must be positive, got {self.duration}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"IntervalSpec.confidence must be in [0, 1], got {self.confidence}")

    @property
    def minutes(self) -> int:
        """Interval in minutes (integer, for bar calculations)."""
        return int(self.duration.to_minutes())

    @property
    def seconds(self) -> float:
        """Interval in seconds."""
        return self.duration.to_seconds()

    @staticmethod
    def from_minutes(
        minutes: int,
        source: IntervalSource = IntervalSource.EXPLICIT,
        confidence: float = 1.0,
        is_uniform: bool = True
    ) -> "IntervalSpec":
        """Create IntervalSpec from minutes."""
        return IntervalSpec(
            duration=Duration.from_seconds(minutes * 60),
            source=source,
            confidence=confidence,
            is_uniform=is_uniform
        )

    @staticmethod
    def from_string(
        interval_str: str,
        source: IntervalSource = IntervalSource.CONFIG,
        confidence: float = 1.0,
        is_uniform: bool = True
    ) -> "IntervalSpec":
        """Create IntervalSpec from string like '5m', '1h'."""
        duration = parse_duration(interval_str)
        return IntervalSpec(
            duration=duration,
            source=source,
            confidence=confidence,
            is_uniform=is_uniform
        )

    def to_dict(self) -> dict:
        """Serialize for artifacts."""
        return {
            "minutes": self.minutes,
            "seconds": self.seconds,
            "source": self.source.value,
            "confidence": self.confidence,
            "is_uniform": self.is_uniform,
        }


# =============================================================================
# Conversion Utilities (Single Source of Truth)
# =============================================================================

def minutes_to_bars(
    lookback_minutes: float,
    interval_minutes: int,
    rounding: str = "ceil"
) -> int:
    """
    Convert time (minutes) to bars with explicit rounding policy.

    CRITICAL: Always use ceil() to avoid under-lookback (data leakage).

    Args:
        lookback_minutes: Lookback period in minutes
        interval_minutes: Bar interval in minutes
        rounding: "ceil" (default, safe) or "floor" (only if you know what you're doing)

    Returns:
        Number of bars

    Raises:
        ValueError: If inputs are invalid
    """
    if lookback_minutes < 0:
        raise ValueError(f"lookback_minutes must be >= 0, got {lookback_minutes}")
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")

    if lookback_minutes == 0:
        return 0

    if rounding == "ceil":
        return int(math.ceil(lookback_minutes / interval_minutes))
    elif rounding == "floor":
        return int(lookback_minutes // interval_minutes)
    else:
        raise ValueError(f"Unknown rounding policy: {rounding}. Use 'ceil' or 'floor'.")


def bars_to_minutes(bars: int, interval_minutes: int) -> int:
    """
    Convert bars to minutes.

    Args:
        bars: Number of bars
        interval_minutes: Bar interval in minutes

    Returns:
        Time in minutes
    """
    if bars < 0:
        raise ValueError(f"bars must be >= 0, got {bars}")
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")
    return bars * interval_minutes


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_interval_target_compatibility(
    *,
    interval_minutes: int,
    target_horizon_minutes: int,
    strict: bool = True,
    min_forward_bars: int = 1
) -> tuple[bool, list[str]]:
    """
    Validate that data interval is compatible with target horizon.

    Rules:
    1. interval <= horizon (can't predict finer than data resolution)
    2. horizon / interval >= min_forward_bars (enough temporal structure)

    Args:
        interval_minutes: Data bar interval
        target_horizon_minutes: Target prediction horizon
        strict: If True, raise on violation; if False, return warnings
        min_forward_bars: Minimum bars forward (default: 1)

    Returns:
        (is_valid, warnings) tuple

    Raises:
        ValueError: If strict=True and validation fails
    """
    warnings = []
    is_valid = True

    # Rule 1: interval <= horizon
    if interval_minutes > target_horizon_minutes:
        msg = (
            f"Invalid: interval ({interval_minutes}m) > horizon ({target_horizon_minutes}m). "
            f"Cannot predict {target_horizon_minutes}m forward with {interval_minutes}m bars."
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
        is_valid = False

    # Rule 2: enough forward bars
    if is_valid:
        forward_bars = minutes_to_bars(target_horizon_minutes, interval_minutes)
        if forward_bars < min_forward_bars:
            msg = f"Only {forward_bars} forward bar(s); recommended minimum is {min_forward_bars}."
            warnings.append(msg)
            # Not a hard failure, just a warning

    # Non-aligned warning
    if is_valid and (target_horizon_minutes % interval_minutes != 0):
        warnings.append(
            f"Horizon ({target_horizon_minutes}m) not evenly divisible by interval ({interval_minutes}m). "
            f"Will use ceil() rounding."
        )

    return is_valid, warnings
```

### 2. PurgeSpec and Utilities

**File**: `TRAINING/ranking/utils/purge.py`

```python
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Purge and Embargo Specification

Single source of truth for temporal safety calculations.
All purge/embargo values are stored in MINUTES, converted to bars at boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from TRAINING.common.interval import minutes_to_bars


@dataclass(frozen=True)
class PurgeSpec:
    """
    Temporal safety specification for CV splits.

    Stored in minutes (time semantics), converted to bars only at CV split creation.

    Attributes:
        purge_minutes: Time to purge after test fold (prevents train seeing test's past)
        embargo_minutes: Time to embargo before test fold (prevents test seeing train's future)
        buffer_minutes: Additional safety buffer
        rounding_policy: How to round when converting to bars ("ceil" recommended)
    """
    purge_minutes: float
    embargo_minutes: float = 0.0
    buffer_minutes: float = 0.0
    rounding_policy: str = "ceil"

    def __post_init__(self) -> None:
        if self.purge_minutes < 0:
            raise ValueError(f"purge_minutes must be >= 0, got {self.purge_minutes}")
        if self.embargo_minutes < 0:
            raise ValueError(f"embargo_minutes must be >= 0, got {self.embargo_minutes}")
        if self.buffer_minutes < 0:
            raise ValueError(f"buffer_minutes must be >= 0, got {self.buffer_minutes}")

    def purge_bars(self, interval_minutes: int) -> int:
        """Convert purge to bars for CV split."""
        return minutes_to_bars(self.purge_minutes, interval_minutes, self.rounding_policy)

    def embargo_bars(self, interval_minutes: int) -> int:
        """Convert embargo to bars for CV split."""
        return minutes_to_bars(self.embargo_minutes, interval_minutes, self.rounding_policy)

    def total_exclusion_minutes(self) -> float:
        """Total time excluded around test fold."""
        return self.purge_minutes + self.embargo_minutes

    def to_dict(self) -> dict:
        """Serialize for artifacts."""
        return {
            "purge_minutes": self.purge_minutes,
            "embargo_minutes": self.embargo_minutes,
            "buffer_minutes": self.buffer_minutes,
            "rounding_policy": self.rounding_policy,
        }

    def to_timedelta(self) -> "pd.Timedelta":
        """Convert purge to pandas Timedelta for PurgedTimeSeriesSplit."""
        import pandas as pd
        return pd.Timedelta(minutes=self.purge_minutes)


def compute_purge_minutes(
    *,
    target_horizon_minutes: float,
    buffer_minutes: float = 5.0,
    max_feature_lookback_minutes: Optional[float] = None,
    include_feature_lookback: bool = False
) -> float:
    """
    Compute purge window in minutes.

    Base formula: purge = horizon + buffer

    If include_feature_lookback=True, also consider max feature lookback
    (features looking back N minutes could see test data).

    Args:
        target_horizon_minutes: Target prediction horizon
        buffer_minutes: Additional safety buffer (default: 5 minutes)
        max_feature_lookback_minutes: Maximum feature lookback (if known)
        include_feature_lookback: Whether to include feature lookback in purge

    Returns:
        Purge window in minutes
    """
    if target_horizon_minutes <= 0:
        raise ValueError(f"target_horizon_minutes must be > 0, got {target_horizon_minutes}")
    if buffer_minutes < 0:
        raise ValueError(f"buffer_minutes must be >= 0, got {buffer_minutes}")

    base_purge = target_horizon_minutes + buffer_minutes

    if include_feature_lookback:
        if max_feature_lookback_minutes is None:
            raise ValueError("include_feature_lookback=True requires max_feature_lookback_minutes")
        if max_feature_lookback_minutes < 0:
            raise ValueError(f"max_feature_lookback_minutes must be >= 0, got {max_feature_lookback_minutes}")

        # Purge must cover both horizon and max feature lookback
        feature_purge = max_feature_lookback_minutes + buffer_minutes
        return max(base_purge, feature_purge)

    return base_purge


def make_purge_spec(
    *,
    target_horizon_minutes: float,
    buffer_minutes: float = 5.0,
    embargo_minutes: float = 0.0,
    max_feature_lookback_minutes: Optional[float] = None,
    include_feature_lookback: bool = False,
    rounding_policy: str = "ceil"
) -> PurgeSpec:
    """
    Create a complete PurgeSpec.

    Convenience function that computes purge_minutes and wraps in PurgeSpec.
    """
    purge_minutes = compute_purge_minutes(
        target_horizon_minutes=target_horizon_minutes,
        buffer_minutes=buffer_minutes,
        max_feature_lookback_minutes=max_feature_lookback_minutes,
        include_feature_lookback=include_feature_lookback
    )

    return PurgeSpec(
        purge_minutes=purge_minutes,
        embargo_minutes=embargo_minutes,
        buffer_minutes=buffer_minutes,
        rounding_policy=rounding_policy
    )
```

### 3. Feature Flags Configuration

**Update**: `CONFIG/pipeline/pipeline_config.yaml`

```yaml
# Interval-agnostic migration flags
interval_agnostic:
  # Use v2 lookback_minutes instead of lag_bars
  use_v2_lookback: false  # Enable after Phase 1 migration

  # Use ceil() rounding for bar conversion (safer, prevents under-lookback)
  use_ceil_rounding: false  # Enable after Phase 3 validation

  # Log all bar/minute conversions for audit
  audit_conversions: false  # Enable during migration

  # Strict mode: fail if v1/v2 results differ
  strict_v1_v2_equivalence: true
```

---

## Testing Requirements

### Unit Tests

**File**: `tests/test_interval.py`

```python
"""Tests for interval module."""

import pytest
from TRAINING.common.interval import (
    IntervalSpec,
    IntervalSource,
    minutes_to_bars,
    bars_to_minutes,
    validate_interval_target_compatibility,
)


class TestMinutesToBars:
    """Test minutes_to_bars conversion."""

    def test_exact_division(self):
        assert minutes_to_bars(100, 5) == 20
        assert minutes_to_bars(60, 1) == 60
        assert minutes_to_bars(300, 15) == 20

    def test_ceil_rounding(self):
        # 101 / 5 = 20.2 → ceil = 21
        assert minutes_to_bars(101, 5, rounding="ceil") == 21
        # 99 / 5 = 19.8 → ceil = 20
        assert minutes_to_bars(99, 5, rounding="ceil") == 20

    def test_floor_rounding(self):
        assert minutes_to_bars(101, 5, rounding="floor") == 20
        assert minutes_to_bars(99, 5, rounding="floor") == 19

    def test_zero_lookback(self):
        assert minutes_to_bars(0, 5) == 0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            minutes_to_bars(-1, 5)
        with pytest.raises(ValueError):
            minutes_to_bars(100, 0)
        with pytest.raises(ValueError):
            minutes_to_bars(100, -5)


class TestIntervalSpec:
    """Test IntervalSpec."""

    def test_from_minutes(self):
        spec = IntervalSpec.from_minutes(5)
        assert spec.minutes == 5
        assert spec.seconds == 300
        assert spec.source == IntervalSource.EXPLICIT

    def test_from_string(self):
        spec = IntervalSpec.from_string("5m")
        assert spec.minutes == 5

        spec = IntervalSpec.from_string("1h")
        assert spec.minutes == 60

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            IntervalSpec.from_minutes(0)
        with pytest.raises(ValueError):
            IntervalSpec.from_minutes(-5)


class TestValidateIntervalTarget:
    """Test interval-target validation."""

    def test_valid_combination(self):
        is_valid, warnings = validate_interval_target_compatibility(
            interval_minutes=5,
            target_horizon_minutes=60,
            strict=False
        )
        assert is_valid

    def test_interval_exceeds_horizon(self):
        with pytest.raises(ValueError):
            validate_interval_target_compatibility(
                interval_minutes=15,
                target_horizon_minutes=10,
                strict=True
            )

    def test_non_aligned_warning(self):
        is_valid, warnings = validate_interval_target_compatibility(
            interval_minutes=7,
            target_horizon_minutes=30,
            strict=False
        )
        assert is_valid
        assert any("not evenly divisible" in w for w in warnings)
```

**File**: `tests/test_purge.py`

```python
"""Tests for purge module."""

import pytest
from TRAINING.ranking.utils.purge import (
    PurgeSpec,
    compute_purge_minutes,
    make_purge_spec,
)


class TestComputePurgeMinutes:
    """Test purge computation."""

    def test_basic_purge(self):
        # 60m horizon + 5m buffer = 65m
        assert compute_purge_minutes(target_horizon_minutes=60) == 65

    def test_custom_buffer(self):
        assert compute_purge_minutes(target_horizon_minutes=60, buffer_minutes=10) == 70

    def test_with_feature_lookback(self):
        # max(60+5, 100+5) = max(65, 105) = 105
        purge = compute_purge_minutes(
            target_horizon_minutes=60,
            buffer_minutes=5,
            max_feature_lookback_minutes=100,
            include_feature_lookback=True
        )
        assert purge == 105


class TestPurgeSpec:
    """Test PurgeSpec."""

    def test_purge_bars_conversion(self):
        spec = PurgeSpec(purge_minutes=65, embargo_minutes=10)

        # At 5m interval: ceil(65/5) = 13 bars
        assert spec.purge_bars(5) == 13

        # At 1m interval: ceil(65/1) = 65 bars
        assert spec.purge_bars(1) == 65

    def test_ceil_rounding(self):
        spec = PurgeSpec(purge_minutes=67)

        # 67 / 5 = 13.4 → ceil = 14
        assert spec.purge_bars(5) == 14

    def test_to_timedelta(self):
        spec = PurgeSpec(purge_minutes=65)
        td = spec.to_timedelta()
        assert td.total_seconds() == 65 * 60
```

### Integration Tests

```python
"""Integration tests for interval-agnostic pipeline."""

def test_same_feature_different_intervals():
    """Same feature produces correct windows at different intervals."""
    from TRAINING.common.feature_registry import get_registry
    from TRAINING.common.interval import minutes_to_bars

    registry = get_registry()

    # Feature with lookback_minutes (v2)
    lookback = registry.get_lookback_minutes("adx_14")
    assert lookback == 70  # 14 bars * 5m

    # At 5m: ceil(70/5) = 14 bars
    assert minutes_to_bars(lookback, 5) == 14

    # At 1m: ceil(70/1) = 70 bars
    assert minutes_to_bars(lookback, 1) == 70

    # At 15m: ceil(70/15) = 5 bars
    assert minutes_to_bars(lookback, 15) == 5


def test_purge_spec_end_to_end():
    """PurgeSpec correctly propagates through CV split."""
    from TRAINING.ranking.utils.purge import make_purge_spec

    spec = make_purge_spec(
        target_horizon_minutes=60,
        buffer_minutes=5,
        max_feature_lookback_minutes=100,
        include_feature_lookback=True
    )

    # At 5m interval: ceil(105/5) = 21 bars
    assert spec.purge_bars(5) == 21

    # At 1m interval: ceil(105/1) = 105 bars
    assert spec.purge_bars(1) == 105


def test_v1_v2_equivalence_at_5m():
    """V1 and V2 produce identical results at 5m interval."""
    from TRAINING.common.feature_registry import get_registry
    from TRAINING.common.interval import minutes_to_bars

    registry = get_registry()

    # Get a feature with both lag_bars (v1) and lookback_minutes (v2)
    metadata = registry.get_feature_metadata("adx_14")

    # V1 calculation
    v1_bars = metadata.get("lag_bars", 0)

    # V2 calculation
    v2_minutes = registry.get_lookback_minutes("adx_14")
    v2_bars = minutes_to_bars(v2_minutes, 5)

    # Must be identical at 5m
    assert v1_bars == v2_bars, f"V1/V2 mismatch: {v1_bars} vs {v2_bars}"
```

---

## Rollout Strategy

### Phase 0-0.5: Foundation (Week 1)
1. Implement `interval.py` and `purge.py`
2. Add feature flags
3. Run unit tests
4. **Gate**: All new tests pass

### Phase 1: Registry Migration (Week 2)
1. Run migration script (dry-run first)
2. Update `FeatureRegistry` class
3. **Gate**: All features have `lookback_minutes`, v1 fallback works

### Phase 2A-2B: Conversion Points (Week 3)
1. Update leakage/feature conversion
2. Consolidate purge hardcodes
3. **Gate**: CV splits identical at 5m, leakage decisions unchanged

### Phase 3: Rounding Policy (Week 4)
1. Enable `use_ceil_rounding=true`
2. Document metric changes
3. **Gate**: Sign-off on CV boundary changes

### Phase 4-5: Sequential & Cleanup (Week 5)
1. Update sequential config
2. Clean up remaining hardcodes
3. **Gate**: Sequential models produce identical results at 5m

### Phase 6-7: Horizons & Artifacts (Week 6)
1. Migrate `allowed_horizons`
2. Add interval to manifests
3. **Gate**: Full pipeline runs with 1m, 5m, 15m data

### Phase 8-9: New Capabilities (Week 7-8)
1. Multi-horizon training
2. Cross-horizon ensemble
3. **Gate**: New models train and produce valid predictions

### Phase 10: Multi-Interval (Future)
1. Cross-interval experiments
2. Feature transfer
3. Regime-based selection

---

## Checklist

### Phase 0: Core Types ✅ COMPLETED (2026-01-18)
- [x] Create `TRAINING/common/interval.py`
- [x] Create `TRAINING/ranking/utils/purge.py`
- [x] Add unit tests for both modules (`tests/test_interval.py`, `tests/test_purge.py`)
- [x] Verify imports work from other modules (85 tests passing)

### Phase 0.5: Validation Infrastructure ✅ COMPLETED (2026-01-18)
- [x] Add feature flags to `CONFIG/pipeline/pipeline.yaml` (interval_agnostic section)
- [x] Add conversion audit logging to `TRAINING/common/interval.py`
- [x] Create v1/v2 comparison test harness (`tests/test_interval_v1_v2_equivalence.py`, 35 tests)

### Phase 1: Registry Migration ✅ COMPLETED (2026-01-18)
- [x] Create migration script (`scripts/migrate_feature_registry_v1_to_v2.py`)
- [x] Run migration on `CONFIG/data/feature_registry.yaml` (211 features migrated, backup created)
- [x] Update `FeatureRegistry` class with `get_lookback_minutes()`
- [x] Add `get_schema_version()` method
- [x] Add `get_feature_lookback_info()` method for debugging/audit
- [x] Verify backward compatibility (v1 features still work via v2 explicit fields)

### Phase 2A: Leakage/Feature Conversion ✅ COMPLETED (2026-01-18)
- [x] Update `leakage_budget.py` to use `lookback_minutes` via `_get_registry_lookback_minutes()` helper
- [x] Verify `feature_time_meta.py` already prefers `lookback_minutes` (no changes needed)
- [x] Compare leakage decisions before/after (v2 features use `v2_explicit`, v1 fallback works)
- [x] Added `_get_registry_lookback_minutes()` SST helper (encapsulates v2 schema with v1 fallback + 0.0 guard)

### Phase 2B: Purge Consolidation ✅ COMPLETED (2026-01-18)
- [x] Add `get_purge_overlap_bars()` helper to `purge.py`
- [x] Replace hardcoded `purge_overlap=17` in `ensemble_trainer.py`
- [x] Replace hardcoded `purge_overlap=17` in `multi_model_feature_selection.py` (3 locations)
- [x] Replace hardcoded `purge_overlap=17` in `unified_training_interface.py`
- [x] Replace hardcoded `purge_overlap=17` in `cross_sectional.py`
- [ ] Integrate with `PurgedTimeSeriesSplit.purge_overlap_time` (deferred - time-based CV)
- [ ] Update `resolved_config.py` to use PurgeSpec (deferred)
- [x] Verify CV splits identical at 5m (via get_purge_overlap_bars() default = 17)

### Phase 3: Rounding Policy Flip ✅ COMPLETED (2026-01-18)
- [x] Enable `use_ceil_rounding=true` by default in `CONFIG/pipeline/pipeline.yaml`
- [x] Enable `use_v2_lookback=true` by default
- [x] Update `registry_coverage.py` to use `horizon_minutes_to_bars()` SST function
- [x] Update `leakage_filtering.py` to use `horizon_minutes_to_bars()` SST function
- [x] All 120 interval tests pass

Note: No sign-off needed - user confirmed historical comparability not required

### Phase 4: Sequential Models ✅ COMPLETED (2026-01-18)
- [x] Update `CONFIG/pipeline/training/sequential.yaml` with `lookback_minutes: 300`
- [x] Update `TRAINING/models/family_router.py` to derive `lookback_T` from `lookback_minutes`
- [x] TTL in `seq_ring_buffer.py` already takes `ttl_seconds` as parameter (no change needed)
- [x] TTL in `sequential.yaml` already configurable under `live.ttl_seconds` (no change needed)
- [x] Update `LIVE_TRADING/models/inference.py` to read TTL from config instead of hardcoded 300.0
- [x] All 120 interval tests pass

### Phase 5: Hardcoded Cleanup ✅ COMPLETED (2026-01-18)
- [x] Replaced `max_lookback_bars = 288` with time-based `feature_lookback_max_minutes = 1440.0` (2 locations in model_evaluation.py)
- [x] `max_sequence_gap` already configurable in `sequential.yaml` under `validation.max_sequence_gap: 300`
- [x] Remaining `= 5` defaults are intentional fallbacks for backward compatibility
- [x] All 120 interval tests pass
- [ ] Audit for remaining hardcoded interval assumptions

### Phase 6: Time-based CV ✅ COMPLETED (2026-01-18)
- [x] Added `purge_overlap_minutes` parameter to `PurgedTimeSeriesSplit`
- [x] Updated docstring with new recommended API
- [x] Updated `shared_ranking_harness.py` to use `purge_overlap_minutes`
- [x] Updated `model_evaluation.py` to use `purge_overlap_minutes`
- [x] Updated `leakage_detection.py` to use `purge_overlap_minutes`
- [x] All 120 interval tests pass
- [ ] (Future) Update remaining callers in `multi_model_feature_selection.py` when timestamps available

### Phase 7: allowed_horizons (Future)
- [ ] Add `allowed_horizon_minutes` to registry schema
- [ ] Update horizon filtering logic
- [ ] Deprecate bar-based `allowed_horizons`

### Phase 8: Artifact Versioning
- [ ] Add `interval_minutes` to manifest schema
- [ ] Add stage-boundary validation gates
- [ ] Add inference-time compatibility checks
- [ ] Add interval to run fingerprint

### Phase 8: Multi-Horizon Training
- [ ] Add `HorizonBundle` type
- [ ] Update target ranking for bundle diversity
- [ ] Add multi-task training strategy
- [ ] Wire `MultiTask` model family

### Phase 9: Cross-Horizon Ensemble
- [ ] Add cross-horizon stacking to `ensemble_trainer.py`
- [ ] Implement ridge weights across horizons
- [ ] Add horizon decay function
- [ ] Update ensemble config

### Phase 10: Multi-Interval Experiments (Future)
- [ ] Add `multi_intervals` experiment config
- [ ] Cross-interval validation
- [ ] Feature transfer warm-start
- [ ] Regime-based interval selection

### Phase 11: Feature Computation Windows (NEW - Critical)
- [ ] Add `window_minutes` to feature registry schema
- [ ] Update `comprehensive_builder.py` to use time-based windows
- [ ] Convert all `rolling(N)` calls to use `minutes_to_bars()`
- [ ] Update `LIVE_TRADING/models/feature_builder.py`
- [ ] Add feature computation audit logging
- [ ] Verify features identical at 5m

### Phase 12: Target Computation Tracking
- [ ] Record `interval_minutes` at target computation time
- [ ] Store target metadata in artifacts
- [ ] Add validation gate for interval mismatch

### Phase 13: Cache Key Interval Component
- [ ] Add `interval_minutes` to cache key construction
- [ ] Update `build_cache_key_with_symbol()` signature
- [ ] Invalidate existing caches on deployment

### Phase 14: Feature-Target Alignment Validation
- [ ] Add timestamp alignment check before merge
- [ ] Add row count validation
- [ ] Log/fail on misalignment

### Phase 15: Data Gap Detection
- [ ] Detect gaps in timestamp sequence
- [ ] Add gap handling options (fill/exclude)
- [ ] Log gap statistics

### Phase 16: Model Metadata Interval Provenance
- [ ] Add `interval_minutes` to `model_meta.json`
- [ ] Add `interval_source` to metadata
- [ ] Add `feature_lookbacks_minutes` dict
- [ ] Update model serialization

### Phase 17: Live Inference Cross-Interval Gate
- [ ] Add interval check in `InferenceEngine.predict()`
- [ ] Reject mismatched intervals with clear error
- [ ] Add logging for interval validation

### Phase 18: Hyperparameter Scaling Guidance
- [ ] Document interval scaling guidelines
- [ ] Add `interval_scaling` config section
- [ ] Optional automated scaling for tree models

### Phase 19: Feature Naming Convention
- [ ] Rename features to interval-agnostic names
- [ ] Store interval in metadata, not name
- [ ] Update feature discovery logic
- [ ] Add migration for existing named features

### Phase 20: Cross-Sectional Purge Fix
- [ ] Replace hardcoded `purge_overlap=17` in cross_sectional.py
- [ ] Use PurgeSpec abstraction

### Phase 21: Live Trading Horizon Configuration
- [ ] Move HORIZONS from constants to config
- [ ] Add horizon discovery from model metadata
- [ ] Support non-standard intervals

### Phase 22: Multi-Interval Test Fixtures
- [ ] Generate test fixtures for 1m, 15m intervals
- [ ] Add parameterized tests across intervals
- [ ] Update testing documentation

### Phase 23: Metric Scaling Documentation
- [ ] Document volatility scaling by interval
- [ ] Document Sharpe ratio annualization rules
- [ ] Add interval-aware metric helpers

### Phase 24: Config Validation Rules
- [ ] Add validation for bar_interval vs base_interval_minutes consistency
- [ ] Add interval conflict detection in config loading

---

## Open Questions

1. **Feature naming**: Keep original names with `lookback_minutes` as metadata.
   - **Decision**: Yes, names are IDs.

2. **Rounding policy default**: When to flip from floor to ceil?
   - **Decision**: Phase 3, with explicit sign-off.

3. **Multi-interval in single run**: Support feature-space explosion?
   - **Decision**: No. Different intervals = different runs. Compare results, don't merge.

4. **Strict mode default**: What should be the default for interval mismatch?
   - **Decision**: `warn` for now, `strict` after Phase 7 validation.

5. **Multi-horizon strategy**: Which horizons to bundle?
   - **Recommendation**: Start with [5m, 15m, 60m] for diversity.

6. **Horizon alpha decay**: What half-life?
   - **Recommendation**: 30 minutes (based on live trading experience).

---

## References

- Original audit: Agent exploration of TRAINING/ for interval dependencies
- Existing Duration class: `TRAINING/common/utils/duration_parser.py`
- Existing FeatureTimeMeta: `TRAINING/ranking/utils/feature_time_meta.py`
- Existing ResolvedConfig: `TRAINING/ranking/utils/resolved_config.py`
- Existing PurgedTimeSeriesSplit: `TRAINING/ranking/utils/purged_time_series_split.py`
- Multi-horizon reference: `LIVE_TRADING/prediction/predictor.py`
- Registry location: `CONFIG/data/feature_registry.yaml`
