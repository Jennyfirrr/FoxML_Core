# Column Projection Optimization: Single-Symbol Probe + Feature Cascade

## Status: Implemented (2026-01-19)

### Completed Tasks

1. ✅ Created `TRAINING/ranking/utils/feature_probe.py` with:
   - `probe_features_for_target()` - Single-symbol importance filtering
   - `probe_all_targets()` - Batch probe for multiple targets
   - SST compliance: sorted features, deterministic symbol selection

2. ✅ Added probe config options to `CONFIG/pipeline/pipeline.yaml`:
   - `intelligent_training.lazy_loading.probe_features: true`
   - `intelligent_training.lazy_loading.probe_top_n: 100`
   - `intelligent_training.lazy_loading.probe_rows: 10000`
   - `intelligent_training.lazy_loading.probe_importance_threshold: 0.0001`

3. ✅ Added SS Universe Size Gate to `TRAINING/ranking/target_routing.py`:
   - `target_routing.max_symbols_for_ss: 100`
   - `target_routing.ss_fallback_route: CROSS_SECTIONAL`
   - Gates both `_compute_target_routing_decisions()` and `_compute_single_target_routing_decision()`

4. ✅ Integrated probe into `TRAINING/training_strategies/execution/training.py`:
   - Probe runs after preflight features are determined
   - Only probes when `len(features) > probe_top_n`
   - Falls back to preflight features on probe failure (fail-open)

5. ✅ Added feature ordering assertions at stage boundaries:
   - Sorting check before `data_loader.load_for_target()`
   - Warning + auto-sort if features unsorted

6. ✅ Created `tests/test_feature_probe.py` with tests for:
   - Sorted output, determinism, threshold skipping
   - Symbol selection, empty inputs, missing target
   - Insufficient samples, batch processing

## Problem Statement

Current lazy loading reduces memory by loading per-target, but still loads **all preflight-approved columns** (~300) for each target. We can do better:

1. **Quick pruner exists** (`feature_pruning.py`) - uses LightGBM to identify important features
2. **Preflight filtering exists** (`preflight_leakage.py`) - schema-only leakage filtering
3. **Column projection exists** (`UnifiedDataLoader`) - load only specified columns

**Missing link**: Use quick pruner on a **single symbol** to identify top N features, then load only those columns for all symbols.

## Memory Impact Estimate

| Stage | Columns Loaded | Memory (728 symbols) |
|-------|---------------|----------------------|
| Current eager | ~500 | OOM |
| Lazy loading | ~300 (preflight filtered) | ~50% reduction |
| **With probe** | ~100 (importance filtered) | **~83% reduction** |

## Key Insight: Cross-Sectional Representativeness

For cross-sectional models with standardized feature engineering:
- Same feature formula across all symbols (RSI_14 computed identically)
- Same target definition across all symbols
- Same market microstructure (liquid US equities)

**One symbol IS representative** for feature importance ranking. If `rsi_14` predicts `fwd_ret_60m` for AAPL, it predicts it for GOOGL too.

## Existing Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `quick_importance_prune()` | `TRAINING/ranking/utils/feature_pruning.py:29` | LightGBM feature importance |
| `preflight_filter_features()` | `TRAINING/ranking/utils/preflight_leakage.py:47` | Schema-only leakage filter |
| `UnifiedDataLoader.load_for_target()` | `TRAINING/data/loading/unified_loader.py:313` | Column projection loading |
| `filter_features_for_target()` | `TRAINING/ranking/utils/leakage_filtering.py:579` | Per-target leakage rules |

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: PREFLIGHT                              │
│  Input: symbols, targets                                                │
│  Output: Dict[target -> List[~300 safe columns]]                        │
│  Cost: ~1ms/symbol (schema only, no data)                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: SINGLE-SYMBOL PROBE                         │
│  For each target:                                                       │
│    1. Load 1 symbol with ~300 preflight columns (~10k rows)             │
│    2. Run quick_importance_prune() → get importance scores              │
│    3. Keep top N features (e.g., 100)                                   │
│  Output: Dict[target -> List[~100 important columns]]                   │
│  Cost: ~2 seconds/target (1 symbol × 10k rows × LightGBM)               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: FULL DATA LOADING                           │
│  For each target:                                                       │
│    1. Load ALL symbols with only ~100 probed columns                    │
│    2. Train models                                                      │
│    3. Release memory                                                    │
│  Cost: ~33% of current memory per target                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Create Probe Function (New)

**File**: `TRAINING/ranking/utils/feature_probe.py`

```python
def probe_features_for_target(
    loader: UnifiedDataLoader,
    symbols: List[str],
    target: str,
    preflight_features: List[str],
    top_n: int = 100,
    probe_rows: int = 10000,
    importance_threshold: float = 0.0001,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Load single symbol, run quick importance, return top N features.

    Args:
        loader: UnifiedDataLoader instance
        symbols: Full symbol list (first alphabetically used for probe)
        target: Target column name
        preflight_features: Features from preflight filter (~300)
        top_n: Maximum features to keep
        probe_rows: Rows to load for probe (10k is fast but representative)
        importance_threshold: Minimum importance to keep (cumulative)

    Returns:
        Tuple of (top_features_list, importance_dict)
    """
    # Deterministic: always use first symbol alphabetically
    probe_symbol = sorted(symbols)[0]

    # Load single symbol with preflight features
    probe_data = loader.load_for_target(
        symbols=[probe_symbol],
        target=target,
        features=preflight_features,
        max_rows_per_symbol=probe_rows,
    )

    # Prepare X, y for quick pruner
    df = probe_data[probe_symbol]
    available_features = [f for f in preflight_features if f in df.columns]
    X = df[available_features].values
    y = df[target].values

    # Run existing quick_importance_prune
    from TRAINING.ranking.utils.feature_pruning import quick_importance_prune
    X_pruned, pruned_names, stats = quick_importance_prune(
        X, y, available_features,
        top_n=top_n,
        cumulative_threshold=importance_threshold,
    )

    # Return pruned feature list + importance scores
    return pruned_names, stats.get('importances', {})
```

### Phase 2: Integrate into Training Flow

**File**: `TRAINING/training_strategies/execution/training.py`

Modify the lazy loading section to add probe step:

```python
if lazy_loading_enabled:
    # Check if feature probing is enabled
    probe_enabled = lazy_loading_config.get('probe_features', True)
    probe_top_n = lazy_loading_config.get('probe_top_n', 100)
    probe_rows = lazy_loading_config.get('probe_rows', 10000)

    if probe_enabled and target_features:
        from TRAINING.ranking.utils.feature_probe import probe_features_for_target

        # Probe features for this target
        preflight_features = target_features.get(target, [])
        if len(preflight_features) > probe_top_n:
            probed_features, importances = probe_features_for_target(
                loader=data_loader,
                symbols=effective_symbols,
                target=target,
                preflight_features=preflight_features,
                top_n=probe_top_n,
                probe_rows=probe_rows,
            )
            logger.info(
                f"🔬 Probe: {len(preflight_features)} preflight → "
                f"{len(probed_features)} important features"
            )
            target_feature_list = probed_features
        else:
            target_feature_list = preflight_features
```

### Phase 3: Add Config Options

**File**: `CONFIG/pipeline/pipeline.yaml`

```yaml
intelligent_training:
  lazy_loading:
    enabled: true
    fail_on_fallback: true
    verify_memory_release: false
    log_memory_usage: true

    # Feature probing (single-symbol importance filtering)
    probe_features: true        # Enable single-symbol probe
    probe_top_n: 100            # Keep top N features per target
    probe_rows: 10000           # Rows to load for probe (fast but representative)
    probe_importance_threshold: 0.0001  # Min cumulative importance
```

### Phase 4: Cascade to Feature Selection Stage

The feature selection stage should also use probed features, not all preflight features.

**File**: `TRAINING/ranking/feature_selector.py`

```python
def select_features_for_target(
    ...
    probe_features: bool = True,
    probe_top_n: int = 100,
):
    # 1. Preflight filter (schema only)
    preflight_features = preflight_filter_features(...)

    # 2. Probe on single symbol if enabled
    if probe_features:
        probed_features, _ = probe_features_for_target(
            loader, symbols, target, preflight_features[target],
            top_n=probe_top_n,
        )
    else:
        probed_features = preflight_features[target]

    # 3. Load data with probed columns only
    mtf_data = loader.load_for_target(
        symbols=symbols,
        target=target,
        features=probed_features,  # Only ~100 columns, not ~300
        ...
    )
```

### Phase 5: Target Ranking Stage Integration

Target ranking currently loads all data. It should also use column projection.

**File**: `TRAINING/ranking/target_ranker.py`

Add preflight + probe before loading:
- Run preflight to get safe columns per target
- For each target being ranked, probe to get top N
- Load only those columns for ranking

## Files to Create/Modify

### New Files
- `TRAINING/ranking/utils/feature_probe.py` - Single-symbol probe function

### Modified Files
- `TRAINING/training_strategies/execution/training.py` - Integrate probe into lazy loading
- `TRAINING/ranking/feature_selector.py` - Use probe before full load
- `TRAINING/ranking/target_ranker.py` - Add preflight + probe
- `CONFIG/pipeline/pipeline.yaml` - Add probe config options
- `CONFIG/experiments/production_baseline.yaml` - Enable probe

### Test Files
- `tests/test_feature_probe.py` - Unit tests for probe function

## Determinism Considerations

| Component | Requirement | Solution |
|-----------|-------------|----------|
| Probe symbol selection | Same symbol each run | `sorted(symbols)[0]` |
| Feature importance ties | Same ranking | Sort by (-importance, feature_name) |
| LightGBM probe | Reproducible | `deterministic=True`, seed from config |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Probe symbol not representative | For CS models with standardized features, this is not a concern |
| Important feature missed | Set `probe_top_n` conservatively (100-150) |
| Probe overhead | 10k rows × 300 cols × LightGBM = ~2 seconds (acceptable) |

## Success Metrics

1. **Memory reduction**: Additional 50% on top of lazy loading (~83% total)
2. **No quality regression**: Same model AUC within 0.001
3. **Probe overhead**: < 5 seconds per target
4. **Determinism**: Same features selected across runs

## Implementation Order

1. ✅ Lazy loading (Phase 1-9 of previous plan) - DONE
2. Create `feature_probe.py` with `probe_features_for_target()`
3. Add config options for probe settings
4. Integrate into training.py lazy loading flow
5. Add unit tests
6. Integrate into feature_selector.py
7. Integrate into target_ranker.py
8. Run full pipeline test with 728 symbols
9. Verify memory reduction and quality metrics

## Infrastructure Integration Verification ✅

### Memory Management - READY
- `release_data()` already called per-target at `training.py:2547`
- Column projection is transparent to memory release
- `MemoryTracker` available for verification
- **No changes needed**

### Auto-Fixers - COMPATIBLE
- Auto-fixers update `excluded_features.yaml` globally
- Preflight respects exclusions before returning feature list
- Probed features go through same preflight → **no bypass possible**
- **No changes needed**

### Leakage Detection - BOTH STAGES COVERED
- **Preflight** (BEFORE loading): Schema-only, enables column projection
- **Runtime** (AFTER training): Full checks, triggers auto-fix if needed
- Probed features flow through both checks
- **No changes needed**

### Stage Boundaries - CRITICAL: Feature Ordering
- Feature lists serialized to JSON between stages
- **MUST sort features** at every boundary:
  1. When returned from feature selection
  2. When serialized to JSON
  3. When passed to data loader
- Add assertion: `assert features == sorted(features)`

## SS Routing Gate for Large Universes

### Problem
Symbol-specific (SS) routing creates one model per symbol. With 728 symbols:
- 728 models × 20 families = 14,560 model instances
- Negates memory savings from column projection

### Solution: Universe Size Gate

**File**: `CONFIG/pipeline/pipeline.yaml`

```yaml
target_routing:
  # Existing thresholds
  T_cs: 0.65
  T_sym: 0.60
  T_suspicious_cs: 0.90

  # NEW: Universe size gate for SS routing
  max_symbols_for_ss: 100  # Disable SS routing above this threshold
  ss_fallback_route: CROSS_SECTIONAL  # What to use instead
```

**File**: `TRAINING/ranking/target_routing.py`

```python
def _compute_target_routing_decisions(...):
    # Get universe size gate
    max_symbols_for_ss = get_cfg("target_routing.max_symbols_for_ss", default=100)
    ss_fallback = get_cfg("target_routing.ss_fallback_route", default="CROSS_SECTIONAL")

    universe_size = len(symbols)

    for target in targets:
        route = _compute_route(...)

        # Apply universe size gate
        if route in (Route.SYMBOL_SPECIFIC, Route.BOTH) and universe_size > max_symbols_for_ss:
            logger.warning(
                f"🚫 SS disabled for {target}: universe size {universe_size} > {max_symbols_for_ss}. "
                f"Falling back to {ss_fallback}"
            )
            route = Route[ss_fallback]

        decisions[target] = {"route": route.name, ...}
```

### Behavior

| Universe Size | SS Allowed | Route Options |
|--------------|------------|---------------|
| ≤ 100 symbols | ✅ Yes | CS, SS, BOTH, BLOCKED |
| > 100 symbols | ❌ No | CS, BLOCKED only |

## Updated Implementation Order

1. ✅ Lazy loading (previous plan) - DONE
2. ✅ Create `feature_probe.py` with `probe_features_for_target()` - DONE (2026-01-19)
3. ✅ Add config options for probe settings - DONE (2026-01-19)
4. ✅ Add SS universe size gate to `target_routing.py` - DONE (2026-01-19)
5. ✅ Integrate probe into training.py lazy loading flow - DONE (2026-01-19)
6. ✅ Add feature ordering assertions at stage boundaries - DONE (2026-01-19)
7. ✅ Add unit tests - DONE (2026-01-19)
8. ✅ Integrate into ranking.py (target ranking stage) - DONE (2026-01-19)
9. ✅ Integrate into shared_ranking_harness.py (feature selection stage) - DONE (2026-01-19)
10. ⏳ Run full pipeline test with 728 symbols
11. ⏳ Verify memory reduction and quality metrics

## Implementation Summary

All three pipeline stages now support lazy loading + preflight + probe:

| Stage | File | Lazy Loading |
|-------|------|--------------|
| TARGET_RANKING | `TRAINING/ranking/predictability/model_evaluation/ranking.py` | ✅ |
| FEATURE_SELECTION | `TRAINING/ranking/shared_ranking_harness.py` | ✅ |
| TRAINING | `TRAINING/training_strategies/execution/training.py` | ✅ (fallback only) |

**Note**: Training stage probe is a safety net - it only activates if feature selection gave >100 features.

## Open Questions

1. Should we cache probe results across targets if they share the same features?
2. Should probe use multiple symbols (e.g., 3) and take intersection?
3. Should we expose probe importance scores in artifacts for debugging?
4. **ANSWERED**: SS gate default at 100 symbols - adjust based on testing?
