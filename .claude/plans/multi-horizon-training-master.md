# Multi-Horizon Training Master Plan

**Status**: Planning Phase
**Created**: 2026-01-19
**Parent Plan**: `interval_agnostic_pipeline.md` (Phases 8-10)

---

## Executive Summary

This master plan covers three advanced capabilities that build on the interval-agnostic foundation:

| Phase | Name | Risk | Effort | Value |
|-------|------|------|--------|-------|
| **8** | Multi-Horizon Training | Medium | 2-3 days | Share encoder across horizons, reduce training time |
| **9** | Cross-Horizon Ensemble | Medium | 2-3 days | Better predictions via horizon blending |
| **10** | Multi-Interval Experiments | High | 4-5 days | Train at coarse intervals, validate at fine |

**Dependencies**: All depend on completed interval-agnostic infrastructure (Phases 0-24).

---

## Subplan Index

| Subplan | File | Status |
|---------|------|--------|
| Phase 8 | [phase8_multi_horizon_training.md](./phase8_multi_horizon_training.md) | Not Started |
| Phase 9 | [phase9_cross_horizon_ensemble.md](./phase9_cross_horizon_ensemble.md) | Not Started |
| Phase 10 | [phase10_multi_interval_experiments.md](./phase10_multi_interval_experiments.md) | Not Started |

---

## Architecture Overview

### Current State (Single-Target Training)
```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (current)                                 │
│                                                          │
│  for target in sorted(targets):                          │
│      X, y = prepare_data(target)                         │
│      for family in families:                             │
│          model = train(X, y, family)                     │
│          save(model, target, family)                     │
└─────────────────────────────────────────────────────────┘
```

### Phase 8: Multi-Horizon Training
```
┌─────────────────────────────────────────────────────────┐
│  Multi-Horizon Training (Phase 8)                        │
│                                                          │
│  bundle = group_by_horizon([fwd_ret_5m, fwd_ret_15m,    │
│                             fwd_ret_60m])                │
│                                                          │
│  ┌─────────────────┐                                     │
│  │ Shared Encoder  │ ─────┬──► Head_5m  ──► pred_5m     │
│  │                 │      ├──► Head_15m ──► pred_15m    │
│  └─────────────────┘      └──► Head_60m ──► pred_60m    │
│                                                          │
│  Loss = w1*MSE_5m + w2*MSE_15m + w3*MSE_60m              │
└─────────────────────────────────────────────────────────┘
```

### Phase 9: Cross-Horizon Ensemble
```
┌─────────────────────────────────────────────────────────┐
│  Cross-Horizon Ensemble (Phase 9)                        │
│                                                          │
│  Trained models (all families, all horizons):            │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ LightGBM │  │ LightGBM │  │ LightGBM │               │
│  │  5m head │  │ 15m head │  │ 60m head │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │             │             │                      │
│       ▼             ▼             ▼                      │
│  ┌────────────────────────────────────────┐             │
│  │        Cross-Horizon Stacker           │             │
│  │  (Ridge with decay weights)            │             │
│  │  w_5m = 0.5, w_15m = 0.3, w_60m = 0.2  │             │
│  └───────────────────┬────────────────────┘             │
│                      ▼                                   │
│                 Final Prediction                         │
└─────────────────────────────────────────────────────────┘
```

### Phase 10: Multi-Interval Experiments
```
┌─────────────────────────────────────────────────────────┐
│  Multi-Interval Experiments (Phase 10)                   │
│                                                          │
│  Data intervals: 1m, 5m, 15m, 60m                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Interval 60m (coarse)                             │   │
│  │  • More data per bar, less noise                  │   │
│  │  • Train: "warm-start" features                   │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │ transfer                       │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Interval 5m (reference)                           │   │
│  │  • Standard training                              │   │
│  │  • Main model output                              │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │ validate                       │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Interval 1m (fine)                                │   │
│  │  • Cross-interval validation                      │   │
│  │  • Generalization check                           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Order

### Recommended Sequence

```
Phase 8.1: HorizonBundle type + target grouping
    │
    ▼
Phase 8.2: MultiTaskTrainer horizon heads
    │
    ▼
Phase 8.3: Training loop integration
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Phase 9.1: Cross-horizon         Phase 10.1: Multi-interval
           stacking                         data loading
    │                                  │
    ▼                                  ▼
Phase 9.2: Decay functions       Phase 10.2: Feature transfer
    │                                  │
    ▼                                  ▼
Phase 9.3: Ensemble config       Phase 10.3: Cross-interval
           integration                      validation
```

### Dependencies

| Phase | Depends On | Blocks |
|-------|------------|--------|
| 8.1 | interval_agnostic (complete) | 8.2, 8.3 |
| 8.2 | 8.1 | 9.1 |
| 8.3 | 8.2 | 9.1, 10.1 |
| 9.1 | 8.3 | 9.2, 9.3 |
| 9.2 | 9.1 | 9.3 |
| 9.3 | 9.2 | - |
| 10.1 | 8.3 | 10.2, 10.3 |
| 10.2 | 10.1 | 10.3 |
| 10.3 | 10.2 | - |

---

## Key Design Decisions

### Decision 1: Horizon Grouping Strategy

**Options**:
1. **Automatic**: Group by base target (e.g., all `fwd_ret_*` together)
2. **Manual**: User specifies bundles in config
3. **Hybrid**: Auto-discover with user overrides

**Recommendation**: Hybrid (3) - Most flexible

```yaml
intelligent_training:
  strategy: multi_horizon_bundle
  bundle_config:
    auto_discover: true  # Group by target prefix
    bundles:
      # Override auto-discovery for specific targets
      return_bundle:
        targets: [fwd_ret_5m, fwd_ret_15m, fwd_ret_60m]
        diversity_threshold: 0.7
```

### Decision 2: Shared vs Separate Encoders

**Options**:
1. **Fully shared**: One encoder for all horizons
2. **Partially shared**: Shared lower layers, separate upper layers
3. **Separate**: Independent encoders (current single-task)

**Recommendation**: Partially shared (2) - Balance sharing and horizon-specific learning

```
Input → [Shared Layer 1] → [Shared Layer 2] → Split
                                              ├── [Horizon-specific Layer] → Head_5m
                                              ├── [Horizon-specific Layer] → Head_15m
                                              └── [Horizon-specific Layer] → Head_60m
```

### Decision 3: Cross-Horizon Blending Method

**Options**:
1. **Simple average**: Equal weights
2. **Ridge blend**: Learn optimal weights
3. **Decay-weighted**: Exponential decay from target horizon
4. **Adaptive**: Per-symbol/regime weights

**Recommendation**: Ridge blend with decay prior (2+3)

```python
# Ridge with decay-informed regularization
weights = ridge_blend(
    predictions=[pred_5m, pred_15m, pred_60m],
    horizons=[5, 15, 60],
    target_horizon=15,  # Predicting 15m
    decay_half_life=30  # minutes
)
# Result: w_5m=0.4, w_15m=0.45, w_60m=0.15
```

### Decision 4: Multi-Interval Data Strategy

**Options**:
1. **Resample**: Convert 1m → 5m, 5m → 15m as needed
2. **Native**: Load each interval natively from data
3. **Hybrid**: Native for training, resample for validation

**Recommendation**: Native (2) - Preserves data fidelity

```yaml
experiment:
  multi_interval:
    primary_interval: 5m
    training_intervals: [5m, 15m]  # Train on both
    validation_interval: 1m  # Validate on finer
    data_sources:
      5m: data/data_labeled/interval=5m
      15m: data/data_labeled/interval=15m
      1m: data/data_labeled/interval=1m
```

---

## Risk Mitigation

### Phase 8 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Horizon interference | Medium | High | Per-head batch normalization, gradient scaling |
| Unbalanced targets | Medium | Medium | Dynamic loss weighting based on variance |
| Training instability | Low | High | Separate optimizer states per head |

### Phase 9 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting to blend | Medium | Medium | Purged CV for blend weights |
| Horizon correlation | High | Low | Diversity threshold for bundle selection |
| Inference latency | Low | Medium | Pre-compute blend weights, cache |

### Phase 10 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data leakage across intervals | High | Critical | Strict purge validation per interval |
| Feature mismatch | Medium | High | Feature registry interval validation |
| Cache key collisions | Medium | High | Interval in cache key (Phase 13 complete) |

---

## Testing Strategy

### Unit Tests
- `test_horizon_bundle.py` - Bundle creation, diversity calculation
- `test_multi_horizon_trainer.py` - Multi-head forward/backward pass
- `test_cross_horizon_ensemble.py` - Blend weight learning, decay functions
- `test_multi_interval_data.py` - Data loading, interval validation

### Integration Tests
- `test_e2e_multi_horizon.py` - Full pipeline with horizon bundles
- `test_e2e_cross_horizon.py` - Ensemble with cross-horizon stacking
- `test_e2e_multi_interval.py` - Training at 5m, validating at 1m

### Contract Tests
- Determinism: Same seeds → same results across horizons
- Leakage: No future data in any horizon's predictions
- Fingerprint: Model artifacts include all horizon metadata

---

## Success Metrics

### Phase 8 Success
- [ ] Multi-horizon training completes without error
- [ ] Shared encoder reduces total training time by 20%+
- [ ] Per-horizon metrics within 5% of single-task baseline

### Phase 9 Success
- [ ] Cross-horizon ensemble improves Sharpe by 10%+ over single horizon
- [ ] Blend weights adapt to horizon relevance
- [ ] Inference latency < 2x single-horizon

### Phase 10 Success
- [ ] Models generalize across intervals (validation IC > 0.8 * training IC)
- [ ] Feature transfer reduces training time by 30%+
- [ ] No cross-interval data leakage detected

---

## File Changes Summary

### New Files
```
TRAINING/common/horizon_bundle.py           # Phase 8: HorizonBundle type
TRAINING/orchestration/horizon_ranker.py    # Phase 8: Bundle diversity ranking
TRAINING/model_fun/multi_horizon_trainer.py # Phase 8: Multi-head trainer
TRAINING/model_fun/cross_horizon_ensemble.py # Phase 9: Cross-horizon stacker
TRAINING/data/multi_interval_loader.py      # Phase 10: Multi-interval data
tests/test_horizon_bundle.py                # Phase 8 tests
tests/test_cross_horizon_ensemble.py        # Phase 9 tests
tests/test_multi_interval.py                # Phase 10 tests
CONFIG/experiments/multi_horizon_example.yaml
CONFIG/experiments/multi_interval_example.yaml
```

### Modified Files
```
TRAINING/model_fun/multi_task_trainer.py    # Phase 8: Horizon-aware heads
TRAINING/model_fun/ensemble_trainer.py      # Phase 9: Cross-horizon support
TRAINING/training_strategies/execution/training.py  # Phase 8+10: Loop changes
TRAINING/orchestration/intelligent_trainer.py       # Phase 8+9+10: Strategy routing
CONFIG/pipeline/training/intelligent.yaml           # Phase 8+9: New strategies
CONFIG/experiments/_template.yaml                   # Phase 8+9+10: New options
```

---

## Next Steps

1. **Review this master plan** - Confirm architecture decisions
2. **Create Phase 8 subplan** - Detailed implementation steps
3. **Create Phase 9 subplan** - Cross-horizon ensemble details
4. **Create Phase 10 subplan** - Multi-interval experiment details
5. **Begin Phase 8.1** - HorizonBundle type implementation

---

## Appendix: Existing Architecture Reference

### MultiTask Model (Current)
Location: `TRAINING/model_fun/multi_task_trainer.py`
- TensorFlow-based shared encoder
- Dict of MSE losses per target
- Auto-detection of multi-target mode

### Ensemble Trainer (Current)
Location: `TRAINING/model_fun/ensemble_trainer.py`
- HGB + RF + Ridge stacking
- Interval-agnostic purge calculation
- Threading optimization

### Training Strategies (Current)
Location: `TRAINING/training_strategies/strategies/`
- SingleTaskStrategy: One model per target
- MultiTaskStrategy: Shared encoder + separate heads (PyTorch)
- CascadeStrategy: Barrier gates + return predictions

### Config Precedence
```
CLI args > Experiment config > Intelligent training config > Pipeline config > Defaults
```
