# Cross-Sectional Ranking Objective - Master Plan

**Status**: Implementation Complete - Training Pipeline Ready
**Created**: 2026-01-21
**Depends On**: `raw-ohlcv-sequence-mode.md` (Phase 1-2 complete)

## Problem Statement

The current training approach treats each `(symbol, time)` row as independent and optimizes pointwise regression loss (MSE on returns). This is **misaligned** with the actual trading objective:

> At each timestamp t, rank N symbols by predicted score, then long top decile / short bottom decile.

### What's Wrong with Current Approach

```
Current Training:
  Batch = random (symbol, time) pairs from flattened data
  Loss = MSE(predicted_return, actual_return)

  Problem: Model never sees "at time t, was AAPL better than MSFT?"
           It only sees "AAPL at t=100 had return 0.02"
```

### What We Need

```
Better Training:
  Batch = timestamps, each containing ALL symbols
  Loss = "Did you rank the winners above the losers at each t?"

  Result: Model learns relative patterns, not absolute predictions
```

## Solution Overview

Three interconnected upgrades:

| Upgrade | What | Why |
|---------|------|-----|
| **1. CS-Normalized Targets** | Percentile rank of residualized returns | Comparable across symbols, robust to vol differences |
| **2. Grouped Batching** | `(B, M, L, F)` structure by timestamp | Enables within-t comparisons |
| **3. Ranking Loss** | Pairwise or listwise objective | Directly optimizes ranking quality |

## Architecture

### Current Data Flow (Pointwise)
```
Raw OHLCV → build_sequences_from_ohlcv → Flatten (N×T, L, 5) → Random Batch → MSE Loss
                                              ↓
                                         Target: fwd_ret_5m (raw)
```

### New Data Flow (Cross-Sectional Ranking)
```
Raw OHLCV → build_sequences_from_ohlcv → Group by timestamp (T, M, L, 5)
                                              ↓
                                         Target: CS percentile rank
                                              ↓
                                         Batch: B timestamps × M symbols
                                              ↓
                                         Pairwise/Listwise Loss (within each t)
```

### Data Shape Transformation

```
Current:  X.shape = (728000, 64, 5)     # N_symbols × N_timestamps flattened
          y.shape = (728000,)            # Raw returns

Proposed: X.shape = (T, M, 64, 5)       # T timestamps × M symbols × L bars × F channels
          y.shape = (T, M)               # CS percentile per timestamp

          Where: T = unique timestamps (~1000-5000)
                 M = symbols per timestamp (728 or sampled subset)
```

## Sub-Plans

### Phase 1: Cross-Sectional Target Construction
**File**: `cs-ranking-phase1-targets.md`

- Implement `compute_cs_target()` function
- Options: percentile rank, robust z-score, residualized
- Winsorization and clipping
- No future data leakage validation

### Phase 2: Timestamp-Grouped Data Structure
**File**: `cs-ranking-phase2-batching.md`

- New `CrossSectionalDataset` class
- Reshape from flat to `(T, M, L, F)`
- Handle missing symbols at some timestamps
- Memory-efficient batching strategies

### Phase 3: Ranking Loss Functions
**File**: `cs-ranking-phase3-losses.md`

- Pairwise logistic loss (sampled top/bottom)
- Listwise softmax loss
- Hybrid approaches
- Gradient considerations

### Phase 4: Metrics and Evaluation
**File**: `cs-ranking-phase4-metrics.md`

- Spearman IC per timestamp
- Top-minus-bottom spread
- Turnover and transaction cost modeling
- Comparison framework vs pointwise baseline

### Phase 5: Pipeline Integration
**File**: `cs-ranking-phase5-integration.md`

- Config schema for ranking mode
- Integration with existing `intelligent_trainer.py`
- Model trainer modifications
- Inference path for live trading

## Configuration Schema

```yaml
# CONFIG/experiments/cs_ranking_experiment.yaml
pipeline:
  input_mode: "raw_sequence"

  # NEW: Cross-sectional ranking configuration
  cross_sectional_ranking:
    enabled: true

    # Target construction
    target:
      type: "cs_percentile"      # "cs_percentile", "cs_zscore", "cs_rank"
      residualize: true          # Subtract market mean before ranking
      winsorize: [0.01, 0.99]    # Clip extremes
      horizon_bars: 1            # Predict 1 bar ahead (5m)

    # Loss function
    loss:
      type: "pairwise"           # "pairwise", "listwise", "pointwise"
      # Pairwise settings
      sample_top_pct: 0.2        # Top 20% as "winners"
      sample_bottom_pct: 0.2     # Bottom 20% as "losers"
      pairs_per_timestamp: 100   # Max pairs to sample per t
      # Listwise settings
      temperature: 1.0           # Softmax temperature

    # Batching
    batching:
      timestamps_per_batch: 32   # B dimension
      symbols_per_timestamp: null # M dimension (null = all)
      min_symbols_per_timestamp: 50  # Skip timestamps with fewer

    # Metrics
    metrics:
      primary: "spearman_ic"     # Main optimization target
      report: ["spearman_ic", "top_bottom_spread", "turnover"]
```

## Key Design Decisions

### 1. Target Type: CS Percentile Rank (Recommended)

**Why percentile over z-score?**
- Bounded [0, 1] - no outliers blowing up gradients
- Directly encodes what we trade: ordering
- Robust to fat tails without winsorization

```python
# For each timestamp t:
y[t, :] = scipy.stats.rankdata(returns[t, :]) / n_symbols
```

### 2. Loss Type: Pairwise (Recommended for Start)

**Why pairwise over listwise?**
- More interpretable: "winner beats loser"
- Scales better: O(k²) pairs vs O(M) full softmax
- Easier to debug gradient issues
- Listwise can be added later as upgrade

```python
# Sample k winners from top 20%, k losers from bottom 20%
# Much cheaper than all O(M²) pairs
loss = sum(log(1 + exp(-(s_winner - s_loser))))
```

### 3. Batching: Full Cross-Section When Possible

**Memory considerations:**
- Full batch: `(32, 728, 64, 5)` = 32 × 728 × 64 × 5 × 4 bytes = ~300MB
- Manageable on GPU, but can sample M < 728 if needed

### 4. Residualization: Yes (Default On)

**Why subtract market mean?**
- Removes "market up, everyone looks good" effect
- Forces model to find relative alpha, not beta
- Standard practice in cross-sectional equity

```python
r_residual[t, i] = r[t, i] - mean(r[t, :])
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory for full cross-section | OOM on GPU | Sample M symbols, gradient accumulation |
| Sparse timestamps (few symbols) | Noisy gradients | Filter timestamps with < min_symbols |
| Target leakage in residualization | Invalid backtest | Strict temporal separation |
| Pairwise loss slow to converge | Long training | Curriculum: easy pairs → hard pairs |
| Ranking loss harder to tune | Hyperparameter sensitivity | Start with pointwise baseline, compare |

## Estimated Effort

| Phase | Complexity | Files Changed | Estimate |
|-------|------------|---------------|----------|
| 1. Targets | Low | 1-2 new | 1-2 hours |
| 2. Batching | Medium | 2-3 new/modified | 2-3 hours |
| 3. Losses | Medium | 1-2 new | 2-3 hours |
| 4. Metrics | Low | 1-2 new | 1-2 hours |
| 5. Integration | Medium | 3-4 modified | 2-3 hours |
| **Total** | | | **8-13 hours** |

## Success Criteria

1. **Spearman IC improvement**: > 0.02 lift over pointwise baseline
2. **Top-bottom spread**: Positive and stable across time periods
3. **Determinism**: Same data → same rankings (no random pair sampling in eval)
4. **Integration**: Works with existing pipeline, minimal breaking changes

## Dependencies

- Raw OHLCV sequence mode (Phase 1-2 complete) ✅
- PyTorch (for custom loss functions)
- Existing model trainers (LSTM, Transformer, CNN1D)

## Open Questions

1. **Symbol sampling strategy**: Random per batch, or fixed subset?
2. **Handling new symbols**: Symbol appears mid-dataset - include or exclude?
3. **Multi-horizon**: Train on 1-bar, 3-bar, 5-bar simultaneously?
4. **Market neutrality**: Sector residualization worth the complexity?

---

## Session Log

### 2026-01-21: Initial Planning
- Identified misalignment between pointwise training and ranking objective
- Analyzed user's detailed input on CS targets and ranking losses
- Created this master plan with 5 sub-phases
- Key insight: The raw OHLCV data pipeline is done; this is the objective layer

### 2026-01-21: Sub-Plans Complete
- Created all 5 phase sub-plans:
  - Phase 1: `cs-ranking-phase1-targets.md` - CS percentile, z-score, vol-scaled targets
  - Phase 2: `cs-ranking-phase2-batching.md` - CrossSectionalDataset with (T, M, L, F) structure
  - Phase 3: `cs-ranking-phase3-losses.md` - Pairwise logistic, listwise softmax losses
  - Phase 4: `cs-ranking-phase4-metrics.md` - Spearman IC, top-bottom spread, turnover
  - Phase 5: `cs-ranking-phase5-integration.md` - Config schema, trainer mods, inference path
- Total estimated effort: 8-13 hours across all phases

### 2026-01-21: Phase 3 Implementation Complete
- Implemented `TRAINING/losses/ranking_losses.py`:
  - 7 loss functions: pairwise_logistic, pairwise_hinge, listwise_softmax, listwise_kl, pointwise_mse, pointwise_huber, hybrid
  - `get_ranking_loss()` factory function
  - `RankingLoss` nn.Module wrapper
  - `RankingLossType` enum
  - Numerical stability helpers: `stable_log_sigmoid()`, `stable_log1p_exp()`
- Implemented `TRAINING/losses/cs_model_wrapper.py`:
  - `CrossSectionalSequenceModel` - wraps any (B, L, F) -> (B, 1) model
  - `CrossSectionalMLPModel` - simple MLP baseline
  - `CrossSectionalAttentionModel` - cross-symbol attention model
- Written 34 unit tests in `tests/test_ranking_losses.py` (all passing)

### 2026-01-21: Phase 1 Implementation Complete
- Implemented `TRAINING/common/targets/cross_sectional.py`:
  - `compute_cs_percentile_target()` - Percentile rank in [0, 1]
  - `compute_cs_zscore_target()` - Robust z-score with MAD
  - `compute_vol_scaled_cs_target()` - Vol-adjusted CS returns
  - `compute_cs_target()` - Unified config-driven interface
- Implemented `TRAINING/common/targets/validators.py`:
  - `validate_no_future_leakage()` - Ensures targets don't use future data
  - `validate_cs_target_quality()` - Checks statistical properties
  - `validate_vol_column_no_leakage()` - Validates volatility column
- Written 38 unit tests in `tests/test_cs_targets.py` (all passing)
- Created config schema in `CONFIG/pipeline/ranking.yaml`

### 2026-01-21: Phase 2 Implementation Complete
- Created `TRAINING/data/datasets/cs_dataset.py`:
  - `CrossSectionalDataset` - Main dataset yielding (M, L, F) per timestamp
  - `CrossSectionalDatasetSampled` - Memory-efficient variant with symbol sampling
  - `CrossSectionalDataModule` - Train/val/test splitting with temporal awareness
  - `CrossSectionalBatch` - Typed dataclass for batches
  - `collate_cross_sectional()` - Custom collate function
  - `prepare_cross_sectional_data()` - Integration with Phase 1 targets
- Written 21 unit tests (all passing)
- Skipped lazy-loading variant (not needed for typical universe sizes)

**Next steps:**
1. ~~Begin implementation with Phase 1 (CS target construction)~~ ✅
2. ~~Begin Phase 2 (Timestamp-Grouped Dataset)~~ ✅
3. ~~Begin Phase 3 (Ranking Loss Functions)~~ ✅
4. ~~Begin Phase 4 (Metrics and Evaluation)~~ ✅
5. ~~Begin Phase 5 (Pipeline Integration)~~ ✅ (Complete)
6. ~~Wire CS trainer into main training.py target loop~~ ✅
7. (Optional) LIVE_TRADING inference integration

### 2026-01-21: Phase 4 Implementation Complete
- Extended `TRAINING/models/specialized/metrics.py` (not new module, per user guidance):
  - `top_bottom_spread()` - simulates long top / short bottom trading
  - `portfolio_turnover()` - measures portfolio churn for cost analysis
  - `cost_adjusted_spread()` - net returns after transaction costs
  - `spearman_ic_matrix()` - Spearman IC for (T, M) matrix input
  - `compute_ranking_metrics()` - unified interface returning all metrics
  - `compute_ranking_metrics_from_flat()` - adapter for existing flattened format
- Written 27 unit tests in `tests/test_ranking_metrics.py` (all passing)
- Deferred visualization to Phase 5 integration

### 2026-01-21: Phase 5 Implementation (Foundation)
- Added `cross_sectional_ranking` config section to `CONFIG/pipeline/pipeline.yaml`:
  - `enabled`: Master switch for CS ranking mode
  - `target`: Target construction config (type, return_col, residualize, winsorize)
  - `loss`: Loss function config (type, top_pct, bottom_pct, max_pairs)
  - `batching`: Batching config (timestamps_per_batch, min_symbols)
  - `metrics`: Metrics config (primary, cost_per_trade_bps)
- Added `is_cs_ranking_enabled()` and `get_cs_ranking_config()` helpers to `intelligent_trainer.py`
- Updated stage skip logic in `intelligent_trainer.py`:
  - Target ranking is skipped when CS ranking enabled
  - Feature selection is skipped when CS ranking enabled
  - Targets marked with `__CS_RANKING_RAW_OHLCV__` placeholder
- Added `cs_ranking_config` parameter to `train_models_for_interval_comprehensive()`
- Created experiment config: `CONFIG/experiments/cs_ranking_baseline.yaml`
- Updated `INTEGRATION_CONTRACTS.md` with `cross_sectional_ranking` field schema
- Updated package `__init__.py` to export CS ranking helpers
- All tests pass (smoke imports, ranking metrics)

### 2026-01-21: Phase 5 Implementation (Training Loop)
- Created `TRAINING/training_strategies/execution/cs_ranking_trainer.py` with:
  - `train_cs_ranking_model()` - Full training loop with early stopping
  - `_evaluate_cs_model()` - Evaluation using ranking metrics
  - `create_cs_model_metadata()` - Model metadata (contracts-compliant)
- Added `cs_collate_fn()` to `TRAINING/data/datasets/cs_dataset.py`
- All imports verified working

### 2026-01-21: Phase 5 Implementation (Training Loop Wiring Complete)
- Added `create_cs_dataset_from_mtf()` to `TRAINING/data/datasets/cs_dataset.py`:
  - Convenience wrapper around `prepare_cross_sectional_data()`
  - Accepts direct parameters instead of nested config dicts
  - Used by training.py CS ranking branch
- CS ranking training branch in `training.py` (lines 1843-1997):
  - Creates `CrossSectionalDataset` from MTF data
  - Trains `SimpleRankingModel` (MLP) with ranking loss
  - Saves model.pt and model_meta.json (contracts-compliant)
  - Records metrics (Spearman IC, spread)
- Toggle verification:
  - Default: `cross_sectional_ranking.enabled: false` - existing training unaffected
  - Experiment config override works: can enable with `enabled: true`
  - Config values (target type, loss type) can be customized per experiment
- All tests pass:
  - 26/28 contract tests pass (2 skipped determinism tests)
  - 99/99 CS ranking tests pass (targets, losses, metrics)
  - Smoke imports verified

**Remaining work (LIVE_TRADING):**
- `CrossSectionalRankingPredictor` for inference
- Model loader updates for CS models
- Signal generation for relative rankings

**Status**: CS ranking training pipeline complete and toggleable via YAML config.
Existing feature-based training is unaffected when `cross_sectional_ranking.enabled: false`.
