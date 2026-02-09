# Raw OHLCV Sequence Mode - Master Plan

**Status**: Planning
**Created**: 2026-01-21
**Author**: Claude Code

## Overview

Add a new training mode that feeds raw OHLCV bar data directly to sequence models (Transformer, LSTM, CNN1D) instead of computed technical indicators. This treats each bar as a "token" and predicts future bars/returns, similar to language model pretraining.

### Motivation

- Technical indicators encode human assumptions about what patterns matter
- Sequence models (Transformers) can learn their own representations from raw data
- Reduces feature engineering overhead and potential indicator lag
- Enables "bar-as-token" prediction paradigm (like text LLMs but for price data)

### Scope

**In scope:**
- Config toggle: `input_mode: "features" | "raw_sequence"`
- New data loader path for raw OHLCV sequences
- Pipeline branching to skip feature selection in raw mode
- Normalization strategy for price sequences
- Determinism compliance for PyTorch attention

**Out of scope (future work):**
- Multi-timeframe sequence fusion
- Cross-symbol attention (treating all symbols as one sequence)
- Autoregressive bar generation (predicting OHLCV, not just returns)

## Architecture

### Current Flow (Tabular/Features)
```
Parquet → Feature Computation → Feature Selection → build_sequences_from_features → (N, T, F) → Model
           (RSI, MACD, etc.)    (per-target)          (rolling window)                  (F=100+)
```

### New Flow (Raw Sequence)
```
Parquet → build_sequences_from_ohlcv → Normalize → (N, T, 5) → Model
           (rolling window)             (returns)    (5=OHLCV)
```

### Key Differences

| Aspect | Features Mode | Raw Sequence Mode |
|--------|---------------|-------------------|
| Input shape | (N, T, F) where F=100+ | (N, T, 5) where 5=OHLCV |
| Feature selection | Yes, per-target | No, skip entirely |
| Normalization | StandardScaler | Price-relative (returns or log-diff) |
| Models | All families | Sequence families only |
| Target | Same targets (fwd_ret_*, etc.) | Same targets |

## Sub-Plans

### Phase 1: Data Infrastructure
**File**: `raw-ohlcv-sequence-data.md`
**Status**: MOSTLY COMPLETE (2026-01-21)

- [x] New function: `build_sequences_from_ohlcv(df, seq_len, interval_minutes)`
- [x] Normalization: Convert to returns or log-differences
- [x] Handle missing data / gaps in time series
- [x] Deterministic ordering (sorted by timestamp)
- [x] Memory-efficient windowing (np.lib.stride_tricks)
- [ ] `UnifiedDataLoader.load_for_sequence_training` (deferred to Phase 2)

### Phase 2: Pipeline Integration
**File**: `raw-ohlcv-sequence-pipeline.md`
**Status**: MOSTLY COMPLETE (2026-01-21)

- [x] Config key: `pipeline.input_mode: "features" | "raw_sequence"`
- [x] Branch in `intelligent_trainer.py` to skip feature selection
- [x] Branch in `data_preparation.py` to use raw loader (`prepare_training_data_raw_sequence`)
- [x] Branch in `training.py` for raw sequence data preparation
- [x] Update model trainers to accept (N, T, 5) input (LSTM, Transformer, CNN1D)
- [ ] Model meta: Record `input_mode` in `model_meta.json` (needs testing)

### Phase 3: Determinism
**File**: `raw-ohlcv-sequence-determinism.md`
**Status**: MOSTLY COMPLETE (2026-01-21)

- [x] Add `torch.use_deterministic_algorithms(True)` in `repro_bootstrap.py`
- [x] Handle CUDA deterministic mode (`CUBLAS_WORKSPACE_CONFIG=:4096:8`)
- [x] Add `init_torch_determinism()` function for post-import setup
- [ ] Verify attention operations are deterministic (testing needed)
- [ ] Add contract test for sequence model determinism

### Phase 4: Evaluation & Comparison
**File**: `raw-ohlcv-sequence-evaluation.md`

- [ ] Benchmark experiment: raw vs features on same targets
- [ ] Metrics comparison: AUC, stability, training time
- [ ] Ablation: sequence length impact (32, 64, 128, 256 bars)
- [ ] Documentation: When to use which mode

### Phase 5: LIVE_TRADING Integration
**File**: `live-trading-inference-master.md` (separate master plan with sub-phases)
**Status**: PLANNED (2026-02-08)

LIVE_TRADING has zero `input_mode` awareness. Full plan created with 5 sub-phases:
- Phase 0: Barrier gate quick fix (standalone bug)
- Phase 1: Input mode detection in loader, inference engine, predictor
- Phase 2: Raw OHLCV normalization and inference path
- Phase 3: Testing and contract verification
- Phase 4: Cross-sectional ranking inference (future)

## Configuration Schema

```yaml
# In CONFIG/experiments/raw_sequence_experiment.yaml
pipeline:
  input_mode: "raw_sequence"  # "features" (default) or "raw_sequence"

  # Sequence-specific settings (only used if input_mode = raw_sequence)
  sequence:
    length_minutes: 320       # 64 bars @ 5m interval
    channels: ["open", "high", "low", "close", "volume"]
    normalization: "returns"  # "returns", "log_returns", "minmax"

    # Optional: predict next bar OHLCV (autoregressive mode)
    autoregressive: false     # Future work

  # These are IGNORED in raw_sequence mode
  feature_selection:
    enabled: true  # Automatically disabled if input_mode = raw_sequence
```

## Model Input Changes

### Current LSTM/Transformer Input
```python
# In lstm_trainer.py line 83-84
X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)  # (N, features, 1)
```

### New Raw Sequence Input
```python
# X_tr already in (N, seq_len, 5) shape from build_sequences_from_ohlcv
# No reshape needed, just pass directly
```

### Model Architecture Considerations

The existing models work, but could be optimized:

| Model | Current | Raw Sequence Optimization |
|-------|---------|---------------------------|
| Transformer | Single attention layer | Could add positional encoding for bar index |
| LSTM | 2-layer stacked | Works as-is, seq_len becomes lookback |
| CNN1D | 1D conv over features | Kernel=3-5 becomes "bar patterns" |

## Determinism Requirements

### PyTorch Attention
```python
# In TRAINING/common/repro_bootstrap.py
import torch
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA 10.2+
```

### Known Non-Deterministic Operations
- `torch.nn.functional.scaled_dot_product_attention` - needs `deterministic=True`
- `torch.nn.MultiheadAttention` - uses SDPA internally
- `scatter_add` operations in some attention variants

### Contract Test
```python
# TRAINING/contract_tests/test_sequence_determinism.py
def test_transformer_sequence_determinism():
    """Same input → same output across runs."""
    X = torch.randn(32, 64, 5)  # (batch, seq_len, channels)
    model = TransformerTrainer(...)

    # Two forward passes should be identical
    out1 = model(X)
    out2 = model(X)
    assert torch.allclose(out1, out2)
```

## Integration Contract Impact

### model_meta.json Changes

```json
{
  "model_family": "Transformer",
  "input_mode": "raw_sequence",      // NEW FIELD
  "sequence_length": 64,             // NEW FIELD (bars)
  "sequence_channels": ["open", "high", "low", "close", "volume"],  // NEW
  "feature_list": [],                // Empty for raw mode
  "interval_minutes": 5,
  "model_checksum": "..."
}
```

**Contract rule**: LIVE_TRADING must handle both modes - check `input_mode` field.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Attention non-determinism | Breaks reproducibility | Add deterministic flags in repro_bootstrap |
| Memory for long sequences | OOM on GPU | Dynamic batch size reduction (already exists) |
| Worse performance than features | Wasted effort | Benchmark experiment before full rollout |
| LIVE_TRADING incompatibility | Production failure | Update consumer before deploying trained models |

## Estimated Effort

| Phase | Complexity | Files Changed |
|-------|------------|---------------|
| 1. Data Infrastructure | Medium | 3-4 new/modified |
| 2. Pipeline Integration | Medium | 5-6 modified |
| 3. Determinism | Low | 2-3 modified |
| 4. Evaluation | Low | 1 new experiment |

## Phase 5: LIVE_TRADING Integration
**File**: `raw-ohlcv-sequence-live.md`

### Analysis: Current State

LIVE_TRADING already has infrastructure for sequential models:

| Component | Location | Current State |
|-----------|----------|---------------|
| `SeqBufferManager` | `TRAINING/common/live/seq_ring_buffer.py` | Stores `(T, F)` feature sequences per symbol |
| `InferenceEngine._predict_sequential` | `LIVE_TRADING/models/inference.py:281` | Uses buffer, handles warmup |
| `SEQUENTIAL_FAMILIES` | `LIVE_TRADING/common/constants.py:67` | CNN1D, LSTM, Transformer, Tab* |
| `FeatureBuilder` | `LIVE_TRADING/models/feature_builder.py` | Computes indicators from OHLCV |
| Alpaca data feed | `LIVE_TRADING/data/alpaca.py` | Already provides raw OHLCV bars |

### Required Changes

#### 1. Model Metadata Detection
```python
# In inference.py
def _should_use_raw_ohlcv(self, metadata: Dict) -> bool:
    return metadata.get("input_mode") == "raw_sequence"
```

#### 2. Raw OHLCV Buffer
Two options:
- **Option A**: Extend `SeqBufferManager` to support OHLCV mode (channels=5)
- **Option B**: New `OHLCVSeqBufferManager` class (cleaner separation)

```python
# Option A extension
class SeqBufferManager:
    def __init__(self, T, F, ttl_seconds, input_mode="features"):
        self.input_mode = input_mode
        if input_mode == "raw_sequence":
            self.F = 5  # Always 5 for OHLCV
```

#### 3. Inference Path Branching
```python
def _predict_sequential(self, model, features, target, family, symbol):
    metadata = self._metadata[f"{target}:{family}:..."]

    if self._should_use_raw_ohlcv(metadata):
        # Features IS the OHLCV bar (5 values)
        self._push_ohlcv_bar(target, family, symbol, features)
        sequence = self._get_ohlcv_sequence(target, family, symbol)
        # Apply normalization matching training
        sequence = self._normalize_ohlcv(sequence, metadata)
    else:
        # Existing feature-based path
        ...
```

#### 4. FeatureBuilder Bypass
For raw_sequence mode, FeatureBuilder is not needed - just pass OHLCV directly:

```python
# In predictor.py or engine
if model_metadata.get("input_mode") == "raw_sequence":
    # Just pass raw OHLCV, don't compute features
    features = np.array([bar.open, bar.high, bar.low, bar.close, bar.volume])
else:
    features = feature_builder.build_features(bar_data)
```

#### 5. Normalization Consistency
Must use SAME normalization as training:

```python
def _normalize_ohlcv(self, sequence: np.ndarray, metadata: Dict) -> np.ndarray:
    method = metadata.get("sequence_normalization", "returns")
    # Use exact same logic as TRAINING/training_strategies/utils.py
    from TRAINING.common.sequence_utils import normalize_ohlcv_sequence
    return normalize_ohlcv_sequence(sequence, method)
```

### Files to Modify

| File | Change |
|------|--------|
| `LIVE_TRADING/models/inference.py` | Add raw_sequence detection and path |
| `LIVE_TRADING/common/constants.py` | Add `RAW_SEQUENCE_FAMILIES` if needed |
| `TRAINING/common/live/seq_ring_buffer.py` | Support OHLCV mode (or new class) |
| `LIVE_TRADING/prediction/predictor.py` | Bypass feature builder for raw mode |
| `TRAINING/common/sequence_utils.py` | Shared normalization (new file) |

### Data Flow: Raw Sequence Mode in Live Trading

```
Alpaca WS → Bar{O,H,L,C,V} → InferenceEngine.predict()
                                    │
                                    ▼
                            Check input_mode
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            "features"                      "raw_sequence"
                    │                               │
                    ▼                               ▼
            FeatureBuilder               Push OHLCV to buffer
                    │                               │
                    ▼                               ▼
            Push to buffer               Get (T, 5) sequence
                    │                               │
                    ▼                               ▼
            Get (T, F) seq               Normalize (returns)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                            Model.predict()
                                    │
                                    ▼
                              Prediction
```

### Contract Rules for LIVE_TRADING

1. **Check `input_mode`** before inference - default to `"features"` if missing
2. **Normalization must match** - use same method as `model_meta.sequence_normalization`
3. **Buffer channels** - raw_sequence uses 5 channels (OHLCV), features uses F channels
4. **No feature computation** in raw mode - FeatureBuilder is bypassed
5. **Warmup still required** - need `sequence_length` bars before first valid prediction

## Next Steps

1. User approval of this master plan
2. Create Phase 1 sub-plan (data infrastructure)
3. Implement and test Phase 1
4. Iterate through remaining phases
5. **Update INTEGRATION_CONTRACTS.md** with new model_meta fields

## Open Questions

1. **Normalization**: Returns vs log-returns vs min-max per-sequence?
2. **Sequence length**: Fixed or configurable per experiment?
3. **Volume scaling**: Volume has different magnitude than prices - normalize separately?
4. **Cross-sectional**: Train one model per symbol, or one model for all symbols with symbol embedding?

---

## Session Log

### 2026-01-21: Initial Planning
- Explored existing feature system
- Found LSTM/Transformer use pseudo-sequences (features as timesteps)
- Identified `build_sequences_from_features` as template
- Confirmed raw OHLCV available in data but excluded from feature selection
- Created this master plan
- Created Phase 1 sub-plan: `raw-ohlcv-sequence-data.md`
- Analyzed LIVE_TRADING integration:
  - Found `SeqBufferManager` already exists
  - `InferenceEngine._predict_sequential` handles sequential models
  - Alpaca already provides raw OHLCV bars
  - FeatureBuilder can be bypassed for raw mode
- Added Phase 5: LIVE_TRADING Integration to master plan
- Updated `INTEGRATION_CONTRACTS.md` v1.3 with new fields:
  - `input_mode`, `sequence_length`, `sequence_channels`, `sequence_normalization`

### 2026-01-21: Phase 1 Implementation
- Implemented `build_sequences_from_ohlcv()` in `TRAINING/training_strategies/utils.py`
- Implemented `_normalize_ohlcv_sequence()` with 4 methods: returns, log_returns, minmax, none
- Implemented `_create_rolling_windows()` using numpy stride tricks (memory efficient)
- Implemented `_detect_timestamp_gaps()` and `_split_on_gaps()` for gap handling
- Added config keys to `CONFIG/pipeline/pipeline.yaml`:
  - `pipeline.sequence.default_length_minutes`
  - `pipeline.sequence.default_channels`
  - `pipeline.sequence.normalization`
  - `pipeline.sequence.gap_handling`
  - `pipeline.sequence.gap_tolerance`
- Added PyTorch determinism to `TRAINING/common/repro_bootstrap.py`:
  - `init_torch_determinism()` function for post-import setup
  - Handles strict mode (full determinism) vs best_effort mode (seeds only)
  - Sets `torch.use_deterministic_algorithms(True)` in strict mode
  - Sets CUDNN deterministic flags
- Created unit tests in `tests/test_sequence_builder.py` (25 tests, all passing):
  - Tests for `_create_rolling_windows`
  - Tests for `_normalize_ohlcv_sequence`
  - Tests for `build_sequences_from_ohlcv`
  - Tests for gap detection
  - Tests for PyTorch determinism initialization
  - Integration tests

**Next steps:**
1. Phase 2: Pipeline Integration - wire sequence builder into training pipeline
2. Add config toggle `pipeline.input_mode`
3. Update model trainers to accept (N, seq_len, 5) input

### 2026-01-21: Phase 2 Implementation (continued)
- Added `pipeline.input_mode` config key to pipeline.yaml
- Created `TRAINING/common/input_mode.py` module:
  - `InputMode` enum (FEATURES, RAW_SEQUENCE)
  - `get_input_mode()` for config precedence
  - `is_raw_sequence_mode()` convenience function
  - `filter_families_for_input_mode()` to filter to sequence families
  - `get_raw_sequence_config()` for sequence settings
  - `RAW_SEQUENCE_FAMILIES` constant (LSTM, Transformer, CNN1D, etc.)
- Updated `intelligent_trainer.py`:
  - Import InputMode helpers
  - Skip feature selection when `input_mode=raw_sequence`
  - Filter model families to sequence-compatible only
- Created `prepare_training_data_raw_sequence()` in `data_preparation.py`:
  - Uses `build_sequences_from_ohlcv` from Phase 1
  - Returns compatible 8-tuple format
  - Includes routing_meta with sequence config
- Updated `training.py`:
  - Import raw sequence data preparation
  - Branch based on input_mode
  - Call appropriate data prep function
- Updated model trainers (LSTM, Transformer, CNN1D):
  - Detect 3D input (already shaped for raw sequence mode)
  - Skip reshape when input is already 3D
  - Keep existing 2D→3D reshape for feature mode

### 2026-01-21: Phase 2 (continued) - Stage Skipping
- Added target ranking skip logic in `intelligent_trainer.py`:
  - When `input_mode=raw_sequence` AND `auto_targets=True`, target ranking is SKIPPED
  - Target ranking uses feature importance which doesn't apply to raw OHLCV
  - Auto-discovers targets from sample symbol as fallback
  - Recommends manual target specification for raw sequence mode

## Complete Pipeline Flow for Raw Sequence Mode

When `input_mode: raw_sequence` is set:

| Stage | Normal Mode | Raw Sequence Mode |
|-------|-------------|-------------------|
| **Target Ranking** | Uses feature importance to rank targets | **SKIPPED** - Requires manual targets or auto-discovers |
| **Feature Selection** | Selects top-M features per target | **SKIPPED** - Uses raw OHLCV channels directly |
| **Data Preparation** | `prepare_training_data_cross_sectional()` | `prepare_training_data_raw_sequence()` |
| **Model Input** | 2D (N, F) reshaped to 3D (N, F, 1) | 3D (N, seq_len, 5) used directly |
| **Model Families** | All families | Only: LSTM, Transformer, CNN1D, TabLSTM, TabTransformer, TabCNN |

## CPU/GPU Compatibility

- **Sequence building**: Uses numpy (CPU, very fast with stride tricks)
- **Model training**: TensorFlow auto-detects GPU (LSTM, Transformer, CNN1D)
- **PyTorch**: `init_torch_determinism()` handles CUDA settings

**Remaining for Phase 2:**
- End-to-end testing with actual training run
- Verify model_meta.json gets correct fields

**Next steps:**
1. Create experiment config for testing raw sequence mode
2. Run end-to-end test
3. Phase 5: LIVE_TRADING integration
