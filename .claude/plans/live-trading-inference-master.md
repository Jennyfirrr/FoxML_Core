# LIVE_TRADING Inference Fixes — Master Plan

**Status**: Phases 0-3 COMPLETE, Phase 4 pending
**Created**: 2026-02-08
**Branch**: `analysis/code-review-and-raw-ohlcv`
**Depends on**: `raw-ohlcv-sequence-mode.md` (Phases 1-2 complete), `cross-sectional-ranking-objective.md` (complete)

## Overview

The LIVE_TRADING inference pipeline was built for feature-based models only. The training side now supports two additional modes — raw OHLCV sequences and cross-sectional ranking — but `input_mode` appears **zero times** in LIVE_TRADING code. This master plan addresses all identified bugs and wires the inference pipeline to handle all three model types.

### Problem Summary

| Bug | File | Severity | Root Cause |
|-----|------|----------|------------|
| Barrier gate calls non-existent method | `trading_engine.py:1295` | CRITICAL | `predict_single_target()` doesn't exist |
| No `input_mode` branching in inference | `inference.py:197` | CRITICAL | Phase 5 never implemented |
| Sequential buffer skips raw models | `inference.py:182` | CRITICAL | `feature_list=[]` → bail |
| Predictor always builds features | `predictor.py:342` | CRITICAL | No raw OHLCV data path |
| No OHLCV normalization in LIVE_TRADING | entire module | CRITICAL | Missing function |
| False warning for raw sequence models | `loader.py:377` | HIGH | No `input_mode` awareness |
| No CS ranking inference path | `predictor.py` | HIGH | Phase not implemented |

### Architecture After Fix

```
Market Data (OHLCV DataFrame)
    ↓
MultiHorizonPredictor._predict_single()
    ├─ input_mode == "features" (existing)
    │   └─ FeatureBuilder.build_features() → (F,) → InferenceEngine.predict()
    │       ├─ TREE_FAMILIES → _predict_tree()
    │       ├─ SEQUENTIAL_FAMILIES → _predict_sequential() via buffer (T, F)
    │       └─ TF_FAMILIES → _predict_keras()
    │
    └─ input_mode == "raw_sequence" (NEW)
        └─ _prepare_raw_sequence() → normalize → (T, 5) → InferenceEngine.predict()
            └─ SEQUENTIAL_FAMILIES → _predict_raw_sequential() via buffer (T, 5)

CrossSectionalRankingPredictor (NEW — future phase)
    └─ Collects all symbol predictions → ranks cross-sectionally
```

## Sub-Plans

### Phase 0: Barrier Gate Quick Fix (standalone)
**File**: `live-phase0-barrier-gate-fix.md`
**Status**: ✅ COMPLETE (commit b8bdf39)
**Estimated scope**: 1 file, ~10 lines

Fix the broken `predict_single_target()` call in `trading_engine.py`. This is a regression that affects ALL model types (not just raw OHLCV). Can be done independently of Phases 1-3.

- [x] Add `predict_single_target()` method to `MultiHorizonPredictor`
- [x] Delegates to `_predict_single()` with first available family/horizon
- [x] Added `.alpha` property to `ModelPrediction` (used by barrier gate sigmoid)

### Phase 1: Input Mode Awareness
**File**: `live-phase1-input-mode-awareness.md`
**Status**: ✅ COMPLETE (commit efcc723)
**Estimated scope**: 3 files, ~60 lines

Add `input_mode` detection throughout the inference pipeline. No new functionality — just make every component aware of which mode a model requires.

Files:
- `LIVE_TRADING/models/loader.py` — `get_feature_list()` + new `get_input_mode()`
- `LIVE_TRADING/models/inference.py` — `_init_sequential_buffer()`, `predict()`
- `LIVE_TRADING/prediction/predictor.py` — `_predict_single()`

Key changes:
- [x] `ModelLoader.get_input_mode(target, family)` → returns `"features"` or `"raw_sequence"`
- [x] `ModelLoader.get_sequence_config(target, family)` → returns seq config dict
- [x] `ModelLoader.get_feature_list()` — suppress warning for raw_sequence models
- [x] `InferenceEngine._init_sequential_buffer()` — use `sequence_channels` for raw models (F=5)
- [x] `InferenceEngine.predict()` — route to `_predict_raw_sequential()` for raw models
- [x] `InferenceEngine._predict_raw_sequential()` — full implementation (push to buffer, predict)
- [x] `MultiHorizonPredictor._predict_single()` — branch on input_mode

### Phase 2: Raw OHLCV Inference Path
**File**: `live-phase2-raw-ohlcv-inference.md`
**Status**: ✅ COMPLETE (commit 2ef0fba)
**Estimated scope**: 3 files, ~120 lines

Implement the actual raw OHLCV data preparation and inference path.

Key changes:
- [x] Import `_normalize_ohlcv_sequence` from `TRAINING/training_strategies/utils.py` (SST)
- [x] `MultiHorizonPredictor._prepare_raw_sequence(prices, target, family)` — extract OHLCV columns (case-insensitive), normalize via SST, return `(5,)` array for buffer
- [x] `InferenceEngine._predict_raw_sequential()` — push OHLCV rows to buffer, predict when ready
- [x] Handle buffer warmup (return NaN while filling)
- [x] Validate `sequence_normalization` exists in metadata, default to `"returns"`

### Phase 3: Testing & Contract Verification
**File**: `live-phase3-testing.md`
**Status**: ✅ COMPLETE (commit daee527)
**Estimated scope**: 2 new test files, ~200 lines

End-to-end verification that raw OHLCV models work through the live pipeline.

- [x] Unit test: `test_live_inference_input_mode.py` — 14 tests for loader, inference, predictor
- [x] Integration test: `test_live_raw_ohlcv_e2e.py` — 9 tests for normalization, column mapping, edge cases
- [x] Contract test: verify `model_meta.json` fields consumed correctly
- [x] Backward compat test: models without `input_mode` default to features
- [x] All 23 tests passing

### Phase 4: Cross-Sectional Ranking Inference (future)
**File**: `live-phase4-cs-ranking-inference.md`
**Status**: Planning (not blocking raw OHLCV)
**Estimated scope**: 1 new file + modifications, ~250 lines

Implement ranking-aware inference for CS-trained models. This is a separate concern from raw OHLCV — CS ranking models output relative scores that need cross-symbol comparison.

- [ ] New class: `CrossSectionalRankingPredictor` in `LIVE_TRADING/prediction/`
- [ ] Collect predictions for all symbols at a timestamp
- [ ] Rank cross-sectionally (percentile rank)
- [ ] Feed ranked signals to blending/arbitration
- [ ] Read `cross_sectional_ranking` metadata from `model_meta.json`

## Dependency Graph

```
Phase 0 (barrier gate)  ←— standalone, do first
    |
Phase 1 (input mode awareness) ←— foundation for Phase 2
    |
Phase 2 (raw OHLCV inference) ←— core implementation
    |
Phase 3 (testing) ←— verification
    |
Phase 4 (CS ranking) ←— future, independent
```

## Contract Compliance

All changes must satisfy `INTEGRATION_CONTRACTS.md` v1.3:

| Field | Used By | Phase |
|-------|---------|-------|
| `input_mode` | loader, inference, predictor | 1 |
| `sequence_length` | buffer init | 1 |
| `sequence_channels` | buffer init (F dimension) | 1 |
| `sequence_normalization` | raw OHLCV prep | 2 |
| `cross_sectional_ranking.*` | CS ranking predictor | 4 |

## Key Design Decisions

1. **Normalization SST**: Import `_normalize_ohlcv_sequence` from `TRAINING/training_strategies/utils.py` rather than reimplementing. The docstring already marks it as the SST for both TRAINING and LIVE_TRADING.

2. **Backward compatibility**: `input_mode` defaults to `"features"` everywhere. Existing models without this field continue to work unchanged.

3. **Buffer reuse**: `SeqRingBuffer` and `SeqBufferManager` work for both modes — they just need the correct `F` dimension (5 for raw OHLCV, len(feature_list) for features).

4. **No FeatureBuilder for raw mode**: When `input_mode == "raw_sequence"`, FeatureBuilder is never instantiated. Raw OHLCV bars flow directly from market data to normalization to buffer.

## Session Notes

### 2026-02-08: Initial creation
- Identified 9 bugs across LIVE_TRADING from code review
- Confirmed `input_mode` appears 0 times in LIVE_TRADING code
- Confirmed INTEGRATION_CONTRACTS.md v1.3 already documents expected consumer behavior
- Created master plan with 5 phases (0-4)

### 2026-02-08: Phases 0-3 implemented
- Phase 0: Added `predict_single_target()` + `.alpha` property on ModelPrediction
- Phase 1: Added `get_input_mode()`, `get_sequence_config()` to loader; fixed buffer init for raw; added routing in predict(); full `_predict_raw_sequential()` implementation
- Phase 2: Imported SST normalization function; implemented `_prepare_raw_sequence()` with case-insensitive column matching
- Phase 3: 23 tests all passing (14 unit + 9 integration)
- Phase 4 (CS ranking inference) remains for future implementation
- All changes on branch `analysis/code-review-and-raw-ohlcv`
