# Phase 3: Testing & Contract Verification

**Status**: Ready for implementation
**Parent**: `live-trading-inference-master.md`
**Scope**: 2 new test files, ~200 lines
**Depends on**: Phase 2 (raw OHLCV inference)

## Goal

Verify that raw OHLCV models work end-to-end through the live inference pipeline, and that existing feature-based models are unaffected.

## Test Files

### 1. `tests/test_live_inference_input_mode.py`

Unit tests for input_mode awareness (Phase 1 + 2):

```python
# --- ModelLoader tests ---

def test_get_input_mode_default():
    """Models without input_mode field return 'features'."""
    # Mock metadata without input_mode
    # Assert loader.get_input_mode() == "features"

def test_get_input_mode_raw_sequence():
    """Models with input_mode='raw_sequence' detected correctly."""
    # Mock metadata with input_mode="raw_sequence"
    # Assert loader.get_input_mode() == "raw_sequence"

def test_get_feature_list_no_warning_for_raw():
    """get_feature_list() returns [] without warning for raw_sequence models."""
    # Mock raw_sequence metadata with feature_list=[]
    # Assert no WARNING log emitted

def test_get_sequence_config():
    """Sequence config extracted correctly from metadata."""
    # Mock metadata with sequence_length, sequence_channels, sequence_normalization
    # Assert returned dict matches

def test_get_sequence_config_defaults():
    """Sequence config uses safe defaults for missing fields."""
    # Mock metadata with input_mode="raw_sequence" but no sequence fields
    # Assert defaults: length=64, channels=OHLCV, normalization="returns"

def test_get_sequence_config_features_mode():
    """Sequence config returns empty dict for feature-based models."""

# --- InferenceEngine tests ---

def test_buffer_init_raw_sequence():
    """Buffer initialized with F=5 for raw_sequence models."""
    # Mock metadata: input_mode="raw_sequence", sequence_channels=5
    # Assert buffer.F == 5

def test_buffer_init_features():
    """Buffer initialized with F=len(feature_list) for feature models."""
    # Mock metadata: input_mode="features", feature_list=100 items
    # Assert buffer.F == 100

def test_predict_routes_raw_sequence():
    """predict() routes to _predict_raw_sequential for raw models."""
    # Mock model + metadata with input_mode="raw_sequence"
    # Assert _predict_raw_sequential called (not _predict_sequential)

def test_predict_routes_features():
    """predict() routes to existing paths for feature models."""
    # Mock model + metadata with input_mode="features"
    # Assert _predict_tree / _predict_sequential called as before

def test_raw_sequence_non_sequential_family_raises():
    """raw_sequence mode with tree family raises InferenceError."""
    # Mock LightGBM model with input_mode="raw_sequence"
    # Assert InferenceError raised

# --- MultiHorizonPredictor tests ---

def test_predict_single_branches_raw():
    """_predict_single() calls _prepare_raw_sequence for raw models."""

def test_predict_single_branches_features():
    """_predict_single() calls FeatureBuilder for feature models."""

def test_predict_single_target_exists():
    """predict_single_target() method exists and works."""
    # Barrier gate fix verification
```

### 2. `tests/test_live_raw_ohlcv_e2e.py`

Integration tests with mock models:

```python
def test_raw_ohlcv_normalization_matches_training():
    """Normalization in live path matches training exactly."""
    from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence
    # Create sample OHLCV data
    # Normalize via training function
    # Normalize via _prepare_raw_sequence
    # Assert exact match

def test_raw_ohlcv_buffer_warmup():
    """Buffer returns NaN during warmup, float after T bars."""
    # Create mock LSTM model
    # Push T-1 bars → assert NaN
    # Push T-th bar → assert float prediction

def test_raw_ohlcv_column_case_insensitive():
    """Column matching works for 'Open', 'open', 'OPEN'."""
    # Create DataFrame with capitalized columns
    # Assert _prepare_raw_sequence still works

def test_backward_compat_no_input_mode():
    """Models without input_mode field work exactly as before."""
    # Create metadata without input_mode
    # Run full prediction cycle
    # Assert identical behavior to pre-change code

def test_mixed_models_same_target():
    """Target with both feature and raw models handles both."""
    # Mock target with LightGBM (features) + LSTM (raw_sequence)
    # Assert both predictions work independently

def test_contract_fields_consumed():
    """All INTEGRATION_CONTRACTS.md v1.3 fields are consumed."""
    # Create metadata with all raw_sequence fields
    # Assert each field is read (input_mode, sequence_length,
    #   sequence_channels, sequence_normalization)
```

## Contract Verification Checklist

| Contract Field | Verified By |
|---------------|------------|
| `input_mode` | `test_get_input_mode_*`, `test_predict_routes_*` |
| `sequence_length` | `test_buffer_init_raw_sequence`, `test_get_sequence_config` |
| `sequence_channels` | `test_buffer_init_raw_sequence`, `test_get_sequence_config` |
| `sequence_normalization` | `test_raw_ohlcv_normalization_matches_training` |
| `feature_list` (empty for raw) | `test_get_feature_list_no_warning_for_raw` |
| Backward compat (no input_mode) | `test_backward_compat_no_input_mode` |

## Running Tests

```bash
# Phase 3 tests only
pytest tests/test_live_inference_input_mode.py tests/test_live_raw_ohlcv_e2e.py -v

# All LIVE_TRADING tests
pytest tests/test_live_*.py -v

# With coverage
pytest --cov=LIVE_TRADING tests/test_live_*.py
```

## Files Created

1. `tests/test_live_inference_input_mode.py` — unit tests
2. `tests/test_live_raw_ohlcv_e2e.py` — integration tests
