# Phase 2: Raw OHLCV Inference Path

**Status**: Ready for implementation
**Parent**: `live-trading-inference-master.md`
**Scope**: 3 files, ~120 lines
**Depends on**: Phase 1 (input mode awareness — stubs replaced here)
**Blocks**: Phase 3 (testing)

## Goal

Implement the actual raw OHLCV data preparation and inference path. Replace the `NotImplementedError` stubs from Phase 1 with working code.

## Key Design: Normalization SST

`TRAINING/training_strategies/utils.py:458` defines `_normalize_ohlcv_sequence()` with this docstring:

> This is the SINGLE SOURCE OF TRUTH for OHLCV normalization.
> Used by both TRAINING (sequence building) and LIVE_TRADING (inference).

We import this directly — no reimplementation. The function accepts `(seq_len, 5)` numpy array and normalization method string, returns normalized `(seq_len, 5)` array.

## Changes

### 1. `LIVE_TRADING/prediction/predictor.py`

Replace the `_prepare_raw_sequence()` stub:

```python
from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence

def _prepare_raw_sequence(
    self,
    prices: pd.DataFrame,
    target: str,
    family: str,
) -> Optional[np.ndarray]:
    """
    Prepare raw OHLCV sequence for inference.

    Extracts the last `sequence_length` bars from prices DataFrame,
    applies the same normalization used during training, and returns
    a (5,) array for the latest bar (pushed to buffer one bar at a time).

    CONTRACT: INTEGRATION_CONTRACTS.md v1.3
    - sequence_normalization must match training
    - sequence_channels must be ["open", "high", "low", "close", "volume"]
    """
    seq_config = self.loader.get_sequence_config(target, family)
    if not seq_config:
        logger.error(f"No sequence config for {target}:{family}")
        return None

    seq_len = seq_config["sequence_length"]
    normalization = seq_config["sequence_normalization"]
    channels = seq_config["sequence_channels"]

    # Map channel names to DataFrame columns
    # Handles both lowercase (parquet) and capitalized (broker) column names
    col_map = {
        "open": next((c for c in prices.columns if c.lower() == "open"), None),
        "high": next((c for c in prices.columns if c.lower() == "high"), None),
        "low": next((c for c in prices.columns if c.lower() == "low"), None),
        "close": next((c for c in prices.columns if c.lower() == "close"), None),
        "volume": next((c for c in prices.columns if c.lower() == "volume"), None),
    }

    missing = [ch for ch in channels if col_map.get(ch) is None]
    if missing:
        logger.error(f"Missing OHLCV columns for {target}:{family}: {missing}")
        return None

    # Extract columns in channel order
    ohlcv_cols = [col_map[ch] for ch in channels]
    ohlcv_df = prices[ohlcv_cols].tail(seq_len + 1)  # +1 for normalization context

    if len(ohlcv_df) < 2:
        logger.warning(f"Insufficient data for {target}:{family}: need >= 2 bars, got {len(ohlcv_df)}")
        return None

    ohlcv_array = ohlcv_df.values.astype(np.float32)

    # Normalize using the SST function from TRAINING
    normalized = _normalize_ohlcv_sequence(ohlcv_array, method=normalization)

    # Return the latest bar as a 1D array to push into the ring buffer
    # The buffer accumulates bars over time; we push one per cycle
    return normalized[-1]  # shape: (5,)
```

**Why return single bar, not full sequence?**

The `SeqRingBuffer` accumulates bars across trading cycles. Each cycle pushes one new bar. When the buffer has `T` bars, it's "ready" and returns the full `(1, T, 5)` tensor for prediction. This matches the feature-based sequential path which also pushes one feature vector per cycle.

### 2. `LIVE_TRADING/models/inference.py`

Replace the `_predict_raw_sequential()` stub:

```python
def _predict_raw_sequential(
    self,
    model: Any,
    ohlcv_row: np.ndarray,
    target: str,
    family: str,
    symbol: str,
    metadata: Dict[str, Any],
) -> float:
    """
    Predict with raw OHLCV sequential model.

    Pushes normalized OHLCV bar into ring buffer, predicts when buffer full.
    Same pattern as _predict_sequential() but with raw OHLCV data.
    """
    buffer_key = f"{target}:{family}"
    buffer_manager = self._seq_buffers.get(buffer_key)

    if buffer_manager is None:
        raise InferenceError(family, symbol, "Buffer not initialized for raw sequence model")

    # Push normalized OHLCV bar to buffer
    ohlcv_1d = np.atleast_1d(ohlcv_row).astype(np.float32)
    buffer_manager.push_features(symbol, ohlcv_1d)

    # Check if buffer has enough bars
    if not buffer_manager.is_ready(symbol):
        return float("nan")  # Still warming up

    # Get full sequence and predict
    sequence = buffer_manager.get_sequence(symbol)
    if sequence is None:
        return float("nan")

    # PyTorch model
    if hasattr(model, "forward"):
        import torch
        with torch.no_grad():
            seq_tensor = sequence.to(self.device)
            pred = model(seq_tensor)
            return float(pred.cpu().numpy().squeeze())
    # Keras model
    else:
        pred = model.predict(sequence.numpy(), verbose=0)
        return float(pred.squeeze())
```

### 3. Import addition

In `LIVE_TRADING/prediction/predictor.py`, add at top:

```python
from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence
```

## Data Flow (complete)

```
Trading cycle for symbol SPY:
    1. data_provider.get_historical("SPY") → DataFrame with OHLCV columns
    2. predictor._predict_single() detects input_mode == "raw_sequence"
    3. predictor._prepare_raw_sequence():
       a. Extract last seq_len+1 bars of OHLCV
       b. Call _normalize_ohlcv_sequence(array, "log_returns")
       c. Return last normalized bar: (5,)
    4. engine.predict() → routes to _predict_raw_sequential()
    5. _predict_raw_sequential():
       a. Push (5,) bar into SeqRingBuffer
       b. If buffer not full → return NaN (warming up)
       c. If buffer full → get_sequence() → (1, T, 5) tensor
       d. model.forward(tensor) → scalar prediction
    6. Prediction flows through standardization, confidence, blending (unchanged)
```

## Edge Cases

| Case | Handling |
|------|----------|
| Buffer warming up (first T-1 cycles) | Return NaN, predictor skips |
| Missing OHLCV columns in prices | Log error, return None |
| Normalization method missing from metadata | Default to "returns" |
| Stale buffer (TTL expired) | SeqRingBuffer handles via TTL |
| Mixed models (some feature, some raw) per target | Each model has its own input_mode; predictor branches per-family |
| DataFrame column case mismatch ("Open" vs "open") | Case-insensitive column matching |

## Verification

- [ ] `_prepare_raw_sequence()` returns `(5,)` array for valid data
- [ ] `_prepare_raw_sequence()` returns `None` for insufficient data
- [ ] Column mapping works for both "Open" and "open" column names
- [ ] Normalization matches training exactly (import same function)
- [ ] Buffer accumulates bars correctly (5 channels)
- [ ] Prediction returns NaN during warmup, float after ready
- [ ] PyTorch model receives correct tensor shape `(1, T, 5)`

## Files Changed

1. `LIVE_TRADING/prediction/predictor.py` — implement `_prepare_raw_sequence()`
2. `LIVE_TRADING/models/inference.py` — implement `_predict_raw_sequential()`
