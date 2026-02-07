# Phase 1: Raw OHLCV Sequence Data Infrastructure

**Parent**: `raw-ohlcv-sequence-mode.md`
**Status**: Ready for implementation
**Estimated complexity**: Medium

## Goal

Create a data loading path that produces `(N, seq_len, 5)` tensors of raw OHLCV bars instead of `(N, features)` tabular data. This is the foundation for training sequence models on raw price data.

## Current State

### Existing Components

```python
# TRAINING/training_strategies/utils.py:425
def build_sequences_from_features(X, lookback=None, lookback_minutes=None, interval_minutes=None):
    """Convert 2D features (N, F) to 3D sequences (N', T, F) using rolling windows."""
    # ... rolling window over FEATURES
```

```python
# TRAINING/data/loading/unified_loader.py
# Loads parquet with columns: ts, symbol, open, high, low, close, volume, ...features...
```

### Problem

Current sequence building:
1. Takes **features** (RSI, MACD, etc.) as input
2. Creates rolling windows over feature vectors
3. Shape: `(N, seq_len, num_features)` where `num_features` = 100+

We need:
1. Take **raw OHLCV** columns only
2. Create rolling windows over bars
3. Shape: `(N, seq_len, 5)` where 5 = O, H, L, C, V

## Implementation

### 1. New Function: `build_sequences_from_ohlcv`

**Location**: `TRAINING/training_strategies/utils.py` (next to existing `build_sequences_from_features`)

```python
def build_sequences_from_ohlcv(
    df: pd.DataFrame,
    seq_len: int = None,
    seq_len_minutes: int = None,
    interval_minutes: int = None,
    channels: List[str] = None,
    normalization: str = "returns",
    symbol: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences from raw OHLCV data for sequence model training.

    Args:
        df: DataFrame with columns [ts, open, high, low, close, volume, ...]
            Must be sorted by timestamp.
        seq_len: Sequence length in bars (mutually exclusive with seq_len_minutes)
        seq_len_minutes: Sequence length in minutes (preferred, interval-agnostic)
        interval_minutes: Data interval for conversion (required if seq_len_minutes)
        channels: OHLCV columns to use. Default: ["open", "high", "low", "close", "volume"]
        normalization: How to normalize prices:
            - "returns": (p[t] - p[t-1]) / p[t-1]
            - "log_returns": log(p[t] / p[t-1])
            - "minmax": (p - min) / (max - min) per sequence
            - "none": No normalization (raw prices)
        symbol: Optional symbol for logging

    Returns:
        X: (N, seq_len, num_channels) sequences
        timestamps: (N,) timestamp of last bar in each sequence
        indices: (N,) original DataFrame indices for alignment

    Raises:
        ValueError: If df is not sorted by timestamp
        ValueError: If required columns missing

    DRY/SST Compliance:
        - Uses minutes_to_bars() for interval conversion
        - Sorted iteration for determinism
        - Config via get_cfg() for defaults
    """
```

### 2. Normalization Strategies

```python
def _normalize_ohlcv_sequence(
    seq: np.ndarray,
    method: str = "returns"
) -> np.ndarray:
    """
    Normalize a single OHLCV sequence.

    Args:
        seq: (seq_len, 5) array of [O, H, L, C, V]
        method: Normalization method

    Returns:
        Normalized (seq_len, 5) array

    Notes:
        - "returns": Divide price columns by first bar's close, subtract 1
        - "log_returns": log(p / first_close)
        - "minmax": Scale to [0, 1] per channel
        - Volume always scaled separately (different magnitude)
    """
    if method == "returns":
        # Price channels: relative to first close
        first_close = seq[0, 3]  # close is index 3
        prices = seq[:, :4]  # O, H, L, C
        prices_norm = (prices / first_close) - 1.0

        # Volume: relative to first volume (or log scale)
        first_vol = seq[0, 4] + 1e-8
        vol_norm = (seq[:, 4] / first_vol) - 1.0

        return np.column_stack([prices_norm, vol_norm[:, None]])

    elif method == "log_returns":
        first_close = seq[0, 3]
        prices = seq[:, :4]
        prices_norm = np.log(prices / first_close + 1e-8)

        first_vol = seq[0, 4] + 1e-8
        vol_norm = np.log(seq[:, 4] / first_vol + 1e-8)

        return np.column_stack([prices_norm, vol_norm[:, None]])

    elif method == "minmax":
        # Per-channel min-max scaling
        mins = seq.min(axis=0, keepdims=True)
        maxs = seq.max(axis=0, keepdims=True)
        return (seq - mins) / (maxs - mins + 1e-8)

    else:
        return seq
```

### 3. Memory-Efficient Windowing

Use `np.lib.stride_tricks` instead of explicit loops:

```python
def _create_rolling_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create rolling windows using stride tricks (zero-copy).

    Args:
        arr: (N, C) array
        window_size: Window length

    Returns:
        (N - window_size + 1, window_size, C) view into arr
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # sliding_window_view handles the striding automatically
    # Returns (N - window_size + 1, window_size, C)
    return sliding_window_view(arr, window_shape=window_size, axis=0)
```

### 4. Gap Handling

Time series may have gaps (weekends, holidays, halts). Options:

```python
def _detect_gaps(timestamps: np.ndarray, expected_interval_seconds: int) -> np.ndarray:
    """
    Detect gaps in timestamp series.

    Returns:
        Boolean array, True where gap detected
    """
    diffs = np.diff(timestamps.astype('datetime64[s]').astype(int))
    return diffs > (expected_interval_seconds * 1.5)  # 50% tolerance


def _split_on_gaps(df: pd.DataFrame, gap_mask: np.ndarray) -> List[pd.DataFrame]:
    """Split DataFrame into continuous segments."""
    split_indices = np.where(gap_mask)[0] + 1
    return np.split(df, split_indices)
```

**Strategy**: Don't create sequences that span gaps - split data at gaps first.

### 5. Integration with DataLoader

Modify or wrap `UnifiedDataLoader` to support raw mode:

```python
# TRAINING/data/loading/unified_loader.py

class UnifiedDataLoader:
    def load_for_sequence_training(
        self,
        symbol: str,
        seq_len_minutes: int,
        channels: List[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Load data specifically for sequence model training.

        Returns:
            {
                "X": (N, seq_len, C) sequences,
                "timestamps": (N,) timestamps,
                "symbol": str
            }
        """
        # Load only needed columns (column projection)
        columns = ["ts"] + (channels or ["open", "high", "low", "close", "volume"])

        df = self.load_symbol(symbol, columns=columns)

        return build_sequences_from_ohlcv(
            df,
            seq_len_minutes=seq_len_minutes,
            interval_minutes=self.interval_minutes,
            channels=channels,
            **kwargs
        )
```

## File Changes

| File | Change |
|------|--------|
| `TRAINING/training_strategies/utils.py` | Add `build_sequences_from_ohlcv`, `_normalize_ohlcv_sequence` |
| `TRAINING/data/loading/unified_loader.py` | Add `load_for_sequence_training` method |
| `TRAINING/common/interval.py` | Verify `minutes_to_bars` works for this use case |
| `CONFIG/pipeline/pipeline.yaml` | Add `sequence.default_channels`, `sequence.normalization` |

## Config Additions

```yaml
# CONFIG/pipeline/pipeline.yaml

pipeline:
  # ... existing config ...

  # Sequence model settings (used when input_mode = "raw_sequence")
  sequence:
    default_length_minutes: 320    # 64 bars @ 5m
    default_channels:
      - open
      - high
      - low
      - close
      - volume
    normalization: "returns"       # returns, log_returns, minmax, none
    gap_handling: "split"          # split (don't span gaps) or "pad" (fill gaps)
```

## Testing

### Unit Tests

```python
# tests/test_sequence_builder.py

def test_build_sequences_shape():
    """Output shape is (N-seq_len+1, seq_len, channels)."""
    df = make_dummy_ohlcv(100)  # 100 bars
    X, ts, idx = build_sequences_from_ohlcv(df, seq_len=20)
    assert X.shape == (81, 20, 5)  # 100 - 20 + 1 = 81

def test_build_sequences_determinism():
    """Same input produces same output."""
    df = make_dummy_ohlcv(100)
    X1, _, _ = build_sequences_from_ohlcv(df, seq_len=20)
    X2, _, _ = build_sequences_from_ohlcv(df, seq_len=20)
    np.testing.assert_array_equal(X1, X2)

def test_normalization_returns():
    """Returns normalization centers around 0."""
    df = make_dummy_ohlcv(100)
    X, _, _ = build_sequences_from_ohlcv(df, seq_len=20, normalization="returns")
    # First bar in each sequence should be ~0 after normalization
    assert np.allclose(X[:, 0, :4].mean(), 0, atol=0.01)

def test_gap_handling():
    """Sequences don't span time gaps."""
    df = make_dummy_ohlcv_with_gap(100, gap_at=50)
    X, _, _ = build_sequences_from_ohlcv(df, seq_len=20)
    # Should have fewer sequences due to gap split
    assert X.shape[0] < 81
```

### Contract Tests

```python
# TRAINING/contract_tests/test_sequence_data.py

def test_sequence_loader_produces_valid_shape():
    """Loader output matches expected contract."""
    loader = UnifiedDataLoader(...)
    result = loader.load_for_sequence_training("AAPL", seq_len_minutes=320)

    assert "X" in result
    assert "timestamps" in result
    assert result["X"].ndim == 3
    assert result["X"].shape[2] == 5  # OHLCV channels
```

## Determinism Checklist

- [ ] Input DataFrame must be sorted by timestamp (validate + raise)
- [ ] Use `sorted()` for any dict/set iteration
- [ ] No random sampling without seed
- [ ] Normalization is deterministic (no randomness)
- [ ] Gap detection uses consistent threshold

## Dependencies

- NumPy (already present)
- Pandas (already present)
- No new dependencies required

## Completion Criteria

1. [x] `build_sequences_from_ohlcv` implemented and tested
2. [ ] `UnifiedDataLoader.load_for_sequence_training` added
3. [x] Config keys added to pipeline.yaml
4. [x] Unit tests passing (25/25 in tests/test_sequence_builder.py)
5. [ ] Contract test passing
6. [x] Determinism verified (same input = same output)

## Next Phase

After this phase: **Phase 2: Pipeline Integration** - wire the new data loader into the training pipeline with a config toggle.
