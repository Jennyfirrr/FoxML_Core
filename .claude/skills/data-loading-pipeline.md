# Data Loading Pipeline

This skill documents the memory-efficient data loading flow used in the TRAINING pipeline.

## Overview

The pipeline uses a **Polars → Pandas → numpy** flow that has been proven to handle large datasets (70M+ rows) with acceptable memory usage (~64GB peak).

**IMPORTANT**: Direct Polars → numpy conversion was attempted and **failed** - it causes higher memory usage than the Pandas path. Do not attempt to bypass the Pandas intermediate step.

## Data Format Journey

```
Parquet (disk)
    ↓ pl.scan_parquet() [lazy, no memory]
Polars LazyFrame
    ↓ .collect(streaming=True) [streaming materialization]
Polars DataFrame (~32GB)
    ↓ .to_pandas() [brief 64GB peak]
Pandas DataFrame (~32GB)
    ↓ .to_numpy() / .values
Numpy Arrays (~32GB)
```

## Key Files and Functions

### Data Loading (`TRAINING/data/loading/unified_loader.py`)

| Function | Purpose |
|----------|---------|
| `UnifiedDataLoader.load_data()` | Load multiple symbols with column projection |
| `UnifiedDataLoader.load_for_target()` | Load with target-specific feature filtering |
| `streaming_concat(mtf_data)` | Memory-efficient combine of symbol DataFrames |
| `release_data(mtf_data)` | Explicit memory cleanup with triple-pass GC |

### Training Data Prep (`TRAINING/training_strategies/execution/data_preparation.py`)

| Function | Purpose |
|----------|---------|
| `prepare_training_data_cross_sectional()` | Main entry point for training data |
| `_prepare_training_data_polars()` | Polars-based preparation (uses Pandas conversion) |
| `_process_combined_data_pandas()` | Pandas processing and numpy extraction |

### Ranking Data Prep (`TRAINING/ranking/utils/cross_sectional_data.py`)

| Function | Purpose |
|----------|---------|
| `prepare_cross_sectional_data_for_ranking()` | Main entry point for ranking data |

## Memory Management Pattern

The critical pattern for memory efficiency is **immediate release**:

```python
# CORRECT: Immediate release after conversion
combined_df = data_pl.to_pandas()
del data_pl          # Release Polars immediately
gc.collect()         # Force garbage collection

# WRONG: Holding both in memory
combined_df = data_pl.to_pandas()
# ... do stuff with combined_df while data_pl still in scope ...
```

### Release Points

Memory cleanup happens at these stages:

1. **After symbol conversion** in `streaming_concat()`:
   ```python
   lf = pl.from_pandas(df).lazy()
   del df
   mtf_data[symbol] = None
   gc.collect()
   ```

2. **After LazyFrame collect**:
   ```python
   combined_pl = combined_lf.collect(streaming=True)
   del combined_lf
   gc.collect()
   ```

3. **After Polars → Pandas conversion**:
   ```python
   combined_df = data_pl.to_pandas()
   del data_pl
   gc.collect()
   ```

4. **Final cleanup** via `release_data()`:
   ```python
   gc.collect(0)  # Generation 0
   gc.collect(1)  # Generation 1
   gc.collect(2)  # Full collection
   ```

## streaming_concat() Details

This is the DRY helper for combining multi-symbol data:

```python
def streaming_concat(
    mtf_data: Dict[str, pd.DataFrame],
    symbol_column: str = "symbol",
    target_column: Optional[str] = None,
    use_float32: Optional[bool] = None,  # Default True from config
    release_after_convert: bool = True,
) -> pl.LazyFrame:
```

**Key behaviors**:
- Processes symbols in **sorted order** (determinism)
- Adds symbol column to each DataFrame
- Optionally casts to float32 (50% memory reduction)
- Releases each DataFrame immediately after conversion
- Returns a **LazyFrame** (not materialized yet)

## Pandas Processing Stage

After conversion to Pandas, the `_process_combined_data_pandas()` function:

1. Extracts target using `safe_target_extraction()`
2. Extracts features: `feature_df = combined_df[feature_names].copy()`
3. Coerces types: `pd.to_numeric(..., errors='coerce')`
4. Sanitizes: Replace inf/-inf with NaN
5. Drops all-NaN columns
6. Keeps only numeric columns
7. Converts to numpy: `X = feature_df.to_numpy(dtype=np.float32)`
8. Cleans rows (removes NaN > 50%, invalid targets)
9. Imputes: `SimpleImputer(strategy="median")`

## Configuration

| Key | Default | Purpose |
|-----|---------|---------|
| `intelligent_training.lazy_loading.use_float32` | `true` | Cast to float32 (50% memory savings) |
| `preprocessing.imputation.strategy` | `"median"` | Imputation method |
| `pipeline.data_limits.max_cross_sectional_samples` | `null` | Per-timestamp sampling limit |

## What NOT To Do

### Do NOT bypass Pandas

```python
# WRONG - causes ~160GB peak memory
X = polars_df.select(features).to_numpy()
```

The Polars `select()` operation creates intermediate DataFrames that Python's GC doesn't release in time. Multiple chained operations compound the problem.

### Do NOT hold DataFrames longer than needed

```python
# WRONG - both in memory simultaneously
combined_df = data_pl.to_pandas()
X = combined_df[features].to_numpy()
y = data_pl.get_column(target).to_numpy()  # data_pl still alive!
```

### Do NOT skip gc.collect()

The explicit `gc.collect()` calls are necessary. Python's automatic GC is too slow for large DataFrames.

## Memory Profile

For 70M rows × 114 features:

| Stage | Memory |
|-------|--------|
| Polars DataFrame | ~32GB |
| During `to_pandas()` | ~64GB peak (both formats briefly) |
| After `del + gc.collect()` | ~32GB (Pandas only) |
| Final numpy arrays | ~32GB |

This is acceptable. Attempts to reduce the 64GB peak by bypassing Pandas have consistently made things worse.

## Debugging Memory Issues

If you encounter OOM errors:

1. **Check column projection** - Are you loading only needed columns?
   ```python
   loader.load_for_target(target, symbols)  # Uses projection
   ```

2. **Check float32** - Is `use_float32` enabled?
   ```yaml
   intelligent_training:
     lazy_loading:
       use_float32: true
   ```

3. **Check sampling** - Can you reduce data size?
   ```yaml
   pipeline:
     data_limits:
       max_cross_sectional_samples: 100
   ```

4. **Check for leaks** - Are DataFrames being released?
   ```python
   import gc
   gc.collect()
   # Check with: import psutil; psutil.Process().memory_info().rss / 1e9
   ```

## RAW_SEQUENCE Mode Data Loading

When `input_mode: RAW_SEQUENCE` is configured, the data loading path changes:

### Differences from FEATURES Mode

| Aspect | FEATURES Mode | RAW_SEQUENCE Mode |
|--------|---------------|-------------------|
| Columns loaded | Selected features + target | OHLCV + target only |
| Output shape | `(samples, features)` | `(samples, seq_len, channels)` |
| Preprocessing | Feature selection, leakage checks | Sequence windowing only |

### Sequence Data Preparation

```python
# RAW_SEQUENCE mode loads OHLCV columns only
columns = ["open", "high", "low", "close", "volume", target]

# Data is reshaped into sequences
# Input: DataFrame with OHLCV per row
# Output: (samples, sequence_length, 5) array

def prepare_sequence_data(df, sequence_length=60):
    """Convert tabular OHLCV to sequence format."""
    ohlcv = df[["open", "high", "low", "close", "volume"]].values

    # Create sliding windows
    n_samples = len(df) - sequence_length + 1
    X = np.zeros((n_samples, sequence_length, 5))
    for i in range(n_samples):
        X[i] = ohlcv[i:i+sequence_length]

    return X
```

### Memory Considerations

RAW_SEQUENCE mode is more memory-efficient for loading (fewer columns) but sequence arrays can be large:

```
# For 1M samples with seq_len=60:
# X shape: (1_000_000, 60, 5) = 300M floats = ~1.2GB (float32)
```

See `adding-new-models.md` for implementing sequence-capable trainers.
