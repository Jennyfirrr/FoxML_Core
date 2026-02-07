# Streaming Concat Optimization for Large Universes (728+ Symbols)

## Status: IMPLEMENTED ✅ (2026-01-19)

### Progress
- [x] **Phase 1**: Add `streaming_concat()` helper to `unified_loader.py` (DRY - single implementation)
- [x] **Phase 2**: float32 casting handled in `streaming_concat()` (no separate loader change needed)
- [x] **Phase 3**: Update `cross_sectional_data.py` to use `streaming_concat()` helper
- [x] **Phase 4**: Update `data_preparation.py` and `strategy_functions.py` to use `streaming_concat()` helper
- [x] **Phase 5**: Configuration options added to `pipeline.yaml`
- [x] **Phase 6**: Tests added and verified (11 tests pass)

## Problem Statement

### Current State (What's Working)
| Component | Location | Status |
|-----------|----------|--------|
| `UnifiedDataLoader` | `unified_loader.py` | ✅ Column projection, per-symbol loading |
| `load_for_target()` | `unified_loader.py` | ✅ Target + features + metadata |
| `release_data()` | `unified_loader.py` | ✅ Memory cleanup with GC |
| `probe_features_for_target()` | `feature_probe.py` | ✅ Single-symbol importance |
| `preflight_filter_features()` | `preflight_leakage.py` | ✅ Schema-only filtering |
| CS sampling | `cross_sectional_data.py:817-857` | ✅ Per-timestamp sampling |
| Min CS enforcement | `cross_sectional_data.py:805-812` | ✅ Filter small timestamps |

### The Bottleneck
All the above works, but **the concat step** at `cross_sectional_data.py:686` and `data_preparation.py:576` still materializes all 728 DataFrames at once:

```python
# Current (line 686) - ALL 728 DataFrames in memory simultaneously
combined_df = pd.concat(all_data, ignore_index=True, copy=False)
```

### Goal
Add a **single reusable helper** for streaming concat that:
1. Works with existing `UnifiedDataLoader` and `release_data()`
2. Preserves all existing sampling and filtering logic
3. Can be called from any stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)

## DRY Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EXISTING (unchanged)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  UnifiedDataLoader                                                      │
│  ├── load_data(symbols, columns, max_rows)      ← Column projection    │
│  ├── load_for_target(symbols, target, features) ← Convenience method   │
│  ├── read_schema(symbols)                       ← Preflight support    │
│  └── _load_projection_polars()                  ← Add use_float32 here │
│                                                                         │
│  release_data(mtf_data, verify, log_memory)     ← Memory cleanup       │
│                                                                         │
│  probe_features_for_target(...)                 ← Single-symbol probe  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     NEW: Single Helper (DRY)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  streaming_concat(                                                      │
│      mtf_data: Dict[str, DataFrame],                                   │
│      symbol_column: str = "symbol",                                    │
│      use_float32: bool = True,                                         │
│  ) -> pl.LazyFrame                                                     │
│                                                                         │
│  Purpose:                                                               │
│  - Convert mtf_data dict to Polars lazy frames                         │
│  - Release each DataFrame after conversion (memory efficient)          │
│  - Return lazy concat (caller decides when to collect)                 │
│  - Optional float32 casting for 50% memory reduction                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            TARGET_RANKING   FEATURE_SELECT    TRAINING
            (ranking.py)     (harness.py)      (training.py)
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                                    ▼
                    Existing sampling/filtering logic
                    (min_cs, max_cs_samples, etc.)
```

## Implementation Plan

### Phase 1: Add `streaming_concat()` Helper

**File**: `TRAINING/data/loading/unified_loader.py`

Add ONE helper function that all stages can use:

```python
def streaming_concat(
    mtf_data: Dict[str, pd.DataFrame],
    symbol_column: str = "symbol",
    target_column: Optional[str] = None,
    use_float32: bool = True,
    release_after_convert: bool = True,
) -> pl.LazyFrame:
    """
    Convert mtf_data dict to a memory-efficient Polars LazyFrame.

    This is the DRY helper for all stages that need to combine symbol data.
    It converts each DataFrame to a lazy frame immediately, releasing memory
    as it goes, then returns a lazy concat that can be collected with streaming.

    Args:
        mtf_data: Dictionary mapping symbol -> DataFrame (from load_data or load_for_target)
        symbol_column: Name for the symbol column to add (default: "symbol")
        target_column: If provided, skip symbols missing this column
        use_float32: Cast float64 columns to float32 for 50% memory reduction
        release_after_convert: Release each DataFrame after converting to lazy frame

    Returns:
        Polars LazyFrame - call .collect(streaming=True) to materialize

    Example:
        ```python
        # Load data using existing methods
        mtf_data = loader.load_for_target(symbols, target, features)

        # Convert to streaming lazy frame
        lf = streaming_concat(mtf_data, target_column=target)

        # Apply filters lazily (no memory allocation yet)
        lf = lf.filter(pl.len().over("ts") >= min_cs)

        # Collect with streaming mode (memory efficient)
        combined_df = lf.collect(streaming=True).to_pandas()
        ```

    SST Compliance:
        - Deterministic: processes symbols in sorted order
        - Reuses existing UnifiedDataLoader output
        - Works with existing release_data() for cleanup
    """
    import polars as pl

    lfs = []

    # DETERMINISTIC: Sort symbols for consistent processing order
    for symbol in sorted(mtf_data.keys()):
        df = mtf_data[symbol]

        if df is None:
            continue

        # Skip if target column required but missing
        if target_column and target_column not in df.columns:
            logger.debug(f"Skipping {symbol}: target '{target_column}' not found")
            continue

        # Convert to Polars lazy frame
        lf = pl.from_pandas(df).lazy()

        # Add symbol column
        lf = lf.with_columns(pl.lit(symbol).alias(symbol_column))

        # Cast to float32 for memory efficiency (50% reduction)
        if use_float32:
            float_cols = [
                c for c in df.select_dtypes(include=['float64']).columns
                if c not in METADATA_COLUMNS
            ]
            if float_cols:
                lf = lf.with_columns([
                    pl.col(c).cast(pl.Float32, strict=False) for c in float_cols
                ])

        lfs.append(lf)

        # Release DataFrame immediately to free memory
        if release_after_convert:
            mtf_data[symbol] = None
            del df

    if release_after_convert:
        gc.collect()

    if not lfs:
        logger.warning("No data to concat")
        return pl.LazyFrame()

    # Return lazy concat - no memory allocation until collect()
    return pl.concat(lfs, how="vertical_relaxed")
```

**Also add to `__init__.py` exports:**
```python
from .unified_loader import (
    UnifiedDataLoader,
    release_data,
    get_memory_mb,
    MemoryTracker,
    MemoryLeakError,
    streaming_concat,  # NEW
)
```

---

### Phase 2: Add `use_float32` to Existing Loader

**File**: `TRAINING/data/loading/unified_loader.py`

Extend `_load_projection_polars()` to optionally cast to float32 at load time:

```python
def _load_projection_polars(
    self,
    parquet_path: Path,
    columns: List[str],
    max_rows: Optional[int],
    use_float32: bool = None,  # NEW parameter
) -> Optional[pd.DataFrame]:
    """Load with column projection using Polars (faster)."""
    import polars as pl

    # Get config default if not specified
    if use_float32 is None:
        use_float32 = get_cfg(
            "intelligent_training.lazy_loading.use_float32",
            default=True
        )

    try:
        lf = pl.scan_parquet(str(parquet_path))
        available_cols = set(lf.collect_schema().names())
        select_cols = sorted([c for c in columns if c in available_cols])

        if not select_cols:
            return None

        lf = lf.select(select_cols)

        # NEW: Cast float64 to float32 for memory efficiency
        if use_float32:
            float_cols = [c for c in select_cols if c not in METADATA_COLUMNS]
            lf = lf.with_columns([
                pl.col(c).cast(pl.Float32, strict=False) for c in float_cols
            ])

        # Time column handling (existing)
        time_col = self._resolve_time_col(select_cols)
        if time_col:
            lf = lf.with_columns(
                pl.col(time_col).cast(pl.Datetime, strict=False).alias(time_col)
            ).drop_nulls([time_col])

        if max_rows:
            lf = lf.tail(max_rows)

        # Collect with streaming for large files
        df_pl = lf.collect(streaming=True)
        return df_pl.to_pandas(use_pyarrow_extension_array=False)

    except Exception as e:
        logger.error(f"Polars load failed for {parquet_path}: {e}")
        return None
```

---

### Phase 3: Update `cross_sectional_data.py`

**File**: `TRAINING/ranking/utils/cross_sectional_data.py`

Replace the concat section (lines 656-691) with call to helper:

```python
# BEFORE (lines 662-691):
all_data = []
symbols_to_process = sorted(mtf_data.keys())
for symbol in symbols_to_process:
    df = mtf_data[symbol]
    if target_column not in df.columns:
        continue
    df_with_symbol = df.assign(symbol=symbol)
    all_data.append(df_with_symbol)
    mtf_data[symbol] = None
gc.collect()
combined_df = pd.concat(all_data, ignore_index=True, copy=False)
del all_data
gc.collect()

# AFTER (DRY - uses helper):
from TRAINING.data.loading import streaming_concat

# Convert to streaming lazy frame (memory efficient)
combined_lf = streaming_concat(
    mtf_data,
    symbol_column="symbol",
    target_column=target_column,
    use_float32=True,
)

# Normalize time column for downstream filtering
time_col = "timestamp" if "timestamp" in combined_lf.columns else (
    "ts" if "ts" in combined_lf.columns else None
)

# Collect with streaming mode
combined_df = combined_lf.collect(streaming=True).to_pandas()
logger.info(f"Combined data shape: {combined_df.shape}")
```

**Note**: All existing sampling logic (min_cs enforcement at line 805, max_cs_samples at line 817) stays **unchanged** - it operates on `combined_df` after the concat.

---

### Phase 4: Update `data_preparation.py`

**File**: `TRAINING/training_strategies/execution/data_preparation.py`

Same pattern - replace concat with helper call:

**Polars path (around line 340-365)**:
```python
# BEFORE:
all_data_pl = []
for symbol in sorted(mtf_data.keys()):
    ...
    all_data_pl.append(df_pl)
combined_pl = pl.concat(all_data_pl, rechunk=False)

# AFTER:
from TRAINING.data.loading import streaming_concat

combined_lf = streaming_concat(
    mtf_data,
    symbol_column="symbol",
    target_column=target,
    use_float32=True,
)
combined_pl = combined_lf.collect(streaming=True)
```

**Pandas fallback (around line 561-581)**:
```python
# BEFORE:
all_data = []
for symbol in symbols_to_process:
    ...
combined_df = pd.concat(all_data, ignore_index=True, copy=False)

# AFTER:
from TRAINING.data.loading import streaming_concat

combined_lf = streaming_concat(
    mtf_data,
    symbol_column="symbol",
    target_column=target,
    use_float32=True,
)
combined_df = combined_lf.collect(streaming=True).to_pandas()
```

---

### Phase 5: Configuration

**File**: `CONFIG/pipeline/pipeline.yaml`

Add to existing `lazy_loading` section:

```yaml
intelligent_training:
  lazy_loading:
    enabled: true
    verify_memory_release: false
    log_memory_usage: true
    fail_on_fallback: true

    # Feature probing (existing)
    probe_features: true
    probe_top_n: 100
    probe_rows: 10000

    # NEW: Memory optimization settings
    use_float32: true       # Cast float64→float32 at load time (50% reduction)
    streaming_collect: true # Use Polars streaming mode for large datasets
```

---

### Phase 6: Testing

**File**: `tests/test_streaming_concat.py`

```python
"""Tests for streaming_concat helper."""
import pytest
import pandas as pd
import numpy as np
from TRAINING.data.loading import streaming_concat

def test_streaming_concat_basic():
    """Test basic concat functionality."""
    mtf_data = {
        "AAPL": pd.DataFrame({"close": [1.0, 2.0], "ts": [1, 2]}),
        "GOOGL": pd.DataFrame({"close": [3.0, 4.0], "ts": [1, 2]}),
    }
    lf = streaming_concat(mtf_data)
    df = lf.collect().to_pandas()

    assert len(df) == 4
    assert "symbol" in df.columns
    assert set(df["symbol"]) == {"AAPL", "GOOGL"}

def test_streaming_concat_deterministic():
    """Test output is deterministic regardless of dict order."""
    mtf_data_1 = {"AAPL": pd.DataFrame({"x": [1.0]}), "GOOGL": pd.DataFrame({"x": [2.0]})}
    mtf_data_2 = {"GOOGL": pd.DataFrame({"x": [2.0]}), "AAPL": pd.DataFrame({"x": [1.0]})}

    df1 = streaming_concat(mtf_data_1).collect().to_pandas()
    df2 = streaming_concat(mtf_data_2).collect().to_pandas()

    pd.testing.assert_frame_equal(df1, df2)

def test_streaming_concat_float32():
    """Test float32 casting reduces memory."""
    mtf_data = {
        "AAPL": pd.DataFrame({"close": np.array([1.0, 2.0], dtype=np.float64)}),
    }

    lf = streaming_concat(mtf_data, use_float32=True)
    df = lf.collect().to_pandas()

    assert df["close"].dtype == np.float32

def test_streaming_concat_skips_missing_target():
    """Test symbols missing target are skipped."""
    mtf_data = {
        "AAPL": pd.DataFrame({"close": [1.0], "target": [0.1]}),
        "GOOGL": pd.DataFrame({"close": [2.0]}),  # No target column
    }

    lf = streaming_concat(mtf_data, target_column="target")
    df = lf.collect().to_pandas()

    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"

def test_streaming_concat_releases_memory():
    """Test memory is released after conversion."""
    mtf_data = {
        "AAPL": pd.DataFrame({"close": [1.0]}),
    }

    streaming_concat(mtf_data, release_after_convert=True)

    # Original dict should have None values
    assert mtf_data["AAPL"] is None
```

---

## Files Modified

| File | Change |
|------|--------|
| `TRAINING/data/loading/unified_loader.py` | Add `streaming_concat()`, add `use_float32` to loader |
| `TRAINING/data/loading/__init__.py` | Export `streaming_concat` |
| `TRAINING/ranking/utils/cross_sectional_data.py` | Use `streaming_concat()` instead of pd.concat |
| `TRAINING/training_strategies/execution/data_preparation.py` | Use `streaming_concat()` instead of pd.concat |
| `CONFIG/pipeline/pipeline.yaml` | Add `use_float32`, `streaming_collect` options |

## New Files

| File | Purpose |
|------|---------|
| `tests/test_streaming_concat.py` | Unit tests for helper |

---

## What Stays Unchanged (DRY)

| Component | Status |
|-----------|--------|
| `UnifiedDataLoader.load_data()` | ✅ Unchanged - still returns `Dict[str, DataFrame]` |
| `UnifiedDataLoader.load_for_target()` | ✅ Unchanged - still returns `Dict[str, DataFrame]` |
| `release_data()` | ✅ Unchanged - still works for cleanup |
| `probe_features_for_target()` | ✅ Unchanged - still does single-symbol probe |
| `preflight_filter_features()` | ✅ Unchanged - still does schema-only filtering |
| CS sampling logic (min_cs, max_cs_samples) | ✅ Unchanged - operates on combined_df after concat |
| All existing tests | ✅ Should pass - API unchanged |

---

## Memory Impact

| Scenario | Before | After |
|----------|--------|-------|
| 728 symbols × 75k rows × 100 cols (float64) | 132+ GB peak | - |
| + Polars streaming | - | ~92 GB peak |
| + float32 casting | - | ~46 GB peak |
| **Final** | OOM | **~46 GB** ✅ |

---

## Determinism Checklist

- [x] `sorted(mtf_data.keys())` in helper ensures consistent symbol order
- [x] float32 casting uses `strict=False` for consistent handling
- [x] Lazy concat preserves input order
- [x] All existing sampling uses deterministic seeds

---

## Implementation Order

1. **Phase 1**: Add `streaming_concat()` to `unified_loader.py` (30 min)
2. **Phase 2**: Add `use_float32` to loader (15 min)
3. **Phase 3**: Update `cross_sectional_data.py` (15 min)
4. **Phase 4**: Update `data_preparation.py` (15 min)
5. **Phase 5**: Add config options (10 min)
6. **Phase 6**: Add tests, run 728 symbol verification (1 hr)

**Total: ~2.5 hours**
