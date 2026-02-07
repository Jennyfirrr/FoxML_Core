# Polars-Native Memory Optimization Plan

## Status: FAILED - Cleaned Up (2026-01-20)

**Original Rollback Point**: `.claude/backups/polars-memory-rollback/` (no longer needed)

**Cleanup Complete**: Dead code for Polars→numpy direct conversion has been removed.
The working Polars→Pandas→numpy path is documented in `.claude/skills/data-loading-pipeline.md`.

## Problem Statement

The data pipeline has a memory spike during Polars → Pandas conversion. With 70M rows × 114 columns:
- Polars DataFrame: ~32GB
- After `to_pandas()`: ~64GB peak (both in memory simultaneously)
- Goal: Reduce peak to ~32-40GB by extracting directly to numpy

---

## Attempt History

### Attempt 1: Original `polars_to_sklearn_dense()` (FAILED)

Created 4 intermediate DataFrames via sequential `select()` calls:
```python
selected_df = pl_df.select(cast_exprs)           # Copy 1
selected_df = selected_df.select(replace_inf)    # Copy 2
filled_df = selected_df.select(fill_exprs)       # Copy 3
X = filled_df.to_numpy()                         # Copy 4
```
**Result**: ~160GB peak (WORSE than Pandas)

### Attempt 2: `extract_all_arrays()` with Chained Expressions (FAILED)

Used chained Polars expressions to reduce intermediate DataFrames:
```python
# Chained: cast → replace_inf → fill_null in one expression
expr = pl.col(col).cast(pl.Float32)
expr = pl.when(expr.is_infinite()).then(None).otherwise(expr)
expr = expr.fill_null(median_value)
result_df = pl_df.select(exprs)  # One select
```

**Problem discovered in code review**: Still created multiple intermediates:
1. Median computation: `pl_df.select([...median()...])` - full copy
2. Main select: `result_df = pl_df.select(all_exprs)` - full copy
3. Null check: `result_df.select([...is_null()...])` - small
4. Optional reselect if all-null columns

**Result**: ~64GB+ peak, caps out RAM in production. No better than Pandas path.

---

## Why Polars-Native Doesn't Help Here

### The Fundamental Problem

To go from Polars → numpy, you MUST:
1. Have the Polars DataFrame in memory (~32GB)
2. Create the numpy array (~32GB)

This is **unavoidable** - the data must exist in both formats simultaneously during conversion.

### Why Pandas Path Actually Works

```python
combined_df = data_pl.to_pandas()  # Peak: 64GB (Polars + Pandas)
del data_pl                         # Immediate release
gc.collect()                        # Force cleanup
# After: 32GB (Pandas only)
```

The Pandas path has a **brief** 64GB spike, then drops to 32GB. This is acceptable.

### Why Polars-Native Path Fails

Any operation on the Polars DataFrame (select, filter, etc.) before `to_numpy()` creates intermediates that Python's GC doesn't immediately release. Even with `del` and `gc.collect()`, the memory pressure during the operation exceeds the Pandas path.

---

## Current State (Working)

The Pandas path is enabled and working:
```yaml
# CONFIG/pipeline/memory.yaml
memory:
  polars_conversion:
    polars_direct_conversion: false  # Use Pandas path
    fallback_to_pandas: true
```

Peak memory: ~64GB during conversion, acceptable for 70M row datasets.

---

## If Memory Reduction Is Still Needed

### Option A: Chunked Processing
Process data in chunks instead of all at once:
```python
chunks = []
for chunk_df in pl_df.iter_slices(n_rows=1_000_000):
    chunk_arr = chunk_df.to_numpy()
    chunks.append(chunk_arr)
    del chunk_df
X = np.vstack(chunks)
```
**Risk**: May affect cross-sectional operations that need full timestamp coverage.

### Option B: Memory-Mapped Arrays
Write to disk and memory-map:
```python
# Write Polars directly to numpy memmap
mmap = np.memmap('temp.npy', dtype='float32', mode='w+', shape=(n_rows, n_cols))
for i, chunk in enumerate(pl_df.iter_slices()):
    mmap[start:end] = chunk.to_numpy()
```
**Risk**: Slower due to disk I/O.

### Option C: Reduce Data Size Upstream
- Use float16 instead of float32 (50% reduction)
- Sample data before training
- Reduce number of features

### Option D: Accept 64GB Peak
The current Pandas path works. If the machine has 64GB+ RAM, this is fine.

---

## Lessons Learned

1. **Polars operations create copies** - Even "in-place" operations like `select()` create new DataFrames
2. **Python GC is lazy** - Reassigning a variable doesn't immediately free the old value
3. **Chained expressions help but don't eliminate copies** - The result still needs to materialize
4. **Pandas `to_pandas()` is actually efficient** - It's a single copy operation, not multiple
5. **Memory optimization requires careful profiling** - Assumptions about memory behavior are often wrong

---

## Recommendation

**Keep the Pandas path**. It works, it's proven, and the 64GB peak is acceptable for production workloads. Further optimization attempts have consistently made things worse.

If memory becomes a hard constraint, pursue Option A (chunked processing) with careful testing.
