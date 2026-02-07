# Run Hash Computation Fallback Fix

**Date**: 2026-01-17  
**Category**: Reproducibility, Bug Fix  
**Impact**: High (fixes run hash computation failures due to run_id format mismatches)

## Summary

Fixed `compute_full_run_hash()` to handle run_id format mismatches gracefully by implementing a one-pass counter-based approach with safe fallback logic. The function now aggregates run identity correctly across all stages even when different stages use different run_id formats.

## Problem

Run hash computation was failing when `run_id` format mismatches occurred:

1. **Format Mismatch**: Different stages use different `run_id` formats:
   - TARGET_RANKING: `"2026-01-17T01:45:14.745352"` (timestamp)
   - FEATURE_SELECTION: `"438f828a-5322-48c6-b03a-be4393557eab"` (UUID)
   - Code searches for: `"intelligent-output-20260117-014407"` (directory name)

2. **Filtering Excluded All Snapshots**: The filter `if run_id and snapshot.get('run_id') != run_id: continue` excluded all snapshots when formats didn't match, resulting in empty `run_state` → returns `None` → warning logged

3. **No Fallback**: No mechanism existed to handle format mismatches, causing legitimate runs to fail hash computation

## Solution

Implemented a one-pass counter-based approach with safe fallback:

1. **One-Pass Collection**: Collects both `all_entries` (all valid snapshots) and `filtered_entries` (matching run_id) in a single pass
2. **Precise Counters**: Tracks `total_snapshots`, `with_fp`, `match_runid`, `match_runid_with_fp` for accurate decision logic
3. **Safe Fallback**: Falls back to all snapshots when `run_id` format mismatches but valid snapshots exist
4. **No Fallback for Invalid**: Returns `None` when matched `run_id` has no fingerprints (correct behavior - fallback won't help)
5. **Improved Sorting**: Sorts by deterministic fields first, then uses `(index_name, key)` as tiebreakers for hash stability

## Changes

### Files Modified

1. **`TRAINING/orchestration/utils/diff_telemetry.py`**:
   - Added `_extract_deterministic_fields()` helper function (before line 5890)
     - Extracts deterministic fields from snapshot
     - Returns `None` if fingerprint missing
     - Normalizes `method` → `model_family` for FEATURE_SELECTION snapshots
   - Replaced `compute_full_run_hash()` loop body (lines 5940-6006) with one-pass collection
   - Replaced decision logic (lines 6008-6014) with counter-based fallback
   - Improved sorting (lines 6016-6023) to sort by deterministic fields first, then tiebreakers

2. **`tests/test_diff_telemetry.py`** (new file):
   - Added unit tests for run_id mismatch fallback
   - Added test for matched run_id but no fingerprints (no fallback)
   - Added determinism and ordering tests

### Implementation Details

**Helper Function**:
```python
def _extract_deterministic_fields(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract deterministic fields from snapshot, returning None if missing fingerprint."""
    # Normalizes method → model_family for FEATURE_SELECTION snapshots
    model_family = snapshot.get('model_family') or snapshot.get('method')
    # ... extracts all deterministic fields
```

**Counter-Based Decision Logic**:
- If `with_fp == 0`: return `None` (no valid snapshots)
- If `run_id` provided:
  - If `match_runid > 0` and `match_runid_with_fp > 0`: use `filtered_entries`
  - If `match_runid > 0` but `match_runid_with_fp == 0`: return `None` (matched but no fingerprints - **no fallback**)
  - If `match_runid == 0` but `with_fp > 0`: fallback to `all_entries` (format mismatch)
- If no `run_id`: use `all_entries`

**Improved Sorting**:
- Primary sort: deterministic fields (`stage`, `target`, `view`, `symbol`, `model_family`, `config_fingerprint`)
- Tiebreaker: `(index_name, key)` for stability under JSON key order changes

## Backward Compatibility

✅ **Function signature unchanged**: `compute_full_run_hash(output_dir: Path, run_id: Optional[str] = None) -> Optional[str]`  
✅ **Return type unchanged**: Still returns `Optional[str]` (16-char hex digest or None)  
✅ **Hash computation unchanged**: Uses same `json.dumps(run_state, sort_keys=True)` + `sha256`  
✅ **Call sites unchanged**: No changes needed to existing callers  
✅ **Behavior**: Backward compatible - adds fallback without breaking existing behavior

## Edge Cases Handled

- ✅ Empty `snapshot_indices` → returns `None` (existing behavior)
- ✅ All snapshots missing fingerprints → returns `None` with specific warning
- ✅ `run_id` matches but fingerprints missing → returns `None` (**no fallback** - correct behavior)
- ✅ `run_id` mismatch but fingerprints exist → fallback to all snapshots
- ✅ No `run_id` provided → uses all snapshots (existing behavior)
- ✅ Multiple indices → sorted by index_name first
- ✅ JSON key order changes → hash stable (sort by content first)
- ✅ FEATURE_SELECTION `method` field → normalized to `model_family`

## Testing

Added comprehensive unit tests in `tests/test_diff_telemetry.py`:
- `test_compute_full_run_hash_run_id_mismatch`: Verifies fallback works
- `test_compute_full_run_hash_matched_runid_no_fingerprints`: Verifies no fallback when matched but invalid
- `test_compute_full_run_hash_deterministic`: Verifies same snapshots → same hash
- `test_compute_full_run_hash_ordering`: Verifies index/key order doesn't affect hash
- Additional edge case tests

## Logging Improvements

Added detailed logging at all decision points:
- Discovery summary (total snapshots, fingerprints count, match count)
- Decision point (which set chosen - filtered vs all)
- Fallback trigger (with sample run_ids when mismatch detected)
- Final hash computation (snapshot count used)
- Specific warnings for different error cases

## Future Work

**Run ID Normalization** (separate issue):
- Different stages use different run_id formats (timestamp, UUID, directory name)
- Recommendation: Normalize run_id computation across all stages to use consistent format
- Files to check: `model_evaluation/reporting.py`, `hooks.py`, `training/reproducibility/io.py`
- This fix works regardless of format, but normalization would prevent mismatches

## Verification

Before implementing, verified:
- ✅ Types imported: `Dict`, `Any`, `Optional` already imported
- ✅ Logger available: `logger` is module-level
- ✅ Function signature unchanged: No breaking changes to callers
- ✅ Return value unchanged: Still returns `Optional[str]` (16-char hex digest or None)
- ✅ Hash computation unchanged: Uses same `json.dumps(run_state, sort_keys=True)` + `sha256`
- ✅ No new dependencies: Only uses existing imports
- ✅ Syntax checks pass
- ✅ Imports successfully
