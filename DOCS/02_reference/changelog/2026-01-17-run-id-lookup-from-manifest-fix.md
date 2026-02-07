# Run ID Lookup from Manifest Fix

**Date**: 2026-01-17  
**Category**: Reproducibility, Bug Fix  
**Impact**: High (fixes snapshot filtering failures due to run_id format mismatches and mutations)

## Summary

Fixed `run_id` lookup in `intelligent_trainer.py` to read from `manifest.json` (authoritative source) instead of deriving from directory name with format mutations. Eliminated all format corruption (underscore→dash conversion), removed ID fabrication fallback, and added strict mode policy for fail-closed behavior when `run_id` is unknowable.

## Problem

Run hash computation was failing because `run_id` lookup had multiple correctness issues:

1. **Format Mutation**: Code converted underscores to dashes (`run_id = trainer._run_name.replace("_", "-")`), causing mismatches with new `rid_*` format snapshots
2. **ID Fabrication**: Fallback generated new `run_id` via `derive_unstable_run_id(generate_run_instance_id())`, violating "never generate" rule
3. **Missing Guard**: Building `manifest_path = trainer.output_dir / "manifest.json"` could throw before fallbacks if `output_dir` was None/non-Path
4. **No Manifest Reading**: Code never read from `manifest.json` (authoritative SST source)
5. **Unclear Strict Mode Policy**: No explicit behavior for strict vs best-effort when `run_id` is unknowable

This caused warnings like:
```
⚠️ Run ID filter excluded all 170 snapshots (requested run_id='intelligent-output-20260117-182608-f5bd22f0')
```

## Solution

Implemented guarded `run_id` lookup with SST pattern and explicit strict mode policy:

1. **SST Helper**: Added `read_run_id_from_manifest()` in `manifest.py` for centralized manifest reading
2. **Guarded Normalization**: Normalize `output_dir` from str/PathLike → Path with try/except before building manifest path
3. **Authoritative Source**: Manifest is read first (SST pattern)
4. **Correct Fallback Order**: manifest → output_dir.name → _run_name → RunIdentity → None/raise
5. **No Format Mutations**: Never convert underscores to dashes
6. **No ID Fabrication**: Removed `derive_unstable_run_id()` fallback
7. **Strict Mode Policy**: Raise error in strict mode, disable filtering + warning in best-effort mode
8. **Source Tracking**: Log which source provided `run_id` for debugging

## Changes

### Files Modified

1. **`TRAINING/orchestration/utils/manifest.py`**:
   - Added `read_run_id_from_manifest(manifest_path: Path) -> Optional[str]` helper function
     - Centralized manifest reading (SST pattern)
     - Debug logging for JSON parse errors, missing keys, wrong types
     - Uses existing module logger and json import
     - Returns `None` on any error (graceful degradation)

2. **`TRAINING/orchestration/intelligent_trainer.py`** (lines 4601-4674):
   - Replaced buggy `run_id` lookup with guarded implementation
   - Normalized `output_dir` from str/PathLike → Path with try/except guard
   - Read from `manifest.json` first (authoritative source)
   - Correct fallback order: manifest → output_dir.name → _run_name → RunIdentity → None/raise
   - Removed underscore-to-dash conversion (no format mutations)
   - Removed `derive_unstable_run_id(generate_run_instance_id())` fallback (no ID fabrication)
   - Added strict mode check: raise error in strict mode, disable filtering + warning in best-effort
   - Added source tracking: logs which source won for debugging

### Implementation Details

**SST Helper**:
```python
def read_run_id_from_manifest(manifest_path: Path) -> Optional[str]:
    """Read run_id from manifest.json (SST helper)."""
    if not manifest_path.exists():
        logger.debug(f"Manifest not found: {manifest_path}")
        return None
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            run_id = manifest.get('run_id')
            if run_id and isinstance(run_id, str) and run_id.strip():
                logger.debug(f"Read run_id from manifest: {run_id}")
                return run_id.strip()
    except json.JSONDecodeError as e:
        logger.debug(f"Manifest JSON parse error: {e} (path: {manifest_path})")
    except Exception as e:
        logger.debug(f"Error reading manifest: {e} (path: {manifest_path})")
    
    return None
```

**Guarded Lookup with Fallbacks**:
```python
# Guard: Normalize output_dir from str/PathLike → Path
raw_output_dir = getattr(trainer, 'output_dir', None)
try:
    output_dir = Path(raw_output_dir) if raw_output_dir else None
except (TypeError, ValueError):
    output_dir = None

# 1. Manifest (authoritative)
if output_dir:
    run_id = read_run_id_from_manifest(output_dir / "manifest.json")
    if run_id:
        run_id_source = "manifest"

# 2. output_dir.name
if not run_id and output_dir:
    run_id = output_dir.name.strip()
    run_id_source = "output_dir.name"

# 3. _run_name (as-is, no mutations)
if not run_id:
    run_id = trainer._run_name.strip()  # No underscore→dash conversion
    run_id_source = "_run_name"

# 4. RunIdentity (deterministic)
if not run_id:
    run_id = derive_run_id_from_identity(run_identity)
    run_id_source = "RunIdentity"

# 5. Final fallback: strict mode policy
if not run_id:
    if is_strict_mode():
        raise RuntimeError("Cannot determine run_id...")
    else:
        logger.warning("Computing run hash without run_id filter...")
        run_id = None  # compute_full_run_hash will use all snapshots
```

## Backward Compatibility

✅ **Function signature unchanged**: No changes to `compute_full_run_hash()` or `save_run_hash()`  
✅ **Behavior**: Backward compatible - adds manifest reading without breaking existing behavior  
✅ **Fallback compatibility**: Old runs without manifest still work via fallbacks  
✅ **Strict mode**: New behavior (fail closed) only affects strict mode runs

## Edge Cases Handled

- ✅ `output_dir` is None → skip manifest read, continue to fallbacks
- ✅ `output_dir` is str/PathLike → normalized to Path with try/except
- ✅ Manifest missing → fallback to directory name
- ✅ Manifest corrupt (JSON parse error) → fallback with debug log
- ✅ Manifest exists but `run_id` missing/empty → fallback with debug log
- ✅ All sources unavailable in strict mode → raise error (fail closed)
- ✅ All sources unavailable in best-effort → disable filtering + warning
- ✅ `compute_full_run_hash()` correctly handles `run_id=None` (uses all snapshots)

## Testing Requirements

### Test 1: Manifest Present
- **Setup**: Create output dir with `manifest.json` containing `run_id="rid_unstable_intelligent_output_20260117_195358_eed99002"`
- **Verify**: Code uses that exact value
- **Verify**: Snapshots are found, no "excluded all snapshots" warning

### Test 2: Manifest Missing, Directory Name is New Format
- **Setup**: `output_dir.name == "rid_unstable_intelligent_output_20260117_195358_eed99002"`, `_run_name == "intelligent_output_20260117_195358"`
- **Verify**: Code picks `output_dir.name` (not `_run_name`)
- **Verify**: Snapshots match

### Test 3: Manifest Corrupt / Missing run_id
- **Setup**: `manifest.json` exists but `run_id` field is missing or empty
- **Verify**: Falls back cleanly (no crash)
- **Verify**: No underscore-to-dash mutation

### Test 4: No Reliable Sources
- **Setup**: No manifest, no output_dir.name, no _run_name, no RunIdentity
- **Verify**: Does NOT fabricate a run_id
- **Verify**: Sets `run_id = None` and logs warning (best-effort) or raises error (strict)

### Test 5: Multi-run Snapshots Directory
- **Setup**: `globals/` contains snapshots from 2 runs with different `run_id` values
- **Setup**: No manifest, no `_run_name`, no identity
- **Verify (best-effort)**: Warning logged, hash computed from all snapshots, hash is deterministic
- **Verify (strict)**: Error raised, hash not computed

## Impact

- ✅ **Fixes snapshot filtering**: Snapshots are now found when manifest contains correct `run_id`
- ✅ **Eliminates format corruption**: No more underscore→dash mutations
- ✅ **Prevents ID fabrication**: Never generates new `run_id` when sources unavailable
- ✅ **SST compliance**: Manifest is authoritative source, centralized reading
- ✅ **Strict mode safety**: Fail-closed behavior in strict mode prevents silent aggregation
- ✅ **Better debugging**: Source tracking logs which source provided `run_id`

## Related Changes

- Related to: [2026-01-17-run-hash-fallback-fix.md](2026-01-17-run-hash-fallback-fix.md) (fixed `compute_full_run_hash()` fallback logic)
- Related to: [2026-01-17-run-id-normalization-and-organization.md](2026-01-17-run-id-normalization-and-organization.md) (new `rid_*` format)
