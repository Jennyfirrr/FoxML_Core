# 2026-01-19: Threading Safety and Determinism Fixes

## Summary

Fixed a critical deadlock in the feature registry that caused pipeline freezes during the TARGET_RANKING stage, plus determinism improvements to the contracts module.

## Changes

### Critical Bug Fix: Registry Lock Deadlock (TS-001)

**File:** `TRAINING/common/feature_registry.py`

**Problem:** The pipeline would freeze/hang at the TARGET_RANKING stage with 0% CPU usage. The last log message was "Stage transition: TARGET_RANKING" and then nothing.

**Root Cause:** Recursive lock acquisition deadlock:
1. `get_registry()` acquires `_REGISTRY_LOCK` (line ~2290)
2. Calls `FeatureRegistry()` constructor
3. Constructor calls `_load_config()`
4. `_load_config()` tries to acquire `_REGISTRY_LOCK` again (line ~1026)
5. `threading.Lock()` is NOT re-entrant → **DEADLOCK**

**Fix:** Changed `_REGISTRY_LOCK` from `threading.Lock()` to `threading.RLock()` (re-entrant lock), which allows the same thread to acquire it multiple times.

```python
# Before (deadlock)
_REGISTRY_LOCK = threading.Lock()

# After (safe)
_REGISTRY_LOCK = threading.RLock()
```

### Determinism Fix: `get_all_features()` Return Type (SB-010)

**File:** `TRAINING/orchestration/contracts/feature_selection.py`

**Problem:** `FeatureSelectionResult.get_all_features()` returned a `Set[str]`, which has non-deterministic iteration order. Any caller using this in artifact-affecting code would get non-deterministic results.

**Fix:** Changed return type to `List[str]` and return sorted results:

```python
# Before (non-deterministic)
def get_all_features(self) -> Set[str]:
    features = set(...)
    return features

# After (deterministic)
def get_all_features(self) -> List[str]:
    features = set(...)
    return sorted(features)  # DETERMINISM: Return sorted list
```

### Defensive: Multiprocessing Spawn Mode

**File:** `TRAINING/common/parallel_exec.py`

**Change:** Added explicit `multiprocessing.set_start_method('spawn')` to prevent potential fork-related deadlocks in parallel execution.

**Note:** This was added defensively during debugging but is good practice regardless. The 'spawn' method:
- Starts fresh Python interpreters (no inherited lock state)
- Avoids fork deadlocks with threading locks
- Is more deterministic (no inherited RNG state)

```python
try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set
```

### Documentation: Circular Import Avoidance

**File:** `.claude/skills/sst-and-coding-standards.md`

Added comprehensive guidance on avoiding circular imports:
- `TYPE_CHECKING` guard for type hints
- Late/local imports for runtime needs
- Extracting shared types to separate modules
- Using protocols instead of concrete types
- Dependency direction hierarchy
- Bootstrap import order (critical for determinism)

## Testing

After these changes:
- Pipeline no longer freezes at TARGET_RANKING stage
- Registry loads successfully
- Parallel target evaluation proceeds normally

## Related Issues

- TS-001: Thread-safe registry singleton
- SB-010: Deterministic feature selection contract
- Import order and circular import prevention

## Migration Notes

No action required. These are internal fixes with no API changes.
