# Phase 2: Thread Safety

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ Complete
**Priority**: P0 (Critical - Race Conditions)
**Estimated Effort**: 1-2 days
**Depends On**: Phase 1 (Run Identity)

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 10/10 items ✅
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 2 complete - proceed to Phase 3 (Fingerprinting)
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| TS-001 | _REGISTRY singleton without lock | ✅ Complete | Added `_REGISTRY_LOCK` with double-check locking |
| TS-002 | _LOGGED_REGISTRY_PATHS set without lock | ✅ Complete | Protected with `_REGISTRY_LOCK` |
| TS-003 | _AUTO_ENABLED_FEATURES_GLOBAL dict shared | ✅ Complete | Added `_AUTO_ENABLED_FEATURES_LOCK` |
| TS-004 | ModelRegistry broken singleton | ✅ Complete | Added class-level `_lock` with double-check |
| TS-005 | ModelFactory broken singleton | ✅ Complete | Added class-level `_lock` with double-check |
| TS-006 | blas_threads() env var race | ✅ Complete | Added `_ENV_LOCK` and runtime warning |
| TS-007 | temp_environ() restoration not atomic | ✅ Complete | Added `_ENV_LOCK` protection |
| TS-008 | set_global_determinism() env race | ✅ Complete | Added `_DETERMINISM_LOCK` and thread check |
| TS-009 | @lru_cache staleness | ✅ Complete | Replaced with invalidatable cache pattern |
| TS-010 | Mutable default argument | ✅ Complete | Fixed `hidden_dims` default in seq_adapters.py |

---

## Problem Statement

Critical thread safety issues affecting parallel training:
1. **Broken singleton patterns** - No locks, multiple instances possible
2. **Global mutable state** - Dicts modified without synchronization
3. **Environment variable races** - Context managers not reentrant
4. **Cache staleness** - `@lru_cache` doesn't invalidate on file changes

---

## Issue Details

### TS-001: _REGISTRY singleton without lock (P0)

**File**: `TRAINING/common/feature_registry.py`
**Lines**: 34, 2271-2277

```python
# CURRENT
_REGISTRY: Optional['FeatureRegistry'] = None

def get_registry(...):
    global _REGISTRY
    if _REGISTRY is None:          # Thread A checks
        _REGISTRY = FeatureRegistry(...)  # Thread B also creates
    return _REGISTRY
```

**Fix**:
```python
import threading

_REGISTRY: Optional['FeatureRegistry'] = None
_REGISTRY_LOCK = threading.Lock()

def get_registry(...):
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:  # Double-check locking
                _REGISTRY = FeatureRegistry(...)
    return _REGISTRY
```

---

### TS-002: _LOGGED_REGISTRY_PATHS set without lock (P1)

**File**: `TRAINING/common/feature_registry.py`
**Line**: 37

```python
# CURRENT
_LOGGED_REGISTRY_PATHS: set = set()

# Usage (unsafe)
_LOGGED_REGISTRY_PATHS.add(path)
```

**Fix**: Protect with same lock as registry:
```python
with _REGISTRY_LOCK:
    if path not in _LOGGED_REGISTRY_PATHS:
        _LOGGED_REGISTRY_PATHS.add(path)
        logger.info(f"Using registry: {path}")
```

---

### TS-003: _AUTO_ENABLED_FEATURES_GLOBAL dict shared (P0)

**File**: `TRAINING/common/feature_registry.py`
**Line**: 44

```python
# CURRENT
_AUTO_ENABLED_FEATURES_GLOBAL: Dict[str, Dict[str, Any]] = {}
```

**Fix**:
```python
import threading

_AUTO_ENABLED_FEATURES_LOCK = threading.Lock()
_AUTO_ENABLED_FEATURES_GLOBAL: Dict[str, Dict[str, Any]] = {}

# Usage:
def update_auto_enabled_features(key: str, value: Dict[str, Any]) -> None:
    with _AUTO_ENABLED_FEATURES_LOCK:
        _AUTO_ENABLED_FEATURES_GLOBAL[key] = value

def get_auto_enabled_features(key: str) -> Optional[Dict[str, Any]]:
    with _AUTO_ENABLED_FEATURES_LOCK:
        return _AUTO_ENABLED_FEATURES_GLOBAL.get(key)
```

---

### TS-004: ModelRegistry broken singleton (P0)

**File**: `TRAINING/models/registry.py`
**Lines**: 19-25

```python
# CURRENT
class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Fix**:
```python
import threading

class ModelRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            # Actual initialization
            self._models = {}
            self._initialized = True
```

---

### TS-005: ModelFactory broken singleton (P0)

**File**: `TRAINING/models/factory.py`
**Lines**: 20-26

Same pattern as TS-004. Apply same fix.

---

### TS-006: blas_threads() env var race (P1)

**File**: `TRAINING/common/threads.py`
**Lines**: 132-143

```python
# CURRENT
@contextmanager
def blas_threads(n: int):
    old_omp = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = str(n)  # Thread A sets 12
        # Thread B sets 4, overwrites A
        yield
    finally:
        # Thread A restores, undoes B's settings
        if old_omp is not None:
            os.environ["OMP_NUM_THREADS"] = old_omp
```

**Fix**: Use process-level isolation or warn about thread safety:
```python
import threading
import warnings

_ENV_LOCK = threading.Lock()
_ENV_WARN_ISSUED = False

@contextmanager
def blas_threads(n: int):
    global _ENV_WARN_ISSUED

    # Warn if potentially unsafe
    if threading.active_count() > 1 and not _ENV_WARN_ISSUED:
        warnings.warn(
            "blas_threads() modifies os.environ which is not thread-safe. "
            "Use process-based parallelism for deterministic behavior.",
            RuntimeWarning
        )
        _ENV_WARN_ISSUED = True

    with _ENV_LOCK:
        old_omp = os.environ.get("OMP_NUM_THREADS")
        try:
            os.environ["OMP_NUM_THREADS"] = str(n)
            yield
        finally:
            if old_omp is not None:
                os.environ["OMP_NUM_THREADS"] = old_omp
            else:
                os.environ.pop("OMP_NUM_THREADS", None)
```

---

### TS-007: temp_environ() restoration not atomic (P1)

**File**: `TRAINING/common/threads.py`
**Lines**: 416-443

Same pattern as TS-006. Apply similar lock-based protection.

---

### TS-008: set_global_determinism() env race (P1)

**File**: `TRAINING/common/determinism.py`
**Lines**: 106-134

```python
# CURRENT
def set_global_determinism(seed: int, threads: int = 1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    # ... more env vars
```

**Fix**: This should only be called once at startup. Add guard:
```python
_DETERMINISM_INITIALIZED = False
_DETERMINISM_LOCK = threading.Lock()

def set_global_determinism(seed: int, threads: int = 1):
    global _DETERMINISM_INITIALIZED

    with _DETERMINISM_LOCK:
        if _DETERMINISM_INITIALIZED:
            raise RuntimeError(
                "set_global_determinism() already called. "
                "Cannot reinitialize determinism settings."
            )

        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        # ... rest of initialization

        _DETERMINISM_INITIALIZED = True
```

---

### TS-009: @lru_cache staleness (P2)

**File**: `TRAINING/common/determinism.py`
**Line**: 483

```python
# CURRENT
@lru_cache(maxsize=1)
def load_reproducibility_config() -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)
```

**Fix**: Use explicit cache with invalidation:
```python
_REPRODUCIBILITY_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_REPRODUCIBILITY_CONFIG_MTIME: Optional[float] = None

def load_reproducibility_config(force_reload: bool = False) -> Dict[str, Any]:
    global _REPRODUCIBILITY_CONFIG_CACHE, _REPRODUCIBILITY_CONFIG_MTIME

    config_path = Path("CONFIG/pipeline/training/reproducibility.yaml")
    current_mtime = config_path.stat().st_mtime if config_path.exists() else None

    if (force_reload or
        _REPRODUCIBILITY_CONFIG_CACHE is None or
        current_mtime != _REPRODUCIBILITY_CONFIG_MTIME):

        with open(config_path) as f:
            _REPRODUCIBILITY_CONFIG_CACHE = yaml.safe_load(f)
        _REPRODUCIBILITY_CONFIG_MTIME = current_mtime

    return _REPRODUCIBILITY_CONFIG_CACHE
```

---

### TS-010: Mutable default argument (P2)

**File**: `TRAINING/models/seq_adapters.py`
**Line**: 38

```python
# CURRENT
def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
```

**Fix**:
```python
def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None):
    if hidden_dims is None:
        hidden_dims = [128, 64]
```

---

## Implementation Steps

### Step 1: Add locks to feature_registry.py
- Add `_REGISTRY_LOCK`
- Protect `get_registry()` with double-check locking
- Protect `_LOGGED_REGISTRY_PATHS`
- Add `_AUTO_ENABLED_FEATURES_LOCK` with accessor functions

### Step 2: Fix ModelRegistry and ModelFactory singletons
- Add class-level locks
- Implement double-check locking pattern
- Protect `__init__` with `_initialized` flag

### Step 3: Add guards to environment modification
- Add `_ENV_LOCK` to threads.py
- Add `_DETERMINISM_INITIALIZED` guard
- Issue warnings when used in multi-threaded context

### Step 4: Replace @lru_cache with invalidatable cache
- Add mtime-based cache invalidation
- Add `force_reload` parameter

### Step 5: Fix mutable default arguments
- Search for `= []` and `= {}` in function signatures
- Replace with `= None` pattern

---

## Contract Tests

```python
# tests/contract_tests/test_thread_safety_contract.py

import threading
import time

class TestSingletonThreadSafety:
    def test_registry_singleton_thread_safe(self):
        """Multiple threads should get same registry instance."""
        instances = []

        def get_instance():
            from TRAINING.common.feature_registry import get_registry
            instances.append(id(get_registry()))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(instances)) == 1, "Multiple registry instances created"

    def test_model_registry_thread_safe(self):
        """ModelRegistry should be thread-safe singleton."""
        from TRAINING.models.registry import ModelRegistry

        instances = []
        def get_instance():
            instances.append(id(ModelRegistry()))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(instances)) == 1

class TestEnvironmentSafety:
    def test_determinism_only_once(self):
        """set_global_determinism should only be callable once."""
        from TRAINING.common.determinism import set_global_determinism

        # First call should succeed (if not already called)
        # Second call should raise
        # Note: Test may need to run in isolated subprocess
```

---

## Verification

```bash
# Check for singleton patterns
grep -rn "_instance = None" TRAINING/

# Check for global mutable state
grep -rn "^_[A-Z].*: Dict\|^_[A-Z].*: List\|^_[A-Z].*: set" TRAINING/common/

# Check for environment modifications
grep -rn "os.environ\[" TRAINING/common/

# Check for mutable defaults
grep -rn "def.*=\s*\[\]" TRAINING/
grep -rn "def.*=\s*{}" TRAINING/

# Run thread safety tests
pytest tests/contract_tests/test_thread_safety_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 10 issues with fixes
- Created implementation steps
- Created contract tests
- **Next**: Start implementing Step 1

### Session 2: 2026-01-19
- Implemented all 10 thread safety fixes:
  - **TS-001, TS-002, TS-003**: Added locks to `feature_registry.py` - `_REGISTRY_LOCK` and `_AUTO_ENABLED_FEATURES_LOCK`
  - **TS-004, TS-005**: Fixed `ModelRegistry` and `ModelFactory` singletons with double-check locking pattern
  - **TS-006, TS-007**: Added `_ENV_LOCK` to `threads.py` for `blas_threads()` and `temp_environ()`
  - **TS-008**: Added `_DETERMINISM_LOCK` and main-thread check to `set_global_determinism()`
  - **TS-009**: Replaced `@lru_cache` with invalidatable cache pattern + `invalidate_repro_config_cache()` function
  - **TS-010**: Fixed mutable default argument in `seq_adapters.py` CNN1DHead
- **Phase 2 Complete** ✅

---

## Notes

- Test with `pytest -n auto` (parallel execution) to verify thread safety
- Consider using `threading.RLock` instead of `Lock` if re-entry possible
- Environment variable changes are process-global; prefer process isolation for true parallelism
- Added `invalidate_repro_config_cache()` to `determinism.py` for test isolation
