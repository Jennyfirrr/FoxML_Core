# Phase 0: Regex Optimization

**Status**: 🔴 Not Started
**Priority**: P0 (Quick Win)
**Effort**: 2 hours
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)

---

## Quick Resume

```
CURRENT TASK: Add compiled pattern cache to FeatureRegistry
FILE: TRAINING/common/feature_registry.py
BLOCKERS: None
NEXT ACTION: Read file, locate __init__ method, add _compiled_family_patterns
```

---

## Problem Statement

The `FeatureRegistry` class uses regex patterns for feature classification but **re-compiles patterns on every call**. During ranking stage, `is_allowed()` is called O(n_features × n_targets × n_horizons) times.

**Example**: 5,000 features × 100 targets × 6 horizons = **3,000,000+ regex compilations**

Current overhead: ~300-500ms per run
With compiled patterns: ~50-100ms (**5-7x speedup**)

---

## Current Code Analysis

### Location: `TRAINING/common/feature_registry.py`

**Pattern 1: Family pattern matching (line ~1313)**
```python
# Current (re-compiles every call)
for family_name, family_config in self.families.items():
    pattern = family_config.get('pattern')
    if pattern and re.match(pattern, feature_name):
        # ...
```

**Pattern 2: Auto-infer metadata (line ~1436)**
```python
# Current (re-compiles every call)
match = re.match(pattern_str, feature_name, re.I)
```

**Pattern 3: Fallback patterns (lines ~1511-1593)**
```python
# Current (hardcoded, re-compiled every call)
simple_patterns = [
    (r'^ret_(\d+)$', 1),
    (r'^(ret_future_|fwd_ret_)', None),
    (r'^(stoch_d|stoch_k|williams_r)_(\d+)$', 2),
    # ... 16+ patterns
]
for pattern, group_idx in simple_patterns:
    match = re.match(pattern, feature_name, re.I)
```

**Pattern 4: Simple prefix checks using regex**
```python
# Current (unnecessary regex)
if re.match(r"^tth_", feature_name):
    return REJECTED
```

---

## Implementation Plan

### Task 1: Add Compiled Pattern Cache to __init__

**Location**: `FeatureRegistry.__init__()` (around line 400)

```python
def __init__(self, config_path: Optional[str] = None, ...):
    # ... existing initialization ...

    # NEW: Compile family patterns once
    self._compiled_family_patterns: Dict[str, re.Pattern] = {}
    for family_name, family_config in self.families.items():
        pattern_str = family_config.get('pattern')
        if pattern_str:
            try:
                self._compiled_family_patterns[family_name] = re.compile(
                    pattern_str, re.IGNORECASE
                )
            except re.error as e:
                logger.warning(f"Invalid regex for family {family_name}: {e}")

    # NEW: Compile fallback patterns once
    self._compiled_fallback_patterns: List[Tuple[re.Pattern, Optional[int], str]] = [
        (re.compile(r"^ret_(\d+)$", re.I), 1, "lagged_returns"),
        (re.compile(r"^(ret_future_|fwd_ret_)", re.I), None, "forward_returns"),
        (re.compile(r"^(stoch_d|stoch_k|williams_r)_(\d+)$", re.I), 2, "stochastic"),
        (re.compile(r"^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var)_(\d+)$", re.I), 2, "technical"),
        (re.compile(r"^(ret|sma|ema|vol)_(\d+)$", re.I), 2, "simple"),
        (re.compile(r"^bb_(upper|lower|width|percent_b|middle)_(\d+)$", re.I), 2, "bollinger"),
        (re.compile(r"^macd_(signal|hist|diff)_(\d+)$", re.I), 2, "macd"),
        (re.compile(r"^volume_(ema|sma)_(\d+)$", re.I), 2, "volume"),
        (re.compile(r"^realized_vol_(\d+)$", re.I), 1, "realized_vol"),
        (re.compile(r"^vol_(ema|sma|std)_(\d+)$", re.I), 2, "volatility"),
    ]

    # NEW: Rejection prefixes (use startswith, not regex)
    self._rejection_prefixes: Tuple[str, ...] = (
        "tth_", "mfe_", "mdd_", "barrier_", "y_", "target_", "p_",
        "ret_future_", "fwd_ret_"
    )
```

### Task 2: Update is_allowed() to Use Compiled Patterns

**Location**: `is_allowed()` method (around line 1313)

```python
def is_allowed(self, feature_name: str, target_horizon: int, ...) -> bool:
    # PHASE A: Quick rejection via prefix (O(1), no regex)
    if feature_name.startswith(self._rejection_prefixes):
        return False

    # PHASE B: Try compiled family patterns (compiled, not re.match)
    for family_name, compiled_pattern in self._compiled_family_patterns.items():
        if compiled_pattern.match(feature_name):
            family_config = self.families[family_name]
            if family_config.get('rejected', False):
                return False
            # ... rest of logic
```

### Task 3: Update _auto_infer_metadata_fallback() to Use Compiled Patterns

**Location**: `_auto_infer_metadata_fallback()` method (around line 1511)

```python
def _auto_infer_metadata_fallback(self, feature_name: str) -> Dict[str, Any]:
    # Use pre-compiled patterns instead of re.match()
    for compiled_pattern, group_idx, family_type in self._compiled_fallback_patterns:
        match = compiled_pattern.match(feature_name)
        if match:
            if group_idx is not None:
                lag_bars = int(match.group(group_idx))
            else:
                lag_bars = 0
            return {
                'lag_bars': lag_bars,
                'family': family_type,
                # ... rest
            }
```

### Task 4: Replace Simple Prefix Checks

**Find and replace all occurrences**:
```python
# BEFORE (regex for simple prefix)
if re.match(r"^tth_", feature_name):

# AFTER (string method, O(1))
if feature_name.startswith("tth_"):
```

### Task 5: Add Lazy Compilation Cache for Config-Based Patterns

**For patterns loaded from config files**:
```python
def _get_compiled_pattern(self, pattern_str: str, flags: int = 0) -> re.Pattern:
    """Get or compile pattern (cached)."""
    cache_key = (pattern_str, flags)
    if cache_key not in self._pattern_cache:
        self._pattern_cache[cache_key] = re.compile(pattern_str, flags)
    return self._pattern_cache[cache_key]
```

---

## Checklist

- [ ] **1. Add compiled pattern cache to `__init__()`**
  - [ ] Add `_compiled_family_patterns` dict
  - [ ] Add `_compiled_fallback_patterns` list
  - [ ] Add `_rejection_prefixes` tuple
  - [ ] Add `_pattern_cache` dict for lazy compilation

- [ ] **2. Update `is_allowed()` method**
  - [ ] Replace `re.match(pattern, ...)` with `compiled.match(...)`
  - [ ] Add quick rejection via `startswith()` at top

- [ ] **3. Update `_auto_infer_metadata_fallback()`**
  - [ ] Use `_compiled_fallback_patterns` instead of inline patterns

- [ ] **4. Update `auto_infer_metadata()`**
  - [ ] Use `_get_compiled_pattern()` for config-based patterns

- [ ] **5. Replace simple prefix checks**
  - [ ] Find all `re.match(r"^prefix_", ...)` patterns
  - [ ] Replace with `str.startswith("prefix_")`

- [ ] **6. Add benchmark test**
  - [ ] Create `tests/test_feature_registry_benchmark.py`
  - [ ] Measure `is_allowed()` performance before/after
  - [ ] Verify 5x+ speedup

---

## Testing

### Unit Tests
```bash
# Run existing tests
pytest tests/test_feature_registry.py -v

# Run with verbose output to catch any regressions
pytest tests/test_feature_registry.py -v --tb=long
```

### Benchmark Test
```python
# tests/test_feature_registry_benchmark.py
import time
from TRAINING.common.feature_registry import get_registry

def test_is_allowed_benchmark():
    """Verify is_allowed() performance."""
    registry = get_registry()

    # Sample features and horizons
    features = [f"ret_{i}" for i in range(1, 100)]
    features += [f"sma_{i}" for i in range(5, 50)]
    features += [f"rsi_{i}" for i in [7, 14, 21]]
    horizons = [5, 10, 15, 30, 60, 120]

    start = time.perf_counter()
    calls = 0
    for _ in range(100):  # Simulate 100 targets
        for feature in features:
            for horizon in horizons:
                registry.is_allowed(feature, horizon)
                calls += 1
    elapsed = time.perf_counter() - start

    print(f"Total calls: {calls:,}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Calls/sec: {calls/elapsed:,.0f}")

    # Should be >100,000 calls/sec with optimization
    assert calls / elapsed > 50_000, f"Performance too slow: {calls/elapsed:.0f} calls/sec"
```

### Integration Test
```bash
# Run a small pipeline to verify no regressions
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir /tmp/test_regex_opt \
    --top-n-targets 3 \
    --dry-run
```

---

## Rollback Plan

If issues arise:
1. Revert changes to `feature_registry.py`
2. The compiled patterns are internal implementation details
3. No external API changes, so rollback is clean

---

## Success Criteria

- [ ] All existing tests pass
- [ ] Benchmark shows 5x+ speedup in `is_allowed()` calls
- [ ] No functional changes to feature classification behavior
- [ ] Memory usage increase is minimal (<10MB for pattern cache)
