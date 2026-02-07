# Phase 6: Data Consistency

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: Not Started
**Priority**: P1 (High - Determinism Issues)
**Estimated Effort**: 1 day
**Depends On**: Phase 4

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 0/10 items
IN PROGRESS: None
BLOCKED BY: Phase 4
NEXT ACTION: Start with DC-003 - Fix unsorted dict iteration
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| DC-001 | DataFrame column mutation without copy | Not Started | |
| DC-002 | NumPy slicing returns view | Not Started | |
| DC-003 | Unsorted dict iteration (feature_selections) | Not Started | P0 |
| DC-004 | Unsorted dict iteration (target_features) | Not Started | |
| DC-005 | winner_symbols not sorted | Not Started | P0 |
| DC-006 | route_counts logged without sorting | Not Started | |
| DC-007 | None vs [] vs {} semantics undefined | Not Started | |
| DC-008 | Empty DataFrame without schema | Not Started | |
| DC-009 | Variable existence via locals()/dir() | Not Started | |
| DC-010 | view and view_enum drift | Not Started | |

---

## Problem Statement

Data consistency issues affecting determinism:
1. **In-place mutations** - DataFrames modified without `.copy()`
2. **Unsorted iteration** - Dict iteration order affects artifacts
3. **Inconsistent empty states** - None vs [] vs {} handling unclear
4. **Fragile checks** - Using `locals()`/`dir()` for variable existence

---

## Issue Details

### DC-001: DataFrame mutation without copy (P1)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 2065-2066

```python
# CURRENT
summary_df['cs_importance_score'] = 0.0
summary_df['feature_category'] = 'PENDING'
```

**Fix**:
```python
summary_df = summary_df.copy()
summary_df['cs_importance_score'] = 0.0
summary_df['feature_category'] = 'PENDING'
```

---

### DC-002: NumPy slicing returns view (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 735-736, 746

```python
# CURRENT
X = X[:, feature_indices]  # May return view
```

**Fix**:
```python
X = X[:, feature_indices].copy()  # Explicit copy
```

---

### DC-003: Unsorted dict iteration (P0)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Line**: 1934

```python
# CURRENT
for key, selection in feature_selections.items():
```

**Fix**:
```python
from TRAINING.common.utils.determinism_helpers import sorted_items

for key, selection in sorted_items(feature_selections):
```

---

### DC-004: Unsorted dict iteration (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 3176-3191

```python
# CURRENT
for target, feat_data in list(target_features.items())[:3]:
```

**Fix**:
```python
for target, feat_data in sorted(target_features.items())[:3]:
```

---

### DC-005: winner_symbols not sorted (P0)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 2638-2648

```python
# CURRENT
winner_symbols = route_info.get('winner_symbols', [])
# ...
for symbol in winner_symbols:
```

**Fix**:
```python
winner_symbols = sorted(route_info.get('winner_symbols', []))
# ...
for symbol in winner_symbols:  # Now deterministic
```

---

### DC-006: route_counts logged without sorting (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Line**: 2596

```python
# CURRENT
logger.info(f"Route counts: {route_counts}")
```

**Fix**:
```python
logger.info(f"Route counts: {dict(sorted(route_counts.items()))}")
```

---

### DC-007: None vs [] vs {} semantics (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 3938, 4064, 2542, 2566

```python
# CURRENT - inconsistent empty states
config_families = None  # vs [] vs {}
```

**Fix**: Document and enforce:
```python
# Convention:
# - None = not set / not applicable
# - [] = explicitly empty list
# - {} = explicitly empty dict

# Example:
config_families: Optional[List[str]] = None  # None means "use default"
if config_families is None:
    config_families = get_default_families()
# Now config_families is always List[str], never None
```

---

### DC-008: Empty DataFrame without schema (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Line**: 1659

```python
# CURRENT
return [], pd.DataFrame()
```

**Fix**:
```python
# Return DataFrame with expected schema
return [], pd.DataFrame(columns=['feature', 'importance', 'rank'])
```

---

### DC-009: Variable existence via locals()/dir() (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 2455-2467

```python
# CURRENT
if 'cohort_context' in locals() and cohort_context:
    ...
else:
    fallback_min_cs = harness_min_cs if 'harness_min_cs' in dir() and harness_min_cs else 10
```

**Fix**:
```python
# Initialize explicitly at function start
cohort_context: Optional[CohortContext] = None
harness_min_cs: Optional[int] = None

# Then check normally
if cohort_context is not None:
    ...
else:
    fallback_min_cs = harness_min_cs if harness_min_cs is not None else 10
```

---

### DC-010: view and view_enum drift (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 219, 299, 307

```python
# CURRENT
view_enum = View.from_string(view) if isinstance(view, str) else view
# ...
view = View.SYMBOL_SPECIFIC  # Updates string but not enum
view_enum = View.SYMBOL_SPECIFIC
```

**Fix**: Use single source of truth:
```python
# Always work with enum
view_enum = View.from_string(view) if isinstance(view, str) else view

# If mutation needed, only mutate view_enum
if condition:
    view_enum = View.SYMBOL_SPECIFIC
    # Don't maintain parallel `view` variable

# If string needed, derive from enum
view_str = view_enum.value
```

---

## Implementation Steps

### Step 1: Add determinism helper for sorted iteration
```python
# TRAINING/common/utils/determinism_helpers.py

def sorted_items(d: Dict[K, V]) -> List[Tuple[K, V]]:
    """Iterate dict items in sorted key order for determinism."""
    return sorted(d.items(), key=lambda x: x[0])

def sorted_keys(d: Dict[K, V]) -> List[K]:
    """Get dict keys in sorted order for determinism."""
    return sorted(d.keys())
```

### Step 2: Fix P0 items (DC-003, DC-005)
These affect determinism directly - fix first.

### Step 3: Fix P1 items (DC-001, DC-004, DC-007)
Data integrity issues.

### Step 4: Fix P2 items (DC-002, DC-006, DC-008, DC-009, DC-010)
Code quality improvements.

---

## Contract Tests

```python
# tests/contract_tests/test_data_consistency_contract.py

class TestDeterministicIteration:
    def test_target_features_sorted(self):
        """target_features iteration should be deterministic."""
        target_features = {"z_target": [...], "a_target": [...], "m_target": [...]}

        results = []
        for target, features in sorted_items(target_features):
            results.append(target)

        assert results == ["a_target", "m_target", "z_target"]

    def test_winner_symbols_sorted(self):
        """winner_symbols should be processed in sorted order."""
        # Test that processing order is deterministic
        pass

class TestDataFrameImmutability:
    def test_summary_df_not_mutated(self):
        """Original DataFrame should not be mutated."""
        original_df = pd.DataFrame(...)
        original_hash = hash(original_df.to_string())

        # Call function that should copy before mutating
        process_summary(original_df)

        assert hash(original_df.to_string()) == original_hash

class TestEmptyStates:
    def test_none_vs_empty_list_semantics(self):
        """None and [] should have distinct meanings."""
        # None = use default
        result_none = get_families(config_families=None)
        assert result_none == DEFAULT_FAMILIES

        # [] = explicitly empty
        result_empty = get_families(config_families=[])
        assert result_empty == []
```

---

## Verification

```bash
# Find unsorted dict iteration
grep -rn "\.items()" TRAINING/orchestration/intelligent_trainer.py | grep -v sorted

# Find DataFrame mutations
grep -rn "\[.*\] = " TRAINING/ranking/feature_selector.py | grep -v "\.copy()"

# Find locals()/dir() usage
grep -rn "in locals()\|in dir()" TRAINING/

# Run data consistency tests
pytest tests/contract_tests/test_data_consistency_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 10 issues
- **Next**: Wait for Phase 4, then implement sorted_items helper and fix DC-003, DC-005

---

## Notes

- sorted_items() should already exist in codebase - verify and reuse
- Consider adding pre-commit hook to detect unsorted .items() calls
- Document None vs [] convention in coding standards
