# Phase 2: DiffTelemetry Decomposition

**Status**: ✅ Completed
**Priority**: P1 (Maintainability)
**Effort**: 6 hours (estimated)
**Completed**: 2026-01-19
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)

---

## Quick Resume

```
STATUS: COMPLETED
FILE: TRAINING/orchestration/utils/diff_telemetry.py (reduced from 6,556 → 5,949 lines)
RESULT: Successfully extracted run_hash functions to submodule
```

## Session Progress (2026-01-19)

### Completed:
1. ✅ **types.py** - Already extracted (633 lines, pre-existing)
   - All dataclasses: `ComparisonGroup`, `NormalizedSnapshot`, `DiffResult`, `BaselineState`, `ResolvedRunContext`
   - All enums: `ChangeSeverity`, `ComparabilityStatus`
   - Helper: `compute_config_signature()`

2. ✅ **run_hash.py** - Created (~450 lines)
   - `_extract_deterministic_fields()` - Extract deterministic snapshot fields
   - `compute_full_run_hash()` - Compute run hash from all snapshots
   - `_load_manifest_comparability_flags()` - Load manifest flags
   - `_normalize_run_id_for_comparison()` - Normalize run IDs (legacy)
   - `_can_runs_be_compared()` - Check run comparability
   - `compute_run_hash_with_changes()` - Compute hash with change tracking
   - `save_run_hash()` - Save run hash to file

3. ✅ **__init__.py** - Updated with run_hash exports

4. ✅ **diff_telemetry.py** - Updated to import from run_hash submodule
   - Removed ~600 lines of duplicated functions
   - Reduced from 6,556 to 5,949 lines

5. ✅ **Tests** - All pass (3 pre-existing failures unrelated to changes)

---

## SST/Determinism Compliance

### Principles Preserved:
| Principle | Status | Notes |
|-----------|--------|-------|
| **Sorted iteration** | ✅ | All `sorted()` calls preserved in extracted code |
| **Deterministic ordering** | ✅ | `sorted_indices`, `sorted_snapshots` patterns kept |
| **Type safety** | ✅ | All type hints preserved |
| **No logic changes** | ✅ | Pure structural refactor, no algorithm changes |

### SST Patterns in Extracted Code:
```python
# run_hash.py preserves:
sorted_indices = sorted(snapshot_indices.items())  # Deterministic
sorted_snapshots = sorted(index_data.items())      # Deterministic
canonical_json = json.dumps(run_state, sort_keys=True)  # Canonical serialization
```

---

## Problem Statement

`diff_telemetry.py` is **6,556 lines** containing:
- Type definitions (~633 lines) - EXTRACTED to types.py
- Helper utilities (~120 lines) - `_sanitize_for_json`, `_write_atomic_json_with_lock`, `_run_id_part`
- Main `DiffTelemetry` class (~5,700 lines) - Remains in main file
- Stand-alone run hash functions (~610 lines) - EXTRACTING to run_hash.py

---

## Current Structure

```
TRAINING/orchestration/utils/
├── diff_telemetry.py              # 6,556 lines (MONOLITH)
└── diff_telemetry/                # Partial decomposition
    ├── __init__.py                # Imports from types.py + parent
    ├── types.py                   # 633 lines (ALREADY EXTRACTED)
    └── run_hash.py                # NEW: ~450 lines (CREATED)
```

## Target Structure

```
TRAINING/orchestration/utils/
├── diff_telemetry.py              # ~5,500 lines (after extraction)
└── diff_telemetry/
    ├── __init__.py                # Public API exports
    ├── types.py                   # EXISTING (dataclasses, enums)
    ├── run_hash.py                # NEW: Run hash computation
    └── utilities.py               # FUTURE: Helper functions (optional)
```

---

## Function Mapping

### run_hash.py (EXTRACTED)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_extract_deterministic_fields()` | 5928-5985 | Extract deterministic snapshot fields |
| `compute_full_run_hash()` | 5988-6162 | Compute hash from all snapshots |
| `_load_manifest_comparability_flags()` | 6165-6200 | Load manifest flags |
| `_normalize_run_id_for_comparison()` | 6203-6226 | Legacy run ID normalization |
| `_can_runs_be_compared()` | 6229-6278 | Check comparability |
| `compute_run_hash_with_changes()` | 6281-6499 | Hash with changes |
| `save_run_hash()` | 6502-6556 | Save hash to file |

### Remaining in main file
| Component | Lines | Purpose |
|-----------|-------|---------|
| `_sanitize_for_json()` | 63-100 | JSON sanitization |
| `_write_atomic_json_with_lock()` | 103-170 | Locked atomic writes |
| `_run_id_part()` | 173-182 | Run ID formatting |
| `DiffTelemetry` class | 186-5927 | Main telemetry class |

---

## Implementation Steps

### Step 1: Update __init__.py ✅
Add exports from run_hash.py:
```python
from .run_hash import (
    compute_full_run_hash,
    compute_run_hash_with_changes,
    save_run_hash,
    _can_runs_be_compared,
    _normalize_run_id_for_comparison,
)
```

### Step 2: Update main file
1. Add import from run_hash submodule
2. Remove function definitions (5928-6556)
3. Add comment noting extraction

### Step 3: Verify
```bash
pytest TRAINING/contract_tests/ -v -k "diff_telemetry or comparison"
python -m py_compile TRAINING/orchestration/utils/diff_telemetry.py
```

---

## Checklist

### Analysis
- [x] Map all functions to target files
- [x] Identify SST patterns to preserve
- [x] Verify no logic changes in extraction

### Extraction
- [x] **types.py** - Already existed
- [x] **run_hash.py** - Created with all run hash functions
- [x] Update __init__.py exports
- [x] Update main file imports
- [x] Remove duplicated code from main file

### Verification
- [x] Syntax validation (py_compile)
- [x] Contract tests pass (3 pre-existing failures unrelated to changes)
- [x] Smoke imports work

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Circular imports | run_hash.py uses TYPE_CHECKING for DiffTelemetry |
| Determinism regression | No algorithm changes, only code movement |
| SST violations | All sorted() patterns preserved |
| Test failures | Run contract tests after each change |

---

## Success Criteria

- [x] diff_telemetry.py reduced from 6,556 to 5,949 lines (-607 lines)
- [x] All run hash functions moved to run_hash.py
- [x] All imports continue to work
- [x] All tests pass (pre-existing failures only)
- [x] SST patterns preserved
