# 2026-01: Determinism Hardening Phase 1 (Financial Outputs)

## Summary

Systematic elimination of non-determinism sources in Tier A files (determinism-critical for financial outputs). Fixed 20 non-deterministic patterns across 5 Tier A files, ensuring same inputs produce identical outputs for target rankings, feature selection, routing decisions, and training plans.

## Critical Fixes

### 1. Determinism Helper Modules

**New Files**:
- `TRAINING/common/utils/determinism_ordering.py` - Canonical helpers for deterministic filesystem and container iteration
- `TRAINING/common/utils/determinism_serialization.py` - Canonical JSON serialization helpers
- `TRAINING/common/determinism_policy.py` - Tier A file list SST and waiver validation

**Functions Created**:
- `iterdir_sorted()`, `glob_sorted()`, `rglob_sorted()` - Deterministic filesystem operations
- `sorted_items()`, `sorted_keys()`, `sorted_unique()` - Deterministic container iteration
- `select_latest_by_semantic_key()` - Deterministic "latest" selection (not mtime)
- `collect_and_sort_parallel_results()` - Deterministic parallel result sorting
- `canonical_json()`, `canonical_json_bytes()`, `write_canonical_json()` - Deterministic JSON serialization

### 2. Filesystem Operations

**`TRAINING/orchestration/target_routing.py`**:
- Line 142: `view_dir.iterdir()` → `iterdir_sorted(view_dir)` - Deterministic symbol directory iteration
- Line 191: `feature_selections_dir.rglob("target_confidence.json")` → `rglob_sorted(...)` - Deterministic confidence file discovery

**`TRAINING/decisioning/decision_engine.py`**:
- Line 347: `decisions_dir.glob("*.json")` → `glob_sorted(decisions_dir, "*.json")` - Deterministic decision file loading

**Impact**: Routing decisions and decision loading order are now deterministic.

### 3. Dictionary Iterations

**`TRAINING/ranking/target_ranker.py`**:
- Line 409: Model family extraction - `sorted_items()` for deterministic order
- Line 695: Parallel target evaluation - `sorted_items()` for deterministic input order
- Line 735, 746, 1045, 1085: Symbol-specific result aggregation - `sorted_items()` for deterministic storage order
- Line 709: Parallel results sorting - `collect_and_sort_parallel_results()` for deterministic processing order

**`TRAINING/orchestration/training_plan_generator.py`**:
- Lines 732, 750, 768, 786: Training plan view generation - `sorted_items()` for deterministic file generation order

**`TRAINING/ranking/feature_selector.py`**:
- Lines 1931, 1967: Symbol-specific feature importance storage - `sorted_items()` for deterministic file generation order
- Lines 2249, 2274: First symbol selection - `next(iter(...))` → `next(sorted_keys(...))` for deterministic selection
- Lines 431, 2182, 2700: Model family list comprehensions - `sorted_items()` for deterministic family order

**Impact**: Target evaluation, training plan generation, and feature selection order are now deterministic.

### 4. Score-Based Sorting with Tie-Breakers

**`TRAINING/ranking/target_ranker.py`**:
- Line 1178: Screen score sorting - Added target name tie-breaker: `key=lambda r: (-r.score_screen, r.target)`
- Line 1182: Strict score sorting - Added target name tie-breaker: `key=lambda r: (-r.score_strict, r.target)`

**Impact**: Equal scores now break ties deterministically by target name, preventing non-deterministic ranking order.

### 5. JSON Serialization

**`TRAINING/common/utils/file_utils.py`**:
- Line 180: Added default `sort_keys=True` to `safe_json_dump()` for deterministic JSON output

**Impact**: All JSON outputs are now deterministic by default (sorted keys).

### 6. Enforcement Tools

**New Scripts**:
- `bin/check_determinism_patterns.sh` - Fast pattern scanner for regression detection
- `bin/verify_determinism_init.py` - Entry point verification script

**Usage**:
```bash
# Scan for determinism violations
./bin/check_determinism_patterns.sh TRAINING

# Verify entry points initialize determinism
python bin/verify_determinism_init.py
```

## Statistics

- **Total Patterns Fixed**: 20
- **Files Modified**: 6 (5 Tier A + 1 utility)
- **New Files Created**: 5 (3 helpers + 2 tools)
- **Lines Changed**: +573/-97

## Verification

- All fixes pass linter checks
- Pattern scanner confirms no violations in fixed files
- Registry coverage fixes remain intact (no breaking changes)
- All logic preserved (only ordering changed)

## Related Documents

- Plan: `.cursor/plans/determinism-hardening-plan_2026-01.plan.md`
- Fixes Documentation: `INTERNAL/docs/fixes-determinism-hardening-phase1-2026-01.md`
- Previous Fixes: `INTERNAL/docs/fixes-dev-mode-coverage-review-2026-01.md`
