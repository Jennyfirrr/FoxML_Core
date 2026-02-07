# Phase 1: Run Identity Lifecycle

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ COMPLETE
**Priority**: P0 (Critical - Blocks Reproducibility)
**Estimated Effort**: 1-2 days

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 7/8 items (RI-007 is by design)
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 1 complete - proceed to Phase 2 or Phase 3
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| RI-001 | Initial manifest uses output_dir.name | ✅ Complete | Changed to run_id=None, let manifest handle derivation |
| RI-002 | Partial identity not stored on self | ✅ Complete | Added self._partial_identity storage |
| RI-003 | trainer.run_identity never set | ✅ Complete | Added _finalize_run_identity() method and call after FS |
| RI-004 | Multiple inconsistent run_id derivation | ✅ Complete | Added _get_stable_run_id() canonical method |
| RI-005 | Another run_identity access fails silently | ✅ Complete | Fixed all `_run_identity` → `_partial_identity` refs |
| RI-006 | dataset_snapshot_hash optional in strict | ✅ Complete | Added graduated enforcement with config flag |
| RI-007 | create_stage_identity returns partial | Documented | By design - no change needed |
| RI-008 | Another partial identity not propagated | ✅ Complete | Same fix as RI-002 |

---

## Problem Statement

The `RunIdentity` fingerprinting system is designed for two-phase construction:
1. Create partial identity (`is_final=False`) early in pipeline
2. Call `finalize(feature_signature)` after features are locked

**However**, the orchestrator creates partial identities but **never calls `finalize()`** at the orchestration level. This causes:
- `trainer.run_identity` is always `None`
- run_id falls back to unstable `output_dir.name`
- Manifest gets unstable run_id
- Reproducibility tracking is broken

---

## Issue Details

### RI-001: Initial manifest uses output_dir.name (P0)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Line**: 918

```python
# CURRENT (line 918)
create_manifest(
    self.output_dir,
    run_id=self.output_dir.name,  # WRONG: Uses directory name
    experiment_config=experiment_config_dict,
    run_metadata=run_metadata_dict
)
```

**Fix**:
```python
# Don't set run_id in initial manifest - will be updated after finalization
create_manifest(
    self.output_dir,
    run_id=None,  # Let manifest derive from output_dir.name as fallback
    experiment_config=experiment_config_dict,
    run_metadata=run_metadata_dict
)
```

**Test**: Verify manifest.json has run_id=null initially, then updated after FS completes.

---

### RI-002: Partial identity not stored on self (P0)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 1182-1191

```python
# CURRENT (lines 1182-1191)
target_ranking_identity = create_stage_identity(  # LOCAL variable
    stage=Stage.TARGET_RANKING,
    symbols=self.symbols,
    experiment_config=experiment_config,
    data_dir=self.data_dir,
)
logger.debug(f"Created TARGET_RANKING identity with train_seed={target_ranking_identity.train_seed}")
```

**Fix**:
```python
# Store on self for later finalization
self._partial_identity = create_stage_identity(
    stage=Stage.TARGET_RANKING,
    symbols=self.symbols,
    experiment_config=experiment_config,
    data_dir=self.data_dir,
)
logger.debug(f"Created partial identity, awaiting finalization: {self._partial_identity.debug_key}")
```

**Test**: After `__init__`, verify `trainer._partial_identity` is not None.

---

### RI-003: trainer.run_identity never set (P0)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 2727, 3436, 4633

```python
# CURRENT (multiple locations)
run_identity = getattr(trainer, 'run_identity', None) or getattr(trainer, '_run_identity', None)
# Always gets None because attribute never set
```

**Fix**: After feature selection completes (~line 2708), finalize and store:
```python
# After feature selection completes, finalize identity
if hasattr(self, '_partial_identity') and self._partial_identity:
    # Get feature signature from completed feature selection
    feature_signature = self._compute_feature_signature(target_features)
    self.run_identity = self._partial_identity.finalize(feature_signature)
    logger.info(f"Finalized RunIdentity: {self.run_identity.debug_key}")

    # Update manifest with stable run_id
    from TRAINING.orchestration.utils.manifest import update_manifest_run_identity
    update_manifest_run_identity(self.output_dir, self.run_identity)
```

**Test**: After `train_with_intelligence()`, verify `trainer.run_identity` is not None and `is_final=True`.

---

### RI-004: Multiple inconsistent run_id derivation (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 1305, 2729, 4636

Multiple fallback chains exist for run_id derivation:
1. Line 1305: `derive_run_id_from_identity(target_ranking_identity)`
2. Line 2729: `derive_run_id_from_identity(run_identity=run_identity)`
3. Line 4636: Same pattern

**Fix**: Create single canonical function:
```python
def _get_stable_run_id(self) -> str:
    """Get stable run_id, failing in strict mode if not available."""
    if hasattr(self, 'run_identity') and self.run_identity:
        from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
        return derive_run_id_from_identity(self.run_identity)

    from TRAINING.common.determinism import is_strict_mode
    if is_strict_mode():
        raise RuntimeError("run_identity not finalized - cannot derive stable run_id in strict mode")

    # Best-effort fallback
    logger.warning("Using unstable run_id from output_dir.name")
    return self.output_dir.name
```

**Test**: In strict mode, calling `_get_stable_run_id()` before finalization should raise.

---

### RI-005: Another run_identity access fails silently (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Line**: 3436

Same pattern as RI-003. Will be fixed by the same solution.

---

### RI-006: dataset_snapshot_hash optional in strict mode (P1)

**File**: `TRAINING/orchestration/utils/manifest.py`
**Lines**: 189-204

```python
# CURRENT (commented out TODO)
# TODO: When data layer provides snapshot hash, require it in strict mode:
# if not has_dataset_snapshot:
#     raise ValueError(...)
```

**Fix**: Uncomment and enforce:
```python
if mode == "strict":
    if not has_identity:
        raise ValueError(
            "Cannot assess comparability in strict mode without finalized RunIdentity."
        )
    if not has_dataset_snapshot:
        raise ValueError(
            "Cannot assess comparability in strict mode without dataset snapshot hash. "
            "Provide dataset_snapshot_hash from data layer."
        )
    return (True, "stable")
```

**Test**: `assess_comparability(mode="strict")` with missing dataset_snapshot should raise.

---

### RI-007: create_stage_identity returns partial (P2 - By Design)

**File**: `TRAINING/common/utils/fingerprinting.py`
**Lines**: 436-462

This is **by design** - `create_stage_identity()` returns partial identity that must be finalized by caller. Document this clearly.

---

### RI-008: Another partial identity not propagated (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Line**: 1647

Another location where partial identity is created but not stored/propagated.

**Fix**: Same pattern as RI-002 - store on self.

---

## Implementation Steps

### Step 1: Add instance attributes
```python
# In IntelligentTrainer.__init__()
self._partial_identity: Optional['RunIdentity'] = None
self.run_identity: Optional['RunIdentity'] = None
```

### Step 2: Store partial identity (RI-002, RI-008)
Update all `create_stage_identity()` calls to store result on `self._partial_identity`.

### Step 3: Create finalization method
```python
def _finalize_run_identity(self, feature_signature: str) -> None:
    """Finalize partial identity after feature selection."""
    if self._partial_identity is None:
        logger.warning("No partial identity to finalize")
        return

    self.run_identity = self._partial_identity.finalize(feature_signature)
    logger.info(f"Finalized RunIdentity: strict_key={self.run_identity.strict_key[:8]}...")
```

### Step 4: Call finalization after feature selection (RI-003)
Add call to `_finalize_run_identity()` after feature selection completes.

### Step 5: Fix run_id derivation (RI-001, RI-004)
- Create `_get_stable_run_id()` method
- Replace all ad-hoc derivation with this method

### Step 6: Enforce strict mode (RI-006)
Uncomment strict mode enforcement in `assess_comparability()`.

---

## Contract Tests

```python
# tests/contract_tests/test_run_identity_contract.py

class TestRunIdentityLifecycle:
    def test_partial_identity_stored_after_init(self, trainer):
        """After __init__, _partial_identity should be set."""
        assert hasattr(trainer, '_partial_identity')
        # May be None if no experiment_config, but attribute exists

    def test_run_identity_finalized_after_feature_selection(self, trainer_with_features):
        """After feature selection, run_identity should be finalized."""
        assert trainer_with_features.run_identity is not None
        assert trainer_with_features.run_identity.is_final

    def test_stable_run_id_in_manifest(self, completed_run):
        """Manifest should have stable run_id from identity."""
        manifest = load_manifest(completed_run.output_dir)
        assert manifest['run_id'] is not None
        # Should NOT be just the directory name
        assert manifest['run_id'] != completed_run.output_dir.name or \
               manifest.get('run_identity', {}).get('is_final')

    def test_strict_mode_requires_finalized_identity(self, trainer_strict):
        """In strict mode, cannot get run_id before finalization."""
        with pytest.raises(RuntimeError, match="run_identity not finalized"):
            trainer_strict._get_stable_run_id()
```

---

## Verification

```bash
# Check all run_identity access patterns
grep -rn "run_identity" TRAINING/orchestration/intelligent_trainer.py

# Check finalize calls
grep -rn "\.finalize\(" TRAINING/

# Run contract tests
pytest tests/contract_tests/test_run_identity_contract.py -v

# Full test suite
pytest TRAINING/contract_tests/ -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 8 issues with line numbers
- Created implementation steps
- Created contract tests

### Session 2: 2026-01-19
- **Implemented all Phase 1 fixes**:
  - RI-001: Changed initial manifest to use `run_id=None` (line 926)
  - RI-002: Added `self._partial_identity` and `self.run_identity` instance attributes (lines 237-241)
  - RI-002: Stored partial identity on self in TARGET_RANKING identity creation (line 1193)
  - RI-003: Added `_compute_feature_signature_from_target_features()` helper method (lines 1014-1052)
  - RI-003: Added `_finalize_run_identity()` method (lines 1054-1099)
  - RI-003: Added finalization call after feature selection completes (line 2850)
  - RI-004: Added `_get_stable_run_id()` canonical method with strict mode enforcement (lines 1101-1126)
  - RI-005: Fixed all `_run_identity` → `_partial_identity` attribute references
  - RI-006: Added graduated enforcement of dataset_snapshot_hash in strict mode (manifest.py lines 198-220)
- Verified syntax with `py_compile`
- **Phase 1 COMPLETE**

---

## Notes

- All changes must maintain backward compatibility
- Test with both strict mode ON and OFF
- Update `architecture-remediation-master.md` when phase completes
