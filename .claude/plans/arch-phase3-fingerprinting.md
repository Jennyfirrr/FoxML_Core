# Phase 3: Fingerprinting Gaps

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ Complete (All 10/10 items)
**Priority**: P0 (Critical - Hash Inconsistency)
**Estimated Effort**: 1 day
**Depends On**: Phase 1, Phase 2

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 10/10 items (ALL items complete)
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 3 complete - proceed to Phase 4 (Error Handling)
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| FP-001 | Feature fingerprint SHA1 8-char | ✅ Complete | Changed to SHA256 16-char |
| FP-002 | split_signature empty string fallback | ✅ Complete | Strict mode check + None fallback |
| FP-003 | Same pattern in multi_model_feature_selection | ✅ Complete | Same fix applied |
| FP-004 | feature_signature_input never computed | ✅ Complete | Now computed before RunIdentity creation |
| FP-005 | routing_signature="" instead of None | ✅ Complete | All `or ""` patterns removed |
| FP-006 | Single hparams_signature for all families | ✅ Complete | Added hparams_by_family to RunIdentity |
| FP-007 | Manual comparison_group construction | ✅ Complete | Created construct_comparison_group() SST helper |
| FP-008 | comparison_group without RunIdentity | ✅ Complete | Added run_identity to all snapshot classes |
| FP-009 | Cache key missing interval params | ✅ Complete | Added lookback_minutes/window_minutes to cache key |
| FP-010 | hyperparameters_signature inconsistent | ✅ Complete | SST helper always includes fields (can be None) |

---

## Problem Statement

The fingerprinting system has inconsistencies that break reproducibility tracking:
1. **Mixed hash lengths**: SHA1 8-char vs SHA256 64-char
2. **Empty string fallbacks**: `or ""` masks missing signatures
3. **Incomplete signatures**: Some fields never computed
4. **Manual construction**: Different comparison_group rules per stage

---

## Issue Details

### FP-001: Feature fingerprint uses SHA1 8-char (P0)

**File**: `TRAINING/common/utils/fingerprinting.py`
**Lines**: 603, 607

```python
# CURRENT
set_fingerprint = hashlib.sha1(set_str.encode()).hexdigest()[:8]
order_fingerprint = hashlib.sha1(order_str.encode()).hexdigest()[:8]
```

**Fix**:
```python
# Use SHA256 with 16-char truncation (matches cache keys)
set_fingerprint = hashlib.sha256(set_str.encode()).hexdigest()[:16]
order_fingerprint = hashlib.sha256(order_str.encode()).hexdigest()[:16]
```

**Impact**: All feature fingerprints will change. Need migration strategy.

---

### FP-002: split_signature empty string fallback (P0)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 925, 1932

```python
# CURRENT
split_signature=split_signature or "",
```

**Fix**:
```python
# Fail-closed in strict mode
if split_signature is None:
    from TRAINING.common.determinism import is_strict_mode
    if is_strict_mode():
        raise ValueError(
            "split_signature required but not computed. "
            "Ensure CV folds are finalized before creating RunIdentity."
        )
    logger.warning("split_signature not available, identity may not be fully reproducible")
    split_signature = None  # Explicit None, not empty string
```

---

### FP-003: Same pattern in multi_model_feature_selection (P0)

**File**: `TRAINING/ranking/multi_model_feature_selection.py`
**Lines**: 4272, 4275

Same fix as FP-002.

---

### FP-004: feature_signature_input never computed (P1)

**File**: `TRAINING/common/utils/fingerprinting.py`
**Lines**: 205-207

```python
# CURRENT (declared but never used)
feature_signature: Optional[str] = None  # Output signature
feature_signature_input: Optional[str] = None  # NEVER COMPUTED
feature_signature_output: Optional[str] = None
```

**Fix**: Compute input signature at feature selection start:
```python
# At start of feature selection
def select_features_for_target(...):
    # Compute input signature (candidate features)
    from TRAINING.common.utils.fingerprinting import compute_feature_fingerprint
    candidate_features = get_candidate_features(...)
    feature_signature_input = compute_feature_fingerprint(candidate_features)

    # ... selection logic ...

    # Compute output signature (selected features)
    feature_signature_output = compute_feature_fingerprint(selected_features)
```

---

### FP-005: routing_signature="" instead of None (P1)

**File**: `TRAINING/ranking/predictability/model_evaluation.py`
**Lines**: 6800, 6804

```python
# CURRENT
routing_signature=routing_sig or ""
```

**Fix**:
```python
# None is semantically correct for "not applicable"
routing_signature=routing_sig  # Let it be None if not computed
```

---

### FP-006: Single hparams_signature for all families (P1)

**File**: `TRAINING/ranking/predictability/model_evaluation.py`
**Line**: 6803

```python
# CURRENT - One signature for all families in TARGET_RANKING
hparams_signature=hparams_sig
```

**Fix**: Compute per-family signatures:
```python
# Compute hparams signature per family
per_family_hparams = {}
for family in model_families:
    family_config = get_model_config(family)
    per_family_hparams[family] = compute_hparams_fingerprint(family_config)

# Store in identity metadata
identity.hparams_by_family = per_family_hparams
```

---

### FP-007: Manual comparison_group construction (P1)

**File**: `TRAINING/stability/feature_importance/schema.py`
**Lines**: 475-488

```python
# CURRENT - Manual dict construction
comparison_group = {}
if importance_snapshot.dataset_signature:
    comparison_group["dataset_signature"] = importance_snapshot.dataset_signature
if importance_snapshot.target_signature:
    comparison_group["target_signature"] = importance_snapshot.target_signature
```

**Fix**: Use SST helper:
```python
from TRAINING.common.utils.fingerprinting import construct_comparison_group_key

comparison_group = construct_comparison_group_key(
    dataset_signature=importance_snapshot.dataset_signature,
    target_signature=importance_snapshot.target_signature,
    # ... other fields
)
```

---

### FP-008: comparison_group without RunIdentity (P2)

**File**: `TRAINING/training_strategies/reproducibility/schema.py`
**Lines**: 117-129

Snapshot stores comparison_group dict but not full RunIdentity.

**Fix**: Store RunIdentity reference:
```python
@dataclass
class TrainingSnapshot:
    # ... existing fields ...
    run_identity: Optional[RunIdentity] = None  # Add full identity
```

---

### FP-009: Cache key missing interval params (P1)

**File**: `TRAINING/ranking/feature_selector.py`
**Line**: 128

```python
# CURRENT - includes explicit_interval but not derived params
'explicit_interval': explicit_interval
```

**Fix**: Include interval-dependent parameters:
```python
'explicit_interval': explicit_interval,
'lookback_minutes': getattr(registry, 'lookback_minutes', None),
'window_minutes': getattr(registry, 'window_minutes', None),
```

---

### FP-010: hyperparameters_signature stage-specific (P2)

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`
**Lines**: 1710-1716

hparams_signature only included for certain stages, not consistently.

**Fix**: Always include (can be None if not applicable):
```python
# Consistent across all stages
comparison_group = {
    "dataset_signature": ...,
    "target_signature": ...,
    "hyperparameters_signature": hparams_sig,  # Always present
    "split_signature": split_sig,  # Always present
    # ...
}
```

---

## Implementation Steps

### Step 1: Standardize hash algorithm (FP-001)
- Change SHA1 → SHA256
- Update all fingerprint generation
- Add migration note for existing fingerprints

### Step 2: Remove empty string fallbacks (FP-002, FP-003, FP-005)
- Replace `or ""` with explicit None handling
- Add strict mode checks

### Step 3: Compute feature_signature_input (FP-004)
- Add computation at feature selection start
- Store in RunIdentity

### Step 4: Create SST comparison_group constructor (FP-007)
- Create `construct_comparison_group_key()` function
- Update all manual construction sites

### Step 5: Add RunIdentity to snapshots (FP-008)
- Update schema classes
- Update serialization

### Step 6: Include interval params in cache key (FP-009)
- Update `_compute_feature_selection_config_hash()`

---

## Contract Tests

```python
# tests/contract_tests/test_fingerprinting_contract.py

class TestFingerprintConsistency:
    def test_all_fingerprints_same_length(self):
        """All fingerprints should use consistent length."""
        from TRAINING.common.utils.fingerprinting import (
            compute_feature_fingerprint,
            compute_split_fingerprint,
            compute_hparams_fingerprint,
        )

        feature_fp = compute_feature_fingerprint(["f1", "f2"])
        split_fp = compute_split_fingerprint(...)
        hparams_fp = compute_hparams_fingerprint(...)

        # All should be 16 chars (or 64 for full)
        assert len(feature_fp) == len(split_fp) == len(hparams_fp)

    def test_no_empty_string_signatures(self):
        """Signatures should be None or valid, never empty string."""
        identity = RunIdentity(...)

        for field in ['split_signature', 'routing_signature', 'hparams_signature']:
            value = getattr(identity, field)
            assert value != "", f"{field} should not be empty string"

    def test_comparison_group_consistent_keys(self):
        """All comparison groups should have same keys."""
        # Test across different stages
        pass
```

---

## Verification

```bash
# Check for SHA1 usage
grep -rn "sha1\|SHA1" TRAINING/common/utils/fingerprinting.py

# Check for empty string fallbacks
grep -rn 'or ""' TRAINING/ | grep signature

# Check hash lengths
grep -rn "hexdigest()\[:" TRAINING/

# Run fingerprinting tests
pytest tests/contract_tests/test_fingerprinting_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 10 issues
- Created implementation steps
- **Next**: Wait for Phase 1, 2 completion, then start Step 1

### Session 2: 2026-01-19
- Completed all P0 items (FP-001 through FP-005):
  - **FP-001**: Changed SHA1 8-char to SHA256 16-char in `fingerprinting.py`
  - **FP-002**: Added strict mode checks for split_signature in `feature_selector.py` (2 locations)
  - **FP-003**: Same fix in `multi_model_feature_selection.py`
  - **FP-004**: Added feature_signature_input computation before RunIdentity creation (3 locations)
  - **FP-005**: Removed all `or ""` fallbacks for signatures in model_evaluation.py (2 locations) and intelligent_trainer.py (1 location)
- P1/P2 items (FP-006 through FP-010) deferred for future iteration
- **Phase 3 P0 Complete** ✅

### Session 3: 2026-01-19
- Completed all remaining P1/P2 items (FP-006 through FP-010):
  - **FP-006**: Added `hparams_by_family: Dict[str, str]` to RunIdentity for per-family tracking
  - **FP-007**: Created `construct_comparison_group()` SST helper in `fingerprinting.py`
  - **FP-008**: Added `run_identity: Optional[Dict[str, Any]]` to TrainingSnapshot, FeatureImportanceSnapshot, FeatureSelectionSnapshot
  - **FP-009**: Added `lookback_minutes` and `window_minutes` to cache key in `feature_selector.py`
  - **FP-010**: Updated `construct_comparison_group()` to always include critical fields (can be None) + migrated manual construction sites to use SST helper
- **Phase 3 Complete** ✅ (All 10/10 items)

---

## Notes

- Hash length changes will invalidate existing fingerprints
- Consider feature flag for gradual rollout
- Coordinate with `code-review-remediation.md` for atomic write patterns
- All items complete - ready for Phase 4 (Error Handling)
