# Phase 7: Stage Boundaries

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ COMPLETE
**Priority**: P2 (Medium - Type Safety)
**Estimated Effort**: 1-2 days
**Depends On**: Phase 3, Phase 6

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 9/9 items ✅
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 7 complete, proceed to Phase 8
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| SB-001 | Returns List[str] for CROSS_SECTIONAL | ✅ Complete | FeatureSelectionResult contract |
| SB-002 | Returns Dict for SYMBOL_SPECIFIC | ✅ Complete | FeatureSelectionResult contract |
| SB-003 | Returns mixed dict for BOTH | ✅ Complete | FeatureSelectionResult contract |
| SB-004 | Empty state is [] for BLOCKED | ✅ Complete | FeatureSelectionResult contract |
| SB-005 | Post-stage filtering modifies data | ✅ Complete | filter_target_features_at_boundary() |
| SB-006 | Routing validation after feature selection | ✅ Complete | Strict mode validation added |
| SB-007 | Registry reused without mutation check | ✅ Complete | Hash check at load and before training |
| SB-008 | Inconsistent empty states | ✅ Complete | Documented in FeatureSelectionResult |
| SB-009 | Feature data structure varies by route | ✅ Complete | Documented in FeatureSelectionResult |

---

## Problem Statement

The 3-stage pipeline has inconsistent data contracts:
1. **3 different return types** from feature selection
2. **No validation** at stage boundaries
3. **Post-stage mutations** outside stage boundaries
4. **Late validation** - routing validated after feature selection

---

## Current Return Types (Inconsistent)

```python
# CROSS_SECTIONAL (line 2628-2634)
target_features[target] = List[str]

# SYMBOL_SPECIFIC (line 2664-2672)
target_features[target] = Dict[str, List[str]]

# BOTH (line 2695-2699)
target_features[target] = {
    'cross_sectional': List[str],
    'symbol_specific': Dict[str, List[str]],
    'route': 'BOTH'
}

# BLOCKED (line 2703)
target_features[target] = []
```

---

## Proposed Data Contract

**New File**: `TRAINING/orchestration/contracts/feature_selection.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class FeatureSelectionView(str, Enum):
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
    BOTH = "BOTH"
    BLOCKED = "BLOCKED"

@dataclass
class FeatureSelectionResult:
    """
    SST data contract for feature selection stage output.

    Provides consistent interface regardless of routing view.
    """
    target: str
    view: FeatureSelectionView

    # CROSS_SECTIONAL features (empty list if SYMBOL_SPECIFIC only)
    cross_sectional_features: List[str] = field(default_factory=list)

    # SYMBOL_SPECIFIC features (empty dict if CROSS_SECTIONAL only)
    symbol_specific_features: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    blocked_reason: Optional[str] = None
    feature_count: int = 0

    def __post_init__(self):
        """Compute feature_count after initialization."""
        cs_count = len(self.cross_sectional_features)
        ss_count = sum(len(v) for v in self.symbol_specific_features.values())
        self.feature_count = cs_count + ss_count

    def is_empty(self) -> bool:
        """Check if no features were selected."""
        return self.feature_count == 0

    def to_legacy_format(self):
        """Convert to legacy format for backward compatibility."""
        if self.view == FeatureSelectionView.BLOCKED:
            return []
        elif self.view == FeatureSelectionView.CROSS_SECTIONAL:
            return self.cross_sectional_features
        elif self.view == FeatureSelectionView.SYMBOL_SPECIFIC:
            return self.symbol_specific_features
        else:  # BOTH
            return {
                'cross_sectional': self.cross_sectional_features,
                'symbol_specific': self.symbol_specific_features,
                'route': 'BOTH'
            }

    @classmethod
    def from_legacy_format(
        cls,
        target: str,
        data,
        view_hint: Optional[FeatureSelectionView] = None
    ) -> 'FeatureSelectionResult':
        """Parse from legacy format with view detection."""
        if data is None or (isinstance(data, list) and len(data) == 0):
            return cls(
                target=target,
                view=FeatureSelectionView.BLOCKED,
                blocked_reason="No features selected"
            )
        elif isinstance(data, list):
            return cls(
                target=target,
                view=FeatureSelectionView.CROSS_SECTIONAL,
                cross_sectional_features=data
            )
        elif isinstance(data, dict):
            if 'cross_sectional' in data and 'symbol_specific' in data:
                return cls(
                    target=target,
                    view=FeatureSelectionView.BOTH,
                    cross_sectional_features=data.get('cross_sectional', []),
                    symbol_specific_features=data.get('symbol_specific', {})
                )
            else:
                return cls(
                    target=target,
                    view=FeatureSelectionView.SYMBOL_SPECIFIC,
                    symbol_specific_features=data
                )
        else:
            raise ValueError(f"Unknown format for {target}: {type(data)}")

    def validate(self) -> List[str]:
        """Validate result and return list of issues."""
        issues = []

        if self.view == FeatureSelectionView.BLOCKED and not self.blocked_reason:
            issues.append("BLOCKED view should have blocked_reason")

        if self.view == FeatureSelectionView.CROSS_SECTIONAL:
            if self.symbol_specific_features:
                issues.append("CROSS_SECTIONAL should not have symbol_specific_features")

        if self.view == FeatureSelectionView.SYMBOL_SPECIFIC:
            if self.cross_sectional_features:
                issues.append("SYMBOL_SPECIFIC should not have cross_sectional_features")

        return issues
```

---

## Issue Details

### SB-001, SB-002, SB-003, SB-004: Inconsistent return types

**Solution**: Use `FeatureSelectionResult` consistently.

```python
# INSTEAD OF
target_features[target] = features_list

# USE
result = FeatureSelectionResult(
    target=target,
    view=FeatureSelectionView.CROSS_SECTIONAL,
    cross_sectional_features=features_list
)
target_features[target] = result  # Or result.to_legacy_format() during transition
```

---

### SB-005: Post-stage filtering modifies data (P2)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 2994-3006

```python
# CURRENT - inline filtering
if isinstance(target_features, dict):
    try:
        filtered_target_features = {}
        for t, f in target_features.items():
            # ... filtering logic
```

**Fix**: Extract to dedicated function at stage boundary:
```python
def filter_target_features_at_boundary(
    target_features: Dict[str, FeatureSelectionResult],
    allowed_targets: Set[str],
    skip_blocked: bool = True
) -> Dict[str, FeatureSelectionResult]:
    """
    Filter target features at stage boundary.

    Called BETWEEN stages, not mid-stage.
    """
    filtered = {}
    for target in sorted(allowed_targets):
        if target not in target_features:
            continue

        result = target_features[target]
        if skip_blocked and result.view == FeatureSelectionView.BLOCKED:
            logger.debug(f"Skipping {target} (BLOCKED)")
            continue

        filtered[target] = result

    return filtered
```

---

### SB-006: Routing validation after feature selection (P1)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 3205-3232

**Current flow**:
```
Feature Selection → (complete) → Load Routing Decisions → Validate
```

**Fixed flow**:
```
Load Routing Decisions → Validate → Feature Selection
```

**Fix**:
```python
# BEFORE feature selection
if auto_features:
    # Step 2a: Load and validate routing FIRST
    routing_decisions = load_routing_decisions(output_dir=self.output_dir)
    if routing_decisions:
        missing = set(targets) - set(routing_decisions.keys())
        if missing and is_strict_mode():
            raise ValueError(f"Missing routing decisions for: {sorted(missing)}")

    # Step 2b: NOW do feature selection with validated routing
    for target in sorted(targets):
        route_info = routing_decisions.get(target, {})
        # ... feature selection
```

---

### SB-007: Registry reused without mutation check (P2)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 1195-1250

**Fix**: Add mutation detection:
```python
# After loading registry
registry_hash = hash(frozenset(registry.features.keys()))

# Before training (after all FS complete)
if hash(frozenset(registry.features.keys())) != registry_hash:
    raise RuntimeError("Registry was mutated between stages!")
```

---

### SB-008, SB-009: Documented by contract

Handled by `FeatureSelectionResult` contract - validates and documents all formats.

---

## Implementation Steps

### Step 1: Create contract file
Create `TRAINING/orchestration/contracts/feature_selection.py`

### Step 2: Add legacy conversion
Implement `to_legacy_format()` and `from_legacy_format()`

### Step 3: Update feature selection callers
Gradually migrate to use typed results

### Step 4: Move routing validation (SB-006)
Reorder validation before feature selection

### Step 5: Add mutation detection (SB-007)
Add registry hash check between stages

### Step 6: Extract filtering function (SB-005)
Move inline filtering to stage boundary function

---

## Contract Tests

```python
# tests/contract_tests/test_stage_boundaries_contract.py

class TestFeatureSelectionContract:
    def test_result_types_normalized(self):
        """All views should produce valid FeatureSelectionResult."""
        for view in FeatureSelectionView:
            result = create_test_result(view)
            assert isinstance(result, FeatureSelectionResult)
            assert result.view == view

    def test_legacy_roundtrip(self):
        """Legacy format conversion should be lossless."""
        original = FeatureSelectionResult(
            target="test",
            view=FeatureSelectionView.BOTH,
            cross_sectional_features=["f1"],
            symbol_specific_features={"AAPL": ["f2"]}
        )

        legacy = original.to_legacy_format()
        restored = FeatureSelectionResult.from_legacy_format("test", legacy)

        assert restored.cross_sectional_features == original.cross_sectional_features
        assert restored.symbol_specific_features == original.symbol_specific_features

    def test_validation_catches_issues(self):
        """Validation should catch malformed results."""
        bad_result = FeatureSelectionResult(
            target="test",
            view=FeatureSelectionView.BLOCKED,
            blocked_reason=None  # Should be set
        )

        issues = bad_result.validate()
        assert len(issues) > 0

class TestRoutingValidation:
    def test_routing_validated_before_feature_selection(self):
        """Routing decisions should be loaded before FS starts."""
        # Test that routing is validated early
        pass
```

---

## Verification

```bash
# Check for mixed return types
grep -rn "target_features\[" TRAINING/orchestration/intelligent_trainer.py | head -20

# Check routing validation location
grep -rn "load_routing_decisions" TRAINING/orchestration/intelligent_trainer.py

# Run stage boundary tests
pytest tests/contract_tests/test_stage_boundaries_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Designed FeatureSelectionResult contract

### Session 2: 2026-01-19
- ✅ Created `TRAINING/orchestration/contracts/feature_selection.py`
  - FeatureSelectionView enum with CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, BLOCKED
  - FeatureSelectionResult dataclass with to_legacy_format() and from_legacy_format()
  - filter_target_features_at_boundary() function for SB-005
- ✅ SB-006: Added strict mode validation for routing decisions before feature selection
- ✅ SB-007: Added registry mutation detection
  - Hash computed at load (line ~1401)
  - Hash validated before training (line ~3420)
- **PHASE COMPLETE**

---

## Notes

- Use gradual migration - support both typed and legacy formats during transition
- Consider using Pydantic for runtime validation if already in dependencies
- Coordinate with Phase 3 (Fingerprinting) for feature_signature handling
