# Phase 5: Config Hierarchy

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: Not Started
**Priority**: P1 (High - SST Violations)
**Estimated Effort**: 1-2 days
**Depends On**: Phase 4

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 0/13 items
IN PROGRESS: None
BLOCKED BY: Phase 4
NEXT ACTION: Start with CH-001 - Replace yaml.safe_load in generate_routing_plan.py
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| CH-001 | generate_routing_plan.py yaml.safe_load | Not Started | |
| CH-002 | routing_integration.py yaml.safe_load | Not Started | |
| CH-003 | family_config.py hardcoded path | Not Started | |
| CH-004 | family_config.py yaml.safe_load | Not Started | |
| CH-005 | multi_model_feature_selection.py yaml.safe_load | Not Started | |
| CH-006 | multi_model_feature_selection.py fallback 50000 | Not Started | |
| CH-007 | feature_selector.py yaml.safe_load fallback | Not Started | |
| CH-008 | feature_selector.py fallback 10, 1000 | Not Started | |
| CH-009 | target_ranker.py yaml.safe_load fallback | Not Started | |
| CH-010 | target_ranker.py hardcoded values | Not Started | |
| CH-011 | cross_sectional_feature_ranker.py fallbacks | Not Started | |
| CH-012 | multi_model_feature_selection.py fallback 2000 | Not Started | |
| CH-013 | multi_model_feature_selection.py duplicate pattern | Not Started | |

---

## Problem Statement

Multiple files bypass the centralized config system (`get_cfg()` from `CONFIG.config_loader`):
1. Direct `yaml.safe_load()` calls bypass experiment config precedence
2. Hardcoded fallback values duplicate what's in `CONFIG/defaults.yaml`
3. Hardcoded paths instead of config-based paths

**Config precedence should be**: CLI args > Experiment config > Intelligent training config > Pipeline configs > Defaults

---

## Files with Direct yaml.safe_load() (19 total)

```
TRAINING/orchestration/intelligent_trainer.py      # Some legitimate
TRAINING/orchestration/routing_integration.py      # CH-002
TRAINING/orchestration/generate_routing_plan.py    # CH-001
TRAINING/common/family_config.py                   # CH-003, CH-004
TRAINING/common/feature_registry.py                # Legitimate
TRAINING/ranking/feature_selector.py               # CH-007, CH-008
TRAINING/ranking/multi_model_feature_selection.py  # CH-005, CH-006
TRAINING/ranking/target_ranker.py                  # CH-009, CH-010
TRAINING/ranking/cross_sectional_feature_ranker.py # CH-011
```

---

## Issue Details

### CH-001: generate_routing_plan.py (P1)

**File**: `TRAINING/orchestration/generate_routing_plan.py`
**Lines**: 87-88

```python
# CURRENT
with open(routing_config_path) as f:
    routing_config = yaml.safe_load(f)
```

**Fix**:
```python
from CONFIG.config_loader import load_config_file

routing_config = load_config_file(routing_config_path)
```

---

### CH-002: routing_integration.py (P1)

**File**: `TRAINING/orchestration/routing_integration.py`
**Lines**: 58-59

Same pattern as CH-001.

---

### CH-003, CH-004: family_config.py (P1)

**File**: `TRAINING/common/family_config.py`
**Lines**: 18, 34-35

```python
# CURRENT
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "family_config.yaml"

with open(_CONFIG_PATH, 'r') as f:
    _FAMILY_CONFIG = yaml.safe_load(f)
```

**Fix**: Move to CONFIG/ and use get_cfg():
```python
from CONFIG.config_loader import get_cfg

def load_family_config() -> Dict[str, Any]:
    """Load family configuration using centralized config."""
    return {
        "families": get_cfg("pipeline.training.families", default={}),
        "thread_policies": get_cfg("pipeline.training.thread_policies", default={})
    }
```

---

### CH-005, CH-006: multi_model_feature_selection.py (P1)

**File**: `TRAINING/ranking/multi_model_feature_selection.py`
**Lines**: 609-610, 676-677

```python
# CURRENT (609-610)
with open(config_path) as f:
    config = yaml.safe_load(f)

# CURRENT (676-677) - fallback
except Exception as e:
    default_max_samples = 50000
```

**Fix**:
```python
from CONFIG.config_loader import get_cfg, load_config_file

# For file loading
config = load_config_file(config_path)

# For fallback
from TRAINING.common.utils.error_helpers import get_config_or_fail
default_max_samples = get_config_or_fail(
    "pipeline.data_limits.default_max_samples_feature_selection",
    default=50000,
    stage="FEATURE_SELECTION"
)
```

---

### CH-007, CH-008: feature_selector.py (P1)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 355-365, 380-385

Same pattern. Use `get_config_or_fail()` helper.

---

### CH-009, CH-010: target_ranker.py (P1)

**File**: `TRAINING/ranking/target_ranker.py`
**Lines**: 155-174, 138, 326

Same pattern. Use `get_config_or_fail()` helper.

---

### CH-011: cross_sectional_feature_ranker.py (P1)

**File**: `TRAINING/ranking/cross_sectional_feature_ranker.py`
**Lines**: 310, 317

Same pattern.

---

### CH-012, CH-013: Additional patterns (P2)

More hardcoded fallbacks in multi_model_feature_selection.py.

---

## Implementation Steps

### Step 1: Create helper for config loading with fallback
Already defined in Phase 4: `get_config_or_fail()`

### Step 2: Update routing files (CH-001, CH-002)
Replace yaml.safe_load with load_config_file

### Step 3: Migrate family_config.py (CH-003, CH-004)
- Create CONFIG/pipeline/training/families.yaml if needed
- Update family_config.py to use get_cfg()

### Step 4: Update ranking files (CH-005 through CH-011)
Apply `get_config_or_fail()` pattern consistently

### Step 5: Document fallback values
Ensure all hardcoded values are documented in CONFIG/defaults.yaml

---

## Contract Tests

```python
# tests/contract_tests/test_config_hierarchy_contract.py

class TestConfigHierarchy:
    def test_no_yaml_safe_load_in_orchestration(self):
        """Orchestration files should not use yaml.safe_load directly."""
        import ast
        import pathlib

        orchestration_dir = pathlib.Path("TRAINING/orchestration")
        violations = []

        for py_file in orchestration_dir.rglob("*.py"):
            content = py_file.read_text()
            if "yaml.safe_load" in content:
                # Check if it's in a legitimate context (e.g., SST comment)
                if "# SST:" not in content.split("yaml.safe_load")[0].split("\n")[-1]:
                    violations.append(str(py_file))

        assert not violations, f"Found yaml.safe_load in: {violations}"

    def test_fallbacks_match_defaults_yaml(self):
        """Hardcoded fallbacks should match CONFIG/defaults.yaml."""
        from CONFIG.config_loader import get_cfg

        # Check known fallback values
        assert get_cfg("pipeline.data_limits.default_max_samples_feature_selection") == 50000
        assert get_cfg("pipeline.data_limits.min_cross_sectional_samples") == 10
        assert get_cfg("pipeline.data_limits.max_cs_samples") == 1000

    def test_experiment_config_overrides_pipeline(self):
        """Experiment config should override pipeline defaults."""
        # Create mock experiment config with override
        # Verify get_cfg returns override value
        pass
```

---

## Verification

```bash
# Find yaml.safe_load in orchestration
grep -rn "yaml.safe_load" TRAINING/orchestration/ | grep -v "# SST:"

# Find hardcoded numeric fallbacks
grep -rn "except.*:" TRAINING/ranking/ -A2 | grep -E "= [0-9]+"

# Check if defaults.yaml has all fallback values
grep -E "^  default_max|^  min_cross|^  max_cs" CONFIG/defaults.yaml

# Run config hierarchy tests
pytest tests/contract_tests/test_config_hierarchy_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Documented all 13 issues
- **Next**: Wait for Phase 4, then implement starting with CH-001

---

## Notes

- Coordinate with Phase 4 (Error Handling) for `get_config_or_fail()` helper
- Ensure backward compatibility - existing configs should still work
- Consider adding pre-commit hook to detect new yaml.safe_load() usage
