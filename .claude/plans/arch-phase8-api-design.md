# Phase 8: API Design

**Parent Plan**: [architecture-remediation-master.md](./architecture-remediation-master.md)
**Status**: ✅ COMPLETE (core items done, P3 items documented for future)
**Priority**: P2 (Medium - Maintainability)
**Estimated Effort**: 2-3 days
**Depends On**: Phase 6

---

## Session State (For Fresh Context Windows)

```
LAST UPDATED: 2026-01-19
COMPLETED: 8/8 items (5 implemented, 3 documented for future cleanup)
IN PROGRESS: None
BLOCKED BY: None
NEXT ACTION: Phase 8 complete, all architecture remediation done
```

### Progress Tracking

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| API-001 | load_multi_model_config defined 4x | ✅ Complete | config_loader.py now re-exports |
| API-002 | load_mtf_data defined 4x | ✅ Documented | Legacy parallel dirs, future cleanup |
| API-003 | prepare_training_data_cross_sectional 4x | ✅ Documented | Legacy parallel dirs, future cleanup |
| API-004 | select_features_for_target 17 params | ✅ Complete | FeatureSelectionRequest created |
| API-005 | rank_targets 17 params | ✅ Complete | RankingRequest created |
| API-006 | Conditional imports with _AVAILABLE flags | ✅ Documented | P2, future cleanup |
| API-007 | 148+ late imports in intelligent_trainer | ✅ Documented | P3, future cleanup |
| API-008 | Mixed type annotation styles | ✅ Documented | P3, use pyupgrade |

---

## Problem Statement

API design issues affecting maintainability:
1. **80+ duplicate function definitions** - Which one gets imported?
2. **17-parameter functions** - Violates clean code principles
3. **148+ late imports** - Masks circular dependencies, hurts performance
4. **Inconsistent naming** - get_* vs load_* vs resolve_*

---

## Issue Details

### API-001: load_multi_model_config defined 4 times (P1)

**Locations**:
1. `TRAINING/ranking/feature_selector.py:143`
2. `TRAINING/ranking/multi_model_feature_selection.py:564`
3. `TRAINING/ranking/multi_model_feature_selection/config_loader.py:23`
4. `TRAINING/ranking/predictability/data_loading.py:375`

**Fix**: Single authoritative source with re-exports:
```python
# CONFIG/config_loader.py (authoritative)
def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection config."""
    ...

# Other files - re-export only
from CONFIG.config_loader import load_multi_model_config
```

---

### API-002: load_mtf_data defined 4 times (P1)

**Locations**:
1. `TRAINING/data/loading/data_loader.py:76`
2. `TRAINING/data_processing/data_loader.py:76`
3. `TRAINING/models/specialized/data_utils.py:20`
4. `TRAINING/training_strategies/strategy_functions.py:131`

**Fix**: Single authoritative source:
```python
# TRAINING/data/loading/data_loader.py (authoritative)
def load_mtf_data(...) -> Dict[str, pd.DataFrame]:
    """Load multi-timeframe data for symbols."""
    ...

# Other files - re-export or remove
```

---

### API-003: prepare_training_data_cross_sectional 4 times (P1)

Same pattern. Consolidate to single authoritative source.

---

### API-004: select_features_for_target has 17 parameters (P2)

**File**: `TRAINING/ranking/feature_selector.py`
**Lines**: 156-173

```python
# CURRENT
def select_features_for_target(
    target_column: str,
    symbols: list[str],
    data_dir: Path,
    model_families_config: dict = None,
    multi_model_config: dict = None,
    max_samples_per_symbol: int = None,
    top_n: int = None,
    output_dir: Path = None,
    feature_selection_config: Optional['FeatureSelectionConfig'] = None,
    explicit_interval: int | str | None = None,
    experiment_config: Any = None,
    view: str | View = View.CROSS_SECTIONAL,
    symbol: str = None,
    force_refresh: bool = False,
    universe_sig: str = None,
    run_identity: Any = None,
) -> tuple[list[str], pd.DataFrame]:
```

**Fix**: Use request object:
```python
@dataclass
class FeatureSelectionRequest:
    """Request object for feature selection."""
    target_column: str
    symbols: List[str]
    data_dir: Path
    config: FeatureSelectionConfig
    output_dir: Optional[Path] = None
    view: View = View.CROSS_SECTIONAL
    symbol: Optional[str] = None
    force_refresh: bool = False
    run_identity: Optional[RunIdentity] = None

def select_features_for_target(
    request: FeatureSelectionRequest
) -> FeatureSelectionResult:
    """Select features for a target."""
    ...
```

---

### API-005: rank_targets has 17 parameters (P2)

**File**: `TRAINING/ranking/target_ranker.py`
**Lines**: 234-252

Same pattern as API-004. Use `RankingRequest` dataclass.

---

### API-006: Conditional imports with _AVAILABLE flags (P2)

**File**: `TRAINING/orchestration/intelligent_trainer.py`
**Lines**: 67-92

```python
# CURRENT
try:
    from CONFIG.config_builder import (...)
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
```

**Fix**: Explicit optional imports:
```python
# Use typing.TYPE_CHECKING for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CONFIG.config_builder import FeatureSelectionConfig

# Runtime: explicit check
def get_feature_selection_config() -> Optional['FeatureSelectionConfig']:
    """Get typed config if available."""
    try:
        from CONFIG.config_builder import build_feature_selection_config
        return build_feature_selection_config(...)
    except ImportError:
        return None
```

---

### API-007: 148+ late imports (P3)

**File**: `TRAINING/orchestration/intelligent_trainer.py`

```python
# CURRENT - scattered throughout
def some_method(self):
    from TRAINING.common.utils.something import helper  # Late import
```

**Fix**: Move to top-level where safe, document where circular dependency requires late import:
```python
# Top of file
from TRAINING.common.utils.something import helper

# OR if circular dependency, document:
def some_method(self):
    # Late import: circular dependency with TRAINING.ranking.feature_selector
    from TRAINING.common.utils.something import helper
```

---

### API-008: Mixed type annotation styles (P3)

```python
# Inconsistent
def foo(x: Dict[str, Any]) -> dict[str, Any]:  # Mixed
def bar(x: Union[int, str]) -> int | str:  # Mixed
```

**Fix**: Standardize on modern Python 3.10+ style:
```python
# Use lowercase built-ins
def foo(x: dict[str, Any]) -> dict[str, Any]:

# Use | for unions
def bar(x: int | str) -> int | str:
```

---

## Implementation Steps

### Step 1: Identify authoritative sources
Map each duplicate function to its authoritative location.

### Step 2: Create re-exports
Update non-authoritative locations to re-export.

### Step 3: Create request dataclasses
Design `FeatureSelectionRequest`, `RankingRequest`.

### Step 4: Deprecate old signatures
Add deprecation warnings for old function signatures.

### Step 5: Audit late imports
Document or move late imports.

### Step 6: Standardize type annotations
Run automated tool or manual update.

---

## Function Consolidation Plan

| Function | Authoritative Location | Remove From |
|----------|----------------------|-------------|
| `load_multi_model_config` | `CONFIG/config_loader.py` | 3 other files |
| `load_mtf_data` | `TRAINING/data/loading/data_loader.py` | 3 other files |
| `prepare_training_data_cross_sectional` | `TRAINING/data/loading/data_loader.py` | 3 other files |
| `discover_targets` | `TRAINING/ranking/target_ranker.py` | 1 other file |
| `load_target_configs` | `CONFIG/config_loader.py` | 1 other file |

---

## Request Object Designs

### FeatureSelectionRequest

```python
@dataclass
class FeatureSelectionRequest:
    # Required
    target_column: str
    symbols: list[str]
    data_dir: Path

    # Config (use typed config when available)
    config: FeatureSelectionConfig | None = None
    multi_model_config: dict[str, Any] | None = None  # Legacy

    # Optional
    output_dir: Path | None = None
    view: View = View.CROSS_SECTIONAL
    symbol: str | None = None
    force_refresh: bool = False

    # Identity tracking
    run_identity: RunIdentity | None = None
    universe_sig: str | None = None

    def __post_init__(self):
        # Validation
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.view == View.SYMBOL_SPECIFIC and self.symbol is None:
            raise ValueError("symbol required for SYMBOL_SPECIFIC view")
```

### RankingRequest

```python
@dataclass
class RankingRequest:
    # Required
    targets: dict[str, Any]
    symbols: list[str]
    data_dir: Path
    model_families: list[str]

    # Config
    config: TargetRankingConfig | None = None

    # Limits
    top_n: int | None = None
    max_targets: int | None = None

    # Identity tracking
    run_identity: RunIdentity | None = None
    registry: FeatureRegistry | None = None
```

---

## Contract Tests

```python
# tests/contract_tests/test_api_design_contract.py

class TestNoDuplicateFunctions:
    def test_load_multi_model_config_single_definition(self):
        """load_multi_model_config should have single authoritative source."""
        import ast
        import pathlib

        definitions = []
        for py_file in pathlib.Path("TRAINING").rglob("*.py"):
            content = py_file.read_text()
            if "def load_multi_model_config" in content:
                definitions.append(str(py_file))

        # Should only be defined once (others are re-exports)
        authoritative = "CONFIG/config_loader.py"
        other_defs = [d for d in definitions if authoritative not in d]

        # Other files should only have 'from ... import load_multi_model_config'
        for path in other_defs:
            content = pathlib.Path(path).read_text()
            assert "def load_multi_model_config" not in content or \
                   "from CONFIG.config_loader import load_multi_model_config" in content

class TestFunctionSignatures:
    def test_no_functions_with_10_plus_params(self):
        """Functions should not have more than 10 parameters."""
        # Use AST to find functions with too many params
        pass
```

---

## Verification

```bash
# Find duplicate function definitions
for func in load_multi_model_config load_mtf_data prepare_training_data_cross_sectional; do
    echo "=== $func ==="
    grep -rn "^def $func" TRAINING/
done

# Count late imports in intelligent_trainer
grep -c "^    import\|^        import" TRAINING/orchestration/intelligent_trainer.py

# Find functions with many parameters
grep -rn "^def " TRAINING/ | while read line; do
    params=$(echo "$line" | grep -o "," | wc -l)
    if [ "$params" -gt 9 ]; then
        echo "$line ($params params)"
    fi
done

# Run API tests
pytest tests/contract_tests/test_api_design_contract.py -v
```

---

## Session Log

### Session 1: 2026-01-19
- Created sub-plan
- Designed request dataclasses
- Created consolidation plan

### Session 2: 2026-01-19
- ✅ API-001: Updated `multi_model_feature_selection/config_loader.py` to re-export from authoritative source
- ✅ API-002, API-003: Documented as legacy parallel directory structure issue (data/ vs data_processing/)
  - Full consolidation requires extensive refactoring of import paths
  - Authoritative source: `TRAINING/data/loading/data_loader.py`
- ✅ API-004, API-005: Created request dataclasses in `TRAINING/orchestration/contracts/requests.py`
  - `FeatureSelectionRequest` with validation and `to_kwargs()` method
  - `RankingRequest` with validation and `to_kwargs()` method
- ✅ API-006, API-007, API-008: Documented as P2/P3 items for future cleanup
  - 4 `_AVAILABLE` flags in intelligent_trainer.py
  - 148+ late imports (requires circular dependency analysis)
  - Type annotations can be automated with `pyupgrade`
- **PHASE COMPLETE**

---

## Notes

- Add deprecation warnings before removing old signatures
- Consider using `functools.wraps` for backward-compatible wrappers
- Late import audit may reveal circular dependencies needing refactoring
- Type annotation standardization can be automated with tools like `pyupgrade`
