# Phase 1: Model Evaluation Decomposition

**Status**: ✅ Completed
**Priority**: P1 (Maintainability)
**Effort**: 8 hours (estimated), ~4 hours actual
**Completed**: 2026-01-19
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)

---

## Quick Resume

```
STATUS: COMPLETED
FILE: TRAINING/ranking/predictability/model_evaluation.py (reduced from 9,359 → 68 lines!)
EXTRACTED: training.py (4,972 lines), ranking.py (4,259 lines), safety.py (323 lines), autofix.py (343 lines)
```

## Session Progress (2026-01-19)

### Completed:
1. ✅ **safety.py** - Full extraction (~323 lines)
   - `enforce_final_safety_gate()` moved to submodule
   - model_evaluation.py now imports from submodule

2. ✅ **autofix.py** - Full logic extraction (~343 lines)
   - `evaluate_target_with_autofix()` implemented
   - Imports evaluate_target_predictability via dynamic import

3. ✅ **training.py** - Full extraction (~4,972 lines)
   - `train_and_evaluate_models()` fully extracted with all imports
   - All nested helper functions included

4. ✅ **ranking.py** - Full extraction (~4,259 lines)
   - `evaluate_target_predictability()` fully extracted with all imports
   - Imports train_and_evaluate_models from training.py

5. ✅ **__init__.py** - Updated with proper exports
   - All public APIs exported
   - Backward compatibility maintained

6. ✅ **model_evaluation.py** - Converted to thin wrapper (68 lines)
   - Re-exports all functions from submodules
   - Full backward compatibility

### Line Count Progress:
- Before: 9,656 lines
- After: 68 lines (thin wrapper)
- Extracted: ~9,900 lines total to submodules

---

## Problem Statement

`model_evaluation.py` is **9,656 lines** - the largest file in the codebase. It contains:
- Model training logic (~1,500 lines)
- Safety gate validation (~400 lines)
- Core ranking logic (~5,000 lines)
- Autofix/leakage correction (~800 lines)
- Helper functions (~2,000 lines)

**Existing partial decomposition**: `model_evaluation/` subdir has 3 files

---

## Current Structure

```
TRAINING/ranking/predictability/
├── model_evaluation.py          # 9,656 lines (MONOLITH)
└── model_evaluation/            # Partial decomposition
    ├── __init__.py
    ├── config_helpers.py        # Config loading
    ├── leakage_helpers.py       # Leakage utilities
    └── reporting.py             # Result reporting
```

## Target Structure

```
TRAINING/ranking/predictability/
├── model_evaluation.py          # ~200 lines (THIN WRAPPER)
└── model_evaluation/
    ├── __init__.py              # Public API exports
    ├── config_helpers.py        # EXISTING
    ├── leakage_helpers.py       # EXISTING
    ├── reporting.py             # EXISTING
    ├── training.py              # NEW: Model training (~1,500 lines)
    ├── safety.py                # NEW: Safety gate validation (~400 lines)
    ├── ranking.py               # NEW: Core ranking logic (~5,000 lines)
    └── autofix.py               # NEW: Leakage autofix (~800 lines)
```

---

## Function Mapping

### training.py (NEW)
Extract from lines ~452-5257:
- `train_and_evaluate_models()` - Main orchestration
- `_train_single_model()` - Per-model training
- `extract_native_importance()` - Feature importance from models
- `extract_shap_importance()` - SHAP-based importance
- `extract_permutation_importance()` - Permutation importance
- `_cross_validate_model()` - CV implementation
- `_compute_cv_metrics()` - Metric computation

### safety.py (NEW)
Extract from lines ~148-451:
- `_enforce_final_safety_gate()` - Feature validation before training
- `_validate_feature_lag()` - Lag validation
- `_check_policy_cap()` - Policy cap enforcement
- `_apply_feature_filters()` - Pre-training filters

### ranking.py (NEW)
Extract from lines ~5258-9347:
- `evaluate_target_predictability()` - Primary entry point
- `_load_and_prepare_data()` - Data loading
- `_compute_target_metrics()` - Metric aggregation
- `_rank_targets()` - Target ranking logic
- `_build_predictability_report()` - Report generation
- All supporting helper functions

### autofix.py (NEW)
Extract from lines ~9348-9656:
- `evaluate_target_with_autofix()` - Main autofix entry
- `_detect_leakage_features()` - Leakage detection
- `_apply_autofix()` - Fix application
- `_verify_fix()` - Post-fix verification

---

## Implementation Steps

### Step 1: Read and Map Functions

```bash
# Extract function signatures
grep -n "^def \|^class \|^async def " TRAINING/ranking/predictability/model_evaluation.py
```

### Step 2: Create training.py

```python
# TRAINING/ranking/predictability/model_evaluation/training.py
"""Model training and evaluation logic."""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from CONFIG.config_loader import get_cfg
from TRAINING.common.feature_registry import get_registry
from .config_helpers import load_evaluation_config
from .leakage_helpers import check_feature_leakage

logger = logging.getLogger(__name__)


def train_and_evaluate_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_families: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train and evaluate multiple model families.

    [Move implementation from model_evaluation.py:452-XXX]
    """
    ...
```

### Step 3: Create safety.py

```python
# TRAINING/ranking/predictability/model_evaluation/safety.py
"""Safety gate validation for features."""
from typing import List, Set, Tuple
import logging

from CONFIG.config_loader import get_cfg
from TRAINING.common.feature_registry import get_registry

logger = logging.getLogger(__name__)


def enforce_final_safety_gate(
    feature_columns: List[str],
    target_horizon: int,
    strict: bool = True
) -> Tuple[List[str], List[str]]:
    """Enforce safety gate before model training.

    [Move implementation from model_evaluation.py:148-XXX]
    """
    ...
```

### Step 4: Create ranking.py

```python
# TRAINING/ranking/predictability/model_evaluation/ranking.py
"""Core target predictability ranking."""
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

from CONFIG.config_loader import get_cfg
from .training import train_and_evaluate_models
from .safety import enforce_final_safety_gate
from .reporting import build_predictability_report

logger = logging.getLogger(__name__)


def evaluate_target_predictability(
    data_dir: str,
    target_column: str,
    output_dir: str,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate predictability of a target.

    [Move implementation from model_evaluation.py:5258-XXX]
    """
    ...
```

### Step 5: Create autofix.py

```python
# TRAINING/ranking/predictability/model_evaluation/autofix.py
"""Automatic leakage detection and fixing."""
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import logging

from .ranking import evaluate_target_predictability
from .leakage_helpers import detect_leakage

logger = logging.getLogger(__name__)


def evaluate_target_with_autofix(
    data_dir: str,
    target_column: str,
    output_dir: str,
    max_iterations: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate with automatic leakage fixing.

    [Move implementation from model_evaluation.py:9348-XXX]
    """
    ...
```

### Step 6: Update __init__.py

```python
# TRAINING/ranking/predictability/model_evaluation/__init__.py
"""
Model evaluation module.

Public API:
    - train_and_evaluate_models: Train multiple model families
    - evaluate_target_predictability: Evaluate target predictability
    - evaluate_target_with_autofix: Evaluate with automatic leakage fixing
    - enforce_final_safety_gate: Validate features before training
"""
from .training import (
    train_and_evaluate_models,
    extract_native_importance,
    extract_shap_importance,
    extract_permutation_importance,
)
from .safety import (
    enforce_final_safety_gate,
)
from .ranking import (
    evaluate_target_predictability,
)
from .autofix import (
    evaluate_target_with_autofix,
)

# Re-export existing
from .config_helpers import load_evaluation_config
from .leakage_helpers import check_feature_leakage
from .reporting import build_predictability_report

__all__ = [
    # Training
    "train_and_evaluate_models",
    "extract_native_importance",
    "extract_shap_importance",
    "extract_permutation_importance",
    # Safety
    "enforce_final_safety_gate",
    # Ranking
    "evaluate_target_predictability",
    # Autofix
    "evaluate_target_with_autofix",
    # Config
    "load_evaluation_config",
    # Leakage
    "check_feature_leakage",
    # Reporting
    "build_predictability_report",
]
```

### Step 7: Convert model_evaluation.py to Thin Wrapper

```python
# TRAINING/ranking/predictability/model_evaluation.py
"""
Model evaluation module.

This is a thin wrapper for backward compatibility.
All implementation is in model_evaluation/ subpackage.

For new code, import directly from the subpackage:
    from TRAINING.ranking.predictability.model_evaluation import (
        train_and_evaluate_models,
        evaluate_target_predictability,
    )
"""
# Re-export everything from subpackage
from TRAINING.ranking.predictability.model_evaluation import (
    # Training
    train_and_evaluate_models,
    extract_native_importance,
    extract_shap_importance,
    extract_permutation_importance,
    # Safety
    enforce_final_safety_gate,
    # Ranking
    evaluate_target_predictability,
    # Autofix
    evaluate_target_with_autofix,
    # Config
    load_evaluation_config,
    # Leakage
    check_feature_leakage,
    # Reporting
    build_predictability_report,
)

# Backward compatibility: module-level __all__
__all__ = [
    "train_and_evaluate_models",
    "extract_native_importance",
    "extract_shap_importance",
    "extract_permutation_importance",
    "enforce_final_safety_gate",
    "evaluate_target_predictability",
    "evaluate_target_with_autofix",
    "load_evaluation_config",
    "check_feature_leakage",
    "build_predictability_report",
]
```

---

## Checklist

### Analysis
- [x] Map all functions in model_evaluation.py to target files
- [x] Identify internal dependencies between functions
- [x] Identify external imports needed per file
- [x] Document function signatures for each target file

### Extraction (Order Matters)
- [x] **1. Extract safety.py** (fewest dependencies)
  - [x] Move `_enforce_final_safety_gate()` and helpers
  - [x] Add imports
  - [x] Verify no circular imports
  - [x] Run tests

- [x] **2. Extract training.py** (depends on safety)
  - [x] Move `train_and_evaluate_models()` and helpers (~4,972 lines)
  - [x] Add all required imports
  - [x] Include nested helper functions
  - [x] Run tests

- [x] **3. Extract ranking.py** (depends on training, safety)
  - [x] Move `evaluate_target_predictability()` and helpers (~4,259 lines)
  - [x] Import train_and_evaluate_models from training.py
  - [x] Add all required imports
  - [x] Run tests

- [x] **4. Extract autofix.py** (depends on ranking)
  - [x] Move `evaluate_target_with_autofix()` and helpers
  - [x] Import from ranking.py
  - [x] Run tests

### Integration
- [x] Update `__init__.py` with all exports
- [x] Convert `model_evaluation.py` to thin wrapper (68 lines)
- [x] Verify all existing imports still work
- [x] Run full test suite (3 pre-existing failures, not from this change)

### Cleanup
- [x] Remove duplicate code from wrapper
- [x] Add module docstrings
- [x] Update imports (removed dynamic parent import)
- [x] Verify logging still works

---

## Dependency Graph

```
                 safety.py
                    │
                    ▼
               training.py ◄─── config_helpers.py
                    │                   │
                    ▼                   │
               ranking.py ◄─── leakage_helpers.py
                    │                   │
                    ▼                   │
               autofix.py ◄─── reporting.py
```

**Extraction order**: safety → training → ranking → autofix

---

## Testing Strategy

### After Each Extraction
```bash
# Quick smoke test
python -c "from TRAINING.ranking.predictability.model_evaluation import train_and_evaluate_models"

# Run unit tests
pytest tests/test_model_evaluation.py -v

# Run contract tests
pytest TRAINING/contract_tests/ -v -k "model_evaluation or ranking"
```

### Final Verification
```bash
# Full test suite
pytest TRAINING/contract_tests/ tests/ -v

# Integration test
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir /tmp/test_decomposition \
    --top-n-targets 2 \
    --dry-run
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Circular imports | Extract in dependency order (safety first) |
| Broken imports | Thin wrapper re-exports everything |
| Missing functions | Comprehensive function mapping before extraction |
| Test failures | Run tests after each extraction |

---

## Success Criteria

- [x] model_evaluation.py reduced from 9,656 to 68 lines (thin wrapper!)
- [x] All 4 new submodules created and functional
- [x] All existing imports continue to work
- [x] All tests pass (3 pre-existing failures only)
- [x] No circular import errors
