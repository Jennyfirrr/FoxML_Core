# Decomposition Verification Checklist

When decomposing large files into submodules, use this checklist to prevent import/export errors.

## Pre-Decomposition

- [ ] **Read the entire file** before making changes
- [ ] **Identify all public exports** (functions, classes, constants used by other modules)
- [ ] **Search for external usages**: `grep -r "from MODULE import" TRAINING/`
- [ ] **Document the current public API** before changes

## During Decomposition

### For Each Extracted Submodule

- [ ] **Create submodule with unique name** (avoid naming conflicts with parent file)
  - BAD: `feature_selector/` when `feature_selector.py` exists
  - GOOD: `feature_selector_modules/` or rename parent first

- [ ] **Include all imports** the extracted code needs
- [ ] **Add `__all__` export list** to each submodule
- [ ] **Verify each function/class exists** before adding to `__init__.py` imports

### For the `__init__.py` File

- [ ] **Import only what exists** - don't add imports for functions you plan to create later
- [ ] **Use try/except for optional imports** if backward compatibility needed
- [ ] **Match __all__ to actual exports** - every item in `__all__` must be importable

### Naming Conflicts

- [ ] **Check for directory/file conflicts**: A directory `foo/` and file `foo.py` in same location causes circular imports
- [ ] **Resolution options**:
  1. Rename directory: `foo_modules/` or `foo_impl/`
  2. Rename file: `foo_main.py` or `_foo.py`
  3. Move file into directory as `__init__.py` (major refactor)

## Post-Decomposition Verification

### 1. Static Import Check
```bash
# Verify all imports in __init__.py exist
python -c "from MODULE import *"
```

### 2. __all__ Export Check
```python
import MODULE
for name in MODULE.__all__:
    assert hasattr(MODULE, name), f"Missing: {name}"
```

### 3. External Usage Check
```bash
# Find all files that import from this module
grep -r "from MODULE import" TRAINING/ CONFIG/ LIVE_TRADING/

# Test each unique import pattern
```

### 4. Contract Tests
```bash
python -m pytest TRAINING/contract_tests/ -v --tb=short
```

### 5. Smoke Test Critical Paths
```bash
# Test the main entry points still work
python -c "from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer"
python -c "from TRAINING.ranking.target_ranker import rank_targets"
python -c "from TRAINING.ranking.feature_selector import select_features_for_target"
```

## CRITICAL: Static Analysis is NOT Enough

**Lesson learned**: Reading files and checking if function names exist is NOT sufficient.
You MUST actually run Python imports to catch:
- Circular imports
- Missing dependencies
- Import order issues
- Runtime initialization errors

## Verification Script

Run this after any decomposition:

```bash
python -c "
import importlib
import sys

# Add all decomposed modules here
MODULES = [
    'TRAINING.ranking.predictability.model_evaluation',
    'TRAINING.orchestration.utils.diff_telemetry',
    'TRAINING.orchestration.intelligent_trainer',
    'TRAINING.ranking.multi_model_feature_selection.trainers',
    'TRAINING.orchestration.utils.repro_tracker_modules',
    'TRAINING.ranking.feature_selector_modules',
]

errors = []
for mod_path in MODULES:
    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, '__all__'):
            for name in mod.__all__:
                if not hasattr(mod, name):
                    errors.append(f'{mod_path}: __all__ contains {name} but not exportable')
        print(f'OK: {mod_path}')
    except Exception as e:
        errors.append(f'{mod_path}: {e}')
        print(f'FAIL: {mod_path}: {e}')

if errors:
    print(f'\nERRORS ({len(errors)}):')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('\nAll modules verified!')
"
```

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Import non-existent function | `ImportError: cannot import name 'X'` | Remove from import or implement function |
| Directory/file name conflict | `ImportError: circular import` | Rename directory with `_modules` suffix |
| Missing `__init__.py` | `ModuleNotFoundError` | Create `__init__.py` in subpackage |
| Relative import outside package | `ImportError: attempted relative import` | Use absolute imports |
| `__all__` lists non-existent name | No error until `from X import *` | Verify each `__all__` entry exists |

## Rollback Plan

If decomposition causes issues:

1. **Immediate**: `git checkout HEAD -- PATH/TO/MODULE.py`
2. **Full rollback**: `git revert COMMIT_HASH`
3. **Keep changes but fix**: Create missing functions as stubs that raise `NotImplementedError`
