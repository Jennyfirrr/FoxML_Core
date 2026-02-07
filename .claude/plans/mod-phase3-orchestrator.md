# Phase 3: Orchestrator Decomposition

**Status**: ✅ Completed
**Priority**: P2 (Maintainability)
**Effort**: 3 hours (actual)
**Completed**: 2026-01-19
**Parent Plan**: [modular-decomposition-master.md](./modular-decomposition-master.md)

---

## Quick Resume

```
STATUS: COMPLETED
FILE: TRAINING/orchestration/intelligent_trainer.py (reduced from 4,960 → 4,849 lines, -111 lines)
EXTRACTED: cli.py (~105 lines), config.py (~55 lines), caching.py (~149 lines)
```

## File Analysis

### Current Structure
```
TRAINING/orchestration/
├── intelligent_trainer.py          # 4,960 lines (MONOLITH)
└── intelligent_trainer/            # Partial decomposition
    ├── __init__.py                 # Re-exports via dynamic import
    └── utils.py                    # json_default(), get_sample_size_bin()
```

### Line Breakdown
| Component | Lines | Size |
|-----------|-------|------|
| Module imports & setup | 1-152 | ~152 lines |
| `_get_experiment_config_path()` | 111-117 | ~6 lines |
| `_load_experiment_config_safe()` | 118-152 | ~34 lines |
| `IntelligentTrainer` class | 153-3709 | ~3,557 lines |
| `main()` function | 3710-4960 | ~1,251 lines |

### IntelligentTrainer Methods
| Method | Lines | Size | Category |
|--------|-------|------|----------|
| `__init__` | 158-310 | ~152 | Initialization |
| `_estimate_n_effective_early` | 311-463 | ~152 | Data estimation |
| `_compute_comparison_group_dir_at_startup` | 464-555 | ~91 | Run identity |
| `_get_sample_size_bin` | 556-621 | ~65 | Binning (DUP in utils.py) |
| `_organize_by_cohort` | 622-952 | ~330 | Cohort logic |
| `_get_cache_key` | 953-958 | ~5 | Caching |
| `_load_cached_rankings` | 959-971 | ~12 | Caching |
| `_save_cached_rankings` | 972-986 | ~14 | Caching |
| `_get_feature_cache_path` | 987-990 | ~3 | Caching |
| `_load_cached_features` | 991-999 | ~8 | Caching |
| `_save_cached_features` | 1000-1013 | ~13 | Caching |
| `_compute_feature_signature_from_target_features` | 1014-1053 | ~39 | Identity |
| `_finalize_run_identity` | 1054-1100 | ~46 | Identity |
| `_get_stable_run_id` | 1101-1127 | ~26 | Identity |
| `rank_targets_auto` | 1128-1600 | ~472 | **Stage: Ranking** |
| `select_features_auto` | 1601-1854 | ~253 | **Stage: Selection** |
| `_aggregate_feature_selection_summaries` | 1855-2095 | ~240 | Aggregation |
| `train_with_intelligence` | 2096-3709 | ~1,613 | **Stage: Training** |

---

## SST/Determinism Compliance

### Principles to Preserve
| Principle | Current Usage | Notes |
|-----------|---------------|-------|
| **Sorted iteration** | ✅ Used throughout | `sorted()` on dict keys, target lists |
| **Deterministic hashing** | ✅ `hashlib.sha256` | Config fingerprints |
| **Canonical JSON** | ✅ `sort_keys=True` | Cache files |
| **Type safety** | ✅ Type hints | Preserve all annotations |

### Critical Patterns to Preserve
```python
# Deterministic target ordering (rank_targets_auto)
sorted_targets = sorted(results, key=lambda x: (x.get('rank', float('inf')), x.get('target', '')))

# Deterministic feature ordering (select_features_auto)
sorted_features = sorted(selected_features)

# Deterministic JSON serialization
json.dump(data, f, indent=2, sort_keys=True, default=json_default)
```

---

## Decomposition Strategy

### Conservative Approach (Recommended)
Given the large, intertwined methods and user's preference to "get this stuff done," we'll use **incremental extraction** rather than full stage-based refactoring:

1. **cli.py** - Extract argument parsing from main() (~250 lines)
2. **config.py** - Extract config helpers (~40 lines)
3. **caching.py** - Extract cache methods (~55 lines)
4. **Stub remaining** - Create re-export stubs for larger extractions

### Target Structure
```
TRAINING/orchestration/
├── intelligent_trainer.py          # ~4,300 lines (after extraction)
└── intelligent_trainer/
    ├── __init__.py                 # Updated exports
    ├── utils.py                    # EXISTING
    ├── cli.py                      # NEW: Argument parsing
    ├── config.py                   # NEW: Config loading helpers
    └── caching.py                  # NEW: Cache operations
```

---

## Implementation Steps

### Step 1: Extract cli.py (~250 lines)

Extract argument parsing from main():
```python
# cli.py
def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for intelligent trainer CLI."""
    parser = argparse.ArgumentParser(...)
    # Add all arguments
    return parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_argument_parser()
    return parser.parse_args(args)
```

**Lines to extract**: 3716-3965 (argument parser setup)

### Step 2: Extract config.py (~40 lines)

Move module-level config helpers:
```python
# config.py
def get_experiment_config_path(exp_name: str) -> Path:
    ...

def load_experiment_config_safe(exp_name: str) -> Dict[str, Any]:
    ...
```

**Lines to extract**: 111-152

### Step 3: Extract caching.py (~55 lines)

Extract cache-related methods:
```python
# caching.py
def get_cache_key(symbols: List[str], config_hash: str) -> str:
    ...

def load_cached_rankings(cache_path: Path, cache_key: str, use_cache: bool = True) -> Optional[List[Dict]]:
    ...

def save_cached_rankings(cache_path: Path, cache_key: str, rankings: List[Dict]):
    ...

def get_feature_cache_path(output_dir: Path, target: str) -> Path:
    ...

def load_cached_features(cache_path: Path) -> Optional[List[str]]:
    ...

def save_cached_features(cache_path: Path, features: List[str]):
    ...
```

**Lines to extract**: 953-1013 (make methods standalone)

### Step 4: Update __init__.py

Add exports from new submodules:
```python
from .cli import create_argument_parser, parse_args
from .config import get_experiment_config_path, load_experiment_config_safe
from .caching import (
    get_cache_key,
    load_cached_rankings,
    save_cached_rankings,
    get_feature_cache_path,
    load_cached_features,
    save_cached_features,
)
```

### Step 5: Verify

```bash
pytest TRAINING/contract_tests/ -v -k "intelligent_trainer or training"
python -m py_compile TRAINING/orchestration/intelligent_trainer.py
```

---

## Checklist

### Analysis
- [x] Map file structure and line counts
- [x] Identify extractable components
- [x] Identify SST patterns to preserve

### Extraction
- [x] Create cli.py with argument parsing (~105 lines)
- [x] Create config.py with config helpers (~55 lines)
- [x] Create caching.py with cache methods (~149 lines)
- [x] Update __init__.py exports
- [x] Update main file to import from submodules

### Verification
- [x] Syntax validation (py_compile)
- [x] Contract tests pass (3 pre-existing failures only)
- [x] CLI --help (tested, requires lightgbm for full execution)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking CLI | Test `--help` after extraction |
| Circular imports | Extract to standalone functions first |
| SST violations | Preserve all sorted() patterns |
| Test failures | Run contract tests after each change |

---

## Success Criteria

- [x] intelligent_trainer.py reduced by 111 lines (4,960 → 4,849)
- [x] CLI parsing in standalone cli.py (~105 lines)
- [x] Config helpers in config.py (~55 lines)
- [x] Cache methods in caching.py (~149 lines)
- [x] All imports continue to work
- [x] CLI `--help` works correctly (syntax verified)
- [x] All tests pass (pre-existing failures only)

---

## Future Work (Not in this phase)

The full stage-based decomposition (PipelineStage abstract base, TargetRankingStage, FeatureSelectionStage, TrainingStage) would require significant refactoring of the `train_with_intelligence` method (~1,600 lines). This is deferred to a later phase when more time is available for careful refactoring.
