# SST Compliance and Coding Standards

Guidelines for Single Source of Truth (SST), DRY (Don't Repeat Yourself), and coding standards.

## DRY Principle: Don't Repeat Yourself

### Core Rule

**Every piece of knowledge must have a single, unambiguous, authoritative representation.**

When you find duplicated code, consolidate it:

```python
# WRONG: Multiple implementations of same function
# File A: def load_data(): ...
# File B: def load_data(): ...  # Copy-paste
# File C: def load_data(): ...  # Another copy

# CORRECT: Single implementation, multiple imports
# TRAINING/data/loading/unified_loader.py
def load_data(): ...

# File A, B, C all import from same place
from TRAINING.data.loading import load_data
```

### When to Consolidate

| Indicator | Action |
|-----------|--------|
| Same function defined in 2+ files | Create shared module, all import from there |
| Copy-paste with minor tweaks | Extract common logic, parameterize differences |
| Similar patterns in 3+ places | Create utility/helper function |
| Config values repeated | Move to CONFIG/, use `get_cfg()` |

### Consolidation Checklist

Before creating a new utility:
1. **Search first**: `grep -r "def function_name" TRAINING/`
2. **Check existing modules**: `TRAINING/common/utils/`, `TRAINING/data/loading/`
3. **If exists**: Import and extend, don't duplicate
4. **If new**: Place in appropriate shared location

### Shared Module Locations

| Purpose | Location |
|---------|----------|
| Data loading | `TRAINING/data/loading/` |
| Config access | `CONFIG/config_loader.py` |
| Path utilities | `TRAINING/orchestration/utils/target_first_paths.py` |
| Common helpers | `TRAINING/common/utils/` |
| Type definitions | `TRAINING/common/types.py` |

### Deprecation Pattern for Consolidation

When consolidating duplicates:

```python
# Old file (now deprecated)
import warnings
from TRAINING.data.loading.unified_loader import load_data as _load_data

def load_data(*args, **kwargs):
    """Deprecated: Use TRAINING.data.loading.unified_loader.load_data instead."""
    warnings.warn(
        "load_data from this module is deprecated. "
        "Import from TRAINING.data.loading.unified_loader instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_data(*args, **kwargs)
```

## Finding SST Helpers (Use MCP Tools)

**Instead of reading docs, use the foxml-sst MCP server:**

```
# Find helpers for a task
mcp__foxml-sst__recommend_sst_helper(task_description="construct target directory path")

# Search helpers by keyword
mcp__foxml-sst__search_sst_helpers(query="config access")

# Get import path and examples for a specific helper
mcp__foxml-sst__get_sst_helper_details(helper_name="get_target_dir")

# Browse all categories
mcp__foxml-sst__list_sst_categories(include_subcategories=True)
```

## SST Hard Rules

### No Manual Path Joins

```python
# WRONG
path = os.path.join(output_dir, "targets", target_name)  # ❌
path = f"{output_dir}/targets/{target_name}"  # ❌

# CORRECT
from TRAINING.orchestration.utils.target_first_paths import run_root, get_target_dir

# run_root() requires output_dir argument (walks up to find run root)
root = run_root(output_dir)  # ✅
path = get_target_dir(root, target_name, model_family)  # ✅
```

### No Hardcoded Config Values

```python
# WRONG
threshold = 0.5  # ❌ hardcoded

# CORRECT
from CONFIG.config_loader import get_cfg
threshold = get_cfg("training.threshold", default=0.5)  # ✅
```

Default must match config file default exactly.

### SST Helpers

**Use MCP tools to look up helpers:** `mcp__foxml-sst__get_sst_helper_details(helper_name="...")`

Common helpers (get full list via `mcp__foxml-sst__list_sst_categories`):

| Purpose | Helper |
|---------|--------|
| Config access | `get_cfg()` |
| Run root path | `run_root(output_dir)` |
| Target directory | `get_target_dir(root, target, family)` |
| Scoped artifacts | `get_scoped_artifact_dir()` |
| Normalize target | `normalize_target_name()` |
| Normalize family | `normalize_family()` |
| Resolve horizon | `resolve_target_horizon_minutes()` |
| **Streaming concat** | `streaming_concat()` |

## Memory-Efficient DataFrame Operations (CRITICAL)

### No Direct pd.concat for Large Data

**WRONG - OOM risk on large universes (728+ symbols):**
```python
# ❌ This allocates all DataFrames in memory before concatenating
all_data = []
for symbol in symbols:
    df = load_data(symbol)
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)  # OOM on 728 symbols
```

**CORRECT - Use streaming_concat():**
```python
# ✅ Memory efficient - converts to lazy frames, releases as it goes
from TRAINING.data.loading import streaming_concat

# mtf_data is Dict[str, DataFrame] from UnifiedDataLoader
combined_lf = streaming_concat(
    mtf_data,
    symbol_column="symbol",
    target_column=target,  # Optional: skip symbols missing target
    # use_float32 reads from config: intelligent_training.lazy_loading.use_float32
    release_after_convert=True,
)

# Collect with streaming mode (processes in memory-efficient chunks)
combined_df = combined_lf.collect(streaming=True).to_pandas()
```

### When to Use streaming_concat()

| Scenario | Use streaming_concat? |
|----------|----------------------|
| Combining data from multiple symbols | ✅ YES |
| Concatenating mtf_data dict | ✅ YES |
| Cross-sectional data preparation | ✅ YES |
| Small utility DataFrames (< 100 rows) | ❌ No (overhead not worth it) |
| Appending to metrics/logging DataFrames | ❌ No (not on hot path) |
| Feature importance aggregation | ❌ No (small data) |

### Memory Budget Guidelines

```
728 symbols × 75k rows × 100 cols:
  - pd.concat:           ~132 GB peak (OOM on 128GB)
  - streaming_concat:    ~46 GB peak (fits)

Per-symbol memory estimate:
  - float64: rows × cols × 8 bytes
  - float32: rows × cols × 4 bytes (use_float32=True)
```

### Config Integration

streaming_concat reads from config when `use_float32=None`:
```yaml
# CONFIG/pipeline/pipeline.yaml
intelligent_training:
  lazy_loading:
    use_float32: true       # 50% memory reduction
    streaming_collect: true # Polars streaming mode
```

### Code Review Checklist for DataFrame Operations

When reviewing code, check for these patterns:

1. **Search for pd.concat**:
   ```bash
   grep -rn "pd\.concat" TRAINING/ --include="*.py"
   ```

2. **For each pd.concat, ask:**
   - Is this on the data loading hot path?
   - Could this receive data from 100+ symbols?
   - Is this inside a loop that accumulates DataFrames?

3. **If yes to any above**, refactor to use `streaming_concat()`

### Files Already Converted (Reference)

| File | Status |
|------|--------|
| `cross_sectional_data.py` | ✅ Uses streaming_concat |
| `data_preparation.py` | ✅ Uses streaming_concat (both paths) |
| `strategy_functions.py` | ✅ Uses streaming_concat |

### Files Still Needing Review

Run this to find remaining pd.concat calls:
```bash
grep -rn "pd\.concat" TRAINING/ --include="*.py" | grep -v __pycache__
```

## Function Signature Compatibility

When modifying functions for SST compliance:

**Reference**: `INTERNAL/docs/references/FUNCTION_SIGNATURE_CHANGES.md`

### Backward Compatibility Rules

1. Add new parameters as optional: `Optional[...] = None`
2. Preserve existing parameter names, types, and order
3. Document changes and migration path
4. For breaking changes: deprecate old function, create new one

### Impact Analysis Required

Before changing any function signature:

1. **Find all call sites**: direct calls, imports, method calls, dynamic sites
2. **Trace dependency chain**: direct + indirect callers, wrappers
3. **Check wrapper threading**: decorators, class methods, partials must pass new params
4. **Check dynamic sites**: config-driven refs, callbacks, partials/lambdas, dataclass defaults
5. **Verify**: all call sites still work after changes

## Serialization Standards

### Exact JSON Knobs (determinism-critical)

```python
json.dumps(
    data,
    sort_keys=True,           # Required
    separators=(',', ':'),    # Compact (or (',', ': ') for pretty)
    ensure_ascii=False,       # UTF-8 support
    allow_nan=False,          # Reject NaN/Inf in hashed artifacts
)
# + newline at EOF
# + UTF-8 encoding
```

### Exact YAML Knobs (best-effort determinism)

```python
yaml.safe_dump(
    data,
    sort_keys=True,           # Required
    default_flow_style=False,
    allow_unicode=True,
    width=10_000,             # Avoid line-wrap drift
    indent=2,
    line_break="\n",
)
# + newline at EOF
```

**Note**: YAML determinism is fragile across library versions. Prefer JSON for hashed artifacts.

### Atomic Write Contract

1. Write to temp file in **same directory** (ensures atomic `os.replace()`)
2. `flush()` + `os.fsync(file_fd)`
3. `os.replace(tmp, final)` (not `rename()`)
4. `os.fsync(dir_fd)` on parent directory
5. Consistent encoding: UTF-8, `\n` at EOF

## Documentation Placement

| Type | Location |
|------|----------|
| Internal (audits, dev refs, analysis) | `INTERNAL/docs/` |
| Public (user guides, API refs) | `DOCS/` |
| Root-level `.md` | Only `README.md`, `CHANGELOG.md`, `SUPPORT.md` |

Never reference `INTERNAL/docs/` paths from public `DOCS/` files.

## Error Handling

### Typed Exceptions

```python
from TRAINING.common.exceptions import (
    ConfigError,
    DataIntegrityError,
    LeakageError,
    ArtifactError,
    StageBoundaryError,
)
```

### Centralized Policy

```python
from TRAINING.common.exceptions import should_fail_closed, handle_error_with_policy

# Check if should fail
if should_fail_closed(stage="TRAINING", error_type="IO_ERROR", affects_artifact=True):
    raise ArtifactError(...)

# Or use handler
result = handle_error_with_policy(error, stage, error_type, affects_artifact, fallback)
```

### Fail-Closed Decision Table

| Code Path | Strict Mode | Best-Effort |
|-----------|-------------|-------------|
| Artifacts/manifests/routing | Raise | Raise |
| Optional diagnostics | Warn | Warn |

### Structured Error Payload

Errors must carry: `run_id`, `stage`, `target_id`, `artifact_path`, `error_code`, `context` dict.

## Artifact-Generating Code Boundary

Code that writes files that are:
- Compared, hashed, signed, or diffed
- Used as run identity/manifest inputs

**Non-artifact diagnostic output** may be non-canonical if clearly labeled.

**Ordering policy**: Lexicographic by key/path. No "first match wins" unless documented.

**When NOT to fix dict iteration**: If it's just counting (order irrelevant), fixing is noise. Only fix if iteration feeds serialization/log output.

## Import Patterns and Circular Import Avoidance

### Import Order

Follow standard Python import ordering:
1. Standard library imports
2. Third-party imports (numpy, pandas, etc.)
3. Local application imports

Within each group, sort alphabetically. Separate groups with blank lines.

### Circular Import Prevention

**The Problem**: Module A imports from Module B, and Module B imports from Module A. Python fails at import time.

**Prevention Strategies**:

1. **TYPE_CHECKING guard for type hints only**:
   ```python
   from __future__ import annotations
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer

   def process(trainer: "IntelligentTrainer") -> None:  # String annotation
       ...
   ```

2. **Late/local imports for runtime needs**:
   ```python
   def get_trainer_instance():
       # Import at call time, not module load time
       from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
       return IntelligentTrainer()
   ```

3. **Extract shared types to a separate module**:
   ```python
   # TRAINING/common/types.py - no imports from other project modules
   from dataclasses import dataclass
   from typing import Dict, List

   @dataclass
   class FeatureSelectionResult:
       features: List[str]
       scores: Dict[str, float]
   ```

4. **Use protocols/ABCs instead of concrete types**:
   ```python
   from typing import Protocol

   class TrainerProtocol(Protocol):
       def train(self, data: Any) -> Any: ...

   def process(trainer: TrainerProtocol) -> None:  # No import needed
       trainer.train(data)
   ```

### Common Circular Import Patterns to Avoid

| Pattern | Problem | Fix |
|---------|---------|-----|
| Config module imports trainer which imports config | Load-time cycle | Config module should be leaf (no project imports) |
| Type hints requiring concrete class imports | Import cycle for annotations | Use `TYPE_CHECKING` guard |
| Helper module imports orchestrator | Inverted dependency | Orchestrator should import helper, not vice versa |
| `__init__.py` re-exports cause cycles | Import order sensitivity | Minimize `__init__.py` re-exports, use explicit imports |

### Bootstrap Import Order (Determinism-Critical)

The `repro_bootstrap` module MUST be imported before numpy/pandas to ensure deterministic behavior:

```python
# CORRECT - bootstrap first
import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first
import numpy as np
import pandas as pd

# WRONG - numpy before bootstrap
import numpy as np  # ❌ Seeds not set yet!
import TRAINING.common.repro_bootstrap
```

**Never import numpy/pandas in a module that `repro_bootstrap` might import** - this would break the bootstrap order.

### Dependency Direction

Maintain a clear dependency hierarchy (lower layers never import from higher):

```
CONFIG/              ← Leaf layer (no project imports except config_loader)
TRAINING/common/     ← Foundation (imports CONFIG only)
TRAINING/data/       ← Data layer (imports common, CONFIG)
TRAINING/models/     ← Model layer (imports data, common, CONFIG)
TRAINING/ranking/    ← Ranking layer (imports models, data, common, CONFIG)
TRAINING/orchestration/ ← Top layer (imports everything)
```

**Rule**: If you're in a lower layer and need something from a higher layer, refactor to push the shared code down.

## Related Skills

- `determinism-and-reproducibility.md` - Determinism patterns and bootstrap
- `configuration-management.md` - Config access patterns
- `skill-maintenance.md` - When to update skills (rarely needed)

## Quick Reference: SST + DRY Rules

| Principle | Rule | Check |
|-----------|------|-------|
| **SST** | One source for each value/path/config | Is there a helper for this? |
| **DRY** | One implementation for each behavior | Does this already exist somewhere? |
| **Consolidate** | When 2+ files have same function | Search before creating |
| **Deprecate** | Don't delete, redirect with warning | Backward compat preserved |
