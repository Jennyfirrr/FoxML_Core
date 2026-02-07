# Determinism and Reproducibility

Skill for ensuring bitwise reproducible results across pipeline runs.

## Core Principle

Same inputs MUST produce byte-identical outputs across runs. This is critical for:
- Research reproducibility
- Debugging (compare runs to isolate changes)
- Audit trails and compliance
- Testing (deterministic tests are reliable tests)

## Bootstrap Requirement

**CRITICAL**: Entry points MUST import `repro_bootstrap` BEFORE any ML libraries.

```python
# CORRECT - At the TOP of your entry point
import TRAINING.common.repro_bootstrap  # noqa: F401 - MUST be first
import numpy as np  # Now safe to import
import pandas as pd
# ... rest of imports
```

### Why This Matters

The bootstrap module:
1. Sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc. BEFORE numpy initializes
2. Validates `PYTHONHASHSEED` is set (required for dict ordering)
3. In strict mode: hard-fails if ML libs already imported
4. In strict mode: forces single-threaded execution

### What Happens If Imported Too Late

```
🚨 STRICT MODE HARD-FAIL: Bootstrap imported too late!
   Already imported: ['numpy', 'pandas']
   repro_bootstrap must be imported BEFORE any numeric libraries.
```

## Strict vs Best-Effort Modes

| Aspect | Strict Mode | Best-Effort Mode |
|--------|-------------|------------------|
| Environment | `REPRO_MODE=strict` | `REPRO_MODE=best_effort` (default) |
| Threading | Single-threaded (OMP=1) | Multi-threaded |
| PYTHONHASHSEED | Required (hard-fail if missing) | Warned if missing |
| Bootstrap order | Required (hard-fail) | Warned only |
| Run ID | Derived from identity | Can use timestamp fallback |
| Use case | Reproducibility verification | Normal development |

### Running in Strict Mode

**IMPORTANT**: For production/deterministic training, ALWAYS use `bin/run_deterministic.sh`:

```bash
# Standard deterministic training with experiment config
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
  --experiment-config production_baseline \
  --output-dir TRAINING/results/prod_run

# Alternative: using -m module syntax
bin/run_deterministic.sh python -m TRAINING.orchestration.intelligent_trainer \
  --experiment-config production_baseline \
  --output-dir TRAINING/results/prod_run

# Minimal run (no experiment config)
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
  --output-dir my_run
```

The `run_deterministic.sh` script sets all required environment variables:
- `REPRO_MODE=strict`
- `PYTHONHASHSEED=42`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.

**Manual setup (NOT recommended):**
```bash
export REPRO_MODE=strict
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m TRAINING.orchestration.intelligent_trainer --output-dir my_run
```

## Deterministic Ordering Helpers

### Dictionary Iteration

```python
from TRAINING.common.utils.determinism_ordering import sorted_items, sorted_keys

# WRONG - insertion order varies
for k, v in my_dict.items():  # ❌
    process(k, v)

# CORRECT - lexicographic order
for k, v in sorted_items(my_dict):  # ✅
    process(k, v)

# Also CORRECT for keys-then-values
for k in sorted_keys(my_dict):  # ✅
    v = my_dict[k]
    process(k, v)
```

### Filesystem Enumeration

```python
from TRAINING.common.utils.determinism_ordering import (
    iterdir_sorted,
    glob_sorted,
    rglob_sorted,
)

# WRONG - filesystem order is non-deterministic
for path in directory.iterdir():  # ❌
    process(path)

# CORRECT - sorted order
for path in iterdir_sorted(directory):  # ✅
    process(path)

# Glob patterns
matches = glob_sorted(directory, "*.json")  # ✅
matches = rglob_sorted(directory, "**/*.yaml")  # ✅
```

### Set Operations

```python
# WRONG - set iteration order varies
for item in my_set:  # ❌
    process(item)

# CORRECT - sort immediately
for item in sorted(my_set):  # ✅
    process(item)
```

## Atomic Write Operations

Artifact files MUST use atomic writes to prevent corruption:

```python
from TRAINING.common.utils.file_utils import write_atomic_json, write_atomic_yaml

# WRONG - can corrupt on crash
with open(path, "w") as f:  # ❌
    json.dump(data, f)

# CORRECT - atomic write
write_atomic_json(path, data)  # ✅
write_atomic_yaml(path, data)  # ✅
```

### Canonical Serialization (for Hashing)

When creating signatures or comparing artifacts:

```python
from TRAINING.common.utils.determinism_serialization import canonical_json, canonical_yaml

# For hashing/signatures
json_str = canonical_json(data)
hash_value = hashlib.sha256(json_str.encode()).hexdigest()

# Canonical JSON knobs (used automatically):
# - sort_keys=True
# - stable separators
# - UTF-8 encoding
# - newline at EOF
```

## Run Identity System

### Strict Mode: Derived from Identity

```python
from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity

# Format: run_{strict_key[:12]}_{replicate_key[:12]}
run_id = derive_run_id_from_identity(run_identity=identity)
# e.g., "run_abc123def456_xyz789ghi012"
```

### Keys Explained

| Key | Purpose | Source |
|-----|---------|--------|
| `strict_key` | Identifies config/code state | Hash of config + code fingerprints |
| `replicate_key` | Distinguishes replicates | Hash of data fingerprint |

### Best-Effort Mode: Fallbacks Allowed

In non-strict mode, run_id can fall back to:
1. Provided `fallback_run_id`
2. Directory-derived ID from `output_dir`
3. Timestamp-based ID (last resort)

## Error Handling Policy (Reproducibility)

**See `sst-and-coding-standards.md`** for typed exceptions and full API.

For reproducibility, the key principle is **fail-closed for artifacts**:

```python
from TRAINING.common.exceptions import handle_error_with_policy

# Anything affecting artifacts must raise (not silently fail)
result = handle_error_with_policy(
    error=e,
    stage="TRAINING",
    error_type="model_fit",
    affects_artifact=True,  # ← Key: this forces raise in strict mode
    fallback_value=None,
)
```

### Policy Rules (Reproducibility-Critical)

| Condition | Strict Mode | Best-Effort Mode |
|-----------|-------------|------------------|
| Affects artifacts | Raise | Raise |
| Affects routing | Raise | Warn + continue |
| Affects manifest | Raise | Raise |
| Optional metadata | Warn | Warn |

## Contract Test Verification

```bash
# Run determinism verification tests
pytest TRAINING/contract_tests/ -v -k determinism

# Manual verification script
bash bin/check_determinism_patterns.sh

# Verify bootstrap is correct
python bin/verify_determinism_init.py
```

## Threading Safety

### Re-entrant Locks for Nested Contexts

If a function acquires a lock and then calls another function that also needs the same lock, use `threading.RLock()` instead of `threading.Lock()`:

```python
import threading

# WRONG - deadlock if same thread tries to acquire twice
_MY_LOCK = threading.Lock()  # ❌

# CORRECT - allows re-entry by same thread
_MY_LOCK = threading.RLock()  # ✅

def outer():
    with _MY_LOCK:
        inner()  # Would deadlock with Lock(), works with RLock()

def inner():
    with _MY_LOCK:
        do_work()
```

**Real example:** `feature_registry._REGISTRY_LOCK` must be `RLock()` because:
1. `get_registry()` holds the lock while creating `FeatureRegistry()`
2. `FeatureRegistry.__init__()` calls `_load_config()`
3. `_load_config()` also needs the lock for logging deduplication

### Multiprocessing Start Method

Use 'spawn' instead of 'fork' to avoid deadlocks:

```python
import multiprocessing

# Set BEFORE any ProcessPoolExecutor usage
multiprocessing.set_start_method('spawn')
```

**Why:** On Linux, 'fork' copies the parent's lock state. If a lock is held at fork time, child processes inherit a "held" lock they can never release → deadlock.

## Anti-Patterns

| Anti-Pattern | Why It Breaks Determinism | Fix |
|--------------|---------------------------|-----|
| `d.items()` in artifact code | Insertion order varies | `sorted_items(d)` |
| `path.iterdir()` | Filesystem order varies | `iterdir_sorted(path)` |
| `datetime.now()` for run_id | Timestamps are non-deterministic | `derive_run_id_from_identity()` |
| `uuid.uuid4()` | Random by design | Derive from identity |
| `json.dump()` without sort_keys | Key order varies | `write_atomic_json()` |
| Global `np.random.seed()` | Affects other code | Per-instance seeds |
| `random.choice()` unseeded | Non-deterministic | Use seeded RNG |
| `threading.Lock()` with nested calls | Deadlock on re-entry | Use `threading.RLock()` |
| `ProcessPoolExecutor` with fork | Inherited lock deadlocks | Use 'spawn' start method |

## Debugging Non-Determinism

### Step 1: Enable Strict Mode

```bash
REPRO_MODE=strict bash bin/run_deterministic.sh ...
```

### Step 2: Compare Artifacts

```bash
# Compare two run directories
diff -r RESULTS/runs/run1/ RESULTS/runs/run2/
```

### Step 3: Check Fingerprints

```python
# Fingerprints are stored in reproducibility directories
cat RESULTS/runs/{run_id}/targets/{target}/reproducibility/fingerprints.json
```

### Step 4: Grep for Anti-Patterns

```bash
# Find dict iteration
rg "\.items\(|\.values\(|\.keys\(" --type py TRAINING/

# Find filesystem enumeration
rg "\.iterdir\(|\.glob\(|\.rglob\(|os\.listdir\(" --type py TRAINING/

# Find timestamp usage
rg "datetime\.now\(|time\.time\(|uuid\.uuid4\(" --type py TRAINING/
```

## Exemptions

When determinism cannot be achieved, document with:

```python
# DETERMINISM-EXEMPT: <reason> (ticket/issue-id)
timestamp = datetime.now()  # Required for external API
```

Exemptions must:
1. Be annotated with the comment pattern above
2. Be tested to prove they don't affect hashed/signed artifacts
3. Have clear justification

## Related Skills

- `sst-and-coding-standards.md` - SST compliance patterns
- `configuration-management.md` - Config access patterns
- `testing-guide.md` - Determinism test patterns

## Related Documentation

- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Complete pattern catalog
- `TRAINING/common/repro_bootstrap.py` - Bootstrap implementation
- `TRAINING/common/utils/determinism_ordering.py` - Ordering helpers
- `TRAINING/common/utils/determinism_serialization.py` - Serialization helpers
- `DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md` - User guide
