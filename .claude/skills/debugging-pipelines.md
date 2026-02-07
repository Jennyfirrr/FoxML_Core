# Debugging Pipelines

Skill for diagnosing and fixing pipeline issues systematically.

## Inspecting Runs (Use MCP Tools)

**Instead of manually reading artifact files, use the foxml-artifact MCP server:**

```
# List recent runs (with filters)
mcp__foxml-artifact__query_runs(experiment_name="determinism_test", limit=10)

# Get full details for a run (manifest, config, targets)
mcp__foxml-artifact__get_run_details(run_id="rid_...", include_config=True)

# Compare two runs (config diff, target diff)
mcp__foxml-artifact__compare_runs(run_id_1="rid_...", run_id_2="rid_...")

# Get model metrics for a target
mcp__foxml-artifact__get_model_metrics(run_id="rid_...", target="fwd_ret_10m")

# View stage progression for a target
mcp__foxml-artifact__get_target_stage_history(run_id="rid_...", target="fwd_ret_10m")

# Diff target results between runs
mcp__foxml-artifact__diff_target_results(run_id_1="rid_...", run_id_2="rid_...", target="fwd_ret_10m")
```

## Pipeline Entry Points

| Command | Purpose |
|---------|---------|
| `python -m TRAINING.orchestration.intelligent_trainer --output-dir run1` | Standard run |
| `bash bin/run_deterministic.sh -m TRAINING.orchestration.intelligent_trainer` | Strict mode |
| `pytest TRAINING/contract_tests/ -v` | Contract tests |

## Log Levels and Output

### Setting Log Levels

```bash
# Via environment
export LOGLEVEL=DEBUG
python -m TRAINING.orchestration.intelligent_trainer ...

# Via Python
import logging
logging.getLogger("TRAINING").setLevel(logging.DEBUG)
```

### Log Locations

| Log Type | Location |
|----------|----------|
| Console output | stdout/stderr |
| Run logs | `RESULTS/runs/{run_id}/logs/` |
| Stage logs | `RESULTS/runs/{run_id}/targets/{target}/logs/` |

## Common Failure Modes

### 1. Bootstrap Not Imported First

**Symptom:**
```
🚨 STRICT MODE HARD-FAIL: Bootstrap imported too late!
   Already imported: ['numpy', 'pandas']
```

**Cause:** ML libraries imported before `repro_bootstrap`.

**Fix:** Move bootstrap import to top of entry point:
```python
import TRAINING.common.repro_bootstrap  # noqa: F401 - MUST be first
import numpy as np  # Now safe
```

**SST Note:** This is a determinism requirement. Bootstrap sets thread environment variables before MKL/OpenMP initialize.

### 2. Config Not Found

**Symptom:**
```
Config file not found: CONFIG/models/my_model.yaml
```

**Cause:** Missing config file or wrong config name.

**Fix:**
1. Check config file exists in expected location
2. Verify config name matches (see `CONFIG/config_loader.py` mappings)
3. Use `get_cfg()` with explicit `config_name` parameter

```python
# Check which config would be loaded
from CONFIG.config_loader import get_config_path
path = get_config_path("my_config")
print(f"Would load: {path}, exists: {path.exists()}")
```

### 3. Memory Exhaustion

**Symptom:**
```
MemoryError: Unable to allocate X GiB
# or
Killed (OOM killer)
```

**Cause:** Data too large for available RAM.

**Diagnosis:**
```python
from TRAINING.common.memory.memory_manager import MemoryManager
mm = MemoryManager()
print(mm.get_memory_usage())
# {'rss_gb': 12.5, 'system_available_gb': 2.1, 'system_percent': 85.0}
```

**Fix:**
1. Enable chunked processing in config:
   ```yaml
   # CONFIG/pipeline/memory.yaml
   memory:
     chunking:
       enabled: true
       chunk_size: 500000  # Reduce if needed
   ```
2. Use memory cleanup:
   ```python
   mm.cleanup_memory()  # Force garbage collection
   ```
3. Reduce parallel workers or batch sizes

### 4. Leakage Detection Failures

**Symptom:**
```
LeakageError: Feature 'price_close' has lookahead bias
   horizon_minutes: 10
   lag_bars: 0
```

**Cause:** Feature uses future data relative to target horizon.

**Diagnosis:**
```python
from TRAINING.ranking.predictability.leakage_detection import analyze_feature_leakage

result = analyze_feature_leakage(
    feature_name="price_close",
    target_horizon_minutes=10,
    feature_registry=registry,
)
print(result)  # Shows why leakage detected
```

**Fix:**
1. Add appropriate `lag_bars` to feature definition
2. Exclude feature for this target horizon
3. Update feature registry with correct `allowed_horizons`

See: `.claude/skills/feature-engineering.md` for leakage rules

### 5. Stage Boundary Errors

**Symptom:**
```
StageBoundaryError: Stage input validation failed
   boundary_type: input
   feature_set_hash: abc123...
```

**Cause:** Features selected in Stage 2 don't match features available in Stage 3.

**Diagnosis:**
1. Check feature selection artifacts:
   ```bash
   cat RESULTS/runs/{run_id}/targets/{target}/feature_selection/selected_features.json
   ```
2. Compare with training data columns

**Fix:**
- Ensure feature registry is consistent
- Re-run feature selection if data changed

### 6. Non-Deterministic Results

**Symptom:** Two runs with same inputs produce different outputs.

**Diagnosis:**
```bash
# Enable strict mode
REPRO_MODE=strict bash bin/run_deterministic.sh ...

# Compare artifacts
diff -r RESULTS/runs/run1/ RESULTS/runs/run2/

# Check for anti-patterns
rg "\.items\(|\.values\(" --type py TRAINING/
rg "datetime\.now\(|time\.time\(" --type py TRAINING/
```

**Fix:** See `.claude/skills/determinism-and-reproducibility.md`

### 7. Import Errors / Circular Imports

**Symptom:**
```
ImportError: cannot import name 'X' from partially initialized module
```

**Cause:** Circular import dependency.

**Diagnosis:**
```bash
# Check import order
python -c "import TRAINING.orchestration.intelligent_trainer"
```

**Fix:**
1. Use lazy imports inside functions:
   ```python
   def my_function():
       # CIRCULAR-IMPORT: Lazy import to avoid cycle
       from TRAINING.orchestration.utils.run_context import get_current_stage
       return get_current_stage()
   ```
2. Check import hierarchy in `INTERNAL/docs/references/SST_SOLUTIONS.md`

## Memory Debugging

### MemoryManager Usage

```python
from TRAINING.common.memory.memory_manager import MemoryManager

mm = MemoryManager()

# Check current usage
usage = mm.get_memory_usage()
print(f"Using {usage['rss_gb']:.1f} GB, {usage['system_percent']:.0f}% system")

# Monitor during operation
with mm.monitor_operation("training"):
    train_model(X, y)

# Force cleanup
mm.cleanup_memory()
```

### Finding Memory Leaks

```python
import tracemalloc

tracemalloc.start()
# ... run suspect code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

## Leakage Debugging Workflow

### Step 1: Identify Flagged Features

```python
from TRAINING.ranking.predictability.leakage_detection import run_leakage_detection

results = run_leakage_detection(
    features=feature_list,
    target="fwd_ret_10m",
    feature_registry=registry,
)

for feature, status in results.items():
    if status['leakage_detected']:
        print(f"{feature}: {status['reason']}")
```

### Step 2: Check Feature Registry

```yaml
# CONFIG/data/feature_registry.yaml
features:
  price_close:
    lag_bars: 1           # Must be > 0 for lookahead safety
    allowed_horizons:
      - 30m               # Only safe for 30m+ horizons
      - 1h
```

### Step 3: Update or Exclude

```python
# Option A: Add to exclusions
# CONFIG/data/excluded_features.yaml
excluded_features:
  - price_close  # Leakage risk

# Option B: Update registry with correct lag
# CONFIG/data/feature_registry.yaml
features:
  price_close:
    lag_bars: 3  # Now safe for 10m horizons
```

## Error Handling Policy

**See `sst-and-coding-standards.md`** for the authoritative reference on:
- Typed exceptions (`ConfigError`, `LeakageError`, `ArtifactError`, etc.)
- Fail-closed decision table
- `handle_error_with_policy()` usage

**Quick debugging pattern:**

```python
# Exceptions carry structured context for debugging
except LeakageError as e:
    print(f"Feature: {e.feature_name}, Horizon: {e.horizon_minutes}")
    print(f"Full context: {e.to_dict()}")  # All structured fields
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.getLogger("TRAINING.ranking").setLevel(logging.DEBUG)
logging.getLogger("TRAINING.model_fun").setLevel(logging.DEBUG)
```

### Inspect Intermediate Artifacts

**Use MCP tools (preferred):**
```
mcp__foxml-artifact__get_run_details(run_id="rid_...", include_targets=True)
mcp__foxml-artifact__get_model_metrics(run_id="rid_...", target="fwd_ret_10m")
mcp__foxml-artifact__get_target_stage_history(run_id="rid_...", target="fwd_ret_10m")
```

**Or manually via bash:**
```bash
# Feature selection results
cat RESULTS/runs/{run_id}/targets/{target}/feature_selection/selected_features.json

# Reproducibility fingerprints
cat RESULTS/runs/{run_id}/targets/{target}/reproducibility/fingerprints.json

# Model metrics
cat RESULTS/runs/{run_id}/targets/{target}/metrics/summary.json
```

### Run Single Stage

```python
# Run just target ranking
from TRAINING.ranking.target_ranker import TargetRanker
ranker = TargetRanker(config)
results = ranker.rank_targets(data)
```

## Skill Updates

This skill should be updated when:
- New error types are added to exceptions.py
- New common failure patterns are discovered
- Memory management patterns change
- Debugging tools or commands are added
- Leakage detection logic changes

## Related Documentation

- **MCP Tools (preferred)**: Use `mcp__foxml-artifact__*` tools for run inspection
- `TRAINING/common/exceptions.py` - Exception definitions
- `TRAINING/common/memory/memory_manager.py` - Memory management
- `TRAINING/ranking/predictability/leakage_detection.py` - Leakage detection
- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Determinism debugging
