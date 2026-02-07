# Configuration Management

Skill for working safely with the CONFIG system and maintaining SST compliance.

## Querying Config (Use MCP Tools)

**Instead of reading config files directly, use the foxml-config MCP server:**

```
# Get any config value by path
mcp__foxml-config__get_config_value(path="pipeline.determinism.base_seed")

# List available keys under a prefix
mcp__foxml-config__list_config_keys(prefix="pipeline.determinism")

# See which config layer provides a value (precedence chain)
mcp__foxml-config__show_config_precedence(path="training.batch_size")

# Load experiment config with overrides
mcp__foxml-config__load_experiment_config(experiment_name="e2e_full_targets_test")

# List all available config files
mcp__foxml-config__list_available_configs()
```

## Core Principle: Single Source of Truth (SST)

Configuration values must come from ONE place - the CONFIG system. Never:
- Hardcode configuration values in code
- Use `config.get()` on loaded dicts (use `get_cfg()` instead)
- Construct paths manually (use SST path helpers)
- Define custom precedence logic (it's automatic)

## Config Precedence (Highest to Lowest)

```
1. CLI arguments           ← Highest priority
2. Experiment config       ← CONFIG/experiments/{name}.yaml
3. Intelligent training    ← CONFIG/pipeline/training/intelligent.yaml
4. Pipeline configs        ← CONFIG/pipeline/training/*.yaml
5. Defaults               ← CONFIG/defaults.yaml (lowest)
```

## Using `get_cfg()`

### Basic Usage

```python
from CONFIG.config_loader import get_cfg

# Access nested config with dot notation
base_seed = get_cfg("pipeline.determinism.base_seed", default=42)
test_size = get_cfg("preprocessing.validation.test_size", default=0.2)

# Specify which config file to load from
safety_val = get_cfg("safety.gradient_clipping.clipnorm",
                     default=1.0,
                     config_name="safety_config")
```

### SST Default Rule

**Critical**: The `default` parameter MUST match the config file default exactly. This ensures determinism - code behaves identically whether config is loaded or fallback is used.

```python
# GOOD: default matches CONFIG/pipeline/pipeline.yaml
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=7200)

# BAD: default doesn't match config file
timeout = get_cfg("pipeline.isolation_timeout_seconds", default=3600)  # WRONG!
```

### Detecting Missing Config

```python
# To detect missing config (instead of silent fallback):
value = get_cfg("some.path", default=None)
if value is None:
    raise ValueError("Required config 'some.path' is missing")
```

## Config File Locations

| Config Name | Path | Purpose |
|-------------|------|---------|
| `pipeline_config` | `CONFIG/pipeline/pipeline.yaml` | Core pipeline settings |
| `intelligent_training_config` | `CONFIG/pipeline/training/intelligent.yaml` | Training orchestration |
| `safety_config` | `CONFIG/pipeline/training/safety.yaml` | Safety guards |
| `preprocessing_config` | `CONFIG/pipeline/training/preprocessing.yaml` | Data preprocessing |
| `callbacks_config` | `CONFIG/pipeline/training/callbacks.yaml` | Training callbacks |
| `reproducibility` | `CONFIG/pipeline/training/reproducibility.yaml` | Determinism settings |
| `memory_config` | `CONFIG/pipeline/memory.yaml` | Memory management |
| `gpu_config` | `CONFIG/pipeline/gpu.yaml` | GPU settings |
| Model configs | `CONFIG/models/{family}.yaml` | Per-family hyperparameters |
| Experiment configs | `CONFIG/experiments/{name}.yaml` | Experiment overrides |

## Loading Model Configs

```python
from CONFIG.config_loader import load_model_config

# Basic load
config = load_model_config("lightgbm")

# With variant
config = load_model_config("lightgbm", variant="conservative")

# With overrides (highest priority)
config = load_model_config("lightgbm", overrides={"learning_rate": 0.02})
```

## Loading Experiment Configs

```python
from CONFIG.config_loader import load_experiment_config

# Load experiment (fails if not found - no fallback)
config = load_experiment_config("e2e_full_targets_test")

# Get experiment path
from CONFIG.config_loader import get_experiment_config_path
path = get_experiment_config_path("e2e_full_targets_test")
```

## Adding New Config Parameters

### Step 1: Add to defaults.yaml (if applicable)

```yaml
# CONFIG/defaults.yaml
new_section:
  new_param: default_value
```

### Step 2: Add to appropriate config file

```yaml
# CONFIG/pipeline/training/intelligent.yaml
new_section:
  new_param: specific_value  # overrides default
```

### Step 3: Access via get_cfg()

```python
value = get_cfg("new_section.new_param", default=default_value)
```

### Step 4: Document in CONFIG/README.md

## Anti-Patterns to Avoid

| Anti-Pattern | Correct Approach |
|--------------|------------------|
| `config = {"learning_rate": 0.01}` | `get_cfg("models.lightgbm.learning_rate")` |
| `config.get("key", fallback)` | `get_cfg("section.key", default=fallback)` |
| `os.path.join(base, "targets", name)` | `get_target_dir(base, name)` |
| `f"{base}/models/{family}"` | `get_target_models_dir(base, target, family)` |
| Custom `if experiment: ...` logic | Precedence is automatic |

## Determinism Impact

Configuration affects determinism in several ways:

### Seeds and Randomness
- `pipeline.determinism.base_seed` - Master seed for all randomness
- `randomness.random_state` - Per-model random state
- Always sourced from config, never generated at runtime

### Mode Detection
```python
from CONFIG.config_loader import get_cfg

# Check if strict mode
strict = get_cfg("pipeline.determinism.mode", default="best_effort") == "strict"
```

### Config Caching
Config is cached after first load. To force reload:
```python
from CONFIG.config_loader import clear_config_cache
clear_config_cache()  # Forces reload on next get_cfg()
```

## Path Construction (SST)

Never construct paths manually. Use helpers from `TRAINING/orchestration/utils/target_first_paths.py`:

```python
from TRAINING.orchestration.utils.target_first_paths import (
    get_target_dir,
    get_target_models_dir,
    get_target_metrics_dir,
    get_scoped_artifact_dir,
    get_globals_dir,
    normalize_target_name,
)

# Target directory
target_dir = get_target_dir(output_dir, "fwd_ret_10m")

# Models directory
models_dir = get_target_models_dir(output_dir, target, "lightgbm")

# Scoped artifact (view/symbol aware)
artifact_dir = get_scoped_artifact_dir(output_dir, target, "importances", scope=write_scope)
```

## Related Skills

- `sst-and-coding-standards.md` - SST compliance patterns
- `determinism-and-reproducibility.md` - Determinism requirements

## Related Documentation

- **MCP Tools (preferred)**: Use `mcp__foxml-config__*` tools for config queries
- `CONFIG/README.md` - Full configuration reference
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - Complete helper catalog
- `CONFIG/config_loader.py` - Implementation source
