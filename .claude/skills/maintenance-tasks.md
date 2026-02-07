# Maintenance Tasks

Skill for routine operations, cleanup procedures, and codebase maintenance.

## Run Cleanup

### RESULTS/runs/ Management

```bash
# List all runs sorted by date
ls -lt RESULTS/runs/

# Check disk usage per run
du -sh RESULTS/runs/*/

# Remove old runs (keep last 5)
ls -1t RESULTS/runs/ | tail -n +6 | xargs -I {} rm -rf RESULTS/runs/{}

# Remove runs older than 7 days
find RESULTS/runs/ -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;
```

### Selective Cleanup

```bash
# Remove only model artifacts (keep metrics/logs)
find RESULTS/runs/*/targets/*/models -name "*.joblib" -delete

# Remove reproducibility fingerprints (regenerated on next run)
find RESULTS/runs/ -path "*/reproducibility/*" -name "*.json" -delete
```

## Config Cache Clearing

### Python API

```python
from CONFIG.config_loader import clear_config_cache

# Force config reload on next access
clear_config_cache()
```

### When to Clear Cache

- After modifying config files during development
- After git pull that changes CONFIG/
- When testing config changes

## Memory Management Operations

### Monitor Memory

```python
from TRAINING.common.memory.memory_manager import MemoryManager

mm = MemoryManager()
usage = mm.get_memory_usage()

print(f"Process RSS: {usage['rss_gb']:.1f} GB")
print(f"System Available: {usage['system_available_gb']:.1f} GB")
print(f"System Usage: {usage['system_percent']:.0f}%")
```

### Force Cleanup

```python
mm = MemoryManager()
mm.cleanup_memory()  # gc.collect() + optional aggressive cleanup
```

### Chunked Processing Config

```yaml
# CONFIG/pipeline/memory.yaml
memory:
  chunking:
    enabled: true
    chunk_size: 500000  # Reduce for low-memory systems
  thresholds:
    memory_threshold: 0.8  # Trigger cleanup at 80% usage
```

## Config Compliance Checking

### Verify SST Compliance

```bash
# Check all config access follows SST patterns
python CONFIG/tools/verify_config_sst.py

# Validate config path mappings
python CONFIG/tools/validate_config_paths.py

# Find repeated/inconsistent defaults
python CONFIG/tools/find_repeated_defaults.py
```

### Test Default Parity

```bash
# Verify code defaults match config file defaults
python CONFIG/tools/test_default_parity.py
```

### Show Config Hierarchy

```bash
# Display config precedence and values
python CONFIG/tools/show_config_hierarchy.py pipeline.determinism
```

## Determinism Verification

### Check Code Patterns

```bash
# Find determinism anti-patterns
bash bin/check_determinism_patterns.sh

# Output shows:
# - Dict iteration without sorted_items()
# - Filesystem enumeration without sorting
# - Timestamp usage in artifact code
```

### Verify Bootstrap

```bash
# Ensure bootstrap is imported correctly in entry points
python bin/verify_determinism_init.py
```

### Test Artifact Determinism

```bash
# Run determinism tests
python CONFIG/tools/test_artifact_determinism.py

# Or via pytest
pytest TRAINING/contract_tests/test_determinism_strict.py -v
```

## Git Workflow

### Branch Naming

```bash
# Feature branches
git checkout -b feature/add-new-model
git checkout -b fix/leakage-detection-bug
git checkout -b refactor/config-consolidation

# Current branch (from status)
# fix-run-id-format-mismatch
```

### Pre-Commit Checks

```bash
# Run before committing
ruff check .          # Lint
ruff format --check . # Format check
pytest -x             # Quick test

# Fix issues
ruff check --fix .
ruff format .
```

### Commit Message Format

```bash
git commit -m "$(cat <<'EOF'
Brief description of change

- Detail 1
- Detail 2

Refs: #issue-number (if applicable)
EOF
)"
```

## Dependency Updates

### Check Outdated Packages

```bash
# With conda
conda list --outdated

# With pip (if mixed env)
pip list --outdated
```

### Update Environment

```bash
# Update from environment.yml
conda env update -f environment.yml --prune

# Recreate environment (clean slate)
conda deactivate
conda env remove -n trader
bash bin/install.sh
```

### Pin Dependencies

```yaml
# environment.yml - pin major versions
dependencies:
  - python=3.10
  - numpy>=1.24,<2.0
  - pandas>=2.0,<3.0
  - lightgbm>=4.0,<5.0
```

## Log Management

### Log Locations

| Log | Location |
|-----|----------|
| Pipeline logs | `RESULTS/runs/{run_id}/logs/` |
| Target logs | `RESULTS/runs/{run_id}/targets/{target}/logs/` |
| System logs | stdout/stderr (redirect as needed) |

### Log Rotation

```bash
# Compress old logs
find RESULTS/runs/ -name "*.log" -mtime +3 -exec gzip {} \;

# Remove very old logs
find RESULTS/runs/ -name "*.log.gz" -mtime +30 -delete
```

### Log Level Configuration

```yaml
# CONFIG/core/logging.yaml
logging:
  level: INFO
  handlers:
    console:
      level: INFO
    file:
      level: DEBUG
      path: logs/pipeline.log
```

## Code Maintenance

### Add Copyright Headers

```bash
# Add SPDX headers to new files
python bin/add_copyright_headers.py TRAINING/new_module.py
```

### Import Verification

```bash
# Check for circular imports
python -c "import TRAINING.orchestration.intelligent_trainer"

# Full import smoke test
python -m compileall TRAINING CONFIG
```

### Type Checking

```bash
# Check all modules
mypy TRAINING/

# Check specific module with strict rules
mypy --strict TRAINING/orchestration/utils/scope_resolution.py
```

## SST Audit Tasks

### Verify Path Construction

```bash
# Find manual path construction (should use SST helpers)
rg "os\.path\.join|f['\"].*/" --type py TRAINING/
```

### Verify Config Access

```bash
# Find hardcoded config values
rg "config\[|config\.get\(" --type py TRAINING/
```

### Verify Deterministic Iteration

```bash
# Find dict iteration (should use sorted_items)
rg "\.items\(\)|\.values\(\)|\.keys\(\)" --type py TRAINING/
```

## Periodic Maintenance Schedule

### Daily (During Active Development)
- Clear config cache if configs changed
- Run quick tests: `pytest -x`
- Check lint: `ruff check .`

### Weekly
- Run full test suite: `pytest`
- Clean old runs: `ls -1t RESULTS/runs/ | tail -n +10 | xargs rm -rf`
- Update dependencies: `conda env update`

### Monthly
- Full SST audit: Run all CONFIG/tools/*.py scripts
- Determinism verification: `bash bin/check_determinism_patterns.sh`
- Review and clean feature registry
- Archive/delete very old results

## Related Skills

- `sst-and-coding-standards.md` - SST compliance patterns
- `determinism-and-reproducibility.md` - Determinism verification
- `testing-guide.md` - Test maintenance

## Related Documentation

- `bin/` - Entry scripts and launchers
- `CONFIG/tools/` - Config validation tools
- `environment.yml` - Conda environment definition
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - SST compliance patterns
