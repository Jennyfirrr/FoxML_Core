# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FoxML Core is research-grade ML infrastructure for cross-sectional and panel data with deterministic strict mode and full fingerprint lineage. It provides a 3-stage intelligent pipeline: target ranking → feature selection → model training.

## Commands

### Environment Setup
```bash
bash bin/install.sh              # One-line install
conda activate trader            # Activate environment
bash bin/test_install.sh         # Verify installation
```

### Running the Pipeline

**IMPORTANT**: For production/reproducible training, ALWAYS use `bin/run_deterministic.sh`:

```bash
# Deterministic training with experiment config (RECOMMENDED)
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config production_baseline \
    --output-dir TRAINING/results/prod_run

# Deterministic training with custom experiment
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config CONFIG/experiments/my_experiment.yaml \
    --output-dir TRAINING/results/my_run

# Quick development run (non-deterministic, multi-threaded)
python -m TRAINING.orchestration.intelligent_trainer --output-dir my_run
```

### Testing
```bash
pytest                                    # Run all tests
pytest TRAINING/contract_tests/ -v        # Contract tests only
pytest tests/test_smoke_imports.py        # Specific test file
pytest tests/test_target_routing.py -v    # Target routing unit tests
pytest tests/test_model_factory.py -v     # Model factory unit tests
pytest --cov=TRAINING --cov=CONFIG tests/ # With coverage
```

### Linting
```bash
ruff check .                    # Lint check
ruff check --fix .              # Lint with auto-fix
ruff format .                   # Format code
mypy TRAINING/                  # Type checking
```

### Determinism Verification
```bash
bash bin/check_determinism_patterns.sh   # Check code for determinism violations
python bin/verify_determinism_init.py    # Verify determinism bootstrap is correct
```

## Architecture

### Directory Structure
- `TRAINING/` - Core ML pipeline code
- `CONFIG/` - Centralized configuration system (YAML files)
- `DATA_PROCESSING/` - Data preparation and ETL
- `DOCS/` - Public documentation
- `INTERNAL/` - Internal documentation, tools, and experimental code
  - `docs/` - Internal documentation and design references
  - `tools/` - Internal development tools
  - `experimental/` - Experimental features
  - `audit/` - Compliance and audit trails
- `LIVE_TRADING/` - Live trading execution engine
- `DASHBOARD/` - Rust TUI dashboard for monitoring and management
  - `dashboard/` - Rust TUI project (ratatui)
  - `bridge/` - Python IPC bridge (FastAPI)
- `RESULTS/runs/` - Pipeline output artifacts organized by run_id
- `ARCHIVE/` - Deprecated code and legacy implementations (not tracked)
- `bin/` - Entry point scripts and launchers
- `TOOLS/` - Development and verification tools

### Pipeline Stages
1. **Target Ranking** (`TRAINING/ranking/target_ranker.py`) - Ranks targets by predictability
2. **Feature Selection** (`TRAINING/ranking/feature_selector.py`) - Selects optimal features per target
3. **Model Training** (`TRAINING/training_strategies/execution/`) - Trains 20 model families

### Key Modules
- `TRAINING/orchestration/intelligent_trainer.py` - Main entry point and orchestrator
- `TRAINING/orchestration/utils/manifest.py` - Run ID and manifest tracking
- `TRAINING/orchestration/utils/reproducibility_tracker.py` - Fingerprinting for determinism
- `TRAINING/orchestration/utils/repro_tracker_mixins/` - Modular mixins for reproducibility tracker
- `TRAINING/common/repro_bootstrap.py` - Determinism bootstrap (must be imported first)
- `TRAINING/common/feature_registry.py` - Feature definitions and constraints
- `TRAINING/common/utils/config_helpers.py` - Centralized config/threshold loading helpers
- `TRAINING/ranking/predictability/leakage_detection.py` - Pre-training leakage detection
- `CONFIG/config_loader.py` - Centralized config access

### Configuration Flow
Config precedence (highest to lowest): CLI args > Experiment config > Intelligent training config > Pipeline configs > Defaults

Always use `get_cfg()` from `CONFIG.config_loader`:
```python
from CONFIG.config_loader import get_cfg
value = get_cfg("path.to.config", default=fallback_value)
```

## Critical Patterns (Quick Reference)

**See skills for full details:** `sst-and-coding-standards.md`, `determinism-and-reproducibility.md`

| Pattern | Rule | Helper/Tool |
|---------|------|-------------|
| Bootstrap | Import `repro_bootstrap` BEFORE numpy/pandas | `import TRAINING.common.repro_bootstrap` |
| Config access | No hardcoded values | `get_cfg("path.to.key", default=value)` |
| Path construction | No `os.path.join()` | `get_target_dir()`, `get_scoped_artifact_dir()` |
| Dict iteration | Sorted in artifact code | `sorted_items(d)` |
| Filesystem enum | Sorted | `iterdir_sorted()`, `glob_sorted()` |
| File writes | Atomic only | `write_atomic_json()`, `write_atomic_yaml()` |
| Error handling | Typed exceptions | `ConfigError`, `LeakageError`, `ArtifactError` |
| Thread locks | Use RLock for nested contexts | `threading.RLock()` not `Lock()` |
| Multiprocessing | Use spawn, not fork | `multiprocessing.set_start_method('spawn')` |
| Artifact schemas | Check contract before changing | `INTEGRATION_CONTRACTS.md` |

**Fail-closed policy**: In strict mode, anything affecting artifacts/routing/manifests must raise, not warn.

### Interval Handling (After interval-agnostic-pipeline.md implementation)

| Pattern | Rule | Helper/Tool |
|---------|------|-------------|
| Interval access | Single source of truth | `get_interval_spec()` (precedence: data > config > default) |
| Bar conversion | Use ceil for features/purge | `minutes_to_bars(minutes, interval, "ceil")` |
| Horizon validation | Use round for exact match | `horizon_minutes_to_bars()` (returns None if inexact) |
| Purge windows | Time-based, not bar-based | `PurgeSpec`, `make_purge_spec()` |
| Feature lookback | Use `lookback_minutes` | `registry.get_lookback_minutes()` (v1 fallback included) |

**DRY note**: Two conversion functions exist with different purposes:
- `minutes_to_bars(ceil)` → Conservative, for feature windows and purge (prevents under-lookback)
- `horizon_minutes_to_bars(round)` → Exact, for target validation (returns None if not exact)

## Model Families

Model families are defined in `CONFIG/pipeline/training/families.yaml`. Each family has hyperparameters in `CONFIG/models/{family}.yaml`.

To see all available families: `ls CONFIG/models/*.yaml` or check `families.yaml`.

See `adding-new-models.md` skill for adding new model families.

## Testing Structure

**Two test directories with different purposes:**

| Directory | Purpose | When to Use |
|-----------|---------|-------------|
| `TRAINING/contract_tests/` | Pipeline contracts, determinism, stage boundaries | Testing invariants that must never break |
| `tests/` | Unit tests, smoke tests, component tests | Testing individual functions and modules |

**Default pytest runs contract tests** (configured in pyproject.toml). For full coverage:
```bash
pytest                                    # Contract tests only (default)
pytest tests/ TRAINING/contract_tests/    # Both directories
pytest --cov=TRAINING --cov=CONFIG tests/ TRAINING/contract_tests/  # With coverage
```

See `testing-guide.md` skill for comprehensive testing patterns.

## Key Documentation
- `DOCS/00_executive/QUICKSTART.md` - Quick start guide
- `DOCS/00_executive/ARCHITECTURE_OVERVIEW.md` - System architecture
- `DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md` - Bitwise reproducibility
- `CONFIG/README.md` - Configuration system reference
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - SST helper catalog
- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Determinism patterns
- **`INTEGRATION_CONTRACTS.md`** - Module integration contracts (TRAINING ↔ LIVE_TRADING)

## Claude Code Context (.claude/)

The `.claude/` directory contains project-specific context for Claude Code:

### Plans (.claude/plans/)
Implementation plans for significant features or changes. **Always run `ls .claude/plans/` to see current plans** before starting work that might already be planned. Plans are living documents updated across sessions.

#### Active Plans

**`cross-sectional-ranking-objective.md`** - LambdaRank loss for cross-sectional ranking (Status: Implementation in progress)

Upgrades the training objective from pointwise MSE to pairwise/listwise ranking loss:
- Cross-sectional normalized targets (percentile rank of residualized returns)
- Grouped batching by timestamp: `(B, M, L, F)` structure
- LambdaRank and ListMLE loss functions
- New metrics: NDCG@K, MRR, Precision@K

**`raw-ohlcv-sequence-mode.md`** - Raw OHLCV sequences for neural models (Status: Phases 1-2 complete, Phase 5 planned)

Adds `input_mode: RAW_SEQUENCE` that feeds raw OHLCV bars directly to sequence models (Transformer, LSTM, CNN1D) instead of computed technical indicators.

**`live-trading-inference-master.md`** - LIVE_TRADING inference fixes for raw OHLCV + CS ranking (Status: Ready for implementation)

Wires raw OHLCV and cross-sectional ranking model support into the live inference pipeline:
- Phase 0: Barrier gate crash fix (standalone)
- Phase 1: Input mode awareness across loader/inference/predictor
- Phase 2: Raw OHLCV normalization and inference path
- Phase 3: Testing and contract verification
- Phase 4: Cross-sectional ranking inference (future)

#### Recently Completed Plans

**`dashboard-integration-master.md`** - Dashboard TUI integration (Status: ✅ Complete)
- IPC bridge (FastAPI) connecting Python engine to Rust TUI
- Real-time event streaming for training/trading monitoring
- Stage-based tags (TR, FS, TRN) for pipeline progress

**`streaming-concat-optimization.md`** - Memory-efficient data concatenation (Status: ✅ Complete)
- `streaming_concat()` helper in unified_loader.py
- Handles 728+ symbols without OOM

**`column-projection-optimization.md`** - Feature probe for column projection (Status: ✅ Complete)
- Single-symbol importance filtering before full load
- Reduces memory by loading only needed columns

**`mcp-servers-implementation.md`** - MCP servers for domain knowledge (Status: ✅ Complete)

#### Reference Plans

**`interval-agnostic-pipeline.md`** - Make data interval a first-class experiment dimension (Status: Ready for implementation)

Key changes this plan introduces:
- **New SST helpers**: `get_interval_spec()`, `minutes_to_bars()`, `PurgeSpec`, `make_purge_spec()`
- **Registry v2 schema**: Adds `lookback_minutes`, `window_minutes` to feature registry
- **Breaking change**: Phase 3 flips rounding from floor→ceil (requires sign-off)
- **24 phases**

**IMPORTANT**: Before implementing any interval-related changes, read this plan.

### Skills (.claude/skills/)
Domain-specific guidance that can be invoked as slash commands. Run `ls .claude/skills/` to see all available skills, or check the directory directly. Key categories:

**Core Pipeline:**
- `architecture-overview.md` - System architecture deep dive
- `configuration-management.md` - Config system patterns
- `determinism-and-reproducibility.md` - Determinism requirements
- `sst-and-coding-standards.md` - SST compliance patterns

**Development:**
- `adding-new-models.md` - Adding model families
- `feature-engineering.md` - Feature pipeline guidance
- `testing-guide.md` - Testing strategies and commands
- `debugging-pipelines.md` - Troubleshooting pipeline issues
- `maintenance-tasks.md` - Common maintenance operations

**Live Trading (LIVE_TRADING/):**
- `execution-engine.md` - Trading engine development
- `broker-integration.md` - Broker interface implementation
- `model-inference.md` - Model loading and prediction
- `signal-generation.md` - Signal and horizon blending
- `risk-management.md` - Risk controls and kill switches

**Dashboard (DASHBOARD/):**
- `dashboard-overview.md` - Dashboard architecture and features
- `dashboard-development.md` - Adding views, widgets, themes
- `dashboard-ipc-bridge.md` - IPC bridge API and development

**Meta:**
- `skill-maintenance.md` - When/how to update skills (rarely needed)

### Multi-Session Workflow
When working on significant tasks that span multiple context windows:

1. **Use fresh context windows** - Start new sessions for continued work rather than pushing to context limits
2. **Update plans before ending** - When nearing the end of a session, update the relevant `.claude/plans/*.md` file to reflect:
   - What has been completed
   - What remains to be done
   - Current state and any blockers
   - Next steps for the following session
3. **Check plans at session start** - When resuming work, read the relevant plan file to understand current state

This ensures continuity across sessions and prevents losing progress context.

## MCP Servers (MCP_SERVERS/)

Model Context Protocol servers exposing FoxML domain knowledge. **Use these tools instead of reading documentation files or grepping code.**

### foxml-sst: SST Helper Catalog
Use for finding the correct SST helper for any task:
- `mcp__foxml-sst__search_sst_helpers` - Search helpers by query (e.g., "target directory", "config access")
- `mcp__foxml-sst__recommend_sst_helper` - Get recommendations for a task description
- `mcp__foxml-sst__get_sst_helper_details` - Get import path, usage, and examples for a specific helper
- `mcp__foxml-sst__list_sst_categories` - Browse all helper categories

### foxml-config: Configuration Access
Use for querying config values and understanding precedence:
- `mcp__foxml-config__get_config_value` - Get any config value by path (e.g., "pipeline.determinism.base_seed")
- `mcp__foxml-config__list_config_keys` - Discover available config keys
- `mcp__foxml-config__show_config_precedence` - See which config layer provides a value
- `mcp__foxml-config__load_experiment_config` - Load experiment overrides
- `mcp__foxml-config__list_available_configs` - List all config files

### foxml-artifact: Run Artifacts
Use for inspecting pipeline runs and comparing results:
- `mcp__foxml-artifact__query_runs` - List runs with filters (experiment, date, git SHA)
- `mcp__foxml-artifact__get_run_details` - Get manifest, config, and targets for a run
- `mcp__foxml-artifact__compare_runs` - Diff two runs (config, targets, git)
- `mcp__foxml-artifact__get_model_metrics` - Get AUC/metrics for a target
- `mcp__foxml-artifact__get_target_stage_history` - View stage progression

### When to Use MCP Tools
| Task | MCP Tool |
|------|----------|
| "What SST helper should I use for X?" | `mcp__foxml-sst__recommend_sst_helper` |
| "How do I import get_target_dir?" | `mcp__foxml-sst__get_sst_helper_details` |
| "What's the default base_seed?" | `mcp__foxml-config__get_config_value` |
| "What config keys exist under pipeline?" | `mcp__foxml-config__list_config_keys` |
| "Compare two runs" | `mcp__foxml-artifact__compare_runs` |
| "What targets were trained in run X?" | `mcp__foxml-artifact__query_targets` |

### If MCP Tools Are Unavailable
Fallback to reading files directly:

| MCP Tool | Fallback |
|----------|----------|
| SST helpers | Read `INTERNAL/docs/references/SST_SOLUTIONS.md` |
| Config values | Read `CONFIG/*.yaml` files, check `CONFIG/README.md` |
| Config precedence | Read `CONFIG/config_loader.py` docstrings |
| Run artifacts | Read `RESULTS/runs/{run_id}/manifest.json` directly |
| Determinism patterns | Read `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` |

## Integration Contracts (TRAINING ↔ LIVE_TRADING)

**CRITICAL**: Before modifying any artifact schema, read `INTEGRATION_CONTRACTS.md`.

The contract between TRAINING (producer) and LIVE_TRADING (consumer) is defined by artifact schemas:

| Artifact | Producer | Consumer | Key Fields |
|----------|----------|----------|------------|
| `model_meta.json` | `TRAINING/training_strategies/execution/training.py` | `LIVE_TRADING/models/loader.py` | `feature_list`, `interval_minutes`, `model_checksum` |
| `manifest.json` | `TRAINING/orchestration/utils/manifest.py` | `LIVE_TRADING/models/loader.py` | `target_index`, `run_id` |
| `routing_decision.json` | `TRAINING/ranking/target_routing.py` | `LIVE_TRADING/models/loader.py` | `route`, `reason` |

### Contract Rules

| Rule | Rationale |
|------|-----------|
| Field names must match exactly | `feature_list` not `features` |
| Required fields in ALL code paths | Symbol-specific AND cross-sectional |
| Lists must be sorted | Deterministic serialization |
| Use `write_atomic_json()` | Crash consistency |

### When Modifying Artifacts

1. **Check `INTEGRATION_CONTRACTS.md`** for current schema
2. **Add new fields as OPTIONAL** first (non-breaking)
3. **Update ALL consumers** before making fields required
4. **Run both test suites**: `pytest TRAINING/contract_tests/` AND `pytest LIVE_TRADING/tests/`

See skill: `integration-contracts.md`

## Execution Engine (LIVE_TRADING/)

Live trading execution engine with SST compliance. See the **Live Trading skills** for detailed guidance:

| Skill | Purpose |
|-------|---------|
| `execution-engine.md` | Trading engine architecture and core components |
| `broker-integration.md` | Broker protocol and implementations |
| `model-inference.md` | Model loading and prediction pipeline |
| `signal-generation.md` | Signal blending and horizon selection |
| `risk-management.md` | Kill switches, gates, and risk controls |

### Quick Reference

**Directory structure**: `LIVE_TRADING/{engine,brokers,models,prediction,blending,gating,arbitration,sizing,risk}/`

**Trading pipeline**: Prediction → Blending → Gating → Arbitration → Sizing → Execution

**Key docs**:
- `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
- `DOCS/03_technical/trading/implementation/LIVE_TRADING_INTEGRATION.md`

## Dashboard (DASHBOARD/)

Rust TUI dashboard for monitoring and managing trading/training systems. **Key principle**: Trading runs independently via systemd - the TUI is an optional viewer/controller.

### Quick Start

```bash
# Launch dashboard (auto-starts IPC bridge if needed)
bin/foxml

# Or build and run directly
cd DASHBOARD/dashboard
cargo build --release
cargo run --release
```

### Architecture

```
Trading Service (systemd)  →  IPC Bridge (Python/FastAPI)  →  Rust TUI (ratatui)
                              http://127.0.0.1:8765
```

### Skills

| Skill | Purpose |
|-------|---------|
| `dashboard-overview.md` | Architecture, features, keyboard shortcuts |
| `dashboard-development.md` | Adding views, widgets, themes, patterns |
| `dashboard-ipc-bridge.md` | IPC bridge API endpoints and development |

### Key Features

- **Trading Monitor**: Real-time metrics, events, pipeline status, kill switch
- **Training Monitor**: Run discovery, progress tracking, model grid
- **Config Editor**: YAML editing with syntax highlighting
- **Service Manager**: systemd service control
- **Run Manager**: Start/stop training runs
- **Theme System**: Auto-detects colors from waybar/hyprland/tmux/kitty

### Key Files

| File | Purpose |
|------|---------|
| `DASHBOARD/dashboard/src/app.rs` | Main app state and event loop |
| `DASHBOARD/dashboard/src/views/` | View modules (screens) |
| `DASHBOARD/dashboard/src/widgets/` | Reusable UI components |
| `DASHBOARD/bridge/server.py` | Python FastAPI IPC bridge |
| `bin/foxml` | Launch script |

## Naming Conventions

**Files and directories follow consistent patterns:**

| Type | Convention | Example |
|------|------------|---------|
| Top-level source dirs | SCREAMING_SNAKE | `TRAINING/`, `CONFIG/`, `LIVE_TRADING/` |
| Subdirectories | lowercase | `TRAINING/common/`, `CONFIG/models/` |
| Python files | snake_case | `intelligent_trainer.py`, `config_loader.py` |
| Markdown files | kebab-case | `architecture-overview.md`, `adding-new-models.md` |
| Claude plans | kebab-case | `interval-agnostic-pipeline.md` |
| Claude skills | kebab-case | `determinism-and-reproducibility.md` |

**Avoid:**
- Mixing cases in the same directory level
- Creating lowercase duplicates of uppercase directories
- Using `_new` or `_old` suffixes (use `ARCHIVE/` instead)
