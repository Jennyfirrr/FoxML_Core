# Architecture Overview

Skill for navigating and understanding the FoxML Core codebase structure.

## Directory Structure

```
trader/
├── TRAINING/                    # Core ML pipeline code
│   ├── orchestration/           # Pipeline orchestration and coordination
│   │   ├── intelligent_trainer.py       # Main entry point (uses mixin)
│   │   ├── intelligent_trainer/         # Submodule (extracted 2026-01)
│   │   │   ├── pipeline_stages.py       # PipelineStageMixin
│   │   │   ├── cli.py, config.py        # Config/CLI helpers
│   │   │   └── caching.py, utils.py     # Caching utilities
│   │   └── utils/
│   │       ├── manifest.py              # Run ID and manifest tracking
│   │       ├── training_events.py       # Dashboard event emitter
│   │       ├── reproducibility_tracker.py # Fingerprinting (uses mixins)
│   │       ├── repro_tracker_mixins/    # Extracted mixins
│   │       ├── diff_telemetry.py        # Diff tracking (uses mixins)
│   │       ├── diff_telemetry/          # Extracted mixins
│   │       └── target_first_paths.py    # SST path construction
│   ├── ranking/                 # Target and feature ranking
│   │   ├── target_ranker.py             # Stage 1: Target ranking
│   │   ├── feature_selector.py          # Stage 2: Feature selection
│   │   ├── multi_model_feature_selection.py   # Multi-model selection
│   │   ├── multi_model_feature_selection/     # Submodule (trainers, etc.)
│   │   ├── predictability/              # Leakage detection
│   │   └── utils/                       # Ranking utilities
│   ├── model_fun/               # Model trainers (20 families)
│   │   ├── base_trainer.py              # Base class contract
│   │   ├── lightgbm_trainer.py          # Reference implementation
│   │   ├── lstm_trainer.py              # LSTM (sequence input)
│   │   ├── transformer_trainer.py       # Transformer (sequence input)
│   │   ├── cnn1d_trainer.py             # CNN1D (sequence input)
│   │   └── {family}_trainer.py          # Family-specific trainers
│   ├── training_strategies/     # Training execution
│   │   └── execution/                   # Family executors
│   ├── common/                  # Shared utilities
│   │   ├── repro_bootstrap.py           # MUST import first
│   │   ├── input_mode.py                # Input mode enum (FEATURES/RAW_SEQUENCE)
│   │   ├── determinism.py               # Determinism helpers
│   │   ├── exceptions.py                # Typed exceptions
│   │   ├── memory/                      # Memory management
│   │   └── utils/                       # File/ordering/SST utils
│   ├── decisioning/             # Bayesian policy decisions
│   ├── stability/               # Feature stability tracking
│   └── contract_tests/          # Pipeline contract tests
├── CONFIG/                      # Centralized configuration
│   ├── config_loader.py                 # Config access API
│   ├── defaults.yaml                    # Default values
│   ├── models/                          # Per-family config
│   ├── pipeline/                        # Pipeline config
│   │   └── training/                    # Training-specific
│   ├── ranking/                         # Ranking config
│   ├── experiments/                     # Experiment overrides
│   └── data/                            # Data schema config
├── DATA_PROCESSING/             # Data preparation and ETL
│   └── features/                        # Feature engineering
├── LIVE_TRADING/                # Live trading execution engine
│   ├── engine/                          # Trading engine core
│   ├── brokers/                         # Broker integrations (Alpaca, IBKR)
│   ├── models/                          # Model loading and inference
│   ├── prediction/                      # Prediction pipeline
│   ├── risk/                            # Risk management and guardrails
│   └── observability/                   # Metrics and events
├── DASHBOARD/                   # Rust TUI dashboard
│   ├── dashboard/                       # Rust app (ratatui)
│   │   └── src/views/                   # View modules
│   └── bridge/                          # Python IPC bridge (FastAPI)
├── RESULTS/runs/                # Pipeline output artifacts
├── INTERNAL/docs/               # Internal documentation
│   └── references/                      # Pattern catalogs
├── DOCS/                        # Public documentation
├── tests/                       # Unit tests
└── bin/                         # Entry scripts/launchers
    ├── run_deterministic.sh             # Deterministic run wrapper
    └── foxml                            # Dashboard launcher
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT TRAINER                          │
│                 (orchestration/intelligent_trainer.py)          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   STAGE 1: TARGET RANKING │
        │   (ranking/target_ranker.py)
        │                           │
        │   - Ranks targets by      │
        │     predictability        │
        │   - Multi-model consensus │
        │   - Leakage pre-screening │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │ STAGE 2: FEATURE SELECTION│
        │ (ranking/feature_selector.py)
        │                           │
        │   - Per-target feature    │
        │     importance            │
        │   - Leakage filtering     │
        │   - Multi-model consensus │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   STAGE 3: MODEL TRAINING │
        │   (training_strategies/)  │
        │                           │
        │   - 20 model families     │
        │   - Isolated execution    │
        │   - Artifact persistence  │
        └───────────────────────────┘
```

## Key Entry Points

| Entry Point | Purpose |
|------------|---------|
| `python -m TRAINING.orchestration.intelligent_trainer` | Main pipeline entry |
| `bin/run_deterministic.sh` | Strict mode launcher |
| `bin/foxml` | Dashboard TUI launcher |
| `pytest TRAINING/contract_tests/` | Contract test runner |

## Input Modes

The pipeline supports two input modes, controlled by experiment config:

| Mode | Description | Use Case |
|------|-------------|----------|
| `FEATURES` (default) | Engineered features (tabular) | Tree models, classic ML |
| `RAW_SEQUENCE` | Raw OHLCV sequences | LSTM, Transformer, CNN1D |

```yaml
# CONFIG/experiments/my_experiment.yaml
intelligent_training:
  input_mode: RAW_SEQUENCE  # or FEATURES (default)
  sequence_length: 60       # For RAW_SEQUENCE mode
```

**RAW_SEQUENCE mode**:
- Skips feature engineering and feature selection stages
- Feeds raw OHLCV bars directly to neural models
- Requires sequence-capable trainers (LSTM, Transformer, CNN1D)
- See `adding-new-models.md` for implementing sequence trainers

## Module Dependency Map

```
CONFIG/config_loader.py          ← Foundation (Level 0)
        ↓
TRAINING/common/utils/           ← Core Utils (Level 1)
  - sst_contract.py
  - file_utils.py
  - determinism_ordering.py
        ↓
TRAINING/orchestration/utils/    ← Path Helpers (Level 2)
  - target_first_paths.py
        ↓
TRAINING/orchestration/utils/    ← Orchestration (Level 3)
  - run_context.py
  - manifest.py
        ↓
TRAINING/ranking/                ← Pipeline Stages
TRAINING/model_fun/
TRAINING/training_strategies/
```

## Mixin Architecture (Large File Decomposition)

Large files have been decomposed using mixin inheritance for maintainability:

| Main File | Mixins Location | Purpose |
|-----------|-----------------|---------|
| `intelligent_trainer.py` | `intelligent_trainer/pipeline_stages.py` | `PipelineStageMixin` - ranking, selection methods |
| `reproducibility_tracker.py` | `repro_tracker_mixins/` | Cohort, comparison, logging mixins |
| `diff_telemetry.py` | `diff_telemetry/` | Diff, fingerprint, normalization mixins |
| `multi_model_feature_selection.py` | `multi_model_feature_selection/` | Trainers, aggregation, persistence |

**Important**: When a file and directory share the same name (e.g., `module.py` and `module/`), submodule imports must use `importlib.util` workaround - see `decomposition-verification.md`.

## Critical Files Index

| File | Purpose |
|------|---------|
| `TRAINING/common/repro_bootstrap.py` | Must be first import in entry points |
| `TRAINING/common/exceptions.py` | Typed exception hierarchy |
| `TRAINING/orchestration/utils/manifest.py` | Run ID derivation and manifest |
| `TRAINING/orchestration/utils/reproducibility_tracker.py` | Fingerprint tracking |
| `CONFIG/config_loader.py` | `get_cfg()` and config access |
| `CONFIG/defaults.yaml` | Default configuration values |
| `TRAINING/model_fun/base_trainer.py` | Model trainer contract |
| `TRAINING/ranking/predictability/leakage_detection.py` | Leakage detection |
| `TRAINING/common/utils/file_utils.py` | Atomic write operations |
| `TRAINING/common/utils/determinism_ordering.py` | Sorted iteration helpers |

## Run Output Structure

```
RESULTS/runs/{run_id}/
├── manifest.json                # Run manifest with fingerprints
├── globals/                     # Global artifacts
│   └── trends/                  # Cross-target trends
└── targets/
    └── {target}/
        ├── reproducibility/     # Stage fingerprints
        ├── feature_selection/   # Selected features
        ├── models/              # Trained models
        │   └── {family}/
        ├── metrics/             # Evaluation metrics
        └── decisions/           # Policy decisions
```

## Determinism & SST Design Principles

This codebase enforces two critical architectural principles:

### Single Source of Truth (SST)
- **Config**: All config access via `get_cfg()` - never hardcode values
- **Paths**: All path construction via SST helpers - never manual `os.path.join()`
- **Normalization**: All family/target names via SST normalizers

### Determinism Requirements
- **Bootstrap first**: Entry points must `import TRAINING.common.repro_bootstrap` before any ML libs
- **Ordered iteration**: Use `sorted_items()`, `iterdir_sorted()`, `glob_sorted()` in artifact code
- **Atomic writes**: Use `write_atomic_json()`, `write_atomic_yaml()` for artifacts
- **No timestamps in strict mode**: Run IDs derived from identity, not `datetime.now()`

## Related Skills

- `sst-and-coding-standards.md` - SST compliance patterns
- `determinism-and-reproducibility.md` - Determinism requirements
- `configuration-management.md` - Config system patterns
- `adding-new-models.md` - Adding model trainers (including sequence models)
- `dashboard-overview.md` - Dashboard TUI architecture
- `dashboard-event-integration.md` - Training event system

## Related Documentation

- `CLAUDE.md` - Quick reference and commands
- `CONFIG/README.md` - Configuration system
- `INTERNAL/docs/references/SST_SOLUTIONS.md` - Path/config helpers catalog
- `INTERNAL/docs/references/DETERMINISTIC_PATTERNS.md` - Determinism patterns
