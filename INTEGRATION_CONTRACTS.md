# Integration Contracts

This document defines the **stable interfaces** between FoxML modules. Any change to these contracts requires coordination across all consuming modules.

**Purpose**: Ensure TRAINING and LIVE_TRADING (and future modules) can evolve independently while maintaining compatibility.

---

## Table of Contents

1. [Contract Principles](#contract-principles)
2. [TRAINING → LIVE_TRADING Contracts](#training--live_trading-contracts)
   - [Artifact: model_meta.json](#artifact-model_metajson)
   - [Artifact: manifest.json](#artifact-manifestjson)
   - [Artifact: routing_decision.json](#artifact-routing_decisionjson)
   - [Artifact: Model Files](#artifact-model-files)
3. [Path Convention Contract](#path-convention-contract)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Versioning and Compatibility](#versioning-and-compatibility)
6. [Adding New Contracts](#adding-new-contracts)
7. [LIVE_TRADING → DASHBOARD Contracts](#live_trading--dashboard-contracts)
   - [IPC Bridge: /api/state](#ipc-bridge-apistate)
   - [IPC Bridge: /api/positions](#ipc-bridge-apipositions)
   - [IPC Bridge: /api/risk/status](#ipc-bridge-apiriskstatus)
   - [IPC Bridge: /ws/events](#ipc-bridge-wsevents)
   - [IPC Bridge: /ws/training](#ipc-bridge-wstraining)

---

## Contract Principles

### 1. **Producer-Consumer Pattern**
Each artifact has exactly ONE producer module and potentially MANY consumer modules.

### 2. **Schema Stability**
- **Required fields**: Must always be present; removal is a breaking change
- **Optional fields**: May be absent; consumers must handle gracefully
- **Deprecated fields**: Keep for backward compatibility, mark in docs

### 3. **Atomic Writes**
All artifacts use `write_atomic_json()` or `write_atomic_yaml()` to ensure crash consistency.

### 4. **Deterministic Serialization**
JSON artifacts use `canonical_json()` for reproducible hashing and sorted keys.

---

## TRAINING → LIVE_TRADING Contracts

### Artifact: `model_meta.json`

**Purpose**: Metadata for a trained model, including feature list, metrics, and security checksum.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/training_strategies/execution/training.py` |
| **Consumer** | `LIVE_TRADING/models/loader.py`, `LIVE_TRADING/models/inference.py` |
| **Location** | `RESULTS/runs/<run_id>/<timestamp>/targets/<target>/models/view=<view>/family=<family>/model_meta.json` |

#### Schema

```python
{
    # === REQUIRED FIELDS (Breaking change if removed) ===
    "family": str,                    # Model family (e.g., "LightGBM", "XGBoost")
    "target": str,                    # Target name (e.g., "ret_5m", "fwd_ret_15m")
    "feature_list": List[str],        # CRITICAL: Ordered feature names for inference
    "n_features": int,                # Feature count (must match len(feature_list))
    "metrics": {                      # Training performance metrics
        "mean_IC": float,             # Information coefficient
        "mean_RankIC": float,         # Rank IC
        "IC_IR": float,               # IC information ratio
        # ... other metrics
    },

    # === REQUIRED FOR SECURITY (H2 FIX) ===
    "model_checksum": str | None,     # SHA256 hash of model file

    # === REQUIRED FOR INTERVAL VALIDATION (Phase 17) ===
    "interval_minutes": float,        # Training data interval (e.g., 1440.0 for daily)

    # === OPTIONAL FIELDS ===
    "symbol": str | None,             # Symbol if SYMBOL_SPECIFIC view
    "route": str,                     # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    "view": str,                      # Same as route
    "params_used": dict | None,       # Hyperparameters used
    "feature_importance": dict,       # Feature importance scores
    "cv_scores": List[float],         # Cross-validation scores
    "cv_mean": float,                 # CV mean
    "cv_std": float,                  # CV std dev
    "package_versions": dict,         # Dependency versions
    "n_rows_train": int,              # Training row count
    "n_rows_val": int,                # Validation row count

    # === RAW SEQUENCE MODE FIELDS (Phase: raw-ohlcv-sequence-mode) ===
    # These fields are present when input_mode = "raw_sequence"
    "input_mode": str | None,         # "features" (default) or "raw_sequence"
    "sequence_length": int | None,    # Sequence length in bars (for raw_sequence mode)
    "sequence_channels": List[str] | None,  # OHLCV channel names (for raw_sequence mode)
    "sequence_normalization": str | None,   # Normalization method: "returns", "log_returns", "minmax"

    # === CROSS-SECTIONAL RANKING MODE FIELDS (Phase: cross-sectional-ranking) ===
    # These fields are present when cross_sectional_ranking.enabled = true
    # CS ranking changes training objective from pointwise (MSE) to ranking (pairwise/listwise)
    "cross_sectional_ranking": {           # Present only if CS ranking enabled
        "enabled": bool,                   # True if trained with ranking objective
        "target_type": str,                # "cs_percentile", "cs_zscore", "vol_scaled"
        "loss_type": str,                  # "pairwise_logistic", "listwise_softmax", etc.
        "sequence_length": int | None,     # Bars in each sequence (typically 64)
        "normalization": str | None,       # OHLCV normalization method
        "training_metrics": {              # Ranking-aligned metrics
            "best_ic": float,              # Best Spearman IC achieved
            "best_spread": float,          # Best top-bottom spread achieved
            "epochs_trained": int,         # Number of training epochs
            # Additional ranking metrics
            "ic_ir": float | None,         # IC Information Ratio
            "ic_hit_rate": float | None,   # Fraction of positive IC timestamps
            "turnover": float | None,      # Portfolio turnover
            "net_spread": float | None,    # Spread after transaction costs
        }
    } | None,

    # === DEPRECATED (kept for backward compatibility) ===
    "features": List[str],            # Use feature_list instead
}
```

#### Consumer Usage

| Consumer | Fields Used | Purpose |
|----------|-------------|---------|
| `ModelLoader.load_model()` | `model_checksum` | Verify model integrity before loading |
| `ModelLoader.get_feature_list()` | `feature_list` | Pass to FeatureBuilder for inference |
| `InferenceEngine._validate_interval()` | `interval_minutes` | Validate data interval matches training |
| `InferenceEngine._should_use_raw_ohlcv()` | `input_mode` | Determine inference path (features vs raw OHLCV) |
| `InferenceEngine._init_sequential_buffer()` | `sequence_length`, `sequence_channels` | Initialize buffer for raw_sequence mode |
| `InferenceEngine._normalize_ohlcv()` | `sequence_normalization` | Apply same normalization as training |
| `CrossSectionalRankingPredictor.predict()` | `cross_sectional_ranking` | Use ranking inference path |
| `CrossSectionalRankingPredictor._get_ranking_config()` | `cross_sectional_ranking.*` | Match training loss/target config |

#### Contract Rules

1. `feature_list` MUST be a `List[str]` (not `Set[str]`) in sorted, deterministic order
2. `model_checksum` MUST be computed using SHA256 of the model file bytes
3. `interval_minutes` MUST reflect the actual data interval used during training
4. If `model_checksum` is `None`, LIVE_TRADING logs a warning but continues (non-strict mode)
5. `input_mode` defaults to `"features"` if not present (backward compatibility)
6. If `input_mode` = `"raw_sequence"`:
   - `feature_list` MAY be empty (raw OHLCV doesn't use computed features)
   - `sequence_length` MUST be present and > 0
   - `sequence_channels` MUST list the OHLCV columns used (typically 5)
   - `sequence_normalization` MUST specify the normalization method
   - LIVE_TRADING MUST use matching normalization at inference time
7. If `cross_sectional_ranking` is present and `enabled` = true:
   - Model was trained with ranking objective (not pointwise MSE)
   - LIVE_TRADING should use `CrossSectionalRankingPredictor` for inference
   - Rankings are relative (scores only meaningful within same timestamp)
   - `training_metrics.best_ic` should be > 0 for useful models
   - Typically combined with `input_mode` = `"raw_sequence"`

---

### Artifact: `manifest.json`

**Purpose**: Run-level metadata including target index for discovery.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/orchestration/utils/manifest.py` |
| **Consumer** | `LIVE_TRADING/models/loader.py` |
| **Location** | `RESULTS/runs/<run_id>/<timestamp>/manifest.json` |

#### Schema

```python
{
    # === REQUIRED FIELDS ===
    "run_id": str,                    # Run identifier
    "run_instance_id": str,           # Instance ID for directory naming
    "timestamp": str,                 # ISO format creation timestamp
    "git_sha": str,                   # Git commit SHA
    "target_index": {                 # Per-target metadata
        "<target>": {
            "status": str,            # "complete", "partial", "failed"
            "families_trained": List[str],
            "view": str,              # CROSS_SECTIONAL or SYMBOL_SPECIFIC
            "auc": float | None,      # Best AUC if available
            # ... other target-specific fields
        }
    },

    # === OPTIONAL FIELDS ===
    "config_digest": str,             # Hash of configuration
    "is_comparable": bool,            # Whether run is comparable
    "experiment_config": dict | None, # Experiment configuration
    "run_metadata": dict | None,      # Additional metadata
}
```

#### Consumer Usage

| Consumer | Fields Used | Purpose |
|----------|-------------|---------|
| `ModelLoader._load_target_index()` | `target_index` | Discover available targets and their status |
| `ModelLoader.list_available_targets()` | `target_index.keys()` | List targets that can be loaded |

#### Contract Rules

1. `target_index` MUST contain an entry for every target that has at least one trained model
2. `status` MUST be one of: `"complete"`, `"partial"`, `"failed"`
3. `families_trained` MUST list all model families with successfully trained models

---

### Artifact: `routing_decision.json`

**Purpose**: Records why a target was routed to CROSS_SECTIONAL vs SYMBOL_SPECIFIC.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/ranking/target_routing.py` |
| **Consumer** | `LIVE_TRADING/models/loader.py` (optional) |
| **Locations** | Global: `RESULTS/.../globals/routing_decisions.json`<br>Per-Target: `RESULTS/.../targets/<target>/decision/routing_decision.json` |

#### Schema

```python
{
    "<target>": {
        # === REQUIRED FIELDS ===
        "route": str,                 # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "BOTH", "BLOCKED"
        "reason": str,                # Human-readable explanation

        # === OPTIONAL FIELDS ===
        "skill01_cs": float,          # Cross-sectional skill score [0,1]
        "skill01_sym_mean": float,    # Mean symbol-specific skill
        "auc": float,                 # Primary metric (deprecated, use skill01_*)
        "frac_symbols_good": float,   # Fraction with good performance
        "winner_symbols": List[str],  # Best-performing symbols
        "n_symbols_evaluated": int,   # Symbols evaluated
    }
}
```

#### Contract Rules

1. `route` MUST be one of: `"CROSS_SECTIONAL"`, `"SYMBOL_SPECIFIC"`, `"BOTH"`, `"BLOCKED"`
2. `reason` MUST be a human-readable string explaining the routing decision

---

### Artifact: Model Files

**Purpose**: Serialized trained model for inference.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/training_strategies/execution/training.py` |
| **Consumer** | `LIVE_TRADING/models/loader.py` |
| **Location** | `RESULTS/.../family=<family>/model.<ext>` |

#### File Extensions by Family

| Family | Primary Extension | Alternative |
|--------|-------------------|-------------|
| `LightGBM` | `.txt` (native booster) | `.pkl` |
| `XGBoost` | `.json` (native) | `.pkl` |
| `CatBoost` | `.cbm` (native) | `.pkl` |
| `TensorFlow/Keras` | `.h5` or `.keras` | SavedModel directory |
| `PyTorch` | `.pt` | - |
| `Scikit-learn` | `.joblib` | `.pkl` |
| `Ridge`, `ElasticNet` | `.joblib` | `.pkl` |

#### Contract Rules

1. Model file MUST exist if `model_meta.json` exists
2. If `model_checksum` is in metadata, file hash MUST match (H2 security)
3. Native formats preferred over pickle for portability

---

## Path Convention Contract

All artifact paths follow the **target-first SST structure**:

```
RESULTS/
└── runs/
    └── <run_id>/
        └── <timestamp>/
            ├── manifest.json                              # Run-level
            ├── globals/
            │   └── routing_decisions.json                 # Global routing summary
            └── targets/
                └── <target>/                              # Target-level
                    ├── decision/
                    │   └── routing_decision.json          # Per-target routing
                    └── models/
                        └── view=<view>/                   # View-level
                            ├── routing_decision.json      # (duplicate for fast access)
                            └── family=<family>/           # Family-level
                                ├── model_meta.json
                                ├── model.<ext>
                                ├── scaler.joblib          # Optional
                                ├── imputer.joblib         # Optional
                                ├── fingerprints.json      # Determinism
                                └── reproducibility.json   # Determinism
```

### Path Construction Rules

1. Use `get_target_dir()` and `get_scoped_artifact_dir()` SST helpers
2. Never use `os.path.join()` directly for artifact paths
3. Target names use underscores: `ret_5m`, `fwd_ret_15m`
4. View names use `=` separator: `view=CROSS_SECTIONAL`
5. Family names use `=` separator: `family=LightGBM`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING MODULE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Target Ranking ──► Feature Selection ──► Model Training ──► Artifact Write │
│                                                                              │
│  Outputs:                                                                    │
│  ├── manifest.json ────────────────────────────────────────────────────────┼─┐
│  ├── routing_decision.json ────────────────────────────────────────────────┼─┼─┐
│  ├── model_meta.json ──────────────────────────────────────────────────────┼─┼─┼─┐
│  └── model.<ext> ──────────────────────────────────────────────────────────┼─┼─┼─┼─┐
│                                                                              │ │ │ │ │
└──────────────────────────────────────────────────────────────────────────────┘ │ │ │ │
                                                                                  │ │ │ │
┌─────────────────────────────────────────────────────────────────────────────┐   │ │ │ │
│                            LIVE_TRADING MODULE                               │   │ │ │ │
├─────────────────────────────────────────────────────────────────────────────┤   │ │ │ │
│                                                                              │   │ │ │ │
│  ┌─────────────────┐                                                         │   │ │ │ │
│  │  ModelLoader    │◄────── manifest.json ───────────────────────────────────┼───┘ │ │ │
│  │                 │◄────── routing_decision.json ───────────────────────────┼─────┘ │ │
│  │  - Target index │◄────── model_meta.json ─────────────────────────────────┼───────┘ │
│  │  - Checksum     │◄────── model.<ext> ─────────────────────────────────────┼─────────┘
│  │  - Feature list │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      │
│  │ FeatureBuilder  │      │ InferenceEngine │      │ Predictor       │      │
│  │                 │      │                 │      │                 │      │
│  │ Uses:           │      │ Validates:      │      │ Outputs:        │      │
│  │ - feature_list  │─────►│ - interval_mins │─────►│ - predictions   │      │
│  │                 │      │                 │      │ - confidence    │      │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Versioning and Compatibility

### Contract Version: 1.4

| Version | Date | Changes |
|---------|------|---------|
| 1.4 | 2026-01-21 | Added cross-sectional ranking mode fields: cross_sectional_ranking object with target_type, loss_type, training_metrics |
| 1.3 | 2026-01-21 | Added raw OHLCV sequence mode fields: input_mode, sequence_length, sequence_channels, sequence_normalization |
| 1.2 | 2026-01-21 | Added LIVE_TRADING → DASHBOARD contracts for IPC bridge endpoints |
| 1.1 | 2026-01-19 | Fixed 4 integration issues: feature_list, interval_minutes, sorted features, model_checksum |
| 1.0 | 2026-01-19 | Initial contract definition |

### Breaking vs Non-Breaking Changes

| Change Type | Breaking? | Action Required |
|-------------|-----------|-----------------|
| Add optional field | No | None |
| Add required field | **Yes** | Coordinate all consumers |
| Remove optional field | No | Update consumers to handle absence |
| Remove required field | **Yes** | Coordinate all consumers |
| Change field type | **Yes** | Coordinate all consumers |
| Change field semantics | **Yes** | Coordinate all consumers |

### Deprecation Process

1. Mark field as deprecated in this document
2. Add warning log in consumer when field is used
3. Keep field for at least 2 major versions
4. Remove field with breaking version bump

---

## Adding New Contracts

When adding new integration points:

1. **Define the artifact** in this document with full schema
2. **Identify producer and consumers** explicitly
3. **Mark required vs optional fields** clearly
4. **Add to data flow diagram** above
5. **Update version** in this document

### Template for New Artifact

```markdown
### Artifact: `<name>.json`

**Purpose**: <one-line description>

| Property | Value |
|----------|-------|
| **Producer** | `<module>/<file>.py` |
| **Consumer** | `<module>/<file>.py` |
| **Location** | `<path pattern>` |

#### Schema

\```python
{
    # === REQUIRED FIELDS ===
    "field": type,  # Description

    # === OPTIONAL FIELDS ===
    "field": type,  # Description
}
\```

#### Contract Rules

1. Rule 1
2. Rule 2
```

---

## Known Contract Issues (Resolved)

All previously identified issues have been fixed as of 2026-01-19.

### Issue 1: `feature_list` vs `features` Field Name Mismatch

**Status**: ✅ RESOLVED

**Resolution**:
1. TRAINING now writes `feature_list` (sorted) in all metadata paths
2. LIVE_TRADING has backward-compatible fallback to `features`/`feature_names`

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py` - Added `feature_list` field
- `TRAINING/models/specialized/core.py` - Added `feature_list` field
- `LIVE_TRADING/models/loader.py` - Added fallback for legacy models
- `LIVE_TRADING/models/inference.py` - Added fallback for legacy models

---

### Issue 2: `interval_minutes` Not Written in Symbol-Specific Path

**Status**: ✅ RESOLVED

**Resolution**:
- Added `interval_minutes` and `interval_source` to all training paths in `training.py`
- Uses `_get_training_interval_minutes()` helper for consistent config access

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py` - All 4 metadata blocks updated

---

### Issue 3: Features Not Guaranteed Sorted

**Status**: ✅ RESOLVED

**Resolution**:
- Added `_get_sorted_feature_list()` helper
- `feature_list` field is always sorted for determinism
- `features` field preserved (unsorted) for backward compatibility

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py` - Uses sorted feature list
- `TRAINING/models/specialized/core.py` - Uses sorted feature list

---

### Issue 4: `model_checksum` Not Always Written

**Status**: ✅ RESOLVED

**Resolution**:
- Added `_compute_model_checksum()` helper (SHA256)
- All training paths now compute and write `model_checksum` after model save
- LIVE_TRADING verification remains graceful (warns if checksum missing)

**Files Changed**:
- `TRAINING/training_strategies/execution/training.py` - Computes checksum after model save
- `TRAINING/models/specialized/core.py` - Added `_compute_model_checksum()` helper and checksum in `save_model()`

---

## LIVE_TRADING → DASHBOARD Contracts

These contracts define the interface between the trading engine and the monitoring dashboard.

### IPC Bridge: `/api/state`

**Purpose**: Real-time engine state including pipeline stage.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/trading_engine.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → Rust TUI |

#### Schema

```python
{
    # === REQUIRED FIELDS ===
    "status": str,              # "running", "paused", "stopped", "error"
    "current_stage": str,       # "idle", "prediction", "blending", "arbitration", "gating", "sizing", "risk", "execution"
    "last_cycle": str,          # ISO timestamp of last cycle completion
    "uptime_seconds": float,    # Seconds since engine start

    # === OPTIONAL FIELDS ===
    "cycle_count": int,         # Total cycles completed
    "symbols_active": int,      # Symbols being processed
}
```

---

### IPC Bridge: `/api/positions`

**Purpose**: Per-position information for position table display.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/engine/state.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → Rust TUI |

#### Schema

```python
{
    "positions": [
        {
            "symbol": str,              # Stock symbol
            "shares": int,              # Number of shares (negative = short)
            "entry_price": float,       # Average entry price
            "current_price": float,     # Latest price
            "market_value": float,      # shares * current_price
            "unrealized_pnl": float,    # Current P&L
            "unrealized_pnl_pct": float, # P&L as percentage
            "weight": float,            # Portfolio weight [0, 1]
            "entry_time": str | None,   # ISO timestamp
            "side": str,                # "long" or "short"
        }
    ],
    "total_positions": int,
    "long_count": int,
    "short_count": int,
    "total_market_value": float,
}
```

#### Contract Rules

1. `positions` array MUST be sorted by symbol for determinism
2. `weight` = `market_value / total_portfolio_value`

---

### IPC Bridge: `/api/risk/status`

**Purpose**: Current risk metrics and warnings.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/gating/risk.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → Rust TUI |

#### Schema

```python
{
    "trading_allowed": bool,
    "kill_switch_active": bool,
    "kill_switch_reason": str | None,

    "daily_pnl_pct": float,
    "daily_loss_limit_pct": float,
    "drawdown_pct": float,
    "max_drawdown_limit_pct": float,
    "gross_exposure": float,
    "net_exposure": float,
    "max_gross_exposure": float,

    "warnings": [
        {
            "type": str,
            "message": str,
            "severity": str,
        }
    ],
    "last_check": str,
}
```

---

### IPC Bridge: `/ws/events`

**Purpose**: Real-time event streaming from EventBus.

| Property | Value |
|----------|-------|
| **Producer** | `LIVE_TRADING/observability/events.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → Rust TUI |

#### Schema (WebSocket Message)

```python
{
    "event_type": str,          # EventType enum value
    "timestamp": str,           # ISO timestamp
    "severity": str,            # "debug", "info", "warning", "error", "critical"
    "message": str,             # Human-readable message
    "data": dict,               # Event-specific data
}
```

#### Contract Rules

1. Events MUST be emitted via EventBus for dashboard visibility
2. `timestamp` MUST be UTC ISO format
3. New event types MUST be added to EventType enum

---

### IPC Bridge: `/ws/training`

**Purpose**: Training pipeline progress streaming.

| Property | Value |
|----------|-------|
| **Producer** | `TRAINING/orchestration/intelligent_trainer.py` |
| **Consumer** | `DASHBOARD/bridge/server.py` → Rust TUI |

#### Schema (WebSocket Message)

```python
{
    "event_type": str,          # "progress", "stage_change", "target_complete", "run_complete", "error"
    "timestamp": str,           # ISO timestamp
    "run_id": str,              # Training run identifier

    # Progress event fields
    "stage": str,               # "ranking", "feature_selection", "training"
    "progress_pct": float,      # Overall progress [0, 100]
    "current_target": str | None,
    "targets_complete": int,
    "targets_total": int,

    # Target complete event fields
    "target": str,
    "status": str,              # "success", "failed", "skipped"
    "models_trained": int,
    "best_auc": float | None,
}
```

#### Contract Rules

1. Progress events MUST be emitted at stage boundaries
2. `run_id` MUST match the manifest run_id
3. Progress MUST be updated at least every 30 seconds during long operations

---

## Related Documentation

- **TRAINING Module**: `TRAINING/README.md`
- **LIVE_TRADING Module**: `LIVE_TRADING/README.md`, `LIVE_TRADING/DOCS/`
- **DASHBOARD Module**: `.claude/skills/dashboard-*.md`
- **SST Helpers**: `INTERNAL/docs/references/SST_SOLUTIONS.md`
- **Path Construction**: `TRAINING/orchestration/utils/artifact_paths.py`

---

## Contact

For contract changes, coordinate with:
- TRAINING module owners
- LIVE_TRADING module owners
- DASHBOARD module owners
- Any future module owners consuming these artifacts
