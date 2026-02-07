# LIVE_TRADING Master Implementation Plan

## Overview

The `LIVE_TRADING/` module is the execution engine for deploying models trained by the TRAINING pipeline. It implements:

1. **Multi-horizon prediction** (5m, 10m, 15m, 30m, 60m, 1d)
2. **Ridge risk-parity blending** across model families within each horizon
3. **Cost-aware horizon arbitration** selecting the optimal horizon after trading costs
4. **Barrier gating** using peak/valley probabilities to filter entries
5. **Volatility-scaled position sizing**
6. **Kill switches and risk guardrails**

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LIVE_TRADING ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Market Data → [Prediction] → [Blending] → [Arbitration] → [Sizing] → Orders │
│                     │             │             │             │              │
│                     ▼             ▼             ▼             ▼              │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│              │  Multi-  │  │  Ridge   │  │  Cost-   │  │   Vol-   │         │
│              │ Horizon  │  │  Risk-   │  │  Aware   │  │  Scaled  │         │
│              │ Predictor│  │  Parity  │  │ Arbiter  │  │  Sizing  │         │
│              └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                │                           ▲                 │
│                                ▼                           │                 │
│                          ┌──────────┐              ┌──────────┐             │
│                          │ Barrier  │              │   Risk   │             │
│                          │  Gates   │──────────────│ Guardrails│             │
│                          └──────────┘              └──────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │    Broker    │
                              │  (Paper/Live)│
                              └──────────────┘
```

## Module Structure

```
LIVE_TRADING/
├── __init__.py                     # Package init with repro_bootstrap
├── plans/                          # Implementation plans (this folder)
│   ├── MASTER.md                   # This file
│   ├── 01_common_infrastructure.md
│   ├── 02_broker_layer.md
│   ├── 03_model_integration.md
│   ├── 04_prediction_pipeline.md
│   ├── 05_blending.md
│   ├── 06_arbitration.md
│   ├── 07_gating.md
│   ├── 08_sizing.md
│   ├── 09_risk.md
│   ├── 10_engine.md
│   └── 11_config_and_cli.md
├── common/
│   ├── __init__.py
│   ├── exceptions.py               # LiveTradingError hierarchy
│   └── constants.py                # HORIZONS, FAMILIES, defaults
├── brokers/
│   ├── __init__.py
│   ├── interface.py                # Broker Protocol
│   ├── paper.py                    # PaperBroker
│   └── data_provider.py            # Market data abstraction
├── models/
│   ├── __init__.py
│   ├── loader.py                   # Load from TRAINING artifacts
│   ├── inference.py                # Family-specific inference routing
│   └── feature_builder.py          # Build features from market data
├── prediction/
│   ├── __init__.py
│   ├── predictor.py                # Multi-horizon prediction coordinator
│   ├── standardization.py          # Z-score calibration
│   └── confidence.py               # IC-based confidence scores
├── blending/
│   ├── __init__.py
│   ├── horizon_blender.py          # Per-horizon model blending
│   ├── ridge_weights.py            # Ridge risk-parity computation
│   └── temperature.py              # Temperature compression
├── arbitration/
│   ├── __init__.py
│   ├── cost_model.py               # Trading cost estimation
│   └── horizon_arbiter.py          # Cost-aware horizon selection
├── gating/
│   ├── __init__.py
│   ├── barrier_gate.py             # Peak/valley probability gates
│   └── spread_gate.py              # Spread-based gating
├── sizing/
│   ├── __init__.py
│   ├── position_sizer.py           # Main sizing engine
│   ├── vol_scaling.py              # Volatility scaling
│   └── turnover.py                 # Turnover management
├── risk/
│   ├── __init__.py
│   ├── guardrails.py               # Kill switches, position limits
│   ├── drawdown.py                 # Drawdown monitoring
│   └── exposure.py                 # Gross/net exposure tracking
├── engine/
│   ├── __init__.py
│   ├── trading_engine.py           # Main orchestrator
│   └── state.py                    # Engine state management
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Pytest fixtures
    ├── test_broker_interface.py
    ├── test_paper_broker.py
    ├── test_model_loader.py
    ├── test_predictor.py
    ├── test_blending.py
    ├── test_arbitration.py
    ├── test_gating.py
    ├── test_sizing.py
    ├── test_risk.py
    └── test_engine_integration.py
```

## Sub-Plan Index

| Plan | Domain | Est. Files | Priority | Dependencies |
|------|--------|------------|----------|--------------|
| [01_common_infrastructure.md](01_common_infrastructure.md) | Exceptions, constants, base classes | 3 | P0 | None |
| [02_broker_layer.md](02_broker_layer.md) | Broker Protocol, PaperBroker, data | 4 | P0 | 01 |
| [03_model_integration.md](03_model_integration.md) | Loader, inference routing, features | 4 | P0 | 01 |
| [04_prediction_pipeline.md](04_prediction_pipeline.md) | Multi-horizon predictor, standardization | 4 | P1 | 03 |
| [05_blending.md](05_blending.md) | Ridge weights, temperature, blending | 4 | P1 | 04 |
| [06_arbitration.md](06_arbitration.md) | Cost model, horizon selection | 3 | P1 | 05 |
| [07_gating.md](07_gating.md) | Barrier gates, spread gates | 3 | P1 | 04 |
| [08_sizing.md](08_sizing.md) | Position sizing, vol scaling | 4 | P1 | 06, 07 |
| [09_risk.md](09_risk.md) | Kill switches, guardrails | 4 | P0 | 01 |
| [10_engine.md](10_engine.md) | Trading engine orchestrator | 3 | P2 | All above |
| [11_config_and_cli.md](11_config_and_cli.md) | Config YAMLs, CLI entry points | 5 | P2 | 10 |

## Implementation Phases

### Phase 0: Foundation (P0)
- `01_common_infrastructure.md` - Exceptions, constants
- `02_broker_layer.md` - Broker Protocol + PaperBroker
- `03_model_integration.md` - Model loading from TRAINING artifacts
- `09_risk.md` - Kill switches (needed early for safety)

**Deliverable:** Can load models and submit paper orders with basic risk checks.

### Phase 1: Pipeline Components (P1)
- `04_prediction_pipeline.md` - Multi-horizon predictions
- `05_blending.md` - Ridge risk-parity blending
- `06_arbitration.md` - Cost-aware horizon selection
- `07_gating.md` - Barrier probability gates
- `08_sizing.md` - Volatility-scaled sizing

**Deliverable:** Full mathematical pipeline functional.

### Phase 2: Integration (P2)
- `10_engine.md` - Trading engine orchestrator
- `11_config_and_cli.md` - Configuration and CLI

**Deliverable:** Runnable paper trading system.

## Key Mathematical Formulas

### 1. Z-Score Standardization
```
s_{m,h} = clip((r̂_{m,h} - μ_{m,h}) / σ_{m,h}, -3, 3)
```

### 2. Ridge Risk-Parity Weights
```
w_h ∝ (Σ_h + λI)^{-1} μ_h
```
- `Σ_h` = correlation matrix of standardized scores
- `λ` = ridge regularization (0.15 default)
- `μ_h` = target vector (net IC after costs)

### 3. Temperature Compression
```
w_h^{(T)} ∝ w_h^{1/T}
```
- `T_{5m} = 0.75`, `T_{10m} = 0.85`, `T_{15m+} = 1.0`

### 4. Net Score (Cost Arbitration)
```
net_h = α_h - k₁×spread_bps - k₂×σ×√(h/5) - k₃×impact(q)
```

### 5. Barrier Gate
```
g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)
```

### 6. Volatility Sizing
```
z = clip(α / σ, -z_max, z_max)
weight = z × (max_weight / z_max)
```

## SST Compliance Requirements

Every file must follow these patterns:

```python
# Entry points only:
import TRAINING.common.repro_bootstrap  # noqa: F401 - MUST be first

# All files:
from CONFIG.config_loader import get_cfg
from TRAINING.common.exceptions import ConfigError, DataIntegrityError
from TRAINING.common.utils.determinism_ordering import sorted_items, iterdir_sorted
from TRAINING.common.utils.file_utils import write_atomic_json
```

**Path Construction:**
- Use `get_target_dir()`, `get_target_models_dir()` from SST helpers
- Never manually join paths for artifacts

**Config Access:**
- Always `get_cfg("live_trading.xyz", default=...)` - never hardcode

**Deterministic Iteration:**
- Use `sorted_items(dict)` for dict iteration
- Use `iterdir_sorted(path)` for filesystem

## Integration with TRAINING

### Model Loading
Models are loaded from TRAINING artifacts:
```
RESULTS/runs/<run_id>/<ts>/targets/<target>/models/view=<view>/family=<family>/
├── model.pkl or model.h5
├── model_meta.json
└── routing_decision.json
```

### Key TRAINING Imports
```python
from TRAINING.models.registry import FAMILY_CAPABILITIES, get_trainer_info
from TRAINING.common.live.seq_ring_buffer import SeqBufferManager
```

### Family Routing
- LightGBM/XGBoost: Direct pickle load + predict
- Keras models (MLP, CNN1D, LSTM, etc.): Load .h5 + SeqBufferManager for sequential
- Experimental families: Load with experimental flag check

## Configuration Schema

```yaml
# CONFIG/live_trading/live_trading.yaml
live_trading:
  horizons: ["5m", "10m", "15m", "30m", "60m", "1d"]

  blending:
    ridge_lambda: 0.15
    temperature:
      "5m": 0.75
      "10m": 0.85
      "15m": 0.90
      "30m": 1.0
      "60m": 1.0
      "1d": 1.0

  cost_model:
    k1: 1.0     # Spread penalty
    k2: 0.15    # Volatility timing
    k3: 1.0     # Market impact

  barrier_gate:
    g_min: 0.2
    gamma: 1.0
    delta: 0.5
    peak_threshold: 0.6
    valley_threshold: 0.55

  sizing:
    z_max: 3.0
    max_weight: 0.05
    gross_target: 0.5
    no_trade_band: 0.008

  risk:
    max_daily_loss_pct: 2.0
    max_drawdown_pct: 10.0
    max_position_pct: 20.0
    spread_max_bps: 12.0
    quote_age_max_ms: 200
```

## Reference Files

| Purpose | Reference File |
|---------|----------------|
| Broker Protocol | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/brokers/interface.py` |
| Paper Broker | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/brokers/paper.py` |
| Data Provider | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/brokers/data_provider.py` |
| Risk Guardrails | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/core/risk/guardrails.py` |
| Paper Engine | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/core/engine/paper.py` |
| Regime Detector | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/core/regime_detector.py` |
| Strategy | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/strategies/regime_aware_ensemble.py` |
| ML Runtime | `/home/Jennifer/EXTERNAL_PLANS/ALPACA_trading/ml/runtime.py` |
| Sequential Buffers | `TRAINING/common/live/seq_ring_buffer.py` |
| Model Registry | `TRAINING/models/registry.py` |
| Math Foundations | `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md` |

## Testing Strategy

### Unit Tests (per module)
Each sub-plan specifies unit tests for its domain.

### Integration Tests
- `test_engine_integration.py` - Full pipeline from prediction to order

### Smoke Tests
- Load models from a real run
- Generate predictions with mock data
- Paper trade cycle

## Cleanup

After implementation is complete:
1. ~~Remove `ALPACA_trading_new/` skeleton~~ ✓ Moved to `ARCHIVE/`
2. ~~Update `CLAUDE.md` to reference `LIVE_TRADING/`~~ ✓ Done
3. Add `LIVE_TRADING/` to test coverage requirements

## Next Steps

1. Review and approve this master plan
2. Begin with Phase 0 (P0) plans
3. Implement in priority order, running tests after each module
