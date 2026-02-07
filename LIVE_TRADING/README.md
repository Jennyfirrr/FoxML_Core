# LIVE_TRADING Module

Execution engine for deploying models trained by the FoxML TRAINING pipeline.

## Quick Start

```bash
# Paper trading with models from a training run
python -m bin.run_live_trading --run-id my_run --broker paper

# Specific symbols
python -m bin.run_live_trading --run-id my_run --symbols SPY QQQ AAPL

# With symbols file
python -m bin.run_live_trading --run-id my_run --symbols-file CONFIG/live_trading/symbols.yaml

# Dry run (simulated data, no run-id needed)
python -m bin.run_live_trading --dry-run --symbols SPY QQQ

# Limited cycles
python -m bin.run_live_trading --run-id my_run --max-cycles 10

# Verbose logging
python -m bin.run_live_trading --run-id my_run --log-level DEBUG
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--run-id` | TRAINING run ID to use for models | Required |
| `--run-root` | Full path to run directory (overrides --run-id) | - |
| `--broker` | Broker to use (paper, ibkr, alpaca) | paper |
| `--symbols` | Space-separated symbols to trade | SPY QQQ |
| `--symbols-file` | YAML file with symbol list | - |
| `--interval` | Cycle interval in seconds | 60 |
| `--max-cycles` | Maximum cycles (0 = unlimited) | 0 |
| `--dry-run` | Dry run mode (simulated data) | False |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--log-dir` | Directory for log files | logs |
| `--no-state` | Don't save state between cycles | False |

## Pipeline Architecture

The trading pipeline processes each symbol through these stages:

```
Market Data
    ↓
1. PREDICTION (MultiHorizonPredictor)
    - Get predictions for each horizon (5m, 10m, 15m, 30m, 60m, 1d)
    - Apply Z-score standardization
    - Calculate confidence scores
    ↓
2. BLENDING (HorizonBlender)
    - Ridge risk-parity weights per horizon
    - Temperature compression
    ↓
3. ARBITRATION (HorizonArbiter)
    - Select best horizon after trading costs
    - Calculate net scores
    ↓
4. GATING (BarrierGate + SpreadGate)
    - Peak/valley barrier checks
    - Spread & quote age validation
    ↓
5. SIZING (PositionSizer)
    - Volatility scaling
    - Turnover management
    - Gross exposure normalization
    ↓
6. RISK (RiskGuardrails)
    - Kill switch validation
    - Position limit checks
    ↓
Orders → Broker (PaperBroker/IBKR)
```

## Configuration

Main configuration in `CONFIG/live_trading/live_trading.yaml`:

```yaml
live_trading:
  # Trading horizons
  horizons: ["5m", "10m", "15m", "30m", "60m", "1d"]

  # Blending
  blending:
    ridge_lambda: 0.15
    temperature:
      "5m": 0.75
      "10m": 0.85
      ...

  # Cost model
  cost_model:
    k1: 1.0  # Spread penalty
    k2: 0.15 # Volatility timing
    k3: 1.0  # Market impact

  # Position sizing
  sizing:
    z_max: 3.0
    max_weight: 0.05
    gross_target: 0.5
    no_trade_band: 0.008

  # Risk management
  risk:
    max_daily_loss_pct: 2.0
    max_drawdown_pct: 10.0
    max_position_pct: 20.0
```

Symbol universe in `CONFIG/live_trading/symbols.yaml`:

```yaml
symbols:
  universe: [SPY, QQQ, IWM, DIA]
  sectors: [XLF, XLK, XLE, ...]
  stocks: [AAPL, MSFT, GOOGL, ...]
  default: [SPY, QQQ, AAPL, MSFT]
```

## Module Structure

```
LIVE_TRADING/
├── __init__.py           # Package init (imports repro_bootstrap)
├── README.md             # This file
│
├── cli/                  # Command-line interface
│   ├── __init__.py
│   └── config.py         # Config validation and loading
│
├── common/               # Shared components
│   ├── constants.py      # HORIZONS, FAMILIES, defaults
│   ├── exceptions.py     # Exception hierarchy
│   └── types.py          # TradeDecision, PipelineTrace, etc.
│
├── brokers/              # Broker interfaces
│   ├── interface.py      # Broker Protocol
│   └── paper.py          # Paper trading broker
│
├── models/               # Model loading and inference
│   ├── loader.py         # Load from TRAINING artifacts
│   ├── inference.py      # Family-specific inference
│   └── feature_builder.py# Technical indicators
│
├── prediction/           # Multi-horizon prediction
│   ├── standardization.py# Z-score normalization
│   ├── confidence.py     # Confidence scoring
│   └── predictor.py      # Multi-horizon orchestrator
│
├── blending/             # Ridge risk-parity blending
│   ├── ridge_weights.py  # w ∝ (Σ + λI)^{-1} μ
│   ├── temperature.py    # w^T compression
│   └── horizon_blender.py# Per-horizon blending
│
├── arbitration/          # Cost-aware horizon selection
│   ├── cost_model.py     # Spread + vol + impact costs
│   └── horizon_arbiter.py# Horizon selection
│
├── gating/               # Entry gates
│   ├── barrier_gate.py   # Peak/valley barriers
│   └── spread_gate.py    # Spread and quote age
│
├── sizing/               # Position sizing
│   ├── vol_scaling.py    # Volatility scaling
│   ├── turnover.py       # No-trade band
│   └── position_sizer.py # Full sizing pipeline
│
├── risk/                 # Risk management
│   ├── drawdown.py       # Drawdown monitoring
│   ├── exposure.py       # Exposure tracking
│   └── guardrails.py     # Kill switches
│
├── engine/               # Main orchestrator
│   ├── state.py          # State persistence
│   ├── data_provider.py  # Market data interface
│   └── trading_engine.py # Main engine
│
└── tests/                # Test suite
    ├── conftest.py       # Pytest fixtures
    ├── test_phase0.py    # Foundation tests
    ├── test_cli.py       # CLI tests
    ├── test_e2e.py       # End-to-end tests
    └── ...               # Module-specific tests
```

## Mathematical Foundations

| Component | Formula |
|-----------|---------|
| Ridge Weights | `w ∝ (Σ + λI)^{-1} μ` |
| Temperature | `w^{(T)} ∝ w^T` (T<1 sharpens) |
| Cost Model | `net_h = α - k₁×spread - k₂×σ×√(h/5) - k₃×impact` |
| Barrier Gate | `max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)` |
| Vol Scaling | `z = clip(α/σ, -z_max, z_max); w = z × (max_weight/z_max)` |
| Confidence | `IC × freshness × capacity × stability` |

See `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md` for full documentation.

## Testing

```bash
# Run all LIVE_TRADING tests
pytest LIVE_TRADING/tests/ -v

# Run specific test file
pytest LIVE_TRADING/tests/test_cli.py -v

# Run end-to-end tests
pytest LIVE_TRADING/tests/test_e2e.py -v

# With coverage
pytest LIVE_TRADING/tests/ --cov=LIVE_TRADING --cov-report=html
```

## SST Compliance

All code follows SST (Single Source of Truth) patterns:

- **Config access**: Uses `get_cfg()` with fallback defaults
- **Dict iteration**: Uses `sorted_items()` for determinism
- **File enumeration**: Uses `iterdir_sorted()`, `glob_sorted()`
- **Atomic writes**: Uses `write_atomic_json()`, `write_atomic_yaml()`
- **Timezone**: All datetime uses `timezone.utc`
- **Exceptions**: Extend `FoxMLError` via `LiveTradingError`

## Usage Example

```python
from LIVE_TRADING.engine import TradingEngine, EngineConfig
from LIVE_TRADING.brokers import get_broker
from LIVE_TRADING.engine.data_provider import get_data_provider

# Create components
broker = get_broker("paper", initial_cash=100_000)
data_provider = get_data_provider("simulated")

# Configure engine
config = EngineConfig(
    save_state=True,
    save_history=True,
    enable_barrier_gate=True,
    enable_spread_gate=True,
)

# Create engine (without TRAINING artifacts for testing)
engine = TradingEngine(
    broker=broker,
    data_provider=data_provider,
    config=config,
)

# Run trading cycle
result = engine.run_cycle(["AAPL", "MSFT", "GOOGL"])

# Check results
print(f"Cycle {result.cycle_number}:")
print(f"  Portfolio value: ${result.portfolio_value:,.2f}")
print(f"  Trades: {result.num_trades}")
print(f"  Holds: {result.num_holds}")
print(f"  Blocked: {result.num_blocked}")

# Get state summary
summary = engine.get_state_summary()
```

## Risk Management

The engine includes multiple risk safeguards:

1. **Kill Switches**:
   - Daily loss limit (default: 2%)
   - Maximum drawdown (default: 10%)
   - Position concentration limit (default: 20%)

2. **Pre-Trade Gates**:
   - Spread gate: Reject if spread > 12 bps
   - Quote freshness: Reject if quote age > 200ms
   - Barrier gate: Block on peak/valley probabilities

3. **Position Sizing**:
   - Volatility scaling
   - Gross exposure target
   - No-trade band for turnover control

## Detailed Documentation

For comprehensive documentation, see the `DOCS/` folder within this module:

| Document | Description |
|----------|-------------|
| [DOCS/README.md](DOCS/README.md) | Documentation index |
| [DOCS/architecture/SYSTEM_ARCHITECTURE.md](DOCS/architecture/SYSTEM_ARCHITECTURE.md) | System architecture and design patterns |
| [DOCS/architecture/PIPELINE_STAGES.md](DOCS/architecture/PIPELINE_STAGES.md) | Detailed 6-stage pipeline breakdown |
| [DOCS/components/MODEL_INFERENCE.md](DOCS/components/MODEL_INFERENCE.md) | Model loading and inference details |
| [DOCS/components/ONLINE_LEARNING.md](DOCS/components/ONLINE_LEARNING.md) | Exp3-IX bandit algorithm |
| [DOCS/reference/CONFIGURATION_REFERENCE.md](DOCS/reference/CONFIGURATION_REFERENCE.md) | Complete configuration options |

### Cross-Module Integration

| Document | Description |
|----------|-------------|
| [INTEGRATION_CONTRACTS.md](../INTEGRATION_CONTRACTS.md) | **CRITICAL**: Artifact schemas between TRAINING and LIVE_TRADING |

**Before modifying model loading or artifact reading**, review `INTEGRATION_CONTRACTS.md` for:
- `model_meta.json` schema (feature_list, interval_minutes, model_checksum)
- `manifest.json` schema (target_index)
- Path conventions (SST-compliant artifact paths)

## Troubleshooting

### No predictions being generated
- Verify `--run-id` points to a valid TRAINING run
- Check that models exist in `RESULTS/runs/{run_id}/{timestamp}/targets/`

### Kill switch triggered
- Check logs for the specific trigger (daily loss, drawdown, etc.)
- Reset via `engine.reset()` or start fresh session

### Missing symbols data
- Ensure symbols are valid and have market data available
- For dry runs, simulated data will be used

### State file corruption
- Delete `state/engine_state.json` to start fresh
- Use `--no-state` flag to disable persistence
