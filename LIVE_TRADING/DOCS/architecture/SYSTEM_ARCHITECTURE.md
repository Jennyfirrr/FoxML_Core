# System Architecture

This document describes the overall architecture of the LIVE_TRADING execution engine.

---

## Design Philosophy

The LIVE_TRADING module follows several core design principles:

1. **SST Compliance**: All code follows Single Source of Truth patterns for determinism
2. **Protocol-Based**: Uses Python Protocols for structural subtyping (brokers, data providers)
3. **Pipeline Architecture**: Clear stage separation for testability and debugging
4. **Fail-Closed**: In strict mode, any error affecting artifacts/routing must raise, not warn
5. **Cost-Aware**: Every decision incorporates trading cost considerations

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIVE_TRADING ENGINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   CLI/Entry  │    │    Config    │    │  Run Loader  │                  │
│  │   Point      │───►│   Loading    │───►│  (TRAINING)  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                │                             │
│                                                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        TRADING ENGINE                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │   │
│  │  │  Data       │   │   Model     │   │   Engine    │               │   │
│  │  │  Provider   │   │   Loader    │   │   State     │               │   │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │   │
│  │         │                 │                 │                       │   │
│  │         ▼                 ▼                 ▼                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    6-STAGE PIPELINE                           │  │   │
│  │  │                                                               │  │   │
│  │  │  [Prediction] → [Blending] → [Arbitration] → [Gating]        │  │   │
│  │  │                                                   │           │  │   │
│  │  │                     [Risk] ← [Sizing] ←──────────┘           │  │   │
│  │  │                       │                                       │  │   │
│  │  │                       ▼                                       │  │   │
│  │  │                   [Orders]                                    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                │                             │
│                                                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BROKER LAYER                                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Paper   │  │   IBKR   │  │  Alpaca  │  │ Backtest │            │   │
│  │  │  Broker  │  │  Broker  │  │  Broker  │  │  Broker  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Connectivity

The following diagram shows how all major components connect:

```
TradingEngine (main orchestrator)
│
├── REQUIRED Inputs:
│   ├── broker: Broker           # Order execution interface
│   └── data_provider: DataProvider  # Market data fetching
│
├── OPTIONAL Inputs:
│   ├── run_root: str            # Path to TRAINING artifacts (or use predictor)
│   ├── predictor: MultiHorizonPredictor  # Pre-configured predictor
│   ├── targets: List[str]       # Target names (default: ["ret_5m"])
│   ├── config: EngineConfig     # Configuration
│   └── clock: Clock             # Time source (default: system)
│
│   NOTE: Engine works in "limited mode" without run_root or predictor
│         (logs warning, no predictions made)
│
├── Uses: MultiHorizonPredictor (if run_root or predictor provided)
│   ├── Loads: ModelLoader (from TRAINING artifacts)
│   │   └── Uses: InferenceEngine (family-specific inference)
│   ├── Uses: ZScoreStandardizer (rolling buffer normalization)
│   ├── Uses: ConfidenceScorer (IC × freshness × capacity × stability)
│   └── Generates: AllPredictions → HorizonPredictions (per model, per horizon)
│
├── Uses: HorizonBlender
│   ├── Uses: RidgeWeightCalculator (w ∝ (Σ + λI)^{-1} μ)
│   ├── Uses: TemperatureCompressor (w^T for shorter horizons)
│   └── Generates: BlendedAlpha per horizon
│
├── Uses: HorizonArbiter
│   ├── Uses: CostModel (spread + vol timing + market impact)
│   └── Generates: ArbitrationResult (selected horizon, net score)
│
├── Uses: BarrierGate (if config.enable_barrier_gate)
│   └── Generates: GateResult (allow/block with reason)
│
├── Uses: SpreadGate (if config.enable_spread_gate)
│   └── Generates: SpreadGateResult (allow/block with reason)
│
├── Uses: PositionSizer
│   ├── Uses: VolatilityScaler (z-score based sizing)
│   ├── Uses: TurnoverManager (no-trade band)
│   └── Generates: SizingResult (target shares, weights)
│
├── Uses: RiskGuardrails
│   ├── Uses: DrawdownMonitor (peak tracking)
│   ├── Uses: ExposureTracker (position limits)
│   └── Generates: RiskStatus (kill switch states)
│
├── Uses: Broker (Protocol)
│   ├── Implementations: PaperBroker, AlpacaBroker, IBKRBroker
│   └── Methods: submit_order, cancel_order, get_positions, get_cash
│
├── Uses: DataProvider (Protocol)
│   ├── Implementations: SimulatedDataProvider, CachedDataProvider
│   └── Methods: get_quote, get_historical, get_adv
│
├── Manages: EngineState (persistence)
│   └── Saves to: JSON files (state/engine_state.json)
│
├── Emits: Events via emit_* functions
│   ├── emit_trade, emit_decision, emit_error
│   ├── emit_cycle_start, emit_cycle_end
│   └── Received by: AlertManager, MetricsCollector
│
├── Uses: CILS (Continuous Integrated Learning System) - if config.enable_online_learning
│   ├── Exp3IXBandit for horizon weight adaptation
│   ├── RewardTracker for P&L feedback
│   ├── EnsembleWeightOptimizer for blending bandit+static weights
│   └── BanditPersistence for state saving
│
├── Position Reconciliation (C2 FIX)
│   └── Periodic sync with broker positions
│
└── Trade Cooldown (C4 FIX)
    └── Prevents duplicate orders within cooldown period
```

---

## Module Organization

### Core Modules (96 Python files)

| Module | Files | Purpose |
|--------|-------|---------|
| `cli/` | 2 | Command-line interface and config loading |
| `common/` | 11 | Shared types, constants, exceptions, utilities |
| `brokers/` | 5 | Broker adapters (Protocol + implementations) |
| `data/` | 6 | Data providers (Protocol + implementations) |
| `models/` | 4 | Model loading and family-specific inference |
| `prediction/` | 4 | Multi-horizon prediction pipeline |
| `blending/` | 4 | Ridge risk-parity blending |
| `arbitration/` | 3 | Cost-aware horizon selection |
| `gating/` | 3 | Entry gates (barrier, spread) |
| `sizing/` | 4 | Position sizing engine |
| `risk/` | 4 | Risk management and kill switches |
| `engine/` | 4 | Main trading orchestrator |
| `learning/` | 5 | Online learning (bandits) |
| `observability/` | 3 | Events and metrics |
| `alerting/` | 3 | Alert management |
| `backtest/` | 5 | Backtesting engine |
| `tests/` | 25 | Comprehensive test suite |

---

## Design Patterns Used

### 1. Protocol-Based Architecture

Brokers and DataProviders use Python's `Protocol` for structural subtyping:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Broker(Protocol):
    def submit_order(self, symbol: str, side: str, qty: int, ...) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> Dict[str, int]: ...
    def get_cash(self) -> float: ...
    def get_portfolio_value(self) -> float: ...
```

### 2. Pipeline Stage Separation

Each pipeline stage is independently testable with clear interfaces:

```python
# Each stage has:
# - Input dataclass (e.g., HorizonPredictions)
# - Output dataclass (e.g., BlendResult)
# - Configuration (from get_cfg())
# - Unit tests

class HorizonBlender:
    def blend(self, predictions: Dict[str, Dict[str, float]]) -> BlendResult:
        ...
```

### 3. Dataclass-Heavy Design

All data transfer uses dataclasses with `to_dict()` for serialization:

```python
@dataclass
class TradeDecision:
    symbol: str
    side: str
    quantity: int
    price: float
    horizon: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 4. Factory Functions

Instantiation via factory functions for flexibility:

```python
def get_broker(broker_type: str, **kwargs) -> Broker:
    if broker_type == "paper":
        return PaperBroker(**kwargs)
    elif broker_type == "ibkr":
        return IBKRBroker(**kwargs)
    elif broker_type == "alpaca":
        return AlpacaBroker(**kwargs)
    raise ValueError(f"Unknown broker: {broker_type}")
```

### 5. Event-Driven Communication

EventBus for cross-module communication:

```python
# 25+ event types
class EventType(Enum):
    CYCLE_START = "cycle_start"
    PREDICTION_COMPLETE = "prediction_complete"
    TRADE_FILLED = "trade_filled"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"
    ...
```

---

## SST Compliance Patterns

All modules follow SST (Single Source of Truth) patterns:

| Pattern | Implementation |
|---------|----------------|
| Bootstrap | `LIVE_TRADING/__init__.py` imports `repro_bootstrap` first |
| Config Access | `get_cfg("live_trading.key", default=DEFAULT_CONFIG["key"])` |
| Dict Iteration | `sorted_items(d)` from determinism_ordering |
| File Enumeration | `iterdir_sorted()`, `glob_sorted()` |
| Atomic Writes | `write_atomic_json()`, `write_atomic_yaml()` |
| Timezone | All datetime uses `timezone.utc` |
| Exceptions | Extend `FoxMLError` via `LiveTradingError` |

---

## State Management

### Engine State

The engine maintains state for resilience across restarts:

```python
@dataclass
class EngineState:
    cycle_number: int
    portfolio_value: float
    positions: Dict[str, PositionState]
    trade_history: List[Dict]
    kill_switch_states: Dict[str, bool]
    last_update: datetime

    def save(self, path: Path) -> None:
        write_atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "EngineState":
        ...
```

### Persistence Locations

| State | File | Format |
|-------|------|--------|
| Engine State | `state/engine_state.json` | JSON |
| Trade History | `state/trades.json` | JSON |
| Bandit Weights | `state/bandit_state.json` | JSON |
| Metrics | Prometheus endpoint | Metrics |

---

## Error Handling

### Exception Hierarchy

```
FoxMLError (base)
└── LiveTradingError
    ├── ConfigurationError
    ├── ModelLoadError
    ├── BrokerError
    │   ├── OrderRejectedError
    │   └── ConnectionError
    ├── DataProviderError
    ├── GatingError
    ├── RiskLimitError
    └── KillSwitchTriggeredError
```

### Fail-Closed Policy

In strict mode, errors affecting artifacts/routing must raise:

```python
if strict_mode:
    raise GatingError(f"Spread gate failed: {spread_bps} > {max_spread_bps}")
else:
    logger.warning(f"Spread gate would fail: {spread_bps} > {max_spread_bps}")
    return GateResult(allowed=False, reason="spread_exceeded")
```

---

## Testing Architecture

### Test Distribution

| Category | Tests | Coverage |
|----------|-------|----------|
| Phase 0 (Foundation) | 49 | Types, constants, exceptions |
| Phase 1 (Pipeline) | 210 | All pipeline stages |
| Phase 2 (Engine) | 46 | Engine orchestration |
| Phase 3 (CLI/E2E) | 60 | CLI and end-to-end |
| **Total** | **365** | **~90%** |

### Test Fixtures

`conftest.py` provides:

- Mock brokers with configurable slippage
- Fake data providers with deterministic data
- Run directory fixtures
- Clock mocking for time-based tests

---

## Performance Considerations

### Latency Targets

| Operation | Target | Threshold |
|-----------|--------|-----------|
| Prediction (all horizons) | < 100ms | 200ms warn |
| Blending | < 10ms | 50ms warn |
| Full cycle | < 500ms | 2s warn |
| Inference to order | < 1s | 2s error |

### Memory Management

- Rolling buffers with fixed size (10-day history)
- Lazy model loading (on first prediction)
- Batch processing for multiple symbols

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DEPLOYMENT OPTIONS                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Option 1: Local                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ python -m bin.run_live_trading ...          │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Option 2: Systemd Service                           │
│  ┌─────────────────────────────────────────────┐    │
│  │ systemctl start live-trading.service        │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  Option 3: Docker                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │ docker run -v state:/state live-trading     │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [PIPELINE_STAGES.md](PIPELINE_STAGES.md) - Detailed pipeline stage breakdown
- [MATHEMATICAL_FORMULAS.md](MATHEMATICAL_FORMULAS.md) - Mathematical foundations
- [../components/MODEL_INFERENCE.md](../components/MODEL_INFERENCE.md) - Model loading details
