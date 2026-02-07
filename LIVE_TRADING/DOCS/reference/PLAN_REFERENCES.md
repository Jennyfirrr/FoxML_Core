# Plan References

This document lists the plan documents that were used to design and build the LIVE_TRADING module.

---

## Overview

The LIVE_TRADING module was built following several interconnected plan documents in the `.claude/plans/` directory. These plans provide the theoretical foundation, architectural decisions, and implementation details.

---

## Primary Plans

### Multi-Horizon Training Master Plan

**File**: `.claude/plans/multi-horizon-training-master.md`

**Status**: Planning Phase

**Purpose**: Master plan covering advanced multi-horizon capabilities.

**Key Contributions**:
- Multi-horizon training architecture (shared encoder + per-horizon heads)
- Cross-horizon ensemble design (ridge stacking with decay)
- Multi-interval experiment framework

**Relevant Sections**:
- Section 2: Within-Horizon Ensemble Blending → `HorizonBlender`
- Section 3: Across-Horizon Arbitration → `HorizonArbiter`

---

### Phase 8: Multi-Horizon Training

**File**: `.claude/plans/phase8-multi-horizon-training.md`

**Status**: Not Started (TRAINING pipeline)

**Purpose**: Detailed plan for multi-horizon model training.

**Key Contributions**:
- `HorizonBundle` type design
- Shared encoder architecture
- Per-horizon head configuration
- Loss weighting strategies

**Impact on LIVE_TRADING**:
- Informed the `MultiHorizonPredictor` design
- Defined horizon grouping logic
- Specified temperature compression approach

---

### Phase 9: Cross-Horizon Ensemble

**File**: `.claude/plans/phase9-cross-horizon-ensemble.md`

**Status**: Not Started (TRAINING pipeline)

**Purpose**: Design cross-horizon stacking and blending.

**Key Contributions**:
- `CrossHorizonStacker` architecture
- Decay function definitions (exponential, linear, inverse)
- Ridge meta-learner design
- Confidence calculation from horizon agreement

**Impact on LIVE_TRADING**:
- Direct implementation in `blending/` module
- Ridge risk-parity formula: `w ∝ (Σ + λI)^{-1} μ`
- Temperature compression: `w^(1/T)`

---

### Phase 10: Multi-Interval Experiments

**File**: `.claude/plans/phase10-multi-interval-experiments.md`

**Status**: Not Started (TRAINING pipeline)

**Purpose**: Train on one interval, validate on another.

**Key Contributions**:
- `MultiIntervalLoader` design
- Feature transfer mechanisms
- Cross-interval validation
- Interval mapping utilities

**Impact on LIVE_TRADING**:
- Informed interval handling patterns
- Feature name adjustment logic (`ret_5m` → `ret_1m`)
- Data resampling utilities

---

### Interval-Agnostic Pipeline

**File**: `.claude/plans/interval-agnostic-pipeline.md`

**Status**: Ready for implementation

**Purpose**: Make data interval a first-class experiment dimension.

**Key Contributions**:
- `get_interval_spec()` helper
- `minutes_to_bars()` conversion
- `PurgeSpec` and `make_purge_spec()` for temporal validation
- Feature registry v2 schema

**Impact on LIVE_TRADING**:
- Used for interval-to-bars conversion
- Informed purge window calculations
- Horizon validation logic

---

## Supporting Plans

### Architecture Remediation Plans

**Files**:
- `.claude/plans/arch-phase1-run-identity.md`
- `.claude/plans/arch-phase2-thread-safety.md`
- `.claude/plans/arch-phase3-fingerprinting.md`
- `.claude/plans/arch-phase4-error-handling.md`
- `.claude/plans/arch-phase5-config-hierarchy.md`
- `.claude/plans/arch-phase6-data-consistency.md`
- `.claude/plans/arch-phase7-stage-boundaries.md`
- `.claude/plans/arch-phase8-api-design.md`

**Impact on LIVE_TRADING**:
- Error handling patterns → `LiveTradingError` hierarchy
- Config hierarchy → `get_cfg()` usage
- Thread safety → Event handling in `EventBus`
- API design → Protocol-based broker/data interfaces

---

### MCP Servers Implementation

**File**: `.claude/plans/mcp-servers-implementation.md`

**Status**: Completed

**Purpose**: Model Context Protocol servers for domain knowledge.

**Impact on LIVE_TRADING**:
- SST helper discovery via `mcp__foxml-sst__*` tools
- Configuration queries via `mcp__foxml-config__*` tools
- Run artifact queries via `mcp__foxml-artifact__*` tools

---

## Mathematical Foundations

**File**: `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`

**Status**: Complete

**Purpose**: Mathematical equations for cost-aware ensemble trading.

**Key Contributions**:
- All formulas used in LIVE_TRADING
- Z-score standardization
- Ridge risk-parity weights
- Temperature compression
- Cost model
- Barrier gate formula
- Exp3-IX algorithm

---

## Skills (Claude Code Guidance)

### Live Trading Skills

| Skill | File | Purpose |
|-------|------|---------|
| `execution-engine.md` | `.claude/skills/execution-engine.md` | Trading engine development |
| `broker-integration.md` | `.claude/skills/broker-integration.md` | Broker protocol implementation |
| `model-inference.md` | `.claude/skills/model-inference.md` | Model loading and prediction |
| `signal-generation.md` | `.claude/skills/signal-generation.md` | Signal blending and gating |
| `risk-management.md` | `.claude/skills/risk-management.md` | Risk controls and kill switches |

### Core Skills Used

| Skill | File | Purpose |
|-------|------|---------|
| `sst-and-coding-standards.md` | `.claude/skills/sst-and-coding-standards.md` | SST compliance patterns |
| `determinism-and-reproducibility.md` | `.claude/skills/determinism-and-reproducibility.md` | Determinism requirements |
| `configuration-management.md` | `.claude/skills/configuration-management.md` | Config system patterns |

---

## Implementation Order

The LIVE_TRADING module was implemented in phases:

### Phase 0: Foundation
- Based on: Architecture remediation plans
- Implemented: `common/`, `brokers/`, `risk/` base modules
- Tests: 49 tests

### Phase 1: Pipeline
- Based on: Phase 8, 9 multi-horizon plans
- Implemented: `models/`, `prediction/`, `blending/`, `arbitration/`, `gating/`, `sizing/`
- Tests: 210 tests

### Phase 2: Integration
- Based on: Mathematical foundations doc
- Implemented: `engine/` module
- Tests: 46 tests

### Phase 3: CLI & Deployment
- Based on: Execution engine skill
- Implemented: `cli/`, end-to-end tests
- Tests: 60 tests

---

## Key Decisions from Plans

### 1. Ridge Risk-Parity (from Phase 9)

**Decision**: Use ridge regression with correlation matrix for model blending.

**Rationale**: Handles correlated models better than simple averaging. The `λI` term provides regularization.

**Formula**: `w ∝ (Σ + λI)^{-1} μ`

---

### 2. Temperature Compression (from Phase 9)

**Decision**: Apply `w^(1/T)` with T < 1 for short horizons.

**Rationale**: Short horizons (5m, 10m) have more noise. Temperature compression makes weights more uniform, reducing concentration risk.

**Values**: T=0.75 for 5m, T=0.85 for 10m

---

### 3. Exp3-IX Bandit (from Mathematical Foundations)

**Decision**: Use Exp3-IX for online weight adaptation.

**Rationale**: Works in adversarial environments, doesn't assume stationary rewards.

**Parameters**: γ=0.05, η_max=0.07

---

### 4. Barrier Gates (from Mathematical Foundations)

**Decision**: Block entries when P(peak) > 0.6.

**Rationale**: Prevents buying at local tops. The 60% threshold provides balance between false positives and protection.

**Formula**: `g = max(g_min, (1-p_peak)^γ × (0.5+0.5×p_valley)^δ)`

---

### 5. Protocol-Based Architecture (from Architecture Plans)

**Decision**: Use Python Protocol for brokers and data providers.

**Rationale**: Enables easy swapping of implementations without inheritance. Supports structural subtyping.

```python
@runtime_checkable
class Broker(Protocol):
    def submit_order(...) -> str: ...
```

---

## Future Plans

### Pending Implementation

| Plan | Status | Impact |
|------|--------|--------|
| Phase 8 (Multi-Horizon Training) | Not Started | Will improve model training |
| Phase 9 (Cross-Horizon Ensemble) | Not Started | Already implemented in LIVE_TRADING |
| Phase 10 (Multi-Interval) | Not Started | Will enable cross-interval experiments |
| Interval-Agnostic Pipeline | Ready | Will standardize interval handling |

### Backlog

- Live Alpaca/IBKR data providers
- Comprehensive backtesting module
- Prometheus/Grafana dashboards
- Advanced alerting rules

---

## Document Maintenance

When modifying LIVE_TRADING:

1. **Check relevant plans** for design decisions
2. **Update skills** if patterns change
3. **Update this document** if new plans are created
4. **Cross-reference** Mathematical Foundations for formulas

---

## Related Documentation

- [../README.md](../README.md) - Documentation index
- [../architecture/SYSTEM_ARCHITECTURE.md](../architecture/SYSTEM_ARCHITECTURE.md) - System architecture
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) - Configuration options
