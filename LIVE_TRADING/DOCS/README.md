# LIVE_TRADING Module Documentation

This directory contains comprehensive documentation for the LIVE_TRADING execution engine module.

## Overview

The LIVE_TRADING module is a production-grade execution engine for deploying ML models trained by the FoxML TRAINING pipeline. It implements a sophisticated 6-stage trading pipeline with cost-aware decision making, multi-horizon predictions, and online learning capabilities.

**Implementation Status**: Phase 3 COMPLETE (365+ tests passing)

---

## Documentation Index

### Architecture

| Document | Description |
|----------|-------------|
| [SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) | Overall system architecture, component connectivity, and design patterns |
| [PIPELINE_STAGES.md](architecture/PIPELINE_STAGES.md) | Detailed breakdown of the 6-stage trading pipeline |
| [MATHEMATICAL_FORMULAS.md](architecture/MATHEMATICAL_FORMULAS.md) | Mathematical foundations and formulas used throughout |

### Components

| Document | Description |
|----------|-------------|
| [MODEL_INFERENCE.md](components/MODEL_INFERENCE.md) | Model loading, family-specific inference, and feature building |
| [MULTI_HORIZON_AND_INTERVAL.md](components/MULTI_HORIZON_AND_INTERVAL.md) | Multi-horizon predictions, interval mapping, and cross-horizon blending |
| [ONLINE_LEARNING.md](components/ONLINE_LEARNING.md) | Exp3-IX bandit algorithm for adaptive weight optimization |
| [RISK_MANAGEMENT.md](components/RISK_MANAGEMENT.md) | Kill switches, drawdown monitoring, and exposure tracking |

### Reference

| Document | Description |
|----------|-------------|
| [CONFIGURATION_REFERENCE.md](reference/CONFIGURATION_REFERENCE.md) | Complete configuration options and defaults |
| [PLAN_REFERENCES.md](reference/PLAN_REFERENCES.md) | Links to plan documents used to build this module |

---

## Quick Start

```bash
# Paper trading with models from a training run
python -m bin.run_live_trading --run-id my_run --broker paper

# Dry run (simulated data)
python -m bin.run_live_trading --dry-run --symbols SPY QQQ AAPL

# Specific symbols with limited cycles
python -m bin.run_live_trading --run-id my_run --symbols SPY QQQ --max-cycles 10
```

---

## Module Structure

```
LIVE_TRADING/
├── cli/                 # Command-line interface
├── common/              # Shared types, constants, exceptions
├── brokers/             # Broker adapters (Paper, IBKR, Alpaca)
├── data/                # Data providers
├── models/              # Model loading and inference
├── prediction/          # Multi-horizon prediction pipeline
├── blending/            # Ridge risk-parity blending
├── arbitration/         # Cost-aware horizon selection
├── gating/              # Entry gates (barrier, spread)
├── sizing/              # Position sizing engine
├── risk/                # Risk management and kill switches
├── engine/              # Main trading orchestrator
├── learning/            # Online learning (bandits)
├── observability/       # Events and metrics
├── alerting/            # Alert management
├── backtest/            # Backtesting engine
└── tests/               # Comprehensive test suite
```

---

## Related Documentation

- **Main README**: `LIVE_TRADING/README.md`
- **Implementation Progress**: `LIVE_TRADING/IMPLEMENTATION_PROGRESS.md`
- **Mathematical Foundations**: `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
- **SST Solutions**: `INTERNAL/docs/references/SST_SOLUTIONS.md`

---

## Plan Documents Used

The LIVE_TRADING module was built using several plan documents:

- **Multi-Horizon Training**: `.claude/plans/multi-horizon-training-master.md`
- **Phase 8-10**: Multi-horizon training, cross-horizon ensemble, multi-interval experiments
- **Interval-Agnostic Pipeline**: `.claude/plans/interval-agnostic-pipeline.md`

See [PLAN_REFERENCES.md](reference/PLAN_REFERENCES.md) for detailed information.
