# Phase 4: Live Trading Infrastructure

## Overview

Phase 4 extends the LIVE_TRADING module with production-ready infrastructure:
- **Phase 0**: Critical foundation fixes (MUST DO FIRST)
- **Phase 4a**: Live broker integrations (Alpaca, IBKR)
- **Phase 4b**: Real-time data providers (Alpaca, Polygon)
- **Phase 4c**: Backtesting mode
- **Phase 4d**: Observability and alerting
- **Phase 4e**: Rust performance extensions (optional)

## Critical: Phase 0 Foundation Fixes

Before implementing any new features, these critical issues must be fixed:

| Plan | Component | Issue | Severity |
|------|-----------|-------|----------|
| **0A** | Clock Abstraction | `datetime.now()` hardcoded - breaks backtesting | CRITICAL |
| **0B** | Order Lifecycle | No partial fills, no order state machine | CRITICAL |
| **0C** | Position Reconciliation | Local state never verified against broker | CRITICAL |
| **0D** | State Persistence | Crash during save corrupts state, no recovery | CRITICAL |
| **0E** | Timezone Standardization | Mixed UTC/naive datetimes cause comparison failures | MEDIUM |

**Estimated LOC for Phase 0**: ~3,000

## Sub-Plans

### Phase 0: Foundation Fixes (MUST DO FIRST)

| Plan | Component | Description | Est. LOC | Status |
|------|-----------|-------------|----------|--------|
| 0A | Clock Abstraction | `Clock` protocol, `SimulatedClock` for backtesting | ~430 | **DONE** |
| 0B | Order Lifecycle | Order state machine, partial fills, OrderBook | ~730 | **DONE** |
| 0C | Position Reconciliation | Verify local state matches broker | ~590 | **DONE** |
| 0D | State Persistence | Atomic writes, WAL, crash recovery | ~730 | **DONE** |
| 0E | Timezone Standardization | UTC everywhere, validation utilities | ~590 | **DONE** |
| **Subtotal** | | | **~3,070** | |

### Phase 4: Live Trading Features

| Plan | Component | Description | Est. LOC | Status |
|------|-----------|-------------|----------|--------|
| 12 | Alpaca Broker | Alpaca trading API integration | ~450 | **DONE** |
| 13 | IBKR Broker | Interactive Brokers TWS/Gateway integration | ~500 | **DONE** |
| 14 | Live Data Providers | Alpaca + Polygon real-time data | ~450 | **DONE** |
| 15 | Backtesting Engine | Historical simulation mode | ~600 | **DONE** |
| 16 | Observability | Metrics hooks and event system | ~350 | **DONE** |
| 17 | Alerting | Webhook-based notification system | ~670 | **DONE** |
| 18 | Rust Core (optional) | Performance-critical paths in Rust | ~880 | PENDING |
| **Subtotal** | | | **~3,900** | |

### Total Estimated LOC: ~7,000

## Implementation Order

```
PHASE 0: CRITICAL FIXES (Must complete before live trading)
============================================================
0A_clock_abstraction.md      ← START HERE (enables backtesting)
0B_order_lifecycle.md        ← Proper order handling
0C_position_reconciliation.md ← Safety net for crashes
0D_state_persistence.md      ← Crash recovery
0E_timezone_standardization.md ← Consistency

PHASE 4a: VALIDATION (Backtest before live)
============================================
14_live_data_providers.md    ← Real market data
15_backtesting.md            ← Validate strategy before live

PHASE 4b: INFRASTRUCTURE
========================
16_observability.md          ← Metrics for analysis
17_alerting.md               ← Notifications

PHASE 4c: LIVE BROKERS
======================
12_alpaca_broker.md          ← Paper trading first
13_ibkr_broker.md            ← Alternative broker

PHASE 4d: PERFORMANCE (Optional)
================================
18_rust_core.md              ← Rust hot paths
```

## Rationale for Order

1. **Clock abstraction first**: Without it, backtesting is impossible - all timestamps use wall-clock time.

2. **Order lifecycle before brokers**: Live brokers return partial fills and pending orders. Without proper order management, the engine can't handle real broker responses.

3. **Position reconciliation before live**: If engine crashes and restarts with stale state, it needs to sync with broker before trading.

4. **Backtesting before live brokers**: Validate the strategy works on historical data before risking real money.

5. **Observability before live**: Need metrics to analyze backtest results and monitor live performance.

6. **Rust extensions last**: Performance optimization after correctness is proven.

## Design Principles

1. **Protocol-Based**: All integrations implement existing Protocols (Broker, DataProvider, Clock)
2. **Modular**: Each component is independently testable and swappable
3. **Extensible**: Easy to add new brokers, data sources, or notification channels
4. **Fail-Safe**: Graceful degradation, comprehensive error handling
5. **Deterministic**: Clock abstraction enables reproducible backtests

## Architecture Changes

### New Directory Structure

```
LIVE_TRADING/
├── common/
│   ├── clock.py              # NEW: Clock abstraction
│   ├── order.py              # NEW: Order lifecycle
│   ├── reconciliation.py     # NEW: Position reconciliation
│   ├── persistence.py        # NEW: State persistence + WAL
│   ├── time_utils.py         # NEW: Timezone utilities
│   └── ...
│
├── brokers/
│   ├── interface.py          # Existing Protocol
│   ├── paper.py              # Existing PaperBroker
│   ├── alpaca.py             # NEW: Alpaca broker
│   └── ibkr.py               # NEW: IBKR broker
│
├── data/                      # NEW: Data providers directory
│   ├── __init__.py
│   ├── interface.py          # Move DataProvider Protocol here
│   ├── simulated.py          # Move SimulatedDataProvider here
│   ├── alpaca.py             # NEW: Alpaca data
│   └── polygon.py            # NEW: Polygon data
│
├── backtest/                  # NEW: Backtesting module
│   ├── __init__.py
│   ├── engine.py             # Backtest orchestrator
│   ├── data_loader.py        # Historical data loading
│   └── report.py             # Performance reporting
│
├── observability/             # NEW: Metrics and events
│   ├── __init__.py
│   ├── metrics.py            # Metrics registry and hooks
│   ├── events.py             # Event bus for system events
│   └── exporters.py          # Prometheus/StatsD exporters (future)
│
├── alerting/                  # NEW: Notification system
│   ├── __init__.py
│   ├── manager.py            # Alert manager
│   └── channels.py           # Webhook, Discord, Slack channels
│
└── rust/                      # NEW: Rust extensions (optional)
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        ├── orderbook.rs
        ├── risk.rs
        └── signals.rs
```

## Configuration Additions

```yaml
# CONFIG/live_trading/live_trading.yaml additions

live_trading:
  # Reconciliation
  reconciliation:
    mode: "warn"           # strict, warn, auto_sync
    qty_tolerance: 0.01
    cash_tolerance: 1.0
    periodic_interval: 100

  # Persistence
  persistence:
    backup_count: 5
    use_wal: true
    wal_max_entries: 10000

  # Broker configuration
  brokers:
    alpaca:
      api_key_env: "ALPACA_API_KEY"
      api_secret_env: "ALPACA_API_SECRET"
      base_url: "https://paper-api.alpaca.markets"
      data_feed: "iex"

    ibkr:
      host: "127.0.0.1"
      port: 7497  # Paper: 7497, Live: 7496
      client_id: 1

  # Data provider configuration
  data:
    provider: "alpaca"
    alpaca:
      feed: "iex"
    polygon:
      api_key_env: "POLYGON_API_KEY"

  # Observability
  observability:
    enabled: true
    metrics:
      prefix: "live_trading"
      export_interval_seconds: 60
    events:
      buffer_size: 1000

  # Alerting
  alerting:
    min_severity: "info"
    rate_limit_seconds: 1.0
    webhook:
      url: null  # Set via env or config
    discord:
      webhook_url: null
    slack:
      webhook_url: null

  # Backtesting
  backtest:
    data_dir: "data/historical"
    default_start: "2020-01-01"
    default_end: "2024-01-01"
    slippage_model: "fixed"
    slippage_bps: 5.0
```

## Dependencies

New packages required:
```
# Phase 0 - no new dependencies

# Phase 4
alpaca-trade-api>=3.0.0    # Alpaca broker/data
ib_insync>=0.9.86          # IBKR (async wrapper)
polygon>=1.0.0             # Polygon data (optional)
aiohttp>=3.8.0             # Async HTTP for webhooks

# Rust extensions (optional)
maturin>=1.0               # Rust-Python build tool
```

## Success Criteria

### Phase 0
- [x] Clock abstraction in place, all datetime.now() removed (0A - DONE)
- [x] Order lifecycle handles partial fills (0B - DONE)
- [x] Position reconciliation runs on startup (0C - DONE)
- [x] State survives crash and recovers correctly (0D - DONE)
- [x] All datetimes are UTC timezone-aware (0E - DONE)

### Phase 4
- [x] Can execute real trades via Alpaca paper trading (Plan 12 - DONE)
- [x] Can execute real trades via IBKR paper trading (Plan 13 - DONE)
- [x] Can stream real-time quotes from Alpaca/Polygon (Plan 14 - DONE)
- [x] Backtest produces consistent results vs live (Plan 15 - DONE)
- [x] Metrics hooks capture all key events (Plan 16 - DONE)
- [x] Alerts fire on kill switch triggers (Plan 17 - DONE)
- [x] All tests pass: 707 passed, 5 skipped (optional deps)

### Post-Implementation Code Review
- [x] Deep code review completed - see `CODE_REVIEW_ISSUES.md`
- [ ] Critical issues fixed (C1-C4)
- [ ] High priority issues fixed (H1-H6)
- [ ] Medium issues addressed (M1-M10)
- [ ] Integration gaps closed (I1-I5)
- [ ] Test coverage expanded (T1-T4)

## Session Planning

Recommended session breakdown:

| Session | Plans | Focus |
|---------|-------|-------|
| 1 | 0A, 0E | Clock + Timezone (foundation) |
| 2 | 0B | Order lifecycle |
| 3 | 0C, 0D | Reconciliation + Persistence |
| 4 | 14 | Data providers |
| 5 | 15 | Backtesting |
| 6 | 16 | Observability |
| 7 | 12 | Alpaca broker |
| 8 | 13 | IBKR broker |
| 9 | 17 | Alerting |
| 10 | 18 | Rust extensions (optional) |

Each session produces working, tested code.

## Related Documents

- **`CODE_REVIEW_ISSUES.md`** - Comprehensive issues from deep code review (36 issues identified)
- **`0A_clock_abstraction.md`** - Clock abstraction implementation plan
- **`0B_order_lifecycle.md`** - Order lifecycle implementation plan
- **`0C_position_reconciliation.md`** - Position reconciliation plan
- **`0D_state_persistence.md`** - State persistence plan
- **`0E_timezone_standardization.md`** - Timezone standardization plan
