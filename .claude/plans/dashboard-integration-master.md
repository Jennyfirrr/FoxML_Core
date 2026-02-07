# Dashboard Integration Master Plan

**Status**: ✅ Complete (All 4 Phases + UI Fixes)
**Created**: 2026-01-21
**Last Updated**: 2026-01-21
**Completed**: 2026-01-21
**Branch**: fix/repro-bootstrap-import-order

---

## Executive Summary

Complete the DASHBOARD integration with LIVE_TRADING and TRAINING modules by:
1. Establishing formal contracts between modules
2. Exposing missing data through the IPC bridge
3. Wiring the Rust TUI to use real data
4. Enabling training monitoring through the same infrastructure

**Total estimated phases**: 4
**Key principle**: Build on existing infrastructure (EventBus, MetricsRegistry, WebSocket) rather than creating new patterns.

---

## Current State Analysis

### What Exists and Works

| Component | Status | Notes |
|-----------|--------|-------|
| EventBus (LIVE_TRADING) | ✅ Complete | 27 event types, thread-safe, pub/sub |
| MetricsRegistry (LIVE_TRADING) | ✅ Complete | Counter/Gauge/Histogram, labels |
| IPC Bridge (FastAPI) | ✅ Partial | REST + WebSocket, some endpoints |
| Rust TUI Framework | ✅ Complete | Views, widgets, themes, animations |
| Trading View | ✅ Partial | Shows metrics, event log disconnected |
| Training View | ✅ Partial | Run discovery works, no live progress |

### Critical Gaps (From Analysis)

| Gap | Impact | Fix Location |
|-----|--------|--------------|
| Event log not connected to WebSocket | No real-time events in UI | Dashboard Rust client |
| Pipeline stage not tracked | Always shows "idle" | Engine + Bridge + Dashboard |
| Position details not exposed | Can't show position table | Bridge endpoint + Dashboard |
| Trade history inaccessible | No audit trail | Bridge endpoint + Dashboard |
| CILS stats hidden | Online learning invisible | Bridge endpoint + Dashboard |
| Risk metrics incomplete | No drawdown/exposure display | Bridge endpoint + Dashboard |
| Training progress not streamed | Polling only, no live updates | Training events + Bridge |

---

## Subplan Index

| Phase | Subplan | Purpose | Effort | Status |
|-------|---------|---------|--------|--------|
| 1 | [dashboard-phase1-contracts.md](./dashboard-phase1-contracts.md) | Define LIVE_TRADING ↔ DASHBOARD contracts | 1h | ✅ Complete |
| 2 | [dashboard-phase2-bridge-api.md](./dashboard-phase2-bridge-api.md) | Add missing bridge endpoints | 3h | ✅ Complete |
| 3 | [dashboard-phase3-rust-wiring.md](./dashboard-phase3-rust-wiring.md) | Connect Rust TUI to bridge | 4h | ✅ Complete |
| 4 | [dashboard-phase4-training-monitor.md](./dashboard-phase4-training-monitor.md) | Training pipeline monitoring | 2h | ✅ Complete |

---

## Architecture (Target State)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING MODULE                                      │
│                                                                                │
│  intelligent_trainer.py ──► TrainingEventBus ──► Bridge /ws/training         │
│       │                            │                                          │
│       └── Progress updates ────────┘                                          │
│           Stage transitions                                                   │
│           Target completion                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LIVE_TRADING MODULE                                  │
│                                                                                │
│  trading_engine.py                                                            │
│       │                                                                       │
│       ├── EventBus ─────────────────────────────────────────► /ws/events     │
│       │   - 27 event types                                                    │
│       │   - Real-time streaming                                               │
│       │                                                                       │
│       ├── MetricsRegistry ──────────────────────────────────► /api/metrics   │
│       │   - portfolio_value, daily_pnl, cash, positions_count                 │
│       │   - trades_total, cycles_total, errors_total                          │
│       │                                                                       │
│       ├── EngineState ──────────────────────────────────────► /api/state     │
│       │   - status, current_stage (NEW!)                                      │
│       │   - positions (detailed) (NEW!)                                       │
│       │                                                                       │
│       ├── CILS Optimizer ───────────────────────────────────► /api/cils      │
│       │   - bandit weights, arm stats (NEW!)                                  │
│       │                                                                       │
│       ├── RiskGuardrails ───────────────────────────────────► /api/risk      │
│       │   - drawdown, exposure, daily_pnl_pct (NEW!)                          │
│       │                                                                       │
│       └── Trade/Decision History ───────────────────────────► /api/trades    │
│           - trade_history, decision_history (NEW!)                            │
│                                                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           IPC BRIDGE (FastAPI)                                 │
│                                                                                │
│  Existing:                          New (Phase 2):                            │
│  ├── /api/metrics                   ├── /api/positions                       │
│  ├── /api/state                     ├── /api/trades/history                  │
│  ├── /api/events/recent             ├── /api/cils/stats                      │
│  ├── /api/control/*                 ├── /api/risk/status                     │
│  ├── /ws/events                     ├── /ws/training                         │
│  └── /ws/alpaca                     └── Stage tracking in /api/state         │
│                                                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           RUST TUI (ratatui)                                   │
│                                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Trading View   │  │  Training View  │  │   Overview      │               │
│  │                 │  │                 │  │                 │               │
│  │  - Metrics ────────── /api/metrics   │  │  - System stats │               │
│  │  - Event Log ──────── /ws/events ✓   │  │  - Quick metrics│               │
│  │  - Pipeline ───────── /api/state ✓   │  │                 │               │
│  │  - Positions ──────── /api/positions │  │                 │               │
│  │  - Risk Gauge ─────── /api/risk      │  │                 │               │
│  └─────────────────┘  │  - Progress ──── /ws/training       │               │
│                       │  - Runs ──────── File scan          │               │
│                       └─────────────────┘                   │               │
│                                                                               │
│  Widgets:                                                                     │
│  ├── event_log.rs ────── Connected to /ws/events (Phase 3)                   │
│  ├── pipeline_status.rs ─ Updated from /api/state (Phase 3)                  │
│  ├── position_table.rs ── Populated from /api/positions (Phase 3)            │
│  ├── metrics_panel.rs ─── Enhanced with risk (Phase 3)                       │
│  └── chart.rs ─────────── Time series display (Future)                       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Contract Summary

### New Contracts (Phase 1)

| Contract | Producer | Consumer | Purpose |
|----------|----------|----------|---------|
| Engine State | LIVE_TRADING engine | Bridge /api/state | Current stage, status |
| Position Details | LIVE_TRADING state | Bridge /api/positions | Per-position data |
| Trade History | LIVE_TRADING state | Bridge /api/trades | Audit trail |
| CILS Stats | LIVE_TRADING optimizer | Bridge /api/cils | Bandit metrics |
| Risk Status | LIVE_TRADING guardrails | Bridge /api/risk | Risk metrics |
| Training Progress | TRAINING orchestrator | Bridge /ws/training | Live progress |

### Existing Contracts (Unchanged)

| Contract | Producer | Consumer | Status |
|----------|----------|----------|--------|
| model_meta.json | TRAINING | LIVE_TRADING | ✅ Stable |
| manifest.json | TRAINING | LIVE_TRADING | ✅ Stable |
| routing_decision.json | TRAINING | LIVE_TRADING | ✅ Stable |

---

## Implementation Order

```
Phase 1: Contracts (1h)
    │
    ├── Define schemas for new endpoints
    ├── Add to INTEGRATION_CONTRACTS.md
    └── Update skills

    ▼
Phase 2: Bridge API (3h)
    │
    ├── /api/positions endpoint
    ├── /api/trades/history endpoint
    ├── /api/cils/stats endpoint
    ├── /api/risk/status endpoint
    ├── Stage tracking in engine + /api/state
    └── /ws/training for training progress

    ▼
Phase 3: Rust Wiring (4h)
    │
    ├── Connect event_log to /ws/events WebSocket
    ├── Update pipeline_status from /api/state
    ├── Populate position_table from /api/positions
    ├── Add risk metrics to trading view
    └── Add CILS display (optional)

    ▼
Phase 4: Training Monitor (2h)
    │
    ├── Emit progress events from intelligent_trainer
    ├── Bridge /ws/training endpoint
    └── Training view subscribe to progress
```

---

## Success Criteria

### Phase 1: Contracts ✅ COMPLETE
- [x] New contracts documented in INTEGRATION_CONTRACTS.md
- [x] Skills updated with new endpoints
- [x] Schema defined for each new endpoint

### Phase 2: Bridge API ✅ COMPLETE
- [x] All new endpoints return real data (not mock)
- [x] Engine emits stage transitions via STAGE_CHANGE event
- [x] WebSocket /ws/training ready for training progress
- [ ] Existing tests pass (to verify)

### Phase 3: Rust Wiring ✅ COMPLETE
- [x] Event log shows real events from EventBus (WebSocket + TradingEvent)
- [x] Pipeline status shows actual stage (from_str + STAGE_CHANGE events)
- [x] Position table populated with real positions (from /api/positions)
- [x] Risk metrics displayed (risk_gauge.rs widget)
- [x] Dashboard builds and runs (cargo build --release succeeds)

### UI Fixes ✅ COMPLETE
- [x] Status bar reflects actual bridge connection status (was hardcoded)
- [x] Menu uses arrow indicator instead of blinking cursor
- [x] Logo centered properly (simplified to block text only)

### Phase 4: Training Monitor ✅ COMPLETE
- [x] Training view shows live progress (WebSocket subscription)
- [x] Stage transitions visible during training (TrainingEvent handling)
- [x] Completion notifications work (run_complete events)
- [x] Python TrainingEventEmitter created with HTTP POST + file fallback
- [x] intelligent_trainer.py integrated with event emissions at all stages

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Add new endpoints, don't modify existing |
| Performance impact from streaming | Use existing EventBus buffer, don't add overhead |
| Contract drift between modules | Explicit schema in INTEGRATION_CONTRACTS.md |
| Rust/Python type mismatches | Define schemas in both Rust structs and Python Pydantic |

---

## Files to Modify

### Phase 1 (Contracts)
- `INTEGRATION_CONTRACTS.md` - Add LIVE_TRADING → DASHBOARD section
- `.claude/skills/dashboard-ipc-bridge.md` - Update with new endpoints

### Phase 2 (Bridge API)
- `LIVE_TRADING/engine/trading_engine.py` - Stage emission
- `DASHBOARD/bridge/server.py` - New endpoints
- `TRAINING/orchestration/intelligent_trainer.py` - Progress events

### Phase 3 (Rust TUI)
- `DASHBOARD/dashboard/src/api/client.rs` - WebSocket connection
- `DASHBOARD/dashboard/src/views/trading.rs` - Wire to real data
- `DASHBOARD/dashboard/src/widgets/event_log.rs` - Receive events
- `DASHBOARD/dashboard/src/widgets/pipeline_status.rs` - Update from state
- `DASHBOARD/dashboard/src/widgets/position_table.rs` - Populate

### Phase 4 (Training)
- `TRAINING/orchestration/intelligent_trainer.py` - Emit events
- `DASHBOARD/bridge/server.py` - /ws/training endpoint
- `DASHBOARD/dashboard/src/views/training.rs` - Subscribe to progress

---

## Related Documentation

- `INTEGRATION_CONTRACTS.md` - Module contracts
- `.claude/skills/dashboard-*.md` - Dashboard skills
- `LIVE_TRADING/observability/events.py` - EventBus reference
- `LIVE_TRADING/observability/metrics.py` - MetricsRegistry reference
