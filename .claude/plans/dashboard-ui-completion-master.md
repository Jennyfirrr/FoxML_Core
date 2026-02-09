# Dashboard UI Completion Master Plan

**Status**: Superseded by `dashboard-hardening-master.md` (all views now exist and are functional)
**Created**: 2026-01-21
**Branch**: fix/repro-bootstrap-import-order

---

## Executive Summary

Complete all placeholder screens in the dashboard and make the launcher front pane actually useful by showing live training/trading status at a glance.

**Total subplans**: 7
**Key principle**: Each view should be functional and useful, not just pretty. Prioritize features that provide real value over polish.

---

## Current State

### Working Views
| View | Status | Notes |
|------|--------|-------|
| Launcher | ✅ Works | Menu only - front pane is just logo + status boxes |
| Overview | ✅ Works | Basic system info |
| Trading Monitor | ✅ Works | Real-time events, positions, risk |
| Training Monitor | ✅ Works | File-based progress monitoring |

### Placeholder Views (Need Implementation)
| View | Current State | Priority |
|------|---------------|----------|
| Launcher front pane | Static status boxes | **P0** - First thing users see |
| Config Editor | "Coming soon" | **P1** - Core functionality |
| Log Viewer | "Coming soon" | **P1** - Essential for debugging |
| Service Manager | "Coming soon" | **P2** - Nice to have |
| Model Selector | "Coming soon" | **P2** - Nice to have |
| File Browser | "Coming soon" | **P3** - Lower priority |
| Settings | "Coming soon" | **P3** - Lower priority |

---

## Subplan Index

| Phase | Subplan | Purpose | Effort | Status |
|-------|---------|---------|--------|--------|
| 0 | [ui-phase0-launcher-dashboard.md](./ui-phase0-launcher-dashboard.md) | Make front pane show live status | 2h | ✅ Complete |
| 1 | [ui-phase1-config-editor.md](./ui-phase1-config-editor.md) | Wire up existing config editor | 1h | Pending |
| 2 | [ui-phase2-log-viewer.md](./ui-phase2-log-viewer.md) | Wire up log viewer | 1h | Pending |
| 3 | [ui-phase3-service-manager.md](./ui-phase3-service-manager.md) | Wire up service manager | 1h | Pending |
| 4 | [ui-phase4-model-selector.md](./ui-phase4-model-selector.md) | Implement model browser | 2h | Pending |
| 5 | [ui-phase5-file-browser.md](./ui-phase5-file-browser.md) | Wire up file browser | 1h | Pending |
| 6 | [ui-phase6-settings.md](./ui-phase6-settings.md) | Implement settings | 1.5h | Pending |

**Note**: Partial implementations already exist in `src/launcher/`. Phases 1-3, 5 mainly need wiring to views.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LAUNCHER (Front Pane)                          │
│                                                                             │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │      MENU               │  │           LIVE STATUS DASHBOARD         │  │
│  │                         │  │                                         │  │
│  │  ▸ Trading Monitor      │  │  ┌─────────────┐  ┌─────────────────┐  │  │
│  │    Training Monitor     │  │  │  TRAINING   │  │    TRADING      │  │  │
│  │    Config Editor        │  │  │             │  │                 │  │  │
│  │    Model Selector       │  │  │  Stage: ... │  │  P&L: +$1,234   │  │  │
│  │    Service Manager      │  │  │  [████░░] 60%│  │  Positions: 5   │  │  │
│  │    Log Viewer           │  │  │  Target: ... │  │  Status: Active │  │  │
│  │    File Browser         │  │  │             │  │                 │  │  │
│  │    Settings             │  │  └─────────────┘  └─────────────────┘  │  │
│  │                         │  │                                         │  │
│  │                         │  │  ┌─────────────────────────────────┐   │  │
│  │                         │  │  │      RECENT EVENTS              │   │  │
│  │                         │  │  │  10:30 TRADE_FILLED AAPL +100   │   │  │
│  │                         │  │  │  10:29 DECISION_MADE MSFT BUY   │   │  │
│  │                         │  │  └─────────────────────────────────┘   │  │
│  └─────────────────────────┘  └─────────────────────────────────────────┘  │
│                                                                             │
│  [↑↓] Navigate  │  [Enter] Select  │  [q] Quit  │  Bridge: ● Connected     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Launcher Dashboard (P0 - Critical)

**Goal**: Replace static status boxes with live training/trading dashboard

**Current state**: Logo + static Bridge/Alpaca/Engine status boxes
**Target state**: Live mini-dashboard showing:
- Training progress (if running): stage, progress bar, current target
- Trading status (if running): P&L, position count, last event
- Recent events feed (last 5 events from either system)
- System health indicators

**Data sources**:
- Training: Read `/tmp/foxml_training_events.jsonl` (same as training view)
- Trading: Read from bridge `/api/metrics`, `/api/state`, `/ws/events`

---

## Phase 1: Config Editor (P1 - Core)

**Goal**: Edit YAML configuration files with syntax highlighting and validation

**Features**:
- File tree of CONFIG/ directory
- YAML syntax highlighting
- Schema validation (warn on invalid config)
- Save with atomic write
- Diff view (show unsaved changes)

**Key files**:
- `CONFIG/experiments/*.yaml`
- `CONFIG/pipeline/*.yaml`
- `CONFIG/models/*.yaml`

---

## Phase 2: Log Viewer (P1 - Core)

**Goal**: View and filter log files in real-time

**Features**:
- Tail mode (follow new lines)
- Search/filter by pattern
- Log level filtering (ERROR, WARN, INFO, DEBUG)
- Multiple log sources:
  - Training logs: `RESULTS/runs/*/logs/`
  - Trading logs: `LIVE_TRADING/logs/`
  - System logs: journalctl for trading service

---

## Phase 3: Service Manager (P2)

**Goal**: Control trading service via systemctl

**Features**:
- Show service status (running/stopped/failed)
- Start/stop/restart service
- View recent journal entries
- Show service configuration

**Commands**:
```bash
systemctl --user status foxml-trading
systemctl --user start foxml-trading
systemctl --user stop foxml-trading
journalctl --user -u foxml-trading -n 50
```

---

## Phase 4: Model Selector (P2)

**Goal**: Browse trained models and select for live trading

**Features**:
- List all trained models from RESULTS/runs/
- Show model metrics (AUC, features, training date)
- Compare models side-by-side
- Select model for live trading (symlink or config update)
- Show which model is currently active

---

## Phase 5: File Browser (P3)

**Goal**: Navigate filesystem and view files

**Features**:
- Directory tree navigation
- File preview (syntax highlighted for code)
- Quick jump to common directories (CONFIG, RESULTS, TRAINING)
- File info (size, modified time)

---

## Phase 6: Settings (P3)

**Goal**: Configure dashboard preferences

**Features**:
- Theme selection (or auto-detect toggle)
- Keybind customization
- Default view on startup
- Refresh intervals
- Bridge URL configuration

**Storage**: `~/.config/foxml-dashboard/settings.yaml`

---

## Implementation Order

```
Phase 0: Launcher Dashboard (2h)
    │
    └── Makes front pane useful immediately

    ▼
Phase 1: Config Editor (3h)
    │
    └── Core functionality for config management

    ▼
Phase 2: Log Viewer (2h)
    │
    └── Essential for debugging

    ▼
Phase 3: Service Manager (1.5h)
    │
    └── Control trading service

    ▼
Phase 4: Model Selector (2h)
    │
    └── Model management

    ▼
Phase 5: File Browser (2h)
    │
    └── General utility

    ▼
Phase 6: Settings (1.5h)
    │
    └── Polish and customization
```

---

## Success Criteria

### Phase 0: Launcher Dashboard
- [ ] Front pane shows live training progress when training is running
- [ ] Front pane shows live trading metrics when trading is running
- [ ] Recent events feed shows last 5 events
- [ ] Status indicators update in real-time

### Phase 1: Config Editor
- [ ] Can browse CONFIG/ directory tree
- [ ] YAML syntax highlighting works
- [ ] Can edit and save files
- [ ] Warns on invalid YAML

### Phase 2: Log Viewer
- [ ] Can tail log files in real-time
- [ ] Search/filter works
- [ ] Can switch between log sources

### Phase 3: Service Manager
- [ ] Shows service status
- [ ] Can start/stop/restart service
- [ ] Shows recent logs

### Phase 4: Model Selector
- [ ] Lists all trained models
- [ ] Shows model metrics
- [ ] Can select model for live trading

### Phase 5: File Browser
- [ ] Can navigate directories
- [ ] Can preview files
- [ ] Quick jump to common paths works

### Phase 6: Settings
- [ ] Can change theme
- [ ] Settings persist across restarts

---

## Files to Create/Modify

| Phase | Files |
|-------|-------|
| 0 | `src/launcher/live_dashboard.rs` (new), `src/views/launcher.rs` |
| 1 | `src/views/config_editor.rs` (new), `src/views/mod.rs`, `src/app.rs` |
| 2 | `src/views/log_viewer.rs` (new), `src/views/mod.rs`, `src/app.rs` |
| 3 | `src/views/service_manager.rs` (new), `src/views/mod.rs`, `src/app.rs` |
| 4 | `src/views/model_selector.rs` (new), `src/views/mod.rs`, `src/app.rs` |
| 5 | `src/views/file_browser.rs` (new), `src/views/mod.rs`, `src/app.rs` |
| 6 | `src/views/settings.rs` (new), `src/config.rs` (new) |

---

## Dependencies

- **Phase 0**: None (uses existing event file + bridge API)
- **Phase 1**: None (file I/O only)
- **Phase 2**: None (file I/O + optional journalctl)
- **Phase 3**: systemd user service must be set up
- **Phase 4**: Depends on RESULTS/ structure from training
- **Phase 5**: None
- **Phase 6**: None

---

## Notes

- Remove the sidebar/status canvas from launcher if it becomes redundant after Phase 0
- Consider combining some views (e.g., Log Viewer + File Browser could share code)
- All views should follow the existing theme system
- Use Panel widget for consistent styling
