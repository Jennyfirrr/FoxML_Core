# Dashboard Hardening & Feature Completion — Master Plan

**Status**: Complete (all 6 phases done)
**Created**: 2026-02-09
**Branch**: `analysis/code-review-and-raw-ohlcv`
**Predecessor Plans**: `dashboard-ui-completion-master.md` (all views exist), `dashboard-bug-fixes-master.md` (13 bugs fixed), `wiggly-wondering-hamming.md` (config editor ViewAction)

---

## Executive Summary

All dashboard views exist and render, but several features are stubbed, broken, or missing. This plan addresses 20 issues across 6 phases: critical functional fixes, safety/robustness, chart widget, trading enhancements, training enhancements, and UX consistency.

**Total subplans**: 6
**Key principle**: Fix what's broken before adding new things. Wire existing backend endpoints before building new ones.

---

## Current State

### What Works
| Component | Status | Notes |
|-----------|--------|-------|
| Trading Monitor | 80% | Real-time metrics, positions, events via HTTP+WS |
| Training Monitor | 90% | File-based progress, run discovery, PID detection |
| Config Editor | 85% | Browse, preview, external editor, inline editor |
| Log Viewer | 70% | Tail mode, level coloring, multi-source |
| Service Manager | 75% | Status, start/stop/restart, auto-refresh |
| Model Selector | 85% | Run scanning, metrics, activation via symlink |
| File Browser | 75% | Navigation, preview, quick jumps, hidden toggle |
| Settings | 90% | Theme/refresh/display prefs, save/load |
| Overview | 30% | Raw metrics only, minimal layout |
| Launcher | 95% | Menu + live status dashboard |
| Widgets | 83% | 5/6 complete (chart is TODO) |
| Theme System | 95% | Multi-source detection, 40+ colors |
| IPC Bridge | 90% | HTTP + WS + control endpoints |

### What's Broken or Stubbed
| Issue | Component | Severity |
|-------|-----------|----------|
| Kill switch Ctrl+K is a stub | app.rs | **CRITICAL** — UI suggests it works |
| Log search parsed but never applied | log_viewer.rs | **HIGH** — feature appears to exist |
| File browser dates are wrong | file_browser.rs | **HIGH** — handwritten leap-year math |
| File browser Enter does nothing | file_browser.rs | **MEDIUM** — expected behavior |
| Chart widget is TODO | widgets/chart.rs | **MEDIUM** — no data viz anywhere |
| Help overlay render incomplete | ui/help.rs | **LOW** — data defined, render stub |
| Overview is 30% done | views/overview.rs | **LOW** — functional but barebones |

---

## Subplan Index

| Phase | Subplan | Purpose | Scope | Status |
|-------|---------|---------|-------|--------|
| 0 | [dash-harden-phase0-critical-fixes.md](./dash-harden-phase0-critical-fixes.md) | Fix broken/stubbed features | 4 files | **Complete** |
| 1 | [dash-harden-phase1-safety.md](./dash-harden-phase1-safety.md) | Safety, robustness, error surfacing | 5 files | **Complete** |
| 2 | [dash-harden-phase2-chart-widget.md](./dash-harden-phase2-chart-widget.md) | Implement chart widget + sparklines | 3 files | **Complete** |
| 3 | [dash-harden-phase3-trading.md](./dash-harden-phase3-trading.md) | Trading view enhancements | 2 files | **Complete** |
| 4 | [dash-harden-phase4-training.md](./dash-harden-phase4-training.md) | Training view enhancements | 2 files | **Complete** |
| 5 | [dash-harden-phase5-ux-polish.md](./dash-harden-phase5-ux-polish.md) | UX consistency and polish | 6 files | **Complete** |

---

## Phase 0: Critical Functional Fixes

**Goal**: Fix things that claim to work but don't.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 0a | Wire kill switch to bridge API | app.rs | Ctrl+K → POST `/api/control/kill_switch`, show toggle state |
| 0b | Implement log search filtering | views/log_viewer.rs | `search_query` exists but never filters displayed lines |
| 0c | Fix file browser date formatting | views/file_browser.rs | Replace hand-rolled leap year math with `chrono` (already a dep) |
| 0d | Wire file browser Enter to open | views/file_browser.rs | Use `ViewAction::SpawnEditor(path)` for text files |

**Depends on**: Nothing
**Verification**: Kill switch toggles via bridge, log search filters lines, file dates are correct, Enter opens files

---

## Phase 1: Safety & Robustness

**Goal**: Prevent data loss, surface errors, handle edge cases.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 1a | Add confirmation dialog widget | ui/dialog.rs (new) | Reusable yes/no/cancel modal for destructive actions |
| 1b | Confirmation before model activation | views/model_selector.rs | "Activate run_xyz for live trading?" |
| 1c | Confirmation before service stop | views/service_manager.rs | "Stop foxml-trading?" |
| 1d | Stream large log files | views/log_viewer.rs | Replace `read_to_string()` with seek-to-tail, prevent OOM |
| 1e | Bridge connectivity indicator | ui/status_bar.rs | Show bridge health in status bar (red/green dot) |
| 1f | Surface API errors to user | views/trading.rs | Show notification when bridge calls fail instead of silent `let _ =` |

**Depends on**: Phase 0 (kill switch should also get a confirmation dialog)
**Verification**: Confirmations appear, large logs don't OOM, bridge status visible, errors shown

---

## Phase 2: Chart Widget

**Goal**: Implement the TODO chart widget for data visualization.

| # | Task | File | Notes |
|---|------|------|-------|
| 2a | Implement ASCII sparkline renderer | widgets/chart.rs | Braille-dot or block-char sparklines for inline use |
| 2b | Implement line chart renderer | widgets/chart.rs | Full chart with axis labels for trading view |
| 2c | Add P&L history ring buffer | views/trading.rs | Store last N metric snapshots for charting |
| 2d | Wire sparklines into overview | views/overview.rs | Mini P&L and position count sparklines |

**Depends on**: Nothing (independent)
**Verification**: Trading view shows P&L chart, overview shows sparklines, charts resize with terminal

---

## Phase 3: Trading View Enhancements

**Goal**: Make trading view a full control surface.

| # | Task | File | Notes |
|---|------|------|-------|
| 3a | Position detail panel | views/trading.rs | Enter on position → detail view (entry time, sizing, P&L breakdown) |
| 3b | Position sorting | views/trading.rs | `s` key cycles sort: symbol, P&L, size, weight |
| 3c | Kill switch toggle from trading view | views/trading.rs | `k` key with confirmation, shows current state |
| 3d | Display Sharpe ratio | views/trading.rs | Already fetched in metrics, just needs rendering |
| 3e | Wire pause/resume control | views/trading.rs | `p` key → POST `/api/control/pause` or `/resume` |

**Depends on**: Phase 1 (confirmation dialog for kill switch)
**Verification**: Can inspect positions, sort works, kill switch toggleable, Sharpe visible

---

## Phase 4: Training View Enhancements

**Goal**: Make training view actionable, not just a monitor.

| # | Task | File | Notes |
|---|------|------|-------|
| 4a | Training run cancellation | views/training.rs | `x` key → send SIGTERM to PID from pid file, with confirmation |
| 4b | Cache RESULTS/ directory scan | views/training.rs | Don't WalkDir every 2s, cache + refresh on `r` key |
| 4c | Stage detail panel | views/training.rs | Show completed targets, current target metrics in current stage |
| 4d | Training throughput/ETA | views/training.rs | Compute targets/min from event stream, estimate completion |

**Depends on**: Phase 1 (confirmation dialog for cancellation)
**Verification**: Can cancel run, scan is fast, stage details visible, ETA shown

---

## Phase 5: UX Consistency & Polish

**Goal**: Uniform keybindings, state preservation, complete incomplete views.

| # | Task | File | Notes |
|---|------|------|-------|
| 5a | Add j/k navigation everywhere | views/trading.rs, overview.rs | Trading position nav, overview sections |
| 5b | View state preservation | app.rs | Cache scroll/selection per view, restore on switch-back |
| 5c | Complete overview view | views/overview.rs | System health, resource usage, bridge status, structured layout |
| 5d | Complete help overlay | ui/help.rs | Finish render method, show context-sensitive shortcuts |
| 5e | Model validation on activation | views/model_selector.rs | Check compiled models exist before activating for live trading |
| 5f | Consistent error message timeouts | all views | Messages should auto-clear after 3-5s, not on next keypress |

**Depends on**: Phases 0-4 (polish comes last)
**Verification**: j/k works in all views, state preserved across switches, overview is useful, help renders

---

## Dependency Graph

```
Phase 0 (Critical fixes)  ─── standalone, do first
    │
    v
Phase 1 (Safety)           ─── needs Phase 0 done (kill switch confirmation)
    │
    ├──> Phase 3 (Trading)  ─── needs confirmation dialog from Phase 1
    ├──> Phase 4 (Training) ─── needs confirmation dialog from Phase 1
    │
    v
Phase 5 (UX polish)        ─── do last, after all functional work

Phase 2 (Chart widget)     ─── fully independent, can be done anytime
```

---

## Files Modified (by phase)

| Phase | Files |
|-------|-------|
| 0 | `app.rs`, `views/log_viewer.rs`, `views/file_browser.rs` |
| 1 | `ui/dialog.rs` (new), `views/model_selector.rs`, `views/service_manager.rs`, `views/log_viewer.rs`, `views/trading.rs`, `ui/status_bar.rs` |
| 2 | `widgets/chart.rs`, `views/trading.rs`, `views/overview.rs` |
| 3 | `views/trading.rs` |
| 4 | `views/training.rs` |
| 5 | `views/trading.rs`, `views/overview.rs`, `app.rs`, `ui/help.rs`, `views/model_selector.rs` |

---

## Verification (End-to-End)

After all phases:
- [ ] Ctrl+K toggles kill switch with confirmation, state reflected in trading view
- [ ] Log viewer search filters displayed lines
- [ ] File browser shows correct dates, Enter opens files
- [ ] Chart widget renders P&L history in trading view
- [ ] Confirmation dialogs on destructive actions (model activate, service stop, training cancel)
- [ ] Large log files don't crash dashboard
- [ ] Bridge connectivity shown in status bar
- [ ] j/k navigation works in all views
- [ ] View state preserved across switches
- [ ] `cargo build --release` passes

---

## Session Notes

### 2026-02-09: Plan created
- Comprehensive dashboard review identified 20 improvements across 6 categories
- Previous plans complete: UI completion (all views exist), bug fixes (13 fixed), config editor (ViewAction)
- Focus is now on fixing broken features, adding control capabilities, and UX polish

### 2026-02-09: All phases complete
- Phase 0: Kill switch wiring, log search, file browser date fix + Enter to open
- Phase 1: Confirmation dialogs, log streaming, bridge indicator, error surfacing
- Phase 2: Sparkline + braille line chart widgets, P&L history, overview sparklines
- Phase 3: Position detail panel, sorting, kill switch from trading, Sharpe, pause/resume
- Phase 4: Training cancellation, cached scanning, throughput/ETA, stage details
- Phase 5: Overview health indicators + 3-panel layout, help overlay complete, model validation
- cargo build --release passes with 98 warnings (all pre-existing), no errors
