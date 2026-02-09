# Dashboard Fixes Round 2 — Master Plan

**Status**: Complete (all 5 phases done)
**Created**: 2026-02-09
**Branch**: `analysis/code-review-and-raw-ohlcv`
**Predecessor Plans**: `dashboard-hardening-master.md` (20 issues fixed across 6 phases)

---

## Executive Summary

A comprehensive audit of the dashboard TUI and IPC bridge after the first hardening round identified 15 remaining issues. These range from fake command palette commands that claim to work but don't, to hardcoded paths, bridge protocol bugs, and missing validation. This plan addresses all issues across 5 phases.

**Total subplans**: 5
**Key principle**: Wire stub commands to real logic first (users are being deceived), then fix protocol/data bugs, then improve configurability.

---

## Current State

### What Was Fixed in Round 1
| Phase | Items Fixed |
|-------|-------------|
| 0 | Kill switch wiring, log search, file browser dates + Enter |
| 1 | Confirmation dialogs, log streaming, bridge indicator, error surfacing |
| 2 | Sparkline + braille chart widgets, P&L history |
| 3 | Position detail panel, sorting, kill switch from trading, Sharpe, pause/resume |
| 4 | Training cancellation, cached scanning, throughput/ETA |
| 5 | Overview 3-panel layout, help overlay, model validation |

### What's Still Broken
| Issue | Component | Severity |
|-------|-----------|----------|
| `trading.pause`/`resume` commands are fake | app.rs (command palette) | **CRITICAL** — user deceived |
| `training.stop` command is fake | app.rs (command palette) | **CRITICAL** — user deceived |
| `config.edit`/`nav.config`/`nav.models` go to placeholders | app.rs (command palette) | **CRITICAL** — real views exist but unreachable |
| Position table highlight wrong row when scrolled | position_table.rs | **HIGH** — visual bug |
| WebSocket connection can hang forever | trading.rs | **HIGH** — permanent UI freeze |
| Sharpe ratio always null from bridge | bridge/server.py | **HIGH** — data never arrives |
| Bridge timezone inconsistency | bridge/server.py | **HIGH** — mixed timestamp formats |
| Bridge `get_metrics()` error returns incomplete schema | bridge/server.py | **MEDIUM** — silent data corruption |
| View render errors silently discarded | app.rs | **MEDIUM** — invisible failures |
| Hardcoded bridge URL in 6+ files | multiple Rust files | **MEDIUM** — not configurable |
| Hardcoded `/tmp/foxml_*` paths | multiple Rust files | **MEDIUM** — not configurable |
| No auth on bridge control endpoints | bridge/server.py | **MEDIUM** — security gap |
| Overview health indicators can overflow | overview.rs | **LOW** — narrow terminal crash |
| Desktop notifications silently dropped | notification.rs | **LOW** — no error feedback |
| Training event queue drops at wrong log level | bridge/server.py | **LOW** — inconsistent observability |

---

## Subplan Index

| Phase | Subplan | Purpose | Scope | Status |
|-------|---------|---------|-------|--------|
| 0 | [dash-fixes-r2-phase0-fake-commands.md](./dash-fixes-r2-phase0-fake-commands.md) | Wire stub/fake commands to real logic | 1 file (app.rs) | **Complete** |
| 1 | [dash-fixes-r2-phase1-tui-bugs.md](./dash-fixes-r2-phase1-tui-bugs.md) | Fix TUI bugs: selection, WS timeout, render errors | 3 files | **Complete** |
| 2 | [dash-fixes-r2-phase2-bridge-bugs.md](./dash-fixes-r2-phase2-bridge-bugs.md) | Fix bridge protocol/data bugs | 2 files (server.py, client.rs) | **Complete** |
| 3 | [dash-fixes-r2-phase3-configurability.md](./dash-fixes-r2-phase3-configurability.md) | Extract hardcoded URLs/paths to config | 15 files | **Complete** |
| 4 | [dash-fixes-r2-phase4-minor-polish.md](./dash-fixes-r2-phase4-minor-polish.md) | Minor polish and observability | 3 files | **Complete** |

---

## Phase 0: Wire Fake Commands to Real Logic

**Goal**: Fix commands that show fake success notifications but don't actually do anything.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 0a | Wire `trading.pause`/`resume` to bridge API | app.rs | Call `client.pause_engine()` / `client.resume_engine()` instead of fake notification |
| 0b | Wire `training.stop` to `cancel_run()` | app.rs | Use `TrainingView::running_pid()` + confirmation dialog, like `x` key handler |
| 0c | Route `config.edit` to real ConfigEditor view | app.rs | Same as `MenuAction::ConfigEditor` path |
| 0d | Route `nav.config` and `nav.models` to real views | app.rs | `nav.config` → ConfigEditor, `nav.models` → ModelSelector |
| 0e | Route `training.logs` to LogViewer (training source) | app.rs | Switch to LogViewer with training log source selected |

**Depends on**: Nothing
**Verification**: All command palette commands perform real actions, no more fake notifications

---

## Phase 1: TUI Bugs

**Goal**: Fix visual/interaction bugs in the Rust TUI.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 1a | Fix position table selection highlight when scrolled | position_table.rs | `i` from `.take(visible_rows).enumerate()` is slice-relative, not absolute |
| 1b | Add WebSocket connection timeout | trading.rs | If WS doesn't connect within 5s, reset `ws_connecting` flag |
| 1c | Surface view render errors instead of `let _ =` | app.rs | Log errors and optionally show notification on render failure |

**Depends on**: Nothing
**Verification**: Position highlight tracks correctly when scrolled, WS timeout recovers, render errors visible

---

## Phase 2: Bridge Protocol Bugs

**Goal**: Fix data correctness and protocol issues in the Python bridge.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 2a | Fix timezone inconsistency in control endpoints | bridge/server.py | Standardize to `datetime.now(timezone.utc).isoformat()` in pause/resume/kill_switch |
| 2b | Implement Sharpe ratio calculation | bridge/server.py | Calculate from P&L history: `mean(returns) / std(returns) * sqrt(252)` |
| 2c | Fix `get_metrics()` error path to return full schema | bridge/server.py | Error fallback should include all expected fields with null/zero defaults |
| 2d | Add bearer token auth to control endpoints | bridge/server.py | Generate token on startup, require in Authorization header for POST endpoints |

**Depends on**: Nothing
**Verification**: Timestamps consistent, Sharpe ratio shows numeric value, error responses match full schema, unauthorized requests rejected

---

## Phase 3: Configurability

**Goal**: Extract hardcoded values to configuration/environment variables.

| # | Fix | Files | Notes |
|---|-----|-------|-------|
| 3a | Extract bridge URL to env var / config | app.rs, trading.rs, overview.rs, launcher/*.rs | `FOXML_BRIDGE_URL` env var, fallback to `127.0.0.1:8765` |
| 3b | Extract `/tmp/foxml_*` paths to env var / config | training.rs, overview.rs, launcher/*.rs | `FOXML_TMP_DIR` env var, fallback to `/tmp` |
| 3c | Extract project root paths to env var | training.rs, model_selector.rs, config_editor.rs | `FOXML_ROOT` env var, fallback to current dir |

**Depends on**: Nothing (independent)
**Verification**: Dashboard works with custom bridge URL, custom tmp dir, custom project root

---

## Phase 4: Minor Polish

**Goal**: Fix low-severity issues for better observability and robustness.

| # | Fix | File | Notes |
|---|-----|------|-------|
| 4a | Add bounds check for overview health indicators | overview.rs | Check available width before writing chars |
| 4b | Log desktop notification failures at warn level | notification.rs | Instead of silently dropping |
| 4c | Fix training event queue drop log level | bridge/server.py | Change from `debug` to `warning` to match other queues |

**Depends on**: Nothing
**Verification**: Narrow terminal doesn't crash overview, notification errors logged, training drops visible in logs

---

## Dependency Graph

```
Phase 0 (Fake commands)      ─── standalone, highest priority
Phase 1 (TUI bugs)           ─── standalone, independent
Phase 2 (Bridge protocol)    ─── standalone, independent
Phase 3 (Configurability)    ─── standalone, independent
Phase 4 (Minor polish)       ─── standalone, do last

All phases are independent and can be done in any order.
Recommended order: 0 → 1 → 2 → 3 → 4 (by severity)
```

---

## Files Modified (by phase)

| Phase | Files |
|-------|-------|
| 0 | `app.rs` |
| 1 | `widgets/position_table.rs`, `views/trading.rs`, `app.rs` |
| 2 | `bridge/server.py` |
| 3 | `app.rs`, `views/trading.rs`, `views/overview.rs`, `views/training.rs`, `views/model_selector.rs`, `views/config_editor.rs`, `launcher/status_canvas.rs`, `launcher/system_status.rs`, `launcher/live_dashboard.rs` |
| 4 | `views/overview.rs`, `ui/notification.rs`, `bridge/server.py` |

---

## Verification (End-to-End)

After all phases:
- [ ] Command palette `trading.pause` actually pauses trading via bridge
- [ ] Command palette `training.stop` actually stops training with confirmation
- [ ] Command palette `config.edit` opens real ConfigEditor view
- [ ] Position table highlight is correct when scrolled down
- [ ] WebSocket connection recovers after timeout
- [ ] Sharpe ratio shows a real number in trading view
- [ ] All bridge timestamps are UTC
- [ ] Bridge control endpoints require auth token
- [ ] `FOXML_BRIDGE_URL` env var overrides default bridge address
- [ ] `/tmp/foxml_*` paths configurable via `FOXML_TMP_DIR`
- [ ] `cargo build --release` passes
- [ ] Bridge starts without errors

---

## Session Notes

### 2026-02-09: Plan created
- Post-hardening audit identified 15 remaining issues across Rust TUI and Python bridge
- All issues categorized and organized into 5 phases by theme
- All phases are independent (no dependency chain) — can be done in any order

### 2026-02-09: All phases complete
- Phase 0: Wired 5 fake command palette commands to real logic (pause/resume, training stop, config edit, nav routing)
- Phase 1: Fixed position table scroll highlight, added WS 5s timeout, surfaced render errors via tracing
- Phase 2: Fixed 3 timezone inconsistencies, implemented Sharpe ratio from rolling P&L, fixed error schema, added bearer token auth
- Phase 3: Created `config.rs` module, extracted bridge URL (6 files), tmp paths (5 files), project root (8 files) to env vars
- Phase 4: Added overflow guard in overview, logged desktop notification failures, fixed training queue log level
- cargo build --release passes with 102 warnings (all pre-existing), no errors
- 18 issues fixed across 15+ files
