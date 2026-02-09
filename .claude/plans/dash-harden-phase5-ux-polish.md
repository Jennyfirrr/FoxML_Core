# Phase 5: UX Consistency & Polish

**Master plan**: `dashboard-hardening-master.md`
**Status**: Complete
**Scope**: 6 files modified
**Depends on**: Phases 0-4 (polish comes last)

---

## Context

After all functional work is done, this phase addresses inconsistencies across views: navigation patterns, state management, and incomplete rendering.

---

## 5a: Uniform j/k Navigation

**Files**: `views/trading.rs`, `views/overview.rs`

### Current State
- Config editor, training, log viewer, model selector, file browser, settings, service manager — all support j/k
- Trading view only uses arrow keys for position navigation
- Overview has no navigation at all

### Changes
1. Trading view: add `j`/`k` aliases for Down/Up in position list
2. Overview: add section navigation with j/k (if overview gets multiple panels in 5c)

---

## 5b: View State Preservation

**File**: `DASHBOARD/dashboard/src/app.rs`

### Current Problem
- When user switches from Trading → Training → back to Trading, scroll position and selected position reset
- Views are not re-created but their render state is lost if they do any resetting on entry

### Changes
1. Views already persist as fields on App (not recreated) — verify nothing resets on view switch
2. If any view has an `on_enter()` or `on_focus()` that resets state, remove the reset
3. For scroll positions specifically: ensure `selected` and `scroll` fields are NOT reset when switching views
4. Test: select position #5 in trading → switch to training → switch back → position #5 still selected

### Notes
- This may already work since views are struct fields — just need to verify
- If it does work, this item is a no-op (just verification)

---

## 5c: Complete Overview View

**File**: `DASHBOARD/dashboard/src/views/overview.rs`

### Current State
- 50/50 split showing raw system status + trading metrics
- No structure, no visual hierarchy

### Target Layout
```
┌───────────────────────────────────────────┐
│ System Health                             │
│ ┌──────────┬──────────┬──────────┐        │
│ │ Bridge ● │ Trading ● │ Alpaca ● │       │
│ └──────────┴──────────┴──────────┘        │
├───────────────────┬───────────────────────┤
│ Trading Summary   │ Training Summary      │
│                   │                       │
│ Portfolio: $X     │ Status: Running       │
│ Daily P&L: +$X   │ Stage: Model Training │
│ Positions: N      │ Progress: 60%         │
│ Sharpe: 1.2       │ Targets: 12/20        │
│ ▁▂▃▅▇ (sparkline) │ ETA: ~14 min         │
├───────────────────┴───────────────────────┤
│ Recent Activity                           │
│ 10:30 TRADE_FILLED AAPL +100 shares       │
│ 10:29 target_complete MSFT AUC=0.58       │
│ 10:28 DECISION_MADE GOOG HOLD             │
└───────────────────────────────────────────┘
```

### Changes
1. Add health indicator row (bridge, trading service, alpaca)
2. Split into trading summary + training summary panels
3. Add recent activity feed (merged trading + training events, sorted by time)
4. Wire sparklines from Phase 2 (if completed)

---

## 5d: Complete Help Overlay

**File**: `DASHBOARD/dashboard/src/ui/help.rs`

### Current State
- Shortcut categories and items are defined
- Render method is incomplete

### Changes
1. Finish render method: centered modal (70% width, 80% height)
2. Two-column layout: left = global shortcuts, right = current view shortcuts
3. Make context-sensitive: show different shortcuts based on `current_view`
4. Esc or `?` closes overlay

---

## 5e: Model Validation on Activation

**File**: `DASHBOARD/dashboard/src/views/model_selector.rs`

### Changes
1. Before creating symlink, verify the selected run has:
   - A valid `manifest.json`
   - At least one model file in expected locations
   - Compatible config (check `input_mode` matches current inference config)
2. Show warnings for partial runs (some targets failed)
3. Block activation for runs with no compiled models (show error message)

---

## 5f: Consistent Error Message Timeouts

**Files**: All views that use `message: Option<(String, bool)>`

### Current Problem
- Messages clear on next keypress, which means:
  - Fast typists never see messages
  - Messages persist indefinitely if no keys pressed

### Changes
1. Add `message_time: Option<Instant>` alongside `message`
2. In render, check if message is older than 5 seconds and clear it
3. Still clear on relevant key presses (e.g., clear error when user retries the action)
4. Error messages stay longer (10s), success messages shorter (3s)

---

## Verification

- [ ] j/k works for position navigation in trading view
- [ ] Switching views and back preserves scroll/selection state
- [ ] Overview shows structured panels with health indicators
- [ ] Help overlay renders with context-sensitive shortcuts
- [ ] Model activation blocked when run has no compiled models
- [ ] Messages auto-clear after timeout
- [ ] `cargo build --release` passes
