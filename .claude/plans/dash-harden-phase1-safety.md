# Phase 1: Safety & Robustness

**Master plan**: `dashboard-hardening-master.md`
**Status**: Pending
**Scope**: 5 files modified, 1 new file
**Depends on**: Phase 0

---

## Context

Several dashboard actions are destructive or irreversible but happen on a single keypress with no confirmation. The log viewer can OOM on large files. API errors are silently swallowed.

---

## 1a: Add Confirmation Dialog Widget

**File**: `DASHBOARD/dashboard/src/ui/dialog.rs` (NEW)
**Also**: `DASHBOARD/dashboard/src/ui/mod.rs` (add module)

### Design
```rust
pub struct ConfirmDialog {
    title: String,
    message: String,
    selected: DialogChoice,  // Yes / No / Cancel
    on_confirm: Option<Box<dyn FnOnce()>>,
}

pub enum DialogChoice { Yes, No, Cancel }
pub enum DialogResult { Confirmed, Cancelled, Pending }
```

### Features
- Centered modal overlay (40% width, 20% height)
- Semi-transparent background (dim the view behind it)
- Left/Right or h/l to select choice, Enter to confirm, Esc to cancel
- Theme-aware colors (warning border for destructive actions)
- Reusable across all views

### Integration
- Add `active_dialog: Option<ConfirmDialog>` to App struct
- Render dialog on top of current view when active
- Intercept all keys when dialog is active

---

## 1b: Confirmation Before Model Activation

**File**: `DASHBOARD/dashboard/src/views/model_selector.rs`

### Changes
1. When user presses Enter/`a` to activate a model, show dialog: "Activate {run_id} for live trading? This will update the active model symlink."
2. Only proceed with symlink creation on dialog confirmation

---

## 1c: Confirmation Before Service Stop

**File**: `DASHBOARD/dashboard/src/views/service_manager.rs`

### Changes
1. When user presses `s` (stop) or `r` (restart) on a running service, show dialog: "Stop {service_name}? This will interrupt any running operations."
2. Only execute `systemctl --user stop` on confirmation

---

## 1d: Stream Large Log Files

**File**: `DASHBOARD/dashboard/src/views/log_viewer.rs`

### Current Problem
- `fs::read_to_string(&path)` loads entire file into memory
- Training logs can be 100MB+, causing OOM

### Changes
1. Replace full file read with seek-based tail:
   - On initial open: seek to last 10,000 lines (or last 1MB)
   - Store file position for efficient tail polling
   - Keep a `VecDeque<String>` ring buffer (max 10,000 lines)
2. Add "Load more" (PgUp at top) to read earlier content on demand
3. Show indicator when not viewing from start: "... (showing last N lines)"

### Notes
- The `tail_position: u64` field already exists but only used for follow mode
- Extend to also limit initial load

---

## 1e: Bridge Connectivity Indicator

**File**: `DASHBOARD/dashboard/src/ui/status_bar.rs`

### Changes
1. Add bridge health field to status bar data
2. Periodically (every 5s) call GET `/health` on the bridge
3. Show in status bar: `Bridge: ●` (green) or `Bridge: ○` (red)
4. If bridge is down, show tooltip/message: "IPC bridge unreachable at 127.0.0.1:8765"

### Notes
- The `DashboardClient` already has a `health_check()` method
- Status bar already shows service dots — add bridge as another one

---

## 1f: Surface API Errors to User

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Current Problem
- Lines like `let _ = self.client.get_metrics().await;` silently discard errors
- User sees stale data with no indication something is wrong

### Changes
1. On API error, push a Warning notification: "Failed to fetch trading metrics: {error}"
2. Throttle error notifications (max 1 per 30 seconds per endpoint)
3. Show "Last updated: Xs ago" in the trading view header when data is stale

### Notes
- Use the existing `NotificationManager` for error surfacing
- Don't spam — deduplicate repeated errors

---

## Verification

- [ ] Confirmation dialog appears before model activation
- [ ] Confirmation dialog appears before service stop/restart
- [ ] Opening a 100MB log file doesn't crash the dashboard
- [ ] Bridge status dot visible in status bar, reflects actual connectivity
- [ ] API errors shown as notifications, not silently swallowed
- [ ] `cargo build --release` passes
