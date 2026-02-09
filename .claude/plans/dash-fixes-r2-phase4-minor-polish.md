# Phase 4: Minor Polish

**Master plan**: `dashboard-fixes-round2-master.md`
**Status**: Complete
**Scope**: 3 files modified
**Depends on**: Nothing

---

## Context

Three low-severity issues that improve robustness and observability without changing functionality.

---

## 4a: Add Bounds Check for Overview Health Indicators

**File**: `DASHBOARD/dashboard/src/views/overview.rs`

### Current Problem
`render_health()` writes characters to the buffer using `buf.cell_mut((x, area.y))` in a loop. If the terminal is very narrow, the loop can write past `area.right()`. While there's an `if x < area.right()` check, the `x += 1` still increments, and later sections (kill switch / paused badge) continue writing at the potentially out-of-bounds position.

### Fix
Add early return when `x >= area.right()`:
```rust
for ch in text.chars() {
    if x >= area.right() {
        return; // No more space, stop rendering health indicators
    }
    if let Some(cell) = buf.cell_mut((x, area.y)) {
        cell.set_char(ch);
        cell.set_style(Style::default().fg(color));
    }
    x += 1;
}
```

### Notes
- The `buf.cell_mut()` already returns `Option` so it won't crash, but the loop wastes cycles iterating past the boundary
- Using `return` (or `break` from outer loop) stops all indicator rendering when out of space
- This is a minor optimization / correctness fix

---

## 4b: Log Desktop Notification Failures

**File**: `DASHBOARD/dashboard/src/ui/notification.rs`

### Current Problem
When desktop notifications fail (e.g., `notify-send` not installed, D-Bus unavailable), the error is silently dropped.

### Fix
Log at `warn` level:
```rust
// Before
let _ = notify_send(...);

// After
if let Err(e) = notify_send(...) {
    log::warn!("Desktop notification failed: {}", e);
}
```

### Notes
- This should only fire once or twice (not per-notification) to avoid log spam
- Consider adding a `desktop_notifications_available: bool` flag that gets set to `false` on first failure, suppressing further attempts
- Alternatively, log at `debug` level after the first failure

---

## 4c: Fix Training Event Queue Drop Log Level

**File**: `DASHBOARD/bridge/server.py`

### Current Problem
Event queues log at inconsistent levels when full:
- Trading event queue full → `logger.warning()`
- Alpaca event queue full → `logger.warning()`
- Training event queue full → `logger.debug()`

### Fix
Change training event queue drop log level to `warning`:
```python
# Before
logger.debug("Training event queue full, dropping event")

# After
logger.warning("Training event queue full, dropping event")
```

### Notes
- Simple one-line change
- Ensures all event drops are visible at the same log level
- Important for debugging training monitor issues

---

## Verification

- [ ] Very narrow terminal (< 40 cols) doesn't cause overview rendering issues
- [ ] Failed desktop notifications appear in log output at `warn` level
- [ ] Training event queue drops appear at `warning` level (not `debug`)
- [ ] `cargo build --release` passes
- [ ] Bridge starts without errors
