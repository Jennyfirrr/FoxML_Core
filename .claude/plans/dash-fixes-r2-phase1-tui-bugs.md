# Phase 1: TUI Bugs

**Master plan**: `dashboard-fixes-round2-master.md`
**Status**: Complete
**Scope**: 3 files modified
**Depends on**: Nothing

---

## Context

Three distinct bugs in the Rust TUI cause visual/interaction issues: a position table highlight that tracks the wrong row when scrolled, a WebSocket connection that can hang forever, and view render errors that are silently discarded.

---

## 1a: Fix Position Table Selection Highlight When Scrolled

**File**: `DASHBOARD/dashboard/src/widgets/position_table.rs`

### Current Problem
The render loop uses `.take(visible_rows).enumerate()` which produces a 0-based index `i` relative to the visible slice. But the highlight check compares `i` against `self.selected` which is an absolute index into the full position list.

```rust
for (i, pos) in self.positions.iter().take(visible_rows).enumerate() {
    let style = if i == self.selected {
        // WRONG: i is slice-relative, self.selected is absolute
        Style::default().bg(theme.selection)
    } else {
        Style::default()
    };
}
```

### Fix
Account for scroll offset. The table needs to:
1. Skip positions above the scroll window: `.skip(scroll_offset).take(visible_rows)`
2. Compare `i + scroll_offset == self.selected` (or equivalently `i == self.selected - scroll_offset`)

### Changes
1. Add `scroll_offset` field (or compute from `self.selected` and `visible_rows`)
2. Use `.skip(scroll_offset).take(visible_rows).enumerate()`
3. Highlight when `i + scroll_offset == self.selected`
4. Ensure `scroll_offset` updates when `self.selected` moves (keep selection visible)

### Notes
- Need to read the full position_table.rs to see current scroll implementation
- Some position tables handle this with a `first_visible` or `scroll` field
- Make sure selection clamps to 0..positions.len()-1

---

## 1b: Add WebSocket Connection Timeout

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Current Problem
When user presses `w` to connect WebSocket, `ws_connecting` is set to `true`. If the connection attempt hangs (e.g., bridge is unreachable), it stays `true` forever. The connect button becomes permanently unresponsive.

### Fix
Add a timeout mechanism:
1. Store `ws_connect_started: Option<std::time::Instant>` alongside `ws_connecting: bool`
2. When initiating connection, set both `ws_connecting = true` and `ws_connect_started = Some(Instant::now())`
3. In `update()` or `render()`, check if `ws_connecting && ws_connect_started.elapsed() > Duration::from_secs(5)`
4. If timed out, reset `ws_connecting = false`, show error message

### Changes
```rust
// New field
ws_connect_started: Option<std::time::Instant>,

// In connect handler
self.ws_connecting = true;
self.ws_connect_started = Some(std::time::Instant::now());

// In update/render
if self.ws_connecting {
    if let Some(started) = self.ws_connect_started {
        if started.elapsed() > std::time::Duration::from_secs(5) {
            self.ws_connecting = false;
            self.ws_connect_started = None;
            self.message = Some(("WebSocket connection timed out".to_string(), true));
        }
    }
}
```

---

## 1c: Surface View Render Errors

**File**: `DASHBOARD/dashboard/src/app.rs`

### Current Problem
All view render calls use `let _ = ...` pattern, silently discarding errors:
```rust
let _ = ViewTrait::render(&mut self.trading_view, f, content_area);
```

### Fix
Log render errors and optionally show them. Two approaches:

**Option A (minimal)**: Use `if let Err(e) = ...` and log:
```rust
if let Err(e) = ViewTrait::render(&mut self.trading_view, f, content_area) {
    log::error!("Trading view render failed: {}", e);
}
```

**Option B (user-visible)**: On error, render a fallback error message in the view area:
```rust
if let Err(e) = ViewTrait::render(&mut self.trading_view, f, content_area) {
    let error_msg = Paragraph::new(format!("Render error: {}", e))
        .style(Style::default().fg(self.theme.error));
    f.render_widget(error_msg, content_area);
}
```

### Recommendation
Use Option A (logging) for all views. View crashes should be rare and logging is non-disruptive. Adding visual error display would add complexity for an edge case.

### Notes
- There are ~15 instances of `let _ =` on render calls
- Also applies to `update_metrics()`, `update()` calls
- For update calls, consider also logging failures

---

## Verification

- [ ] Position table: scroll down to position #20, highlight is on the correct row (not row 0)
- [ ] WebSocket: start connection with bridge down → after 5s, button becomes usable again
- [ ] View render error: if a view fails, error appears in log output
- [ ] `cargo build --release` passes
