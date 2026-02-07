# UI Phase 2: Log Viewer View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 1h

---

## Current State

Existing implementation in `src/launcher/log_viewer.rs` (87 lines):
- Has file loading
- Has scroll support
- Basic line display

**Missing**: Real-time tailing, filtering, log level colors.

---

## Implementation

### Step 1: Create LogViewerView in views/

Wrap existing LogViewer with view trait and add:
- Real-time file tailing (poll for new lines)
- Search/filter box
- Log level highlighting (ERROR=red, WARN=yellow, INFO=default)

### Step 2: Add log source selector

Allow switching between:
- Training logs: `RESULTS/runs/*/logs/`
- Trading logs: `LIVE_TRADING/logs/`
- System journal: `journalctl --user -u foxml-trading`

### Step 3: Wire up in app.rs

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/log_viewer.rs` | NEW - View wrapper |
| `src/views/mod.rs` | Add LogViewer variant |
| `src/app.rs` | Wire up LogViewer view |
| `src/launcher/log_viewer.rs` | Add tailing, filtering |

---

## Checklist

- [ ] Create LogViewerView in views/
- [ ] Add real-time tailing
- [ ] Add search/filter
- [ ] Add log level coloring
- [ ] Add log source selector
- [ ] Wire up in app.rs
- [ ] Test with actual logs
