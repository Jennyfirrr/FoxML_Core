# UI Phase 3: Service Manager View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 1h

---

## Current State

Existing implementation in `src/launcher/service_manager.rs` (154 lines):
- Has systemctl status checking
- Has start/stop/restart commands
- Has status display

**Missing**: Proper view wrapper, journal log display, theme support.

---

## Implementation

### Step 1: Create ServiceManagerView in views/

### Step 2: Add features
- Show recent journal entries (last 20 lines)
- Confirm dialog for destructive actions
- Auto-refresh status

### Step 3: Wire up in app.rs

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/service_manager.rs` | NEW - View wrapper |
| `src/views/mod.rs` | Add ServiceManager variant |
| `src/app.rs` | Wire up ServiceManager view |

---

## Checklist

- [ ] Create ServiceManagerView in views/
- [ ] Add journal log display
- [ ] Add confirmation for destructive actions
- [ ] Wire up in app.rs
- [ ] Test with actual service
