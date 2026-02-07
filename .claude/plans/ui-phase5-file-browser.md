# UI Phase 5: File Browser View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 1h

---

## Current State

Existing implementation in `src/launcher/file_browser.rs` (115 lines):
- Has directory listing
- Has navigation
- Has file entry struct

**Missing**: File preview, quick jumps, theme support.

---

## Implementation

### Step 1: Create FileBrowserView in views/

### Step 2: Add features
- File preview panel (syntax highlighted for code)
- Quick jump keys:
  - `1` → CONFIG/
  - `2` → RESULTS/
  - `3` → TRAINING/
  - `4` → LIVE_TRADING/
- File info (size, modified date)

### Step 3: Wire up in app.rs

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/file_browser.rs` | NEW - View wrapper |
| `src/views/mod.rs` | Add FileBrowser variant |
| `src/app.rs` | Wire up FileBrowser view |
| `src/launcher/file_browser.rs` | Add preview, quick jumps |

---

## Checklist

- [ ] Create FileBrowserView in views/
- [ ] Add file preview panel
- [ ] Add quick jump keys
- [ ] Add file info display
- [ ] Wire up in app.rs
- [ ] Test navigation
