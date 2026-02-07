# UI Phase 6: Settings View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 1.5h

---

## Current State

Existing implementation in `src/launcher/settings.rs` (28 lines):
- Just a placeholder

**Needs full implementation.**

---

## Implementation

### Step 1: Define settings schema

```yaml
# ~/.config/foxml-dashboard/settings.yaml
theme:
  mode: auto  # auto, light, dark, custom
  custom_colors: {}

refresh:
  bridge_interval_ms: 2000
  training_interval_ms: 500

display:
  default_view: launcher  # launcher, trading, training
  show_timestamps: true

keybinds:
  # Override default keybinds
  quit: q
  help: ?
```

### Step 2: Create SettingsView

Form-style view with:
- Theme selection (dropdown)
- Refresh intervals (number inputs)
- Default view (dropdown)
- Keybind editor

### Step 3: Persistence

- Load on startup
- Save on change
- Create default if missing

### Step 4: Wire up in app.rs

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/settings.rs` | NEW - Full implementation |
| `src/views/mod.rs` | Add Settings variant |
| `src/app.rs` | Wire up Settings view, load on startup |
| `src/config.rs` | NEW - Settings schema and I/O |
| `src/launcher/settings.rs` | Remove or integrate |

---

## Checklist

- [ ] Define settings schema
- [ ] Create config.rs for settings I/O
- [ ] Create SettingsView with form UI
- [ ] Add persistence (load/save)
- [ ] Wire up in app.rs
- [ ] Test settings changes
