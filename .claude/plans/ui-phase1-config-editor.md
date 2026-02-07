# UI Phase 1: Config Editor View

**Status**: Pending
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 1h

---

## Current State

Existing implementation in `src/launcher/config_editor.rs` (291 lines):
- Uses ropey for text editing
- Has cursor movement, scroll support
- Has save functionality
- Has YAML validation

**Problem**: Not wired up to a View - app.rs uses `switch_to_placeholder` instead.

---

## Implementation

### Step 1: Create ConfigEditorView in views/

Create `src/views/config_editor.rs` that wraps the launcher ConfigEditor:

```rust
pub struct ConfigEditorView {
    editor: ConfigEditor,
    theme: Theme,
    file_path: Option<String>,
}

impl ViewTrait for ConfigEditorView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        self.editor.render(frame, area, &self.theme)
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        self.editor.handle_key(key)
    }
}
```

### Step 2: Add View::ConfigEditor variant

In `src/views/mod.rs`:
```rust
pub enum View {
    // ... existing
    ConfigEditor,
}
```

### Step 3: Wire up in app.rs

Replace placeholder with actual view:
```rust
MenuAction::ConfigEditor => {
    self.config_editor_view = Some(ConfigEditorView::new("CONFIG/experiments/production_baseline.yaml"));
    self.switch_view(View::ConfigEditor);
}
```

### Step 4: Add file picker

Before opening editor, show a file picker for CONFIG/ directory to choose which file to edit.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/views/config_editor.rs` | NEW - View wrapper |
| `src/views/mod.rs` | Add ConfigEditor variant |
| `src/app.rs` | Wire up ConfigEditor view |
| `src/launcher/config_editor.rs` | Add theme support |

---

## Checklist

- [ ] Create ConfigEditorView in views/
- [ ] Add View::ConfigEditor enum variant
- [ ] Add config_editor_view field to App
- [ ] Wire up MenuAction::ConfigEditor
- [ ] Add file picker before opening
- [ ] Ensure consistent theming
- [ ] Test editing and saving
