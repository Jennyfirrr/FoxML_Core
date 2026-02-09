# Phase 0: Wire Fake Commands to Real Logic

**Master plan**: `dashboard-fixes-round2-master.md`
**Status**: Complete
**Scope**: 1 file modified (app.rs)
**Depends on**: Nothing

---

## Context

The command palette exposes ~29 commands. Five of them show fake success notifications but don't perform the advertised action. The user believes the action happened when it didn't. This is the highest-priority fix category.

---

## 0a: Wire `trading.pause` / `trading.resume` to Bridge API

**File**: `DASHBOARD/dashboard/src/app.rs` (in `execute_command()`)

### Current State
```rust
"trading.pause" => {
    self.notifications.push(
        Notification::info("Trading Paused").message("Trading pipeline paused"),
    );
}
"trading.resume" => {
    self.notifications.push(
        Notification::info("Trading Resumed").message("Trading pipeline resumed"),
    );
}
```

### Target State
```rust
"trading.pause" => {
    match self.client.pause_engine().await {
        Ok(_) => {
            self.notifications.push(
                Notification::info("Trading Paused").message("Trading pipeline paused"),
            );
        }
        Err(e) => {
            self.notifications.push(
                Notification::error("Pause Failed").message(&format!("{}", e)),
            );
        }
    }
}
"trading.resume" => {
    match self.client.resume_engine().await {
        Ok(_) => {
            self.notifications.push(
                Notification::info("Trading Resumed").message("Trading pipeline resumed"),
            );
        }
        Err(e) => {
            self.notifications.push(
                Notification::error("Resume Failed").message(&format!("{}", e)),
            );
        }
    }
}
```

### Notes
- `self.client.pause_engine()` and `self.client.resume_engine()` already exist on `DashboardClient`
- These are the same methods already called by the `p` key handler in trading view
- The only change is in the command palette handler — key handler in trading view already works

---

## 0b: Wire `training.stop` to `cancel_run()`

**File**: `DASHBOARD/dashboard/src/app.rs` (in `execute_command()`)

### Current State
```rust
"training.stop" => {
    self.notifications.push(
        Notification::warning("Training Stopped").message("Training run stopped"),
    );
}
```

### Target State
- Check if training is running via `self.training_view.running_pid()`
- If running, show confirmation dialog with `PendingAction::CancelTraining(pid)`
- If not running, show info notification "No training run active"
- This mirrors the `x` key handler already wired in app.rs

### Changes
```rust
"training.stop" => {
    if let Some(pid) = self.training_view.running_pid() {
        self.pending_action = Some(PendingAction::CancelTraining(pid));
        self.confirm_dialog = Some(ConfirmDialog::new(
            "Cancel Training",
            &format!("Stop training run (PID {})? This will send SIGTERM.", pid),
        ));
    } else {
        self.notifications.push(
            Notification::info("No Training").message("No training run is currently active"),
        );
    }
}
```

---

## 0c: Route `config.edit` to Real ConfigEditor View

**File**: `DASHBOARD/dashboard/src/app.rs` (in `execute_command()`)

### Current State
```rust
"config.edit" => {
    self.switch_to_placeholder("Config Editor - Coming soon");
}
```

### Target State
Route to real ConfigEditor, same as `MenuAction::ConfigEditor`:
```rust
"config.edit" => {
    let default_config = std::path::PathBuf::from("CONFIG/experiments/production_baseline.yaml");
    let _ = self.config_editor_view.open(default_config);
    self.switch_view(View::ConfigEditor);
}
```

---

## 0d: Route `nav.config` and `nav.models` to Real Views

**File**: `DASHBOARD/dashboard/src/app.rs` (in `execute_command()`)

### Current State
```rust
"nav.config" => {
    self.switch_to_placeholder("Config Editor - Coming soon");
}
"nav.models" => {
    self.switch_to_placeholder("Model Manager - Coming soon");
}
```

### Target State
```rust
"nav.config" => {
    let default_config = std::path::PathBuf::from("CONFIG/experiments/production_baseline.yaml");
    let _ = self.config_editor_view.open(default_config);
    self.switch_view(View::ConfigEditor);
}
"nav.models" => {
    self.switch_view(View::ModelSelector);
}
```

---

## 0e: Route `training.logs` to LogViewer

**File**: `DASHBOARD/dashboard/src/app.rs` (in `execute_command()`)

### Current State
```rust
"training.logs" => {
    self.switch_to_placeholder("Training Logs - Coming soon");
}
```

### Target State
Switch to LogViewer view. If LogViewer supports source selection, pre-select training source:
```rust
"training.logs" => {
    self.switch_view(View::LogViewer);
    // Optionally: self.log_viewer.set_source(LogSource::Training);
}
```

### Notes
- Check if LogViewer has a method to set the log source
- If not, just switching to the view is still better than a placeholder

---

## Verification

- [ ] `Ctrl+P` → type "pause" → runs and actually pauses trading via bridge
- [ ] `Ctrl+P` → type "resume" → runs and actually resumes trading via bridge
- [ ] `Ctrl+P` → type "stop training" → shows confirmation dialog, cancels on confirm
- [ ] `Ctrl+P` → type "config" → opens real Config Editor with default config
- [ ] `Ctrl+P` → type "models" → opens real Model Selector
- [ ] `Ctrl+P` → type "training logs" → opens real Log Viewer
- [ ] Error responses from bridge show error notifications (not fake success)
- [ ] `cargo build --release` passes
