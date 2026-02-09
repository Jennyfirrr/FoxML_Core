# Phase 3: Configurability

**Master plan**: `dashboard-fixes-round2-master.md`
**Status**: Complete
**Scope**: 8+ files modified
**Depends on**: Nothing

---

## Context

The dashboard has hardcoded values scattered across many files. The bridge URL (`127.0.0.1:8765`), temp file paths (`/tmp/foxml_*`), and project directory paths (`RESULTS/`, `CONFIG/`) are all baked in. This makes it impossible to run the dashboard in non-standard environments without recompiling.

---

## 3a: Extract Bridge URL to Environment Variable

**Files**: Multiple Rust files containing `DashboardClient::new("127.0.0.1:8765")`

### Current State
Bridge URL hardcoded in 6+ locations:
- `src/app.rs` (~line 130)
- `src/views/trading.rs` (~line 46)
- `src/views/overview.rs` (~line 38)
- `src/launcher/status_canvas.rs` (~line 39)
- `src/launcher/system_status.rs` (~line 35)
- `src/launcher/live_dashboard.rs` (~line 77)

### Fix
1. Create a config/constants module with a helper function:
```rust
// In src/config.rs (or src/constants.rs)
pub fn bridge_url() -> String {
    std::env::var("FOXML_BRIDGE_URL")
        .unwrap_or_else(|_| "127.0.0.1:8765".to_string())
}
```

2. Replace all hardcoded strings:
```rust
// Before
client: DashboardClient::new("127.0.0.1:8765"),

// After
client: DashboardClient::new(&config::bridge_url()),
```

### Files to Update
| File | Line | Current |
|------|------|---------|
| `src/app.rs` | ~130 | `DashboardClient::new("127.0.0.1:8765")` |
| `src/views/trading.rs` | ~46 | `DashboardClient::new("127.0.0.1:8765")` |
| `src/views/overview.rs` | ~38 | `DashboardClient::new("127.0.0.1:8765")` |
| `src/launcher/status_canvas.rs` | ~39 | `DashboardClient::new("127.0.0.1:8765")` |
| `src/launcher/system_status.rs` | ~35 | `DashboardClient::new("127.0.0.1:8765")` |
| `src/launcher/live_dashboard.rs` | ~77 | `DashboardClient::new("127.0.0.1:8765")` |

### Notes
- `DashboardClient::new()` takes `&str`, so the helper should return `String` and be borrowed at call site
- Consider also adding `FOXML_BRIDGE_PORT` as a separate override (or just use the full URL)

---

## 3b: Extract `/tmp/foxml_*` Paths to Environment Variable

**Files**: Multiple Rust files with hardcoded `/tmp/foxml_*` paths

### Current State
```rust
const TRAINING_EVENTS_FILE: &str = "/tmp/foxml_training_events.jsonl";
const TRAINING_PID_FILE: &str = "/tmp/foxml_training.pid";
```

Also referenced in:
- `src/views/overview.rs` (~line 189)
- `src/launcher/system_status.rs` (~line 12)
- `src/launcher/live_dashboard.rs` (~line 21)
- `src/launcher/training_executor.rs` (~line 12)

### Fix
1. Add to config module:
```rust
pub fn tmp_dir() -> String {
    std::env::var("FOXML_TMP_DIR")
        .unwrap_or_else(|_| "/tmp".to_string())
}

pub fn training_events_file() -> String {
    format!("{}/foxml_training_events.jsonl", tmp_dir())
}

pub fn training_pid_file() -> String {
    format!("{}/foxml_training.pid", tmp_dir())
}
```

2. Replace all hardcoded paths with function calls

### Notes
- Constants become function calls, which is slightly less efficient but negligible for path strings
- Could use `lazy_static!` or `once_cell::sync::Lazy` to compute once if desired

---

## 3c: Extract Project Root Paths to Environment Variable

**Files**: Files referencing `RESULTS/`, `CONFIG/experiments/`, etc.

### Current State
```rust
let results_dir = PathBuf::from("RESULTS");
let config_dir = PathBuf::from("CONFIG/experiments");
```

### Fix
1. Add to config module:
```rust
pub fn project_root() -> PathBuf {
    std::env::var("FOXML_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn results_dir() -> PathBuf {
    project_root().join("RESULTS")
}

pub fn config_dir() -> PathBuf {
    project_root().join("CONFIG")
}
```

2. Replace relative path constructions with `config::results_dir()` etc.

### Notes
- This is the least critical of the three — the dashboard is typically run from project root
- But it enables running from systemd or other contexts where cwd != project root
- The `bin/foxml` launcher already `cd`s to the correct directory, so this is a defense-in-depth improvement

---

## Implementation Order

1. Create `src/config.rs` module with all helper functions
2. Add `mod config;` to `src/main.rs` (or `src/lib.rs`)
3. Update bridge URL references (3a)
4. Update tmp path references (3b)
5. Update project root references (3c)
6. Build and verify

---

## Verification

- [ ] Default behavior unchanged (works without env vars)
- [ ] `FOXML_BRIDGE_URL=10.0.0.5:9000 cargo run` → connects to custom bridge
- [ ] `FOXML_TMP_DIR=/var/run/foxml cargo run` → reads PID/events from custom dir
- [ ] `FOXML_ROOT=/opt/foxml cargo run` → scans results/config from custom root
- [ ] `cargo build --release` passes
- [ ] No remaining hardcoded `127.0.0.1:8765` in source (except in config.rs default)
- [ ] No remaining hardcoded `/tmp/foxml_` in source (except in config.rs default)
