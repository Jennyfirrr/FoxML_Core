# Dashboard Functionality Audit & Improvement Plan

**Status**: Phase A, B, C Complete
**Created**: 2026-01-21
**Updated**: 2026-01-21
**Scope**: DASHBOARD/dashboard/ - Rust TUI functionality fixes and improvements

## Completed Work

### Phase A: Critical Fixes ✓
- [x] Fix process detection with PID file (training.rs, training_events.py)
- [x] State recovery on startup (reads existing events to find current run)
- [x] Basic Training Launcher config selection

### Phase B: UX Improvements ✓
- [x] Full Training Launcher redesign (config list, output dir, deterministic toggle)
- [x] Config Browser redesign (read-only preview, open in $EDITOR)
- [x] Training executor uses nohup/setsid for detached process

### Phase C: Polish ✓
- [x] Log Viewer improvements (multi-log discovery, Tab to cycle files)
- [x] Service Manager improvements (dynamic discovery, NotInstalled status)
- [x] Fixed j/k navigation routing to all views
- [x] Fixed back navigation (q/Esc) for all views

---

## Executive Summary

The dashboard TUI compiles and renders but has significant functionality gaps. This plan addresses:
1. Training run detection issues (critical)
2. Training Launcher UX improvements (high priority)
3. Config Editor redesign (medium priority)
4. Various smaller improvements (low priority)

---

## Part 1: Training Run Detection Issues (Critical)

### Current Architecture

```
Python (intelligent_trainer.py)
    │
    ├──► HTTP POST to bridge (127.0.0.1:8765/api/training/event)
    │    [Preferred but requires bridge running]
    │
    └──► File append to /tmp/foxml_training_events.jsonl
         [Fallback - always available]

Rust TUI (training.rs)
    │
    ├──► Poll /tmp/foxml_training_events.jsonl (500ms interval)
    │    [Reads new events since last position]
    │
    └──► ps aux | grep intelligent_trainer.py
         [Process detection for "running" status]
```

### Issues Identified

| Issue | Severity | Description |
|-------|----------|-------------|
| **Tmp file volatility** | Medium | `/tmp/foxml_training_events.jsonl` cleared on reboot |
| **File may not exist** | Low | No events file until first training run |
| **Fragile process detection** | High | `ps aux` parsing breaks on path/arg changes |
| **Missing manifest fields** | Medium | `progress`, `targets_complete` not in manifest.json |
| **No file lock** | Medium | Concurrent write/read could corrupt file |
| **Stale events** | Low | Old events from previous runs accumulate |
| **Bridge dependency** | Medium | HTTP events require bridge; file fallback works but less immediate |

### Proposed Fixes

#### 1.1 Improve Process Detection (High Priority)

**Current**: Parses `ps aux` output looking for "intelligent_trainer.py"

**Problem**: Fragile - depends on exact command line format

**Solution**: Use `/proc` filesystem directly or check for PID file

```rust
// Option A: Check for PID file written by Python
fn detect_running_from_pidfile() -> Option<RunningInfo> {
    let pidfile = PathBuf::from("/tmp/foxml_training.pid");
    if pidfile.exists() {
        if let Ok(content) = fs::read_to_string(&pidfile) {
            let pid: u32 = content.trim().parse().ok()?;
            // Check if process exists
            if PathBuf::from(format!("/proc/{}", pid)).exists() {
                return Some(RunningInfo { pid, ... });
            }
        }
    }
    None
}

// Python side: Write PID file in intelligent_trainer.py
def write_pidfile():
    with open("/tmp/foxml_training.pid", "w") as f:
        f.write(str(os.getpid()))
```

**Action Items**:
- [ ] Add PID file writing to `intelligent_trainer.py` startup
- [ ] Add PID file cleanup on exit (atexit handler)
- [ ] Update `detect_running_processes()` to check PID file first
- [ ] Fall back to `ps aux` if no PID file

#### 1.2 Persist Events File in Project (Medium Priority)

**Current**: `/tmp/foxml_training_events.jsonl` (lost on reboot)

**Solution**: Use project-local path with run-id namespacing

```
RESULTS/.dashboard/
├── current_run.json       # Current run info (if any)
├── events/
│   └── {run_id}.jsonl     # Events per run
└── training.pid           # PID file
```

**Action Items**:
- [ ] Create `RESULTS/.dashboard/` directory structure
- [ ] Update Python `TRAINING_EVENT_FILE` path
- [ ] Add run-id to events file path
- [ ] Update Rust to read from new location
- [ ] Keep `/tmp/` as fallback for backward compatibility

#### 1.3 Add Progress to Manifest (Low Priority)

**Current**: manifest.json has no `progress` or `targets_complete` fields

**Solution**: Update manifest writer to include progress fields

```json
{
  "run_id": "...",
  "status": "running",
  "progress_pct": 45.2,
  "targets_complete": 5,
  "targets_total": 12,
  "current_stage": "training",
  "current_target": "AAPL_20d_return"
}
```

**Action Items**:
- [ ] Update `manifest.py` to write progress fields
- [ ] Update manifest on each stage/target change
- [ ] Update Rust `RunManifest` struct to read new fields

---

## Part 2: Training Launcher (Run Manager) - High Priority

### Current Issues

| Issue | Description |
|-------|-------------|
| **Hardcoded config** | Always uses "production_baseline" |
| **Hardcoded output** | Always uses "RESULTS/prod" |
| **No command preview** | User can't see what will run |
| **Missing determinism wrapper** | Doesn't use `bin/run_deterministic.sh` |
| **No config validation** | Doesn't check if config exists |

### Proposed Redesign

```
┌─────────────────────────────────────────────────────────────────┐
│ Training Pipeline Launcher                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Experiment Config:                                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ > production_baseline                                       ││
│  │   quick_e2e_test                                            ││
│  │   development_fast                                          ││
│  │   full_training                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Output Directory: RESULTS/______________________                │
│                                                                  │
│  □ Use deterministic mode (bin/run_deterministic.sh)            │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  Command Preview:                                                │
│  bin/run_deterministic.sh python -m                             │
│    TRAINING.orchestration.intelligent_trainer                    │
│    --experiment-config production_baseline                       │
│    --output-dir RESULTS/prod_20260121                           │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  [Enter] Start   [Tab] Next Field   [Esc] Cancel                │
└─────────────────────────────────────────────────────────────────┘
```

### Action Items

- [ ] Scan `CONFIG/experiments/*.yaml` for config list
- [ ] Add config selector widget with j/k navigation
- [ ] Add output directory text input
- [ ] Add checkbox for deterministic mode
- [ ] Show command preview that updates live
- [ ] Use `bin/run_deterministic.sh` when checkbox enabled
- [ ] Validate config exists before starting
- [ ] Auto-generate output dir name with date: `RESULTS/{config}_{YYYYMMDD}`

### Implementation Plan

**Phase 1**: Config selector
```rust
struct TrainingLauncherView {
    executor: TrainingExecutor,
    // New fields:
    configs: Vec<String>,           // Available configs
    selected_config: usize,         // Currently selected
    output_dir: String,             // User-editable
    deterministic: bool,            // Use wrapper script
    focus: LauncherFocus,           // Config/OutputDir/Start
}

enum LauncherFocus {
    ConfigList,
    OutputDir,
    StartButton,
}
```

**Phase 2**: Scan for configs
```rust
fn scan_configs(&mut self) {
    let config_dir = PathBuf::from("CONFIG/experiments");
    if let Ok(entries) = fs::read_dir(&config_dir) {
        self.configs = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "yaml"))
            .map(|e| e.path().file_stem().unwrap().to_string_lossy().to_string())
            .collect();
        self.configs.sort();
    }
}
```

**Phase 3**: Update executor to use wrapper
```rust
// In TrainingExecutor
pub fn start_training(&mut self, use_deterministic: bool) -> Result<()> {
    let mut cmd = if use_deterministic {
        let mut c = Command::new("bin/run_deterministic.sh");
        c.arg("python");
        c
    } else {
        Command::new("python")
    };

    cmd.arg("-m")
       .arg("TRAINING.orchestration.intelligent_trainer")
       .arg("--experiment-config")
       .arg(&self.experiment_config)
       .arg("--output-dir")
       .arg(&self.output_dir);
    // ...
}
```

---

## Part 3: Config Editor Redesign - Medium Priority

### Current Issues

- Opens single hardcoded file (`CONFIG/experiments/production_baseline.yaml`)
- Full inline editor is complex and error-prone
- User requested: list → preview → system editor

### Proposed Redesign

```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration Browser                                            │
├─────────────────────────────────────────────────────────────────┤
│ Experiments:           │ Preview:                                │
│ ────────────────────── │ ──────────────────────────────────────  │
│ > production_baseline  │ # Production Baseline Config            │
│   quick_e2e_test       │                                         │
│   development_fast     │ description: Full production training   │
│   full_training        │                                         │
│                        │ overrides:                              │
│ Pipeline Configs:      │   pipeline.targets.max_targets: null    │
│ ────────────────────── │   pipeline.training.families:           │
│   pipeline.yaml        │     - lightgbm                          │
│   training/families    │     - xgboost                           │
│                        │     - catboost                          │
│                        │   ...                                   │
├─────────────────────────────────────────────────────────────────┤
│ [Enter] Open in $EDITOR   [u] Use for Run   [r] Refresh         │
└─────────────────────────────────────────────────────────────────┘
```

### Action Items

- [ ] Replace `ConfigEditorView` with `ConfigBrowserView`
- [ ] Scan `CONFIG/experiments/*.yaml` for experiment list
- [ ] Scan `CONFIG/pipeline/*.yaml` for pipeline configs
- [ ] Show read-only preview in right pane
- [ ] `Enter` opens file in `$EDITOR` (via `std::process::Command`)
- [ ] `u` sets config for Training Launcher
- [ ] Remove complex inline editing code

### Implementation

```rust
pub struct ConfigBrowserView {
    theme: Theme,
    experiments: Vec<ConfigEntry>,
    pipeline_configs: Vec<ConfigEntry>,
    selected_section: ConfigSection,
    selected_index: usize,
    preview_content: String,
}

enum ConfigSection {
    Experiments,
    PipelineConfigs,
}

impl ConfigBrowserView {
    fn open_in_editor(&self) {
        let path = self.selected_path();
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vim".to_string());

        // Suspend TUI, run editor, resume TUI
        // This requires special handling in the app
    }
}
```

---

## Part 4: Other Improvements - Low Priority

### 4.1 Log Viewer Issues

**Problem**: Expects `training.log` but actual log location varies

**Fix**:
- Search for any `.log` files in run directory
- Support `logs/` subdirectory pattern
- Show list of available log files if multiple found

### 4.2 Service Manager

**Problem**: Hardcoded service names that may not exist

**Fix**:
- Check which services actually exist
- Allow configuring service names in Settings
- Show "not installed" status for missing services

### 4.3 File Browser Integration

**Enhancement**:
- `e` key to open file in Config Editor / system editor
- `u` key to use selected config for Training Launcher

### 4.4 Model Selector → Training Launcher Integration

**Enhancement**:
- After selecting a model, offer to start new training based on that run's config
- Copy config from successful run to new experiment

### 4.5 Theme and UI Polish

- Loading indicators during scans
- Better error messages
- Confirmation dialogs for destructive actions
- Keyboard shortcut help overlay (`?` key)

---

## Part 5: Implementation Order

### Phase A: Critical Fixes (Do First)
1. Fix process detection with PID file
2. Add progress fields to manifest
3. Basic Training Launcher config selection

### Phase B: UX Improvements
4. Full Training Launcher redesign
5. Config Browser redesign
6. Events file persistence

### Phase C: Polish
7. Log Viewer improvements
8. Service Manager improvements
9. View integrations
10. UI polish

---

## Files to Modify

### Rust (DASHBOARD/dashboard/src/)
| File | Changes |
|------|---------|
| `views/training.rs` | PID file detection, new manifest fields |
| `views/training_launcher.rs` | Complete rewrite |
| `views/config_editor.rs` | Rename to config_browser, simplify |
| `launcher/training_executor.rs` | Add deterministic wrapper support |
| `views/log_viewer.rs` | Better log file discovery |
| `views/service_manager.rs` | Dynamic service detection |

### Python (TRAINING/)
| File | Changes |
|------|---------|
| `orchestration/intelligent_trainer.py` | Write PID file |
| `orchestration/utils/manifest.py` | Add progress fields |
| `orchestration/utils/training_events.py` | Update events file path |

---

## Testing Checklist

- [x] Training view shows live progress during actual run (state recovery implemented)
- [x] Training view correctly detects running process (PID file support added)
- [x] Training Launcher lists all experiments (scans CONFIG/experiments/)
- [x] Training Launcher starts run with correct command (deterministic wrapper support)
- [x] Config Browser shows all configs (experiments + pipeline configs)
- [x] Config Browser opens file in $EDITOR
- [x] Log Viewer finds and tails training logs (multi-file discovery)
- [x] Service Manager shows correct status for installed services (NotInstalled detection)
- [x] Settings persist across restarts (already working)

---

## Open Questions

1. Should we remove the inline config editor entirely, or keep it as an option?
2. Should Training Launcher support custom CLI arguments beyond config/output?
3. Should we add a "Recent Runs" quick-access in Training Monitor?
