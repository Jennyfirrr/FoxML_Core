# Dashboard Event Integration

Guidelines for integrating training/trading events with the dashboard.

## Architecture Overview

```
Python (Training/Trading)          Rust Dashboard
         │                              │
         ├─► /tmp/foxml_training.pid    │  (PID file for process detection)
         │                              │
         ├─► /tmp/foxml_training_events.jsonl ◄─┤  (Event file, polled every 500ms)
         │                              │
         └─► HTTP POST to bridge ───────┘  (Optional, for real-time)
```

## Event File Format

Events are written as newline-delimited JSON (JSONL) to `/tmp/foxml_training_events.jsonl`.

### Event Types

| Event Type | When Emitted | Key Fields |
|------------|--------------|------------|
| `progress` | During stage execution | `stage`, `progress_pct`, `current_target`, `targets_complete`, `targets_total`, `message` |
| `stage_change` | Stage transitions | `previous_stage`, `new_stage` |
| `target_start` | Target processing begins | `target`, `target_index`, `total_targets` |
| `target_complete` | Target processing ends | `target`, `status`, `best_auc`, `duration_seconds` |
| `run_complete` | Run finishes | `status`, `total_targets`, `successful_targets`, `duration_seconds` |
| `error` | Errors occur | `error_message`, `error_type`, `recoverable`, `target` |

### Common Fields (All Events)

```json
{
  "event_type": "progress",
  "run_id": "intelligent_output_20260121_133213_64987a2d",
  "timestamp": "2026-01-21T19:32:13.024320+00:00"
}
```

## Python Side: Emitting Events

### Import and Initialize

```python
from TRAINING.orchestration.utils.training_events import (
    init_training_events,
    emit_progress,
    emit_stage_change,
    emit_target_start,
    emit_target_complete,
    emit_run_complete,
    emit_error,
    close_training_events,
)

# Initialize at run start (writes PID file)
init_training_events(run_id)

# Clean up at run end
close_training_events()
```

### Emitting Progress During Long Operations

**IMPORTANT**: Long-running stages (like ranking) should emit progress events periodically so the dashboard can show activity.

```python
# During ranking stage - emit progress for each target evaluated
for i, target in enumerate(targets):
    emit_progress(
        stage="ranking",
        progress_pct=(i / len(targets)) * 100,
        current_target=target,
        targets_complete=i,
        targets_total=len(targets),
        message=f"Evaluating {target}"
    )
    # ... do ranking work ...

# After ranking completes
emit_progress(
    stage="ranking",
    progress_pct=100,
    targets_total=len(selected_targets),
    message=f"Ranking complete: {len(selected_targets)} targets selected"
)
```

### Stage Transitions

```python
# Always emit stage changes
emit_stage_change("ranking", "feature_selection")
emit_progress("feature_selection", 0, message="Starting feature selection")
```

## Rust Side: Consuming Events

### Timestamp Handling

Python emits timestamps in UTC (via `datetime.now(timezone.utc).isoformat()`). The Rust dashboard converts these to local time for display:

```rust
use chrono::{DateTime, FixedOffset, Local};

// Convert UTC ISO timestamp to local time for display
let timestamp = if !event.timestamp.is_empty() {
    if let Ok(dt) = DateTime::<FixedOffset>::parse_from_rfc3339(&event.timestamp) {
        dt.with_timezone(&Local).format("%H:%M:%S").to_string()
    } else if event.timestamp.len() >= 19 {
        event.timestamp[11..19].to_string()  // Fallback: extract HH:MM:SS
    } else {
        event.timestamp.clone()
    }
} else {
    Local::now().format("%H:%M:%S").to_string()
};
```

### Event Struct (`src/api/events.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEvent {
    pub event_type: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub new_stage: Option<String>,  // For stage_change events
    #[serde(default)]
    pub progress_pct: f64,
    // ... other fields with #[serde(default)]
}

impl TrainingEvent {
    /// Handle stage_change using new_stage field
    pub fn effective_stage(&self) -> &str {
        if self.event_type == "stage_change" {
            self.new_stage.as_deref().unwrap_or(&self.stage)
        } else {
            &self.stage
        }
    }
}
```

### Stage Tags in Dashboard

The dashboard renders stage-specific tags for events:

| Stage | Tag | Color |
|-------|-----|-------|
| `ranking` | TR | Yellow (warning) |
| `feature_selection` | FS | Accent |
| `training` | TRN | Green (success) |
| `initializing` | INIT | Muted |
| `completed` | DONE | Green (success) |

The `DashboardEvent` struct includes a `stage` field populated from `event.effective_stage()`.
```

### Polling Pattern (`src/views/training.rs`)

```rust
impl TrainingView {
    fn poll_events(&mut self) {
        // Only poll every 500ms
        if self.last_event_poll.elapsed().as_millis() < 500 {
            return;
        }

        // Read new lines from file since last position
        let events = self.read_new_events();
        for event in events {
            self.handle_training_event(event);
        }
    }

    fn handle_training_event(&mut self, event: TrainingEvent) {
        self.live_run_id = Some(event.run_id.clone());

        match event.event_type.as_str() {
            "progress" => {
                self.live_stage = event.effective_stage().to_string();
                self.live_progress = event.progress_pct;
                // ... update other fields
            }
            "stage_change" => {
                self.live_stage = event.effective_stage().to_string();
            }
            // ... handle other event types
        }
    }
}
```

### State Recovery on Startup

The dashboard reads all existing events on startup to recover state if a training run is already in progress:

```rust
fn recover_current_state(&mut self) {
    // Read all events, find last run
    // If last run has no run_complete event, replay its events
    // Set file position to end for future polling
}
```

## PID File

Written by Python at `/tmp/foxml_training.pid`:

```json
{
  "pid": 12345,
  "run_id": "intelligent_output_20260121_133213_64987a2d",
  "started_at": "2026-01-21T19:32:13.024230+00:00"
}
```

Rust checks if process is alive via `/proc/{pid}`.

## Adding New Event Types

1. **Python**: Add emit function in `TRAINING/orchestration/utils/training_events.py`
2. **Rust**: Add fields to `TrainingEvent` struct with `#[serde(default)]`
3. **Rust**: Handle in `handle_training_event()` match statement
4. **Document**: Update this skill file

## Debugging

```bash
# Check PID file
cat /tmp/foxml_training.pid

# Watch events in real-time
tail -f /tmp/foxml_training_events.jsonl

# Check if process is running
ps aux | grep intelligent_trainer

# Count events
wc -l /tmp/foxml_training_events.jsonl
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| No live progress shown | No events being emitted | Add `emit_progress()` calls to Python code |
| Progress stuck at 0% | Stage doesn't emit progress updates | Add periodic `emit_progress()` during long operations |
| Wrong stage shown | `stage_change` uses `new_stage` not `stage` | Use `event.effective_stage()` in Rust |
| Dashboard doesn't detect running training | PID file missing or stale | Ensure `init_training_events()` called at start |
| Timestamps wrong by hours | UTC displayed instead of local | Use `DateTime::parse_from_rfc3339` + `with_timezone(&Local)` |
| Code changes don't affect running process | Python process uses old code | New training runs will use updated code |
