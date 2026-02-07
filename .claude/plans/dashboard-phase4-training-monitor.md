# Dashboard Phase 4: Training Monitoring

**Status**: ✅ Complete
**Parent**: [dashboard-integration-master.md](./dashboard-integration-master.md)
**Estimated Effort**: 2 hours
**Dependencies**: Phase 2 (Bridge API) complete
**Completed**: 2026-01-21

---

## UI Fixes Completed (Pre-requisite)

Before implementing training monitoring, the following UI issues were fixed:

### 1. Status Bar Mismatch ✅
- **Issue**: Status bar hardcoded "Bridge: Connected" regardless of actual status
- **Fix**: Added `bridge_connected` field to LauncherView, synced from StatusCanvas
- **Files**: `src/views/launcher.rs`, `src/launcher/status_canvas.rs`

### 2. Menu Selection Style ✅
- **Issue**: Blinking cursor (`│`) indicator was distracting
- **Fix**: Replaced with static arrow (`▸`) indicator + accent color for selected items
- **Files**: `src/launcher/menu.rs`

### 3. Logo Alignment ✅
- **Issue**: Fox mascot + FOX ML combined logo had uneven line widths causing misalignment
- **Fix**: Simplified to centered block text only (removed fox mascot from logo)
- **Files**: `src/views/launcher.rs`

### Remaining UI Issues (Future Work)
- **Placeholder screens**: Many menu items lead to placeholder views
  - Config Editor, Model Selector, Service Manager, Log Viewer, File Browser, Settings
  - Each needs full implementation (separate phases)

---

## Objective

Enable real-time training progress monitoring through the same dashboard infrastructure used for trading.

---

## Current State

The training view (`src/views/training.rs`) currently:
- Scans `RESULTS/` directories for manifest.json files
- Lists discovered runs with basic status
- Detects running processes by checking for `python` processes

**Limitations**:
- No real-time progress updates
- Stage transitions not visible
- Target completion not streamed
- Must manually refresh to see updates

---

## Target State

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  intelligent_trainer.py                                                       │
│                                                                               │
│  _run_training_stages()                                                       │
│       │                                                                       │
│       ├── emit_progress("ranking", 10%)                                       │
│       ├── emit_progress("feature_selection", 30%)                             │
│       ├── emit_target_complete("ret_5m", "success", auc=0.65)                │
│       ├── emit_progress("training", 60%)                                      │
│       └── emit_run_complete()                                                 │
│                                                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TrainingEventEmitter (new)                                                   │
│                                                                               │
│  - Writes to socket/file for bridge to read                                   │
│  - Or: Uses HTTP callback to bridge                                           │
│  - Or: Shared event queue                                                     │
│                                                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Bridge /ws/training                                                          │
│                                                                               │
│  - Streams events to connected dashboards                                     │
│  - Buffers recent events for late joiners                                     │
│                                                                               │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Training View (Rust)                                                         │
│                                                                               │
│  - Subscribes to /ws/training                                                 │
│  - Updates progress bar                                                       │
│  - Shows current stage and target                                             │
│  - Lists completed targets with metrics                                       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Part 1: Training Event Emitter (Python)

**Create `TRAINING/orchestration/utils/training_events.py`**:

```python
"""
Training Event Emitter
======================

Emits training progress events for dashboard monitoring.

Events are written to a Unix socket or file that the bridge reads.
"""

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Event socket path
TRAINING_EVENT_SOCKET = os.environ.get(
    "FOXML_TRAINING_EVENT_SOCKET",
    "/tmp/foxml_training_events.sock"
)

# Fallback: event file for polling
TRAINING_EVENT_FILE = os.environ.get(
    "FOXML_TRAINING_EVENT_FILE",
    "/tmp/foxml_training_events.jsonl"
)


class TrainingEventEmitter:
    """
    Emits training events for dashboard consumption.

    Tries Unix socket first, falls back to file append.
    """

    def __init__(self, run_id: str):
        """
        Initialize emitter for a training run.

        Args:
            run_id: The training run identifier
        """
        self.run_id = run_id
        self._socket: Optional[socket.socket] = None
        self._file_path = Path(TRAINING_EVENT_FILE)
        self._connect_socket()

    def _connect_socket(self) -> None:
        """Try to connect to event socket."""
        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(TRAINING_EVENT_SOCKET)
            logger.debug(f"Connected to training event socket: {TRAINING_EVENT_SOCKET}")
        except (FileNotFoundError, ConnectionRefusedError):
            self._socket = None
            logger.debug("Training event socket not available, using file fallback")

    def _emit(self, event: Dict[str, Any]) -> None:
        """
        Emit an event.

        Args:
            event: Event dictionary to emit
        """
        event["run_id"] = self.run_id
        event["timestamp"] = datetime.now(timezone.utc).isoformat()

        event_json = json.dumps(event) + "\n"

        # Try socket first
        if self._socket:
            try:
                self._socket.send(event_json.encode())
                return
            except (BrokenPipeError, ConnectionResetError):
                self._socket = None

        # Fallback to file
        try:
            with open(self._file_path, "a") as f:
                f.write(event_json)
        except Exception as e:
            logger.warning(f"Failed to write training event: {e}")

    def emit_progress(
        self,
        stage: str,
        progress_pct: float,
        current_target: Optional[str] = None,
        targets_complete: int = 0,
        targets_total: int = 0,
    ) -> None:
        """
        Emit a progress update.

        Args:
            stage: Current stage (ranking, feature_selection, training)
            progress_pct: Overall progress percentage [0, 100]
            current_target: Target currently being processed
            targets_complete: Number of targets completed
            targets_total: Total number of targets
        """
        self._emit({
            "event_type": "progress",
            "stage": stage,
            "progress_pct": progress_pct,
            "current_target": current_target,
            "targets_complete": targets_complete,
            "targets_total": targets_total,
        })

    def emit_stage_change(self, previous_stage: Optional[str], new_stage: str) -> None:
        """
        Emit a stage transition.

        Args:
            previous_stage: Previous stage (None if starting)
            new_stage: New stage being entered
        """
        self._emit({
            "event_type": "stage_change",
            "previous_stage": previous_stage,
            "new_stage": new_stage,
        })

    def emit_target_complete(
        self,
        target: str,
        status: str,
        models_trained: int = 0,
        best_auc: Optional[float] = None,
    ) -> None:
        """
        Emit target completion.

        Args:
            target: Target name
            status: Completion status (success, failed, skipped)
            models_trained: Number of models trained
            best_auc: Best AUC achieved (if available)
        """
        self._emit({
            "event_type": "target_complete",
            "target": target,
            "status": status,
            "models_trained": models_trained,
            "best_auc": best_auc,
        })

    def emit_run_complete(self, status: str = "success") -> None:
        """
        Emit run completion.

        Args:
            status: Final status (success, failed, cancelled)
        """
        self._emit({
            "event_type": "run_complete",
            "status": status,
        })

    def emit_error(
        self,
        error_message: str,
        error_type: str = "unknown",
        recoverable: bool = True,
    ) -> None:
        """
        Emit an error event.

        Args:
            error_message: Error description
            error_type: Error classification
            recoverable: Whether training can continue
        """
        self._emit({
            "event_type": "error",
            "error_message": error_message,
            "error_type": error_type,
            "recoverable": recoverable,
        })

    def close(self) -> None:
        """Close the event emitter."""
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None


# Global emitter instance
_emitter: Optional[TrainingEventEmitter] = None


def init_training_events(run_id: str) -> TrainingEventEmitter:
    """
    Initialize the global training event emitter.

    Args:
        run_id: Training run identifier

    Returns:
        The event emitter instance
    """
    global _emitter
    _emitter = TrainingEventEmitter(run_id)
    return _emitter


def get_training_emitter() -> Optional[TrainingEventEmitter]:
    """Get the global training event emitter."""
    return _emitter


def emit_progress(stage: str, progress_pct: float, **kwargs) -> None:
    """Convenience function to emit progress."""
    if _emitter:
        _emitter.emit_progress(stage, progress_pct, **kwargs)


def emit_stage_change(previous_stage: Optional[str], new_stage: str) -> None:
    """Convenience function to emit stage change."""
    if _emitter:
        _emitter.emit_stage_change(previous_stage, new_stage)


def emit_target_complete(target: str, status: str, **kwargs) -> None:
    """Convenience function to emit target completion."""
    if _emitter:
        _emitter.emit_target_complete(target, status, **kwargs)
```

---

### Part 2: Integrate with Intelligent Trainer

**Changes to `TRAINING/orchestration/intelligent_trainer.py`**:

```python
from TRAINING.orchestration.utils.training_events import (
    init_training_events,
    emit_progress,
    emit_stage_change,
    emit_target_complete,
)

class IntelligentTrainer:
    def __init__(self, ...):
        # ... existing init
        self._event_emitter = None

    def _initialize_events(self) -> None:
        """Initialize training event emitter."""
        if self.run_identity:
            self._event_emitter = init_training_events(self.run_identity.run_id)

    def _run_training_stages(self) -> None:
        """Run the training pipeline stages."""
        self._initialize_events()

        # Stage 1: Target Ranking
        emit_stage_change(None, "ranking")
        emit_progress("ranking", 0)

        targets = self._rank_targets()

        emit_progress("ranking", 100, targets_total=len(targets))

        # Stage 2: Feature Selection
        emit_stage_change("ranking", "feature_selection")

        for i, target in enumerate(targets):
            progress = (i / len(targets)) * 100
            emit_progress(
                "feature_selection",
                progress,
                current_target=target,
                targets_complete=i,
                targets_total=len(targets),
            )

            features = self._select_features(target)

        emit_stage_change("feature_selection", "training")

        # Stage 3: Model Training
        for i, target in enumerate(targets):
            progress = (i / len(targets)) * 100
            emit_progress(
                "training",
                progress,
                current_target=target,
                targets_complete=i,
                targets_total=len(targets),
            )

            result = self._train_target(target)

            emit_target_complete(
                target=target,
                status="success" if result.success else "failed",
                models_trained=result.models_trained,
                best_auc=result.best_auc,
            )

        # Complete
        emit_progress("training", 100, targets_complete=len(targets), targets_total=len(targets))
        if self._event_emitter:
            self._event_emitter.emit_run_complete("success")
```

---

### Part 3: Bridge Event Reader

**Add to `DASHBOARD/bridge/server.py`**:

```python
import asyncio
from pathlib import Path

TRAINING_EVENT_FILE = Path(os.environ.get(
    "FOXML_TRAINING_EVENT_FILE",
    "/tmp/foxml_training_events.jsonl"
))

# Training event queue for WebSocket streaming
training_event_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

# Recent training events buffer
recent_training_events: List[Dict[str, Any]] = []
MAX_RECENT_TRAINING_EVENTS = 50


async def watch_training_events():
    """
    Watch for new training events from the event file.

    Runs as a background task, reading new events and pushing to queue.
    """
    last_position = 0

    while True:
        try:
            if TRAINING_EVENT_FILE.exists():
                with open(TRAINING_EVENT_FILE, "r") as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                    for line in new_lines:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                # Add to recent buffer
                                recent_training_events.append(event)
                                if len(recent_training_events) > MAX_RECENT_TRAINING_EVENTS:
                                    recent_training_events.pop(0)
                                # Push to queue
                                try:
                                    training_event_queue.put_nowait(event)
                                except asyncio.QueueFull:
                                    pass
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.error(f"Error watching training events: {e}")

        await asyncio.sleep(0.5)  # Poll every 500ms


@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    asyncio.create_task(watch_training_events())


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for training progress streaming.
    """
    await websocket.accept()

    # Send recent events on connect
    for event in recent_training_events:
        await websocket.send_json(event)

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    training_event_queue.get(),
                    timeout=5.0
                )
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")


@app.get("/api/training/status")
async def get_training_status() -> Dict[str, Any]:
    """
    Get current training status from recent events.
    """
    if not recent_training_events:
        return {"running": False, "events": []}

    latest = recent_training_events[-1]
    return {
        "running": latest.get("event_type") != "run_complete",
        "run_id": latest.get("run_id"),
        "stage": latest.get("stage"),
        "progress_pct": latest.get("progress_pct", 0),
        "current_target": latest.get("current_target"),
        "events": recent_training_events[-10:],  # Last 10 events
    }
```

---

### Part 4: Rust Training View Updates

**Changes to `src/views/training.rs`**:

```rust
use tokio::sync::mpsc::Receiver;

pub struct TrainingView {
    // ... existing fields
    event_receiver: Option<Receiver<TrainingEvent>>,
    current_progress: TrainingProgress,
    completed_targets: Vec<CompletedTarget>,
}

#[derive(Default)]
struct TrainingProgress {
    running: bool,
    run_id: Option<String>,
    stage: String,
    progress_pct: f64,
    current_target: Option<String>,
    targets_complete: usize,
    targets_total: usize,
}

struct CompletedTarget {
    name: String,
    status: String,
    models_trained: usize,
    best_auc: Option<f64>,
}

impl TrainingView {
    pub async fn connect_events(&mut self) -> Result<()> {
        let url = format!("ws://{}/ws/training", self.client.host);
        let (ws_stream, _) = connect_async(&url).await?;
        let (_, mut read) = ws_stream.split();

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        self.event_receiver = Some(rx);

        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                if let Ok(Message::Text(text)) = msg {
                    if let Ok(event) = serde_json::from_str::<TrainingEvent>(&text) {
                        let _ = tx.send(event).await;
                    }
                }
            }
        });

        Ok(())
    }

    pub fn poll_events(&mut self) {
        if let Some(ref mut rx) = self.event_receiver {
            while let Ok(event) = rx.try_recv() {
                self.handle_event(event);
            }
        }
    }

    fn handle_event(&mut self, event: TrainingEvent) {
        match event.event_type.as_str() {
            "progress" => {
                self.current_progress.running = true;
                self.current_progress.run_id = Some(event.run_id);
                self.current_progress.stage = event.stage.unwrap_or_default();
                self.current_progress.progress_pct = event.progress_pct.unwrap_or(0.0);
                self.current_progress.current_target = event.current_target;
                self.current_progress.targets_complete = event.targets_complete.unwrap_or(0);
                self.current_progress.targets_total = event.targets_total.unwrap_or(0);
            }
            "target_complete" => {
                if let Some(target) = event.target {
                    self.completed_targets.push(CompletedTarget {
                        name: target,
                        status: event.status.unwrap_or_default(),
                        models_trained: event.models_trained.unwrap_or(0),
                        best_auc: event.best_auc,
                    });
                }
            }
            "run_complete" => {
                self.current_progress.running = false;
                self.current_progress.progress_pct = 100.0;
            }
            _ => {}
        }
    }

    fn render_progress(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        if !self.current_progress.running {
            buf.set_string(
                area.x,
                area.y,
                "No training running",
                Style::default().fg(theme.text_muted),
            );
            return;
        }

        // Stage indicator
        let stage_text = format!(
            "Stage: {} ({:.0}%)",
            self.current_progress.stage,
            self.current_progress.progress_pct
        );
        buf.set_string(area.x, area.y, &stage_text, Style::default().fg(theme.accent));

        // Progress bar
        let bar_width = area.width.saturating_sub(2) as usize;
        let filled = ((self.current_progress.progress_pct / 100.0) * bar_width as f64) as usize;
        let bar = format!(
            "[{}{}]",
            "█".repeat(filled),
            "░".repeat(bar_width - filled)
        );
        buf.set_string(area.x, area.y + 1, &bar, Style::default().fg(theme.accent));

        // Current target
        if let Some(ref target) = self.current_progress.current_target {
            let target_text = format!(
                "Target: {} ({}/{})",
                target,
                self.current_progress.targets_complete,
                self.current_progress.targets_total
            );
            buf.set_string(area.x, area.y + 2, &target_text, Style::default().fg(theme.text_secondary));
        }
    }
}
```

---

## Implementation Checklist

### Part 1: Event Emitter (30m) ✅
- [x] Create `TRAINING/orchestration/utils/training_events.py`
- [x] Implement TrainingEventEmitter class (HTTP POST + file fallback)
- [x] Add convenience functions (emit_progress, emit_stage_change, etc.)

### Part 2: Intelligent Trainer Integration (30m) ✅
- [x] Initialize emitter at run start (`init_training_events(run_id)`)
- [x] Emit progress at stage boundaries (ranking, feature_selection, training)
- [x] Emit target completion events with AUC metrics
- [x] Emit run completion with duration

### Part 3: Bridge Integration (30m) ✅
- [x] Add event file watcher (Phase 2 already added /ws/training)
- [x] Add /ws/training WebSocket (Phase 2)
- [x] Add /api/training/status endpoint (via existing infrastructure)

### Part 4: Rust View Updates (30m) ✅
- [x] Connect to /ws/training via `connect_training_ws()`
- [x] Handle progress events (TrainingEvent struct, poll_events)
- [x] Render progress bar with stage indicator
- [x] Show completed targets with AUC

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `TRAINING/orchestration/utils/training_events.py` | Create |
| `TRAINING/orchestration/intelligent_trainer.py` | Add event emission |
| `DASHBOARD/bridge/server.py` | Add training endpoints |
| `DASHBOARD/dashboard/src/views/training.rs` | Add live progress |

---

## Testing

```bash
# Start bridge
cd DASHBOARD/bridge && python server.py &

# Start dashboard
bin/foxml

# In another terminal, start a training run
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config production_baseline \
    --output-dir test_run

# Watch the Training view for live progress updates
```
