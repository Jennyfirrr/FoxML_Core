# UI Phase 0: Launcher Live Dashboard

**Status**: ✅ Complete
**Parent**: [dashboard-ui-completion-master.md](./dashboard-ui-completion-master.md)
**Effort**: 2h

---

## Problem Statement

The launcher front pane currently shows:
- Static status boxes (Bridge/Alpaca/Engine) with hardcoded or broken values
- "STAGE: unknown", "RUN: N/A" even when training is running
- No live data whatsoever

**This is the first thing users see and it shows nothing useful.**

---

## Target State

Replace the right side of the launcher with a live dashboard showing:

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─ TRAINING ─────────────────────────────────────────────────┐ │
│  │  Run: intelligent_output_20260121_043020_abc123            │ │
│  │  Stage: feature_selection                                   │ │
│  │  Progress: [████████████░░░░░░░░] 62%                      │ │
│  │  Target: ret_15m (8/13)                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ TRADING ──────────────────────────────────────────────────┐ │
│  │  Status: Active          P&L: +$1,234.56 (+0.82%)          │ │
│  │  Positions: 5            Cash: $148,765                    │ │
│  │  Stage: evaluating       Cycle: 1,247                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ RECENT EVENTS ────────────────────────────────────────────┐ │
│  │  10:32:15  TRADE_FILLED     AAPL +100 @ $178.50           │ │
│  │  10:31:42  DECISION_MADE    MSFT: BUY                     │ │
│  │  10:30:01  CYCLE_START      Cycle 1247                    │ │
│  │  10:29:58  TRAINING         feature_selection 62%         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Training Data
- **Source**: `/tmp/foxml_training_events.jsonl`
- **Method**: Same file-tailing as TrainingView
- **Fields needed**:
  - `run_id` - current run name
  - `stage` - ranking/feature_selection/training/completed
  - `progress_pct` - overall progress
  - `current_target` - target being processed
  - `targets_complete` / `targets_total`

### Trading Data
- **Source**: Bridge API (when connected)
  - `/api/metrics` - portfolio value, P&L, positions count
  - `/api/state` - engine status, current stage
  - `/api/positions` - position count
- **Fallback**: Show "Bridge offline" if not connected

### Recent Events
- **Source**: Combined from both:
  - Training events file
  - Bridge `/ws/events` or `/api/events/recent`
- **Display**: Last 5 events, newest first

---

## Implementation

### Step 1: Create LiveDashboard widget

New file: `src/launcher/live_dashboard.rs`

```rust
pub struct LiveDashboard {
    theme: Theme,
    // Training state (from events file)
    training_run_id: Option<String>,
    training_stage: String,
    training_progress: f64,
    training_target: Option<String>,
    training_targets_complete: i64,
    training_targets_total: i64,
    event_file_pos: u64,

    // Trading state (from bridge)
    trading_connected: bool,
    trading_status: String,
    trading_stage: String,
    trading_pnl: f64,
    trading_pnl_pct: f64,
    trading_positions: i64,
    trading_cash: f64,
    trading_cycle: i64,

    // Combined recent events
    recent_events: VecDeque<DashboardEvent>,
}

struct DashboardEvent {
    timestamp: String,
    source: EventSource,  // Training or Trading
    event_type: String,
    message: String,
}
```

### Step 2: Replace StatusCanvas in LauncherView

Current launcher layout:
```
[ Menu ] [ StatusCanvas (broken) ]
```

New layout:
```
[ Menu ] [ LiveDashboard ]
```

### Step 3: Poll for updates

In the render loop:
1. Poll training events file (every 500ms)
2. Poll bridge API for trading data (every 1s, if connected)
3. Update display

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/launcher/mod.rs` | Add `live_dashboard` module |
| `src/launcher/live_dashboard.rs` | **NEW** - Live dashboard widget |
| `src/views/launcher.rs` | Replace StatusCanvas with LiveDashboard |
| `src/launcher/status_canvas.rs` | Can be removed or kept for reference |

---

## Checklist

### Step 1: LiveDashboard widget
- [ ] Create `src/launcher/live_dashboard.rs`
- [ ] Implement training state from events file
- [ ] Implement trading state from bridge API
- [ ] Implement recent events list
- [ ] Implement render methods for each section

### Step 2: Integration
- [ ] Add module to `src/launcher/mod.rs`
- [ ] Replace StatusCanvas in LauncherView
- [ ] Wire up polling in render loop

### Step 3: Polish
- [ ] Show "No training running" when idle
- [ ] Show "Bridge offline" when disconnected
- [ ] Color-code events by type
- [ ] Test with actual training run

---

## Testing

```bash
# Start a training run
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config production_baseline \
    --output-dir TRAINING/results/test_run

# Launch dashboard
bin/foxml

# Verify:
# - Training section shows live progress
# - Progress bar updates
# - Stage changes are reflected
# - Recent events show training events
```
