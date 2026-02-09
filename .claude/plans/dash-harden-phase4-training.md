# Phase 4: Training View Enhancements

**Master plan**: `dashboard-hardening-master.md`
**Status**: Complete
**Scope**: 1 file modified
**Depends on**: Phase 1 (confirmation dialog for cancellation)

---

## Context

The training view monitors runs well but can't control them. It also does expensive recursive directory walks every 2 seconds, which will slow down as RESULTS/ grows.

---

## 4a: Training Run Cancellation

**File**: `DASHBOARD/dashboard/src/views/training.rs`

### Current State
- PID file detection exists (`/tmp/foxml_training.pid`)
- `is_process_running()` already checks if PID is alive via `ps`
- No way to stop a running process from the TUI

### Changes
1. `x` key on a running training run opens confirmation dialog: "Cancel training run {run_id}? This will send SIGTERM to PID {pid}."
2. On confirm: send SIGTERM via `kill {pid}` command
3. Wait 2s, check if still running, offer SIGKILL if needed
4. Update run status in UI after cancellation
5. Show notification: "Training run {run_id} cancelled"

### Safety
- Only show cancel option for runs that are actually running (PID alive)
- Never send SIGKILL without a second confirmation
- Log the cancellation action

---

## 4b: Cache RESULTS/ Directory Scan

**File**: `DASHBOARD/dashboard/src/views/training.rs`

### Current Problem
- `scan_results_directory()` does `WalkDir::new("RESULTS/")` on every update cycle
- With many runs this becomes expensive I/O

### Changes
1. Cache scan results in `cached_runs: Vec<RunInfo>` with `last_scan: Instant`
2. Only re-scan on:
   - First load
   - Explicit refresh (`r` key)
   - When a running training completes (event-driven)
   - After 60 seconds staleness (instead of 2s)
3. The 2s poll cycle still reads events from JSONL — just skip the directory walk

---

## 4c: Stage Detail Panel

**File**: `DASHBOARD/dashboard/src/views/training.rs`

### Changes
1. When a run is selected with active training, show stage details in a panel:
   - **Target Ranking**: targets evaluated / total, current target
   - **Feature Selection**: targets selected / total, current target
   - **Model Training**: models trained / total, current family + target, best AUC so far
2. Parse from training events (data already available in `TrainingEvent` struct)
3. Show as a detail panel below the run list

### Notes
- Training events already contain `targets_complete`, `targets_total`, `best_auc`
- Just needs rendering — data is there

---

## 4d: Training Throughput and ETA

**File**: `DASHBOARD/dashboard/src/views/training.rs`

### Changes
1. Track timestamps of target completions: `target_times: VecDeque<Instant>`
2. Compute rolling throughput: targets/minute over last 10 completions
3. Compute ETA: `(total - complete) / throughput`
4. Display in run detail: "Speed: 2.3 targets/min | ETA: ~14 min"
5. Show "Computing..." until at least 3 targets completed

### Notes
- Only meaningful during model training stage (not ranking/feature selection)
- Reset counters on stage change

---

## Verification

- [ ] `x` key cancels running training with confirmation
- [ ] RESULTS/ scan is cached, only refreshes on `r` or completion
- [ ] Stage details show target progress for current stage
- [ ] Throughput and ETA displayed during model training
- [ ] `cargo build --release` passes
