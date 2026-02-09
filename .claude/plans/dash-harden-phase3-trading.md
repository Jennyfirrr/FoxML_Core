# Phase 3: Trading View Enhancements

**Master plan**: `dashboard-hardening-master.md`
**Status**: Complete
**Scope**: 2 files modified
**Depends on**: Phase 1 (confirmation dialog for kill switch)

---

## Context

The trading view is the most-used screen but only supports passive monitoring. Users can't control trading, inspect positions, or sort data. The bridge already has control endpoints — they just need wiring.

---

## 3a: Position Detail Panel

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Design
Pressing Enter on a selected position opens an inline detail panel (replaces events panel temporarily).

### Detail Panel Contents
- Symbol, side (long/short), shares
- Entry price, current price, market value
- Unrealized P&L ($, %)
- Entry time, hold duration
- Portfolio weight
- Position sizing metadata (if available from `/api/decisions/recent`)

### Navigation
- Enter → open detail panel for selected position
- Esc → close detail panel, return to normal view
- j/k → cycle through positions while detail panel stays open

---

## 3b: Position Sorting

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Changes
1. Add `sort_mode: PositionSort` enum field
2. `s` key cycles through sort modes: Symbol (alpha), P&L (desc), Size (desc), Weight (desc)
3. Show current sort in positions panel title: "Positions [sorted by P&L]"
4. Sort applied on each data refresh

```rust
enum PositionSort {
    Symbol,
    PnlDesc,
    SizeDesc,
    WeightDesc,
}
```

---

## 3c: Kill Switch Toggle from Trading View

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Changes
1. `k` key opens confirmation dialog: "ACTIVATE kill switch? This will halt all trading immediately."
2. If already active: "Deactivate kill switch? Trading will resume."
3. On confirm: POST `/api/control/kill_switch` via `DashboardClient`
4. Show kill switch state prominently in risk gauge widget (already partially done)
5. Add visual indicator: red border on entire trading view when kill switch active

### Notes
- This is separate from the global Ctrl+K in app.rs (which Phase 0 wires up)
- Trading view version has context — shows risk metrics alongside the toggle

---

## 3d: Display Sharpe Ratio

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Changes
1. Sharpe ratio is already fetched in metrics but not rendered
2. Add to metrics panel alongside P&L: "Sharpe: 1.42"
3. Color: green if > 1.0, yellow if 0.5-1.0, red if < 0.5

---

## 3e: Wire Pause/Resume Control

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Changes
1. `p` key toggles pause/resume
2. Show current state: "PAUSED" badge in header when paused
3. POST `/api/control/pause` or `/api/control/resume` via client
4. No confirmation dialog needed — pause is non-destructive and easily reversible

---

## Verification

- [ ] Enter on position shows detail panel with full info
- [ ] Esc closes detail panel
- [ ] `s` cycles sort modes, positions reorder correctly
- [ ] `k` toggles kill switch with confirmation, state reflected in UI
- [ ] Sharpe ratio displayed with correct coloring
- [ ] `p` pauses/resumes trading, badge shown when paused
- [ ] `cargo build --release` passes
