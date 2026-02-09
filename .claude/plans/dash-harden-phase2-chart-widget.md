# Phase 2: Chart Widget

**Master plan**: `dashboard-hardening-master.md`
**Status**: Pending
**Scope**: 3 files modified
**Depends on**: Nothing (independent)

---

## Context

The chart widget (`widgets/chart.rs`) is a TODO stub — the only unimplemented widget. There's no data visualization anywhere in the dashboard. This phase adds sparklines and line charts.

---

## 2a: Implement Sparkline Renderer

**File**: `DASHBOARD/dashboard/src/widgets/chart.rs`

### Design
Compact sparkline using Unicode block characters (▁▂▃▄▅▆▇█) that fits in 1 row height.

```rust
pub struct Sparkline {
    data: VecDeque<f64>,
    max_points: usize,
    color: Color,
    label: Option<String>,
}

impl Sparkline {
    pub fn push(&mut self, value: f64);
    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme);
}
```

### Features
- Auto-scaling Y axis (min/max of visible data)
- Configurable color (green for positive P&L, red for negative)
- Optional label prefix ("P&L: ▁▂▃▅▇")
- Fixed-width: one char per data point, newest on right

---

## 2b: Implement Line Chart Renderer

**File**: `DASHBOARD/dashboard/src/widgets/chart.rs`

### Design
Full chart using Braille dots (⠁⠂⠄⡀⠈⠐⠠⢀) for 2x4 subpixel resolution.

```rust
pub struct LineChart {
    series: Vec<Series>,
    x_label: String,
    y_label: String,
    show_axis: bool,
    show_legend: bool,
}

pub struct Series {
    data: VecDeque<(f64, f64)>,  // (timestamp, value)
    color: Color,
    label: String,
}
```

### Features
- Multiple series support (P&L + benchmark on same chart)
- Auto-scaling axes with tick marks
- Braille-dot rendering (2x4 resolution per cell = high detail)
- Y-axis labels with dynamic formatting ($, %, raw)
- X-axis with time labels (HH:MM)
- Legend when multiple series
- Color per series from theme

### Notes
- ratatui has no built-in braille chart — implement from scratch using `⠀` (U+2800) base + bit offsets
- Each cell is 2 columns x 4 rows of dots = 8 bits per character

---

## 2c: Add P&L History Ring Buffer

**File**: `DASHBOARD/dashboard/src/views/trading.rs`

### Changes
1. Add `pnl_history: VecDeque<(Instant, f64)>` field (max 300 points = 10min at 2s intervals)
2. On each metric update, push `(now, daily_pnl)` to history
3. Also track `portfolio_history: VecDeque<(Instant, f64)>` for portfolio value
4. Render `LineChart` in trading view below positions table
5. Add `c` key to toggle chart visibility (some users prefer more position rows)

### Layout Change
```
Before:                          After:
┌─────────┬──────────┐          ┌─────────┬──────────┐
│ Metrics │ Pipeline │          │ Metrics │ Pipeline │
├─────────┴──────────┤          ├─────────┴──────────┤
│ Positions          │          │ Positions          │
│                    │          ├────────────────────┤
│                    │          │ P&L Chart (toggle) │
├────────────────────┤          ├────────────────────┤
│ Events             │          │ Events             │
└────────────────────┘          └────────────────────┘
```

---

## 2d: Wire Sparklines into Overview

**File**: `DASHBOARD/dashboard/src/views/overview.rs`

### Changes
1. Add sparkline data buffers for key metrics
2. Show sparklines inline next to metric values:
   - Portfolio value sparkline
   - Position count sparkline
   - Training progress sparkline (if running)
3. Restructure overview layout from raw numbers to structured panels

---

## Verification

- [ ] Sparkline renders correctly with varying data
- [ ] Line chart renders with Braille dots, axes auto-scale
- [ ] P&L history accumulates and displays in trading view
- [ ] `c` key toggles chart in trading view
- [ ] Overview shows sparklines
- [ ] Charts resize properly when terminal is resized
- [ ] `cargo build --release` passes
