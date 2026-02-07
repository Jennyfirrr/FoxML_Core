# Dashboard Phase 3: Rust TUI Wiring

**Status**: COMPLETE
**Parent**: [dashboard-integration-master.md](./dashboard-integration-master.md)
**Estimated Effort**: 4 hours
**Dependencies**: Phase 2 (Bridge API) complete
**Completed**: 2026-01-21

---

## Objective

Connect the Rust TUI to the new bridge endpoints and wire up real-time data display.

---

## Current State

| Widget | Data Source | Status |
|--------|-------------|--------|
| `event_log.rs` | Manual `add_event()` | ❌ Not connected to WebSocket |
| `pipeline_status.rs` | `/api/state` | ❌ Always shows "idle" |
| `position_table.rs` | None | ❌ Stubbed (35 lines) |
| `metrics_panel.rs` | `/api/metrics` | ⚠️ Partial (4 metrics only) |
| `chart.rs` | None | ❌ Stubbed |

---

## Implementation Tasks

### Task 1: Connect Event Log to WebSocket (1h)

**Goal**: Subscribe to `/ws/events` and display real events.

**Changes to `src/api/client.rs`**:

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};

impl DashboardClient {
    /// Connect to the events WebSocket and return a receiver
    pub async fn connect_events_ws(&self) -> Result<tokio::sync::mpsc::Receiver<Event>> {
        let url = format!("ws://{}/ws/events", self.host);
        let (ws_stream, _) = connect_async(&url).await?;
        let (_, mut read) = ws_stream.split();

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Spawn task to read from WebSocket
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                if let Ok(Message::Text(text)) = msg {
                    if let Ok(event) = serde_json::from_str::<Event>(&text) {
                        if tx.send(event).await.is_err() {
                            break;  // Receiver dropped
                        }
                    }
                }
            }
        });

        Ok(rx)
    }
}
```

**Changes to `src/api/events.rs`**:

```rust
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Event {
    pub event_type: String,
    pub timestamp: String,
    pub severity: String,
    pub message: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

impl Event {
    pub fn display_message(&self) -> String {
        if self.message.is_empty() {
            format!("{}: {}", self.event_type, self.data)
        } else {
            self.message.clone()
        }
    }
}
```

**Changes to `src/widgets/event_log.rs`**:

```rust
use crate::api::events::Event;

pub struct EventLog {
    events: VecDeque<Event>,
    max_events: usize,
}

impl EventLog {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
        }
    }

    /// Add event from WebSocket
    pub fn push_event(&mut self, event: Event) {
        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    /// Render events with severity coloring
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        for (i, event) in self.events.iter().rev().take(area.height as usize).enumerate() {
            let y = area.y + i as u16;
            if y >= area.bottom() {
                break;
            }

            let severity_color = match event.severity.as_str() {
                "error" | "critical" => theme.error,
                "warning" => theme.warning,
                "info" => theme.info,
                _ => theme.text_muted,
            };

            let timestamp = &event.timestamp[11..19];  // Extract HH:MM:SS
            let line = format!("[{}] {}", timestamp, event.display_message());

            buf.set_string(area.x, y, &line, Style::default().fg(severity_color));
        }
    }
}
```

**Changes to `src/views/trading.rs`**:

```rust
use tokio::sync::mpsc::Receiver;

pub struct TradingView {
    // ... existing fields
    event_receiver: Option<Receiver<Event>>,
    ws_connected: bool,
}

impl TradingView {
    pub async fn connect_events(&mut self) -> Result<()> {
        match self.client.connect_events_ws().await {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.ws_connected = true;
                Ok(())
            }
            Err(e) => {
                self.ws_connected = false;
                Err(e)
            }
        }
    }

    /// Poll for new events (call this in update loop)
    pub fn poll_events(&mut self) {
        if let Some(ref mut rx) = self.event_receiver {
            while let Ok(event) = rx.try_recv() {
                self.event_log.push_event(event);
            }
        }
    }
}
```

**Changes to `src/app.rs`**:

```rust
// In the main loop, poll events for trading view
View::Trading => {
    self.trading_view.poll_events();
    let _ = self.trading_view.update_metrics().await;
    let _ = self.trading_view.update_state().await;
}

// Connect WebSocket when entering trading view
fn switch_view(&mut self, view: View) {
    // ... existing code
    if matches!(view, View::Trading) && !self.trading_view.ws_connected {
        tokio::spawn(async move {
            let _ = self.trading_view.connect_events().await;
        });
    }
}
```

---

### Task 2: Update Pipeline Status from State (30m)

**Goal**: Display actual pipeline stage from `/api/state`.

**Changes to `src/widgets/pipeline_status.rs`**:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    Idle,
    Prediction,
    Blending,
    Arbitration,
    Gating,
    Sizing,
    Risk,
    Execution,
}

impl PipelineStage {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "prediction" => Self::Prediction,
            "blending" => Self::Blending,
            "arbitration" => Self::Arbitration,
            "gating" => Self::Gating,
            "sizing" => Self::Sizing,
            "risk" => Self::Risk,
            "execution" => Self::Execution,
            _ => Self::Idle,
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Idle => 0,
            Self::Prediction => 1,
            Self::Blending => 2,
            Self::Arbitration => 3,
            Self::Gating => 4,
            Self::Sizing => 5,
            Self::Risk => 6,
            Self::Execution => 7,
        }
    }
}

pub struct PipelineStatus {
    current_stage: PipelineStage,
}

impl PipelineStatus {
    pub fn set_stage(&mut self, stage: PipelineStage) {
        self.current_stage = stage;
    }

    pub fn set_stage_from_str(&mut self, stage: &str) {
        self.current_stage = PipelineStage::from_str(stage);
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let stages = [
            "Prediction", "Blending", "Arbitration", "Gating", "Sizing", "Risk"
        ];

        let width_per_stage = area.width / stages.len() as u16;

        for (i, stage_name) in stages.iter().enumerate() {
            let x = area.x + (i as u16 * width_per_stage);
            let is_active = self.current_stage.index() == i + 1;
            let is_past = self.current_stage.index() > i + 1;

            let (fg, bg) = if is_active {
                (theme.background, theme.accent)
            } else if is_past {
                (theme.text_primary, theme.surface)
            } else {
                (theme.text_muted, theme.background)
            };

            let style = Style::default().fg(fg).bg(bg);
            buf.set_string(x, area.y, stage_name, style);
        }
    }
}
```

**Changes to `src/views/trading.rs`**:

```rust
pub async fn update_state(&mut self) -> Result<()> {
    match self.client.get_state().await {
        Ok(json) => {
            if let Some(stage) = json["current_stage"].as_str() {
                self.pipeline_status.set_stage_from_str(stage);
            }
        }
        Err(_) => {}
    }
    Ok(())
}
```

---

### Task 3: Implement Position Table (1h)

**Goal**: Display positions from `/api/positions`.

**Changes to `src/api/client.rs`**:

```rust
pub async fn get_positions(&self) -> Result<serde_json::Value> {
    let url = format!("http://{}/api/positions", self.host);
    let response = reqwest::get(&url).await?.json().await?;
    Ok(response)
}
```

**Changes to `src/widgets/position_table.rs`**:

```rust
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Row, Table},
};
use crate::themes::Theme;

#[derive(Clone, Debug)]
pub struct Position {
    pub symbol: String,
    pub shares: i64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_pct: f64,
    pub weight: f64,
}

pub struct PositionTable {
    positions: Vec<Position>,
    selected: usize,
}

impl PositionTable {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            selected: 0,
        }
    }

    pub fn update(&mut self, positions: Vec<Position>) {
        self.positions = positions;
        if self.selected >= self.positions.len() && !self.positions.is_empty() {
            self.selected = self.positions.len() - 1;
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let header_cells = ["Symbol", "Shares", "Entry", "Current", "P&L", "P&L %", "Weight"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(theme.text_primary).bold()));
        let header = Row::new(header_cells).height(1);

        let rows = self.positions.iter().enumerate().map(|(i, pos)| {
            let pnl_color = if pos.unrealized_pnl >= 0.0 {
                theme.success
            } else {
                theme.error
            };

            let selected_style = if i == self.selected {
                Style::default().bg(theme.surface)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(pos.symbol.clone()),
                Cell::from(format!("{}", pos.shares)),
                Cell::from(format!("${:.2}", pos.entry_price)),
                Cell::from(format!("${:.2}", pos.current_price)),
                Cell::from(format!("${:.2}", pos.unrealized_pnl))
                    .style(Style::default().fg(pnl_color)),
                Cell::from(format!("{:.1}%", pos.unrealized_pnl_pct))
                    .style(Style::default().fg(pnl_color)),
                Cell::from(format!("{:.1}%", pos.weight * 100.0)),
            ])
            .style(selected_style)
        });

        let widths = [
            Constraint::Length(8),   // Symbol
            Constraint::Length(8),   // Shares
            Constraint::Length(10),  // Entry
            Constraint::Length(10),  // Current
            Constraint::Length(12),  // P&L
            Constraint::Length(8),   // P&L %
            Constraint::Length(8),   // Weight
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::NONE));

        ratatui::widgets::Widget::render(table, area, buf);
    }

    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn down(&mut self) {
        if self.selected < self.positions.len().saturating_sub(1) {
            self.selected += 1;
        }
    }
}
```

**Changes to `src/views/trading.rs`**:

```rust
pub struct TradingView {
    // ... existing
    position_table: PositionTable,
}

pub async fn update_positions(&mut self) -> Result<()> {
    match self.client.get_positions().await {
        Ok(json) => {
            let positions: Vec<Position> = json["positions"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|p| {
                    Some(Position {
                        symbol: p["symbol"].as_str()?.to_string(),
                        shares: p["shares"].as_i64()?,
                        entry_price: p["entry_price"].as_f64()?,
                        current_price: p["current_price"].as_f64()?,
                        unrealized_pnl: p["unrealized_pnl"].as_f64()?,
                        unrealized_pnl_pct: p["unrealized_pnl_pct"].as_f64()?,
                        weight: p["weight"].as_f64()?,
                    })
                })
                .collect();
            self.position_table.update(positions);
        }
        Err(e) => {
            self.event_log.add_event(format!("Error fetching positions: {}", e));
        }
    }
    Ok(())
}
```

---

### Task 4: Add Risk Metrics Display (1h)

**Goal**: Show drawdown, exposure, and risk warnings.

**Create `src/widgets/risk_gauge.rs`**:

```rust
use ratatui::prelude::*;
use crate::themes::Theme;

pub struct RiskGauge {
    daily_pnl_pct: f64,
    daily_limit_pct: f64,
    drawdown_pct: f64,
    drawdown_limit_pct: f64,
    gross_exposure: f64,
    max_exposure: f64,
    kill_switch_active: bool,
    warnings: Vec<String>,
}

impl RiskGauge {
    pub fn new() -> Self {
        Self {
            daily_pnl_pct: 0.0,
            daily_limit_pct: 2.0,
            drawdown_pct: 0.0,
            drawdown_limit_pct: 5.0,
            gross_exposure: 0.0,
            max_exposure: 2.0,
            kill_switch_active: false,
            warnings: Vec::new(),
        }
    }

    pub fn update(&mut self, data: &serde_json::Value) {
        self.daily_pnl_pct = data["daily_pnl_pct"].as_f64().unwrap_or(0.0);
        self.daily_limit_pct = data["daily_loss_limit_pct"].as_f64().unwrap_or(2.0);
        self.drawdown_pct = data["drawdown_pct"].as_f64().unwrap_or(0.0);
        self.drawdown_limit_pct = data["max_drawdown_limit_pct"].as_f64().unwrap_or(5.0);
        self.gross_exposure = data["gross_exposure"].as_f64().unwrap_or(0.0);
        self.max_exposure = data["max_gross_exposure"].as_f64().unwrap_or(2.0);
        self.kill_switch_active = data["kill_switch_active"].as_bool().unwrap_or(false);

        self.warnings = data["warnings"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|w| w["message"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // Daily P&L
                Constraint::Length(1),  // Drawdown
                Constraint::Length(1),  // Exposure
                Constraint::Length(1),  // Kill switch
                Constraint::Min(0),     // Warnings
            ])
            .split(area);

        // Daily P&L gauge
        self.render_gauge(
            buf, rows[0],
            "Daily P&L",
            self.daily_pnl_pct.abs(),
            self.daily_limit_pct,
            self.daily_pnl_pct < 0.0,
            theme,
        );

        // Drawdown gauge
        self.render_gauge(
            buf, rows[1],
            "Drawdown",
            self.drawdown_pct,
            self.drawdown_limit_pct,
            true,
            theme,
        );

        // Exposure
        self.render_gauge(
            buf, rows[2],
            "Exposure",
            self.gross_exposure,
            self.max_exposure,
            false,
            theme,
        );

        // Kill switch status
        let ks_color = if self.kill_switch_active {
            theme.error
        } else {
            theme.success
        };
        let ks_text = if self.kill_switch_active {
            "KILL SWITCH: ACTIVE"
        } else {
            "Kill Switch: Off"
        };
        buf.set_string(rows[3].x, rows[3].y, ks_text, Style::default().fg(ks_color));

        // Warnings
        for (i, warning) in self.warnings.iter().enumerate() {
            let y = rows[4].y + i as u16;
            if y < area.bottom() {
                buf.set_string(
                    rows[4].x,
                    y,
                    format!("⚠ {}", warning),
                    Style::default().fg(theme.warning),
                );
            }
        }
    }

    fn render_gauge(
        &self,
        buf: &mut Buffer,
        area: Rect,
        label: &str,
        value: f64,
        max: f64,
        is_negative: bool,
        theme: &Theme,
    ) {
        let pct = (value / max).min(1.0);
        let color = if pct > 0.8 {
            theme.error
        } else if pct > 0.5 {
            theme.warning
        } else {
            theme.success
        };

        let sign = if is_negative { "-" } else { "" };
        let text = format!("{}: {}{:.1}% / {:.1}%", label, sign, value, max);
        buf.set_string(area.x, area.y, &text, Style::default().fg(color));
    }
}
```

---

### Task 5: Update Trading View Layout (30m)

**Goal**: Integrate new widgets into the trading view layout.

**Changes to `src/views/trading.rs` render method**:

```rust
fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
    // Clear background
    let bg = Block::default().style(Style::default().bg(self.theme.background));
    frame.render_widget(bg, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Header
            Constraint::Length(7),   // Metrics + Pipeline + Risk
            Constraint::Min(8),      // Positions + Events (split horizontal)
            Constraint::Length(2),   // Footer
        ])
        .margin(1)
        .split(area);

    // Header
    self.render_header(frame, chunks[0]);

    // Top section: Metrics, Pipeline, Risk
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35),  // Metrics
            Constraint::Percentage(35),  // Pipeline
            Constraint::Percentage(30),  // Risk
        ])
        .split(chunks[1]);

    // Metrics panel
    let metrics_block = Panel::new(&self.theme).title("Portfolio").block();
    let metrics_inner = metrics_block.inner(top_chunks[0]);
    frame.render_widget(metrics_block, top_chunks[0]);
    self.render_metrics(frame, metrics_inner);

    // Pipeline status
    let pipeline_block = Panel::new(&self.theme).title("Pipeline").block();
    let pipeline_inner = pipeline_block.inner(top_chunks[1]);
    frame.render_widget(pipeline_block, top_chunks[1]);
    self.pipeline_status.render(pipeline_inner, frame.buffer_mut(), &self.theme);

    // Risk gauge
    let risk_block = Panel::new(&self.theme).title("Risk").block();
    let risk_inner = risk_block.inner(top_chunks[2]);
    frame.render_widget(risk_block, top_chunks[2]);
    self.risk_gauge.render(risk_inner, frame.buffer_mut(), &self.theme);

    // Middle section: Positions and Events
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55),  // Positions
            Constraint::Percentage(45),  // Events
        ])
        .split(chunks[2]);

    // Position table
    let positions_block = Panel::new(&self.theme).title("Positions").block();
    let positions_inner = positions_block.inner(mid_chunks[0]);
    frame.render_widget(positions_block, mid_chunks[0]);
    self.position_table.render(positions_inner, frame.buffer_mut(), &self.theme);

    // Event log
    let events_block = Panel::new(&self.theme).title("Events").block();
    let events_inner = events_block.inner(mid_chunks[1]);
    frame.render_widget(events_block, mid_chunks[1]);
    self.event_log.render(events_inner, frame.buffer_mut(), &self.theme);

    // Footer
    self.render_footer(frame, chunks[3]);

    Ok(())
}
```

---

## Implementation Checklist

### Task 1: Event Log WebSocket (1h) ✅
- [x] Add WebSocket connection to client.rs (`connect_events_ws()`)
- [x] Add Event struct to events.rs (`TradingEvent` with display helpers)
- [x] Update event_log.rs to receive events (`push_event()`, `render_themed()`)
- [x] Connect trading view to WebSocket (`connect_events()`, `poll_events()`)
- [x] Poll events in main loop (called in render)

### Task 2: Pipeline Status (30m) ✅
- [x] Add PipelineStage::from_str() (added `Execution` stage too)
- [x] Update pipeline_status.rs rendering (`render_themed()`)
- [x] Wire to /api/state current_stage (`set_stage_from_str()`)

### Task 3: Position Table (1h) ✅
- [x] Add get_positions() to client (returns `Vec<Position>`)
- [x] Implement position_table.rs (full table with P&L coloring)
- [x] Add to trading view
- [x] Wire update_positions()

### Task 4: Risk Gauge (1h) ✅
- [x] Create risk_gauge.rs widget (gauges + warnings display)
- [x] Add get_risk_status() to client (returns `RiskStatus`)
- [x] Add to trading view layout

### Task 5: Layout Update (30m) ✅
- [x] Update trading view render() (3-column top, 2-column bottom)
- [x] Add new widgets to struct (position_table, risk_gauge, event_receiver)
- [x] Add update methods (refresh_all, update_positions, update_risk)
- [x] Test build and run (compiles successfully)

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/api/client.rs` | Add WebSocket, positions, risk endpoints |
| `src/api/events.rs` | Add Event struct |
| `src/widgets/event_log.rs` | Rewrite to receive events |
| `src/widgets/pipeline_status.rs` | Add from_str, fix rendering |
| `src/widgets/position_table.rs` | Full implementation |
| `src/widgets/risk_gauge.rs` | Create new widget |
| `src/widgets/mod.rs` | Export new widgets |
| `src/views/trading.rs` | Wire everything together |
| `src/app.rs` | Poll events in main loop |

---

## Testing

```bash
# Build
cd DASHBOARD/dashboard && cargo build --release

# Run with bridge
bin/foxml

# Verify:
# 1. Events appear in event log
# 2. Pipeline status updates during trading
# 3. Positions table shows real positions
# 4. Risk gauge displays metrics
```
