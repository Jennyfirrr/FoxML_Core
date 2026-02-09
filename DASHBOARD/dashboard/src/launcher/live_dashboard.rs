//! Live dashboard - real-time training and trading status display
//!
//! Shows live status from both training (via events file) and trading (via bridge API).

use anyhow::Result;
use chrono::{DateTime, FixedOffset, Local};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::time::Instant;

use crate::api::client::DashboardClient;
use crate::api::events::TrainingEvent;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;

/// Path to training events file
fn training_events_file() -> std::path::PathBuf { crate::config::training_events_file() }

/// Maximum recent events to display
const MAX_RECENT_EVENTS: usize = 5;

/// Event source
#[derive(Clone, Debug)]
pub enum EventSource {
    Training,
    Trading,
}

/// Dashboard event for display
#[derive(Clone, Debug)]
pub struct DashboardEvent {
    pub timestamp: String,
    pub source: EventSource,
    pub event_type: String,
    pub message: String,
    pub stage: String,  // Pipeline stage: ranking, feature_selection, training
}

/// Live dashboard showing training and trading status
pub struct LiveDashboard {
    client: DashboardClient,
    last_bridge_update: Instant,
    last_training_poll: Instant,

    // Training state (from events file)
    training_run_id: Option<String>,
    training_stage: String,
    training_progress: f64,
    training_target: Option<String>,
    training_targets_complete: i64,
    training_targets_total: i64,
    training_message: Option<String>,
    event_file_pos: u64,

    // Trading state (from bridge)
    bridge_connected: bool,
    trading_status: String,
    trading_stage: String,
    trading_pnl: f64,
    trading_pnl_pct: f64,
    trading_positions: i64,
    trading_cash: f64,
    trading_cycle: i64,
    kill_switch_active: bool,

    // Combined recent events
    recent_events: VecDeque<DashboardEvent>,
}

impl LiveDashboard {
    pub fn new() -> Self {
        let mut dashboard = Self {
            client: DashboardClient::new(&crate::config::bridge_url()),
            last_bridge_update: Instant::now(),
            last_training_poll: Instant::now(),

            training_run_id: None,
            training_stage: "idle".to_string(),
            training_progress: 0.0,
            training_target: None,
            training_targets_complete: 0,
            training_targets_total: 0,
            training_message: None,
            event_file_pos: 0,

            bridge_connected: false,
            trading_status: "Unknown".to_string(),
            trading_stage: "idle".to_string(),
            trading_pnl: 0.0,
            trading_pnl_pct: 0.0,
            trading_positions: 0,
            trading_cash: 0.0,
            trading_cycle: 0,
            kill_switch_active: false,

            recent_events: VecDeque::with_capacity(MAX_RECENT_EVENTS + 1),
        };

        // Recover current training state from existing events
        dashboard.recover_training_state();

        dashboard
    }

    /// Recover current training state from existing events file
    fn recover_training_state(&mut self) {
        let file = match fs::File::open(&training_events_file()) {
            Ok(f) => f,
            Err(_) => {
                self.event_file_pos = 0;
                return;
            }
        };

        // Read all events to find the most recent run
        let reader = BufReader::new(&file);
        let mut last_run_id: Option<String> = None;
        let mut last_run_events: Vec<TrainingEvent> = Vec::new();
        let mut run_completed = false;

        for line in reader.lines() {
            if let Ok(line_str) = line {
                let trimmed = line_str.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<TrainingEvent>(trimmed) {
                    // Check if this is a new run
                    if !event.run_id.is_empty() {
                        if last_run_id.as_ref() != Some(&event.run_id) {
                            // New run started - reset events
                            last_run_id = Some(event.run_id.clone());
                            last_run_events.clear();
                            run_completed = false;
                        }
                    }

                    // Track if run completed
                    if event.event_type == "run_complete" {
                        run_completed = true;
                    }

                    last_run_events.push(event);
                }
            }
        }

        // Set file position to end for future polling
        if let Ok(metadata) = file.metadata() {
            self.event_file_pos = metadata.len();
        }

        // If the last run is NOT completed, replay its events to set current state
        if !run_completed && !last_run_events.is_empty() {
            for event in last_run_events {
                self.handle_training_event(&event);
            }
        }
    }

    /// Check if bridge is connected
    pub fn is_bridge_connected(&self) -> bool {
        self.bridge_connected
    }

    /// Poll training events from file (synchronous, fast)
    pub fn poll_training_events(&mut self) {
        // Poll every 500ms
        if self.last_training_poll.elapsed().as_millis() < 500 {
            return;
        }
        self.last_training_poll = Instant::now();

        let file = match fs::File::open(&training_events_file()) {
            Ok(f) => f,
            Err(_) => return, // File doesn't exist
        };

        let mut reader = BufReader::new(file);
        if reader.seek(SeekFrom::Start(self.event_file_pos)).is_err() {
            return;
        }

        let mut line = String::new();
        while reader.read_line(&mut line).unwrap_or(0) > 0 {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                if let Ok(event) = serde_json::from_str::<TrainingEvent>(trimmed) {
                    self.handle_training_event(&event);
                }
            }
            line.clear();
        }

        if let Ok(pos) = reader.stream_position() {
            self.event_file_pos = pos;
        }
    }

    /// Handle a training event
    fn handle_training_event(&mut self, event: &TrainingEvent) {
        // Update training state
        if !event.run_id.is_empty() {
            self.training_run_id = Some(event.run_id.clone());
        }

        match event.event_type.as_str() {
            "progress" => {
                self.training_stage = event.effective_stage().to_string();
                self.training_progress = event.progress_pct;
                self.training_target = event.current_target.clone();
                self.training_targets_complete = event.targets_complete;
                self.training_targets_total = event.targets_total;
                self.training_message = event.message.clone();
            }
            "stage_change" => {
                self.training_stage = event.effective_stage().to_string();
            }
            "target_start" => {
                self.training_target = event.target.clone();
            }
            "target_complete" => {
                self.training_targets_complete = event.targets_complete;
            }
            "run_complete" => {
                self.training_stage = "completed".to_string();
                self.training_progress = 100.0;
            }
            _ => {}
        }

        // Add to recent events - convert UTC timestamp to local time for display
        let timestamp = if !event.timestamp.is_empty() {
            // Parse ISO timestamp and convert to local time
            if let Ok(dt) = DateTime::<FixedOffset>::parse_from_rfc3339(&event.timestamp) {
                dt.with_timezone(&Local).format("%H:%M:%S").to_string()
            } else if event.timestamp.len() >= 19 {
                // Fallback: just extract HH:MM:SS (already in UTC though)
                event.timestamp[11..19].to_string()
            } else {
                event.timestamp.clone()
            }
        } else {
            Local::now().format("%H:%M:%S").to_string()
        };

        self.add_event(DashboardEvent {
            timestamp,
            source: EventSource::Training,
            event_type: event.event_type.clone(),
            message: event.display_message(),
            stage: event.effective_stage().to_string(),
        });
    }

    /// Update trading data from bridge (async)
    pub async fn update_bridge_data(&mut self) {
        // Update every 2 seconds
        if self.last_bridge_update.elapsed().as_secs() < 2 {
            return;
        }
        self.last_bridge_update = Instant::now();

        // Check bridge health
        match self.client.get_health().await {
            Ok(health) => {
                self.bridge_connected = health.status == "ok";
            }
            Err(_) => {
                self.bridge_connected = false;
                return;
            }
        }

        // Get metrics
        if let Ok(metrics) = self.client.get_metrics().await {
            if let Some(pnl) = metrics.get("daily_pnl").and_then(|v| v.as_f64()) {
                self.trading_pnl = pnl;
            }
            if let Some(pnl_pct) = metrics.get("daily_pnl_pct").and_then(|v| v.as_f64()) {
                self.trading_pnl_pct = pnl_pct;
            }
            if let Some(cash) = metrics.get("cash").and_then(|v| v.as_f64()) {
                self.trading_cash = cash;
            }
            if let Some(positions) = metrics.get("positions_count").and_then(|v| v.as_i64()) {
                self.trading_positions = positions;
            }
            if let Some(cycle) = metrics.get("cycles_total").and_then(|v| v.as_i64()) {
                self.trading_cycle = cycle;
            }
        }

        // Get state
        if let Ok(state) = self.client.get_state().await {
            if let Some(status) = state.get("status").and_then(|s| s.as_str()) {
                self.trading_status = status.to_string();
            }
            if let Some(stage) = state.get("current_stage").and_then(|s| s.as_str()) {
                if !stage.is_empty() {
                    self.trading_stage = stage.to_string();
                }
            }
        }

        // Get control status
        if let Ok(control) = self.client.get_control_status().await {
            self.kill_switch_active = control
                .get("kill_switch_active")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        }
    }

    /// Add event to recent events list
    fn add_event(&mut self, event: DashboardEvent) {
        self.recent_events.push_front(event);
        while self.recent_events.len() > MAX_RECENT_EVENTS {
            self.recent_events.pop_back();
        }
    }

    /// Render the dashboard
    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) -> Result<()> {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6), // Training section
                Constraint::Length(5), // Trading section
                Constraint::Min(0),    // Recent events
            ])
            .split(area);

        self.render_training(frame, chunks[0], theme);
        self.render_trading(frame, chunks[1], theme);
        self.render_events(frame, chunks[2], theme);

        Ok(())
    }

    /// Render training section
    fn render_training(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Panel::new(theme).title("Training").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.training_run_id.is_none() {
            let text = Paragraph::new("No training running")
                .style(Style::default().fg(theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Run ID
                Constraint::Length(1), // Stage
                Constraint::Length(1), // Progress bar
                Constraint::Length(1), // Target
            ])
            .split(inner);

        // Run ID
        let run_id = self.training_run_id.as_deref().unwrap_or("N/A");
        let run_display = if run_id.len() > 40 {
            format!("{}...", &run_id[..40])
        } else {
            run_id.to_string()
        };
        let run_line = Line::from(vec![
            Span::styled("Run: ", Style::default().fg(theme.text_muted)),
            Span::styled(run_display, Style::default().fg(theme.text_secondary)),
        ]);
        frame.render_widget(Paragraph::new(run_line), rows[0]);

        // Stage with color
        let stage_color = match self.training_stage.as_str() {
            "ranking" => theme.warning,
            "feature_selection" => theme.accent,
            "training" => theme.success,
            "completed" => theme.success,
            _ => theme.text_muted,
        };
        let stage_line = Line::from(vec![
            Span::styled("Stage: ", Style::default().fg(theme.text_muted)),
            Span::styled(&self.training_stage, Style::default().fg(stage_color).bold()),
        ]);
        frame.render_widget(Paragraph::new(stage_line), rows[1]);

        // Progress bar
        let bar_width = rows[2].width.saturating_sub(12) as usize;
        let filled = ((self.training_progress / 100.0) * bar_width as f64) as usize;
        let empty = bar_width.saturating_sub(filled);
        let bar = format!(
            "[{}{}] {:>5.1}%",
            "█".repeat(filled),
            "░".repeat(empty),
            self.training_progress
        );
        frame.render_widget(
            Paragraph::new(bar).style(Style::default().fg(theme.accent)),
            rows[2],
        );

        // Target info
        let target_info = if let Some(target) = &self.training_target {
            format!(
                "Target: {} ({}/{})",
                target, self.training_targets_complete, self.training_targets_total
            )
        } else if self.training_targets_total > 0 {
            format!(
                "Targets: {}/{}",
                self.training_targets_complete, self.training_targets_total
            )
        } else {
            String::new()
        };
        frame.render_widget(
            Paragraph::new(target_info).style(Style::default().fg(theme.text_secondary)),
            rows[3],
        );
    }

    /// Render trading section
    fn render_trading(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Panel::new(theme).title("Trading").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if !self.bridge_connected {
            let text = Paragraph::new("Bridge offline - start trading engine to see live data")
                .style(Style::default().fg(theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Status + P&L
                Constraint::Length(1), // Positions + Cash
                Constraint::Length(1), // Stage + Cycle
            ])
            .split(inner);

        // Row 1: Status + P&L
        let status_color = match self.trading_status.to_lowercase().as_str() {
            "running" | "live" | "active" => theme.success,
            "paused" => theme.warning,
            "stopped" | "error" => theme.error,
            _ => theme.text_muted,
        };
        let pnl_color = if self.trading_pnl >= 0.0 {
            theme.success
        } else {
            theme.error
        };
        let pnl_sign = if self.trading_pnl >= 0.0 { "+" } else { "" };

        let row1 = Line::from(vec![
            Span::styled("Status: ", Style::default().fg(theme.text_muted)),
            Span::styled(&self.trading_status, Style::default().fg(status_color).bold()),
            Span::styled("    P&L: ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}${:.2} ({}{:.2}%)", pnl_sign, self.trading_pnl, pnl_sign, self.trading_pnl_pct),
                Style::default().fg(pnl_color).bold(),
            ),
        ]);
        frame.render_widget(Paragraph::new(row1), rows[0]);

        // Row 2: Positions + Cash
        let row2 = Line::from(vec![
            Span::styled("Positions: ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}", self.trading_positions),
                Style::default().fg(theme.text_secondary),
            ),
            Span::styled("    Cash: ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("${:.0}", self.trading_cash),
                Style::default().fg(theme.text_secondary),
            ),
        ]);
        frame.render_widget(Paragraph::new(row2), rows[1]);

        // Row 3: Stage + Cycle + Kill switch
        let mut row3_spans = vec![
            Span::styled("Stage: ", Style::default().fg(theme.text_muted)),
            Span::styled(&self.trading_stage, Style::default().fg(theme.accent)),
            Span::styled("    Cycle: ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}", self.trading_cycle),
                Style::default().fg(theme.text_secondary),
            ),
        ];

        if self.kill_switch_active {
            row3_spans.push(Span::styled("    ", Style::default()));
            row3_spans.push(Span::styled(
                format!("{} KILL SWITCH", Separators::CIRCLE_FILLED),
                Style::default().fg(theme.error).bold(),
            ));
        }

        frame.render_widget(Paragraph::new(Line::from(row3_spans)), rows[2]);
    }

    /// Render recent events section
    fn render_events(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let block = Panel::new(theme).title("Recent Events").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.recent_events.is_empty() {
            let text = Paragraph::new("No recent events")
                .style(Style::default().fg(theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let items: Vec<ListItem> = self
            .recent_events
            .iter()
            .map(|event| {
                // Determine tag and color based on pipeline stage
                let (source_tag, source_color) = match event.source {
                    EventSource::Training => {
                        // Use stage for training pipeline events
                        match event.stage.as_str() {
                            "ranking" => ("TR", theme.warning),
                            "feature_selection" => ("FS", theme.accent),
                            "training" => ("TRN", theme.success),
                            "initializing" => ("INIT", theme.text_muted),
                            "completed" => ("DONE", theme.success),
                            _ => ("TRN", theme.accent),
                        }
                    }
                    EventSource::Trading => ("TRD", theme.success),
                };

                let line = Line::from(vec![
                    Span::styled(&event.timestamp, Style::default().fg(theme.text_muted)),
                    Span::styled(" ", Style::default()),
                    Span::styled(
                        format!("[{:^4}]", source_tag),
                        Style::default().fg(source_color),
                    ),
                    Span::styled(" ", Style::default()),
                    Span::styled(&event.message, Style::default().fg(theme.text_secondary)),
                ]);

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }
}

impl Default for LiveDashboard {
    fn default() -> Self {
        Self::new()
    }
}
