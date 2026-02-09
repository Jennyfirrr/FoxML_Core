//! Live trading view - real-time trading dashboard

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use tokio::sync::mpsc;

use crate::api::client::DashboardClient;
use crate::api::events::TradingEvent;
use crate::api::metrics::TradingMetrics;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;
use crate::widgets::*;

/// Trading view - live trading monitoring
pub struct TradingView {
    client: DashboardClient,
    metrics: TradingMetrics,
    event_log: event_log::EventLog,
    pipeline_status: pipeline_status::PipelineStatus,
    position_table: position_table::PositionTable,
    risk_gauge: risk_gauge::RiskGauge,
    last_update: std::time::Instant,
    theme: Theme,
    // WebSocket event receiver
    event_receiver: Option<mpsc::Receiver<TradingEvent>>,
    ws_connected: bool,
    ws_connecting: bool,
}

impl TradingView {
    pub fn new() -> Self {
        Self {
            client: DashboardClient::new("127.0.0.1:8765"),
            metrics: TradingMetrics {
                portfolio_value: 0.0,
                daily_pnl: 0.0,
                cash_balance: 0.0,
                positions_count: 0,
                sharpe_ratio: None,
            },
            event_log: event_log::EventLog::new(100),
            pipeline_status: pipeline_status::PipelineStatus::new(),
            position_table: position_table::PositionTable::new(),
            risk_gauge: risk_gauge::RiskGauge::new(),
            last_update: std::time::Instant::now(),
            theme: Theme::load(),
            event_receiver: None,
            ws_connected: false,
            ws_connecting: false,
        }
    }

    /// Connect to events WebSocket
    pub async fn connect_events(&mut self) -> Result<()> {
        if self.ws_connected || self.ws_connecting {
            return Ok(());
        }

        self.ws_connecting = true;
        match self.client.connect_events_ws().await {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.ws_connected = true;
                self.ws_connecting = false;
                self.event_log.add_event("Connected to event stream".to_string());
                Ok(())
            }
            Err(e) => {
                self.ws_connected = false;
                self.ws_connecting = false;
                self.event_log.add_event_with_severity(
                    "warning",
                    format!("WebSocket connection failed: {}", e),
                );
                Err(e)
            }
        }
    }

    /// Poll for new events from WebSocket
    pub fn poll_events(&mut self) {
        if let Some(ref mut rx) = self.event_receiver {
            // Drain all available events
            while let Ok(event) = rx.try_recv() {
                // Handle stage change events specially
                if event.event_type == "STAGE_CHANGE" {
                    if let Some(stage) = event.data.get("stage").and_then(|v| v.as_str()) {
                        self.pipeline_status.set_stage_from_str(stage);
                    }
                }
                self.event_log.push_event(event);
            }
        }
    }

    /// Update metrics from API
    pub async fn update_metrics(&mut self) -> Result<()> {
        match self.client.get_metrics().await {
            Ok(json) => {
                self.metrics.portfolio_value = json["portfolio_value"].as_f64().unwrap_or(0.0);
                self.metrics.daily_pnl = json["daily_pnl"].as_f64().unwrap_or(0.0);
                self.metrics.cash_balance = json["cash_balance"].as_f64().unwrap_or(0.0);
                self.metrics.positions_count = json["positions_count"].as_u64().unwrap_or(0) as usize;
                self.metrics.sharpe_ratio = json["sharpe_ratio"].as_f64();
                self.last_update = std::time::Instant::now();
            }
            Err(e) => {
                self.event_log.add_event_with_severity(
                    "warning",
                    format!("Error fetching metrics: {}", e),
                );
            }
        }
        Ok(())
    }

    /// Update state from API
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

    /// Update positions from API
    pub async fn update_positions(&mut self) -> Result<()> {
        match self.client.get_positions().await {
            Ok(positions) => {
                self.position_table.update(positions);
            }
            Err(e) => {
                self.event_log.add_event_with_severity(
                    "warning",
                    format!("Error fetching positions: {}", e),
                );
            }
        }
        Ok(())
    }

    /// Update risk status from API
    pub async fn update_risk(&mut self) -> Result<()> {
        match self.client.get_risk_status().await {
            Ok(status) => {
                self.risk_gauge.update(status);
            }
            Err(e) => {
                self.event_log.add_event_with_severity(
                    "warning",
                    format!("Error fetching risk: {}", e),
                );
            }
        }
        Ok(())
    }

    /// Full update (all data sources)
    pub async fn refresh_all(&mut self) {
        let _ = self.update_metrics().await;
        let _ = self.update_state().await;
        let _ = self.update_positions().await;
        let _ = self.update_risk().await;
    }
}

impl TradingView {
    /// Render the view header
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let ws_indicator = if self.ws_connected {
            Span::styled(
                format!("{} Live", Separators::CIRCLE_FILLED),
                Style::default().fg(self.theme.success),
            )
        } else if self.ws_connecting {
            Span::styled(
                format!("{} Connecting...", Separators::CIRCLE_EMPTY),
                Style::default().fg(self.theme.warning),
            )
        } else {
            Span::styled(
                format!("{} Disconnected", Separators::CIRCLE_EMPTY),
                Style::default().fg(self.theme.error),
            )
        };

        let title = Line::from(vec![
            Span::styled(
                format!("{} ", Separators::DIAMOND),
                Style::default().fg(self.theme.accent),
            ),
            Span::styled(
                "Trading Monitor",
                Style::default().fg(self.theme.text_primary).bold(),
            ),
            Span::styled(
                "  │  ",
                Style::default().fg(self.theme.border),
            ),
            ws_indicator,
        ]);
        frame.render_widget(Paragraph::new(title), area);
    }

    /// Format a number with thousand separators
    fn format_currency(value: f64) -> String {
        let abs_value = value.abs();
        let sign = if value < 0.0 { "-" } else { "" };

        let integer_part = abs_value as u64;
        let decimal_part = ((abs_value - integer_part as f64) * 100.0).round() as u64;

        // Format integer part with commas
        let int_str = integer_part.to_string();
        let mut result = String::new();
        for (i, c) in int_str.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.insert(0, ',');
            }
            result.insert(0, c);
        }

        format!("{}${}.{:02}", sign, result, decimal_part)
    }

    /// Render metrics with theme styling
    fn render_metrics(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);

        // Portfolio Value
        let pv_color = self.theme.text_primary;
        let portfolio_line = Line::from(vec![
            Span::styled("Portfolio Value  ", Style::default().fg(self.theme.text_muted)),
            Span::styled(
                Self::format_currency(self.metrics.portfolio_value),
                Style::default().fg(pv_color).bold(),
            ),
        ]);
        frame.render_widget(Paragraph::new(portfolio_line), rows[0]);

        // Daily P&L
        let pnl_color = if self.metrics.daily_pnl >= 0.0 {
            self.theme.success
        } else {
            self.theme.error
        };
        let pnl_sign = if self.metrics.daily_pnl >= 0.0 { "+" } else { "" };
        let pnl_line = Line::from(vec![
            Span::styled("Daily P&L        ", Style::default().fg(self.theme.text_muted)),
            Span::styled(
                format!("{}{}", pnl_sign, Self::format_currency(self.metrics.daily_pnl)),
                Style::default().fg(pnl_color).bold(),
            ),
        ]);
        frame.render_widget(Paragraph::new(pnl_line), rows[1]);

        // Cash Balance
        let cash_line = Line::from(vec![
            Span::styled("Cash Balance     ", Style::default().fg(self.theme.text_muted)),
            Span::styled(
                Self::format_currency(self.metrics.cash_balance),
                Style::default().fg(self.theme.text_secondary),
            ),
        ]);
        frame.render_widget(Paragraph::new(cash_line), rows[2]);

        // Positions
        let positions_line = Line::from(vec![
            Span::styled("Positions        ", Style::default().fg(self.theme.text_muted)),
            Span::styled(
                format!("{}", self.position_table.len()),
                Style::default().fg(self.theme.accent),
            ),
        ]);
        frame.render_widget(Paragraph::new(positions_line), rows[3]);
    }

    /// Render the footer with keybindings
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = vec![
            ("r", "Refresh"),
            ("c", "Connect WS"),
            ("b/Esc", "Back"),
        ];

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            }
            spans.push(Span::styled(
                format!("[{}]", key),
                Style::default().fg(self.theme.accent),
            ));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_secondary),
            ));
        }

        // Add last update time
        spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
        spans.push(Span::styled(
            format!("Updated: {:.1}s ago", self.last_update.elapsed().as_secs_f64()),
            Style::default().fg(self.theme.text_muted),
        ));

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for TradingView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for TradingView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Poll for events (non-blocking)
        self.poll_events();

        // Clear background with theme color
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        // Split into sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // Header
                Constraint::Length(7),  // Metrics + Pipeline + Risk
                Constraint::Min(8),     // Positions + Events
                Constraint::Length(2),  // Footer
            ])
            .margin(1)
            .split(area);

        // Header with title
        self.render_header(frame, chunks[0]);

        // Top section: Metrics, Pipeline, Risk
        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(35), // Metrics
                Constraint::Percentage(35), // Pipeline
                Constraint::Percentage(30), // Risk
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
        self.pipeline_status.render_themed(pipeline_inner, frame.buffer_mut(), &self.theme);

        // Risk gauge
        let risk_block = Panel::new(&self.theme).title("Risk").block();
        let risk_inner = risk_block.inner(top_chunks[2]);
        frame.render_widget(risk_block, top_chunks[2]);
        self.risk_gauge.render_themed(risk_inner, frame.buffer_mut(), &self.theme);

        // Middle section: Positions and Events
        let mid_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(55), // Positions
                Constraint::Percentage(45), // Events
            ])
            .split(chunks[2]);

        // Position table
        let positions_block = Panel::new(&self.theme).title("Positions").block();
        let positions_inner = positions_block.inner(mid_chunks[0]);
        frame.render_widget(positions_block, mid_chunks[0]);
        self.position_table.render_themed(positions_inner, frame.buffer_mut(), &self.theme);

        // Event log
        let events_block = Panel::new(&self.theme).title("Events").block();
        let events_inner = events_block.inner(mid_chunks[1]);
        frame.render_widget(events_block, mid_chunks[1]);
        self.event_log.render_themed(events_inner, frame.buffer_mut(), &self.theme);

        // Footer
        self.render_footer(frame, chunks[3]);

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<bool> {
        match key {
            crossterm::event::KeyCode::Char('r') => {
                // Refresh all data (use existing runtime handle to avoid nested runtime panic)
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.refresh_all().await;
                    });
                });
                Ok(false)
            }
            crossterm::event::KeyCode::Char('c') => {
                // Connect to WebSocket
                if !self.ws_connected {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let _ = self.connect_events().await;
                        });
                    });
                }
                Ok(false)
            }
            crossterm::event::KeyCode::Up => {
                self.position_table.up();
                Ok(false)
            }
            crossterm::event::KeyCode::Down => {
                self.position_table.down();
                Ok(false)
            }
            _ => Ok(false),
        }
    }
}
