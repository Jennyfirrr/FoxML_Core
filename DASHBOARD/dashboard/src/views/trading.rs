//! Live trading view - real-time trading dashboard

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::collections::VecDeque;
use tokio::sync::mpsc;

use crate::api::client::DashboardClient;
use crate::api::events::TradingEvent;
use crate::api::metrics::TradingMetrics;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;
use crate::widgets::*;
use crate::widgets::chart::{LineChart, Series};

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
    ws_connect_started: Option<std::time::Instant>,
    // P&L history for chart (max 300 points = ~10 min at 2s intervals)
    pnl_history: VecDeque<f64>,
    portfolio_history: VecDeque<f64>,
    show_chart: bool,
    // Engine control state
    engine_paused: bool,
    kill_switch_active: bool,
    // Position detail panel
    show_detail: bool,
}

impl TradingView {
    pub fn new() -> Self {
        Self {
            client: DashboardClient::new(&crate::config::bridge_url()),
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
            ws_connect_started: None,
            pnl_history: VecDeque::with_capacity(300),
            portfolio_history: VecDeque::with_capacity(300),
            show_chart: true,
            engine_paused: false,
            kill_switch_active: false,
            show_detail: false,
        }
    }

    /// Whether the position detail panel is open
    pub fn has_detail_panel(&self) -> bool {
        self.show_detail
    }

    /// Close the position detail panel
    pub fn close_detail_panel(&mut self) {
        self.show_detail = false;
    }

    /// Navigate position selection up
    pub fn position_up(&mut self) {
        self.position_table.up();
    }

    /// Navigate position selection down
    pub fn position_down(&mut self) {
        self.position_table.down();
    }

    /// Connect to events WebSocket
    pub async fn connect_events(&mut self) -> Result<()> {
        if self.ws_connected || self.ws_connecting {
            return Ok(());
        }

        self.ws_connecting = true;
        self.ws_connect_started = Some(std::time::Instant::now());
        match self.client.connect_events_ws().await {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.ws_connected = true;
                self.ws_connecting = false;
                self.ws_connect_started = None;
                self.event_log.add_event("Connected to event stream".to_string());
                Ok(())
            }
            Err(e) => {
                self.ws_connected = false;
                self.ws_connecting = false;
                self.ws_connect_started = None;
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
        // Check for WebSocket connection timeout (5 seconds)
        if self.ws_connecting {
            if let Some(started) = self.ws_connect_started {
                if started.elapsed() > std::time::Duration::from_secs(5) {
                    self.ws_connecting = false;
                    self.ws_connect_started = None;
                    self.event_log.add_event_with_severity(
                        "warning",
                        "WebSocket connection timed out".to_string(),
                    );
                }
            }
        }

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

                // Record history for chart
                if self.pnl_history.len() >= 300 {
                    self.pnl_history.pop_front();
                }
                self.pnl_history.push_back(self.metrics.daily_pnl);
                if self.portfolio_history.len() >= 300 {
                    self.portfolio_history.pop_front();
                }
                self.portfolio_history.push_back(self.metrics.portfolio_value);
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
            Err(e) => {
                self.event_log.add_event_with_severity(
                    "warning",
                    format!("Error fetching state: {}", e),
                );
            }
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

    /// Update control status (kill switch, pause state)
    pub async fn update_control_status(&mut self) {
        if let Ok(status) = self.client.get_control_status().await {
            self.kill_switch_active = status["kill_switch_active"].as_bool().unwrap_or(false);
            self.engine_paused = status["paused"].as_bool().unwrap_or(false);
        }
    }

    /// Full update (all data sources)
    pub async fn refresh_all(&mut self) {
        let _ = self.update_metrics().await;
        let _ = self.update_state().await;
        let _ = self.update_positions().await;
        let _ = self.update_risk().await;
        self.update_control_status().await;
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

        // Show stale data indicator when data is old
        let stale_secs = self.last_update.elapsed().as_secs();
        let stale_indicator = if stale_secs > 10 {
            Span::styled(
                format!("  │  Updated: {}s ago", stale_secs),
                Style::default().fg(self.theme.warning),
            )
        } else {
            Span::styled("", Style::default())
        };

        // Control state badges
        let paused_badge = if self.engine_paused {
            Span::styled(
                "  PAUSED ",
                Style::default().fg(self.theme.background).bg(self.theme.warning).bold(),
            )
        } else {
            Span::styled("", Style::default())
        };

        let kill_badge = if self.kill_switch_active {
            Span::styled(
                "  KILL SWITCH ",
                Style::default().fg(self.theme.background).bg(self.theme.error).bold(),
            )
        } else {
            Span::styled("", Style::default())
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
            kill_badge,
            paused_badge,
            Span::styled(
                "  │  ",
                Style::default().fg(self.theme.border),
            ),
            ws_indicator,
            stale_indicator,
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

        // Sharpe Ratio
        let (sharpe_text, sharpe_color) = match self.metrics.sharpe_ratio {
            Some(sr) => {
                let color = if sr > 1.0 {
                    self.theme.success
                } else if sr > 0.5 {
                    self.theme.warning
                } else {
                    self.theme.error
                };
                (format!("{:.2}", sr), color)
            }
            None => ("-".to_string(), self.theme.text_muted),
        };
        let sharpe_line = Line::from(vec![
            Span::styled("Sharpe Ratio     ", Style::default().fg(self.theme.text_muted)),
            Span::styled(sharpe_text, Style::default().fg(sharpe_color)),
        ]);
        frame.render_widget(Paragraph::new(sharpe_line), rows[4]);
    }

    /// Render P&L chart
    fn render_chart(&self, area: Rect, buf: &mut Buffer) {
        if self.pnl_history.len() < 2 {
            // Not enough data yet
            let msg = "Collecting data for chart...";
            let x = area.x + (area.width.saturating_sub(msg.len() as u16)) / 2;
            for (i, ch) in msg.chars().enumerate() {
                if let Some(cell) = buf.cell_mut((x + i as u16, area.y + area.height / 2)) {
                    cell.set_char(ch);
                    cell.set_style(Style::default().fg(self.theme.text_muted));
                }
            }
            return;
        }

        let mut chart = LineChart::new().with_y_label("P&L");
        let pnl_color = if self.metrics.daily_pnl >= 0.0 {
            self.theme.success
        } else {
            self.theme.error
        };
        let mut pnl_series = Series::new("P&L", pnl_color, 300);
        for &val in &self.pnl_history {
            pnl_series.push(val);
        }
        chart.add_series(pnl_series);
        chart.render(area, buf, &self.theme);
    }

    /// Render position detail panel (replaces events panel when active)
    fn render_detail_panel(&self, frame: &mut Frame, area: Rect) {
        if let Some(pos) = self.position_table.selected_position() {
            let side_str = if pos.side.is_empty() {
                if pos.shares > 0 { "Long" } else { "Short" }
            } else {
                &pos.side
            };

            let pnl_color = if pos.unrealized_pnl >= 0.0 {
                self.theme.success
            } else {
                self.theme.error
            };

            let pnl_sign = if pos.unrealized_pnl >= 0.0 { "+" } else { "" };

            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Symbol           ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(&pos.symbol, Style::default().fg(self.theme.text_primary).bold()),
                ]),
                Line::from(vec![
                    Span::styled("Side             ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(side_str, Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("Shares           ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("{}", pos.shares), Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Entry Price      ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(Self::format_currency(pos.entry_price), Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("Current Price    ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(Self::format_currency(pos.current_price), Style::default().fg(self.theme.text_primary)),
                ]),
                Line::from(vec![
                    Span::styled("Market Value     ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(Self::format_currency(pos.market_value), Style::default().fg(self.theme.text_primary)),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Unrealized P&L   ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(
                        format!("{}{}", pnl_sign, Self::format_currency(pos.unrealized_pnl)),
                        Style::default().fg(pnl_color).bold(),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("P&L %            ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(
                        format!("{}{:.2}%", pnl_sign, pos.unrealized_pnl_pct),
                        Style::default().fg(pnl_color),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Portfolio Weight  ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("{:.2}%", pos.weight * 100.0), Style::default().fg(self.theme.accent)),
                ]),
            ];

            if let Some(ref entry_time) = pos.entry_time {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("Entry Time       ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(entry_time.as_str(), Style::default().fg(self.theme.text_secondary)),
                ]));
            }

            let detail = Paragraph::new(lines);
            frame.render_widget(detail, area);
        } else {
            let no_pos = Paragraph::new("No position selected")
                .style(Style::default().fg(self.theme.text_muted))
                .alignment(Alignment::Center);
            frame.render_widget(no_pos, area);
        }
    }

    /// Render the footer with keybindings
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let chart_toggle = if self.show_chart { "Hide Chart" } else { "Show Chart" };
        let pause_label = if self.engine_paused { "Resume" } else { "Pause" };
        let mut keybinds = vec![
            ("r", "Refresh"),
            ("s", "Sort"),
            ("p", pause_label),
            ("c", chart_toggle),
            ("w", "Connect WS"),
        ];
        if self.show_detail {
            keybinds.push(("Esc", "Close Detail"));
        } else {
            keybinds.push(("Enter", "Detail"));
            keybinds.push(("b/Esc", "Back"));
        }

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

        // Split into sections (with optional chart panel)
        let chunks = if self.show_chart {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),   // Header
                    Constraint::Length(7),   // Metrics + Pipeline + Risk
                    Constraint::Min(5),      // Positions + Events
                    Constraint::Length(10),  // Chart
                    Constraint::Length(2),   // Footer
                ])
                .margin(1)
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),  // Header
                    Constraint::Length(7),  // Metrics + Pipeline + Risk
                    Constraint::Min(8),    // Positions + Events
                    Constraint::Length(0), // No chart
                    Constraint::Length(2), // Footer
                ])
                .margin(1)
                .split(area)
        };

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

        // Position table (with sort indicator in title)
        let sort_title = format!("Positions [by {}]", self.position_table.sort_label());
        let positions_block = Panel::new(&self.theme).title(&sort_title).block();
        let positions_inner = positions_block.inner(mid_chunks[0]);
        frame.render_widget(positions_block, mid_chunks[0]);
        self.position_table.render_themed(positions_inner, frame.buffer_mut(), &self.theme);

        // Right panel: detail panel or event log
        if self.show_detail {
            let detail_title = if let Some(pos) = self.position_table.selected_position() {
                format!("Position: {}", pos.symbol)
            } else {
                "Position Detail".to_string()
            };
            let detail_block = Panel::new(&self.theme).title(&detail_title).block();
            let detail_inner = detail_block.inner(mid_chunks[1]);
            frame.render_widget(detail_block, mid_chunks[1]);
            self.render_detail_panel(frame, detail_inner);
        } else {
            let events_block = Panel::new(&self.theme).title("Events").block();
            let events_inner = events_block.inner(mid_chunks[1]);
            frame.render_widget(events_block, mid_chunks[1]);
            self.event_log.render_themed(events_inner, frame.buffer_mut(), &self.theme);
        }

        // Chart panel (when visible)
        if self.show_chart && chunks[3].height > 2 {
            let chart_block = Panel::new(&self.theme).title("P&L History").block();
            let chart_inner = chart_block.inner(chunks[3]);
            frame.render_widget(chart_block, chunks[3]);
            self.render_chart(chart_inner, frame.buffer_mut());
        }

        // Footer
        self.render_footer(frame, chunks[4]);

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        match key {
            crossterm::event::KeyCode::Char('r') => {
                // Refresh all data (use existing runtime handle to avoid nested runtime panic)
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.refresh_all().await;
                    });
                });
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Char('c') => {
                // Toggle chart visibility
                self.show_chart = !self.show_chart;
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Char('w') => {
                // Connect to WebSocket
                if !self.ws_connected {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let _ = self.connect_events().await;
                        });
                    });
                }
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Char('s') => {
                // Cycle position sort mode
                self.position_table.cycle_sort();
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Char('p') => {
                // Toggle pause/resume (non-destructive, no confirmation needed)
                let paused = self.engine_paused;
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let result = if paused {
                            self.client.resume_engine().await
                        } else {
                            self.client.pause_engine().await
                        };
                        match result {
                            Ok(_) => {
                                self.engine_paused = !paused;
                                let msg = if !paused { "Trading paused" } else { "Trading resumed" };
                                self.event_log.add_event(msg.to_string());
                            }
                            Err(e) => {
                                self.event_log.add_event_with_severity(
                                    "warning",
                                    format!("Pause/resume failed: {}", e),
                                );
                            }
                        }
                    });
                });
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Enter => {
                // Toggle position detail panel
                if !self.position_table.is_empty() {
                    self.show_detail = true;
                }
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Esc => {
                // Close detail panel if open, otherwise signal back
                if self.show_detail {
                    self.show_detail = false;
                    Ok(ViewAction::Continue)
                } else {
                    Ok(ViewAction::Back)
                }
            }
            crossterm::event::KeyCode::Up => {
                self.position_table.up();
                Ok(ViewAction::Continue)
            }
            crossterm::event::KeyCode::Down => {
                self.position_table.down();
                Ok(ViewAction::Continue)
            }
            _ => Ok(ViewAction::Continue),
        }
    }
}
