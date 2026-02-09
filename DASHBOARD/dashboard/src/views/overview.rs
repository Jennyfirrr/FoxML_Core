//! Overview view - combined system overview

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use crate::launcher::system_status::SystemStatus;
use crate::api::client::DashboardClient;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;
use crate::widgets::chart::Sparkline;

/// Overview view - combined system status
pub struct OverviewView {
    theme: Theme,
    system_status: SystemStatus,
    client: DashboardClient,
    trading_metrics: Option<serde_json::Value>,
    control_status: Option<serde_json::Value>,
    // Sparkline data
    pnl_sparkline: Sparkline,
    portfolio_sparkline: Sparkline,
    positions_sparkline: Sparkline,
    // Health status
    bridge_connected: bool,
    trading_active: bool,
}

impl OverviewView {
    pub fn new() -> Self {
        let theme = Theme::load();
        Self {
            pnl_sparkline: Sparkline::new(60, theme.success).with_label("P&L"),
            portfolio_sparkline: Sparkline::new(60, theme.accent).with_label("Value"),
            positions_sparkline: Sparkline::new(60, theme.warning).with_label("Pos"),
            theme,
            system_status: SystemStatus::new(),
            client: DashboardClient::new(&crate::config::bridge_url()),
            trading_metrics: None,
            control_status: None,
            bridge_connected: false,
            trading_active: false,
        }
    }

    /// Update metrics
    pub async fn update(&mut self) {
        self.system_status.refresh();

        // Check bridge health
        self.bridge_connected = self.client.get_health().await.is_ok();

        // Fetch trading metrics
        if let Ok(metrics) = self.client.get_metrics().await {
            let pnl = metrics["daily_pnl"].as_f64().unwrap_or(0.0);
            let value = metrics["portfolio_value"].as_f64().unwrap_or(0.0);
            let positions = metrics["positions_count"].as_u64().unwrap_or(0) as f64;
            self.pnl_sparkline.push(pnl);
            self.portfolio_sparkline.push(value);
            self.positions_sparkline.push(positions);
            self.trading_active = true;
            self.trading_metrics = Some(metrics);
        } else {
            self.trading_active = false;
        }

        // Fetch control status
        if let Ok(status) = self.client.get_control_status().await {
            self.control_status = Some(status);
        }
    }

    /// Render health indicator row
    fn render_health(&self, area: Rect, buf: &mut Buffer) {
        let indicators = [
            ("Bridge", self.bridge_connected),
            ("Trading", self.trading_active),
        ];

        let mut x = area.x;
        for (name, healthy) in &indicators {
            let (icon, color) = if *healthy {
                (Separators::CIRCLE_FILLED, self.theme.success)
            } else {
                (Separators::CIRCLE_EMPTY, self.theme.error)
            };

            let text = format!("{} {}  ", icon, name);
            for ch in text.chars() {
                if x >= area.right() { return; }
                if let Some(cell) = buf.cell_mut((x, area.y)) {
                    cell.set_char(ch);
                    cell.set_style(Style::default().fg(color));
                }
                x += 1;
            }
        }

        // Kill switch status
        if let Some(ref status) = self.control_status {
            let ks_active = status["kill_switch_active"].as_bool().unwrap_or(false);
            let paused = status["paused"].as_bool().unwrap_or(false);

            let (text, style) = if ks_active {
                ("KILL SWITCH ", Style::default().fg(self.theme.error).bold())
            } else if paused {
                ("PAUSED ", Style::default().fg(self.theme.warning).bold())
            } else {
                return;
            };

            for ch in text.chars() {
                if x >= area.right() { return; }
                if let Some(cell) = buf.cell_mut((x, area.y)) {
                    cell.set_char(ch);
                    cell.set_style(style);
                }
                x += 1;
            }
        }
    }

    /// Render trading summary panel
    fn render_trading_summary(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Trading Summary").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if let Some(ref metrics) = self.trading_metrics {
            let pv = metrics["portfolio_value"].as_f64().unwrap_or(0.0);
            let pnl = metrics["daily_pnl"].as_f64().unwrap_or(0.0);
            let cash = metrics["cash_balance"].as_f64().unwrap_or(0.0);
            let pos = metrics["positions_count"].as_u64().unwrap_or(0);
            let sharpe = metrics["sharpe_ratio"].as_f64();

            let pnl_color = if pnl >= 0.0 { self.theme.success } else { self.theme.error };

            let mut lines = vec![
                Line::from(vec![
                    Span::styled("Portfolio Value  ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("${:.2}", pv), Style::default().fg(self.theme.text_primary).bold()),
                ]),
                Line::from(vec![
                    Span::styled("Daily P&L        ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("{}{:.2}", if pnl >= 0.0 { "+" } else { "" }, pnl), Style::default().fg(pnl_color).bold()),
                ]),
                Line::from(vec![
                    Span::styled("Cash Balance     ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("${:.2}", cash), Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("Positions        ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("{}", pos), Style::default().fg(self.theme.accent)),
                ]),
            ];

            if let Some(sr) = sharpe {
                let sr_color = if sr > 1.0 { self.theme.success } else if sr > 0.5 { self.theme.warning } else { self.theme.error };
                lines.push(Line::from(vec![
                    Span::styled("Sharpe Ratio     ", Style::default().fg(self.theme.text_muted)),
                    Span::styled(format!("{:.2}", sr), Style::default().fg(sr_color)),
                ]));
            }

            frame.render_widget(Paragraph::new(lines), inner);
        } else {
            let no_data = Paragraph::new("Not connected to trading engine")
                .style(Style::default().fg(self.theme.text_muted))
                .alignment(Alignment::Center);
            frame.render_widget(no_data, inner);
        }
    }

    /// Render training summary panel (reads PID file for live status)
    fn render_training_summary(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Training Summary").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Check PID file for running training
        let pid_path = crate::config::training_pid_file();
        let (status, run_id) = if pid_path.exists() {
            if let Ok(content) = std::fs::read_to_string(pid_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    let pid = json["pid"].as_u64().unwrap_or(0) as u32;
                    let run = json["run_id"].as_str().unwrap_or("-").to_string();
                    let proc_path = std::path::PathBuf::from(format!("/proc/{}", pid));
                    if proc_path.exists() {
                        ("Running", run)
                    } else {
                        ("Idle", "-".to_string())
                    }
                } else {
                    ("Idle", "-".to_string())
                }
            } else {
                ("Idle", "-".to_string())
            }
        } else {
            ("Idle", "-".to_string())
        };

        let status_color = if status == "Running" { self.theme.success } else { self.theme.text_muted };

        let lines = vec![
            Line::from(vec![
                Span::styled("Status           ", Style::default().fg(self.theme.text_muted)),
                Span::styled(status, Style::default().fg(status_color).bold()),
            ]),
            Line::from(vec![
                Span::styled("Run ID           ", Style::default().fg(self.theme.text_muted)),
                Span::styled(&run_id, Style::default().fg(self.theme.text_secondary)),
            ]),
        ];

        frame.render_widget(Paragraph::new(lines), inner);
    }
}

impl Default for OverviewView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for OverviewView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        // Layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // Header
                Constraint::Length(1),  // Health indicators
                Constraint::Min(5),    // Content (trading + training summaries)
                Constraint::Length(4), // Sparklines
                Constraint::Length(1), // Footer
            ])
            .margin(1)
            .split(area);

        // Header
        let header = Paragraph::new(Line::from(vec![
            Span::styled(format!("{} ", Separators::DIAMOND), Style::default().fg(self.theme.accent)),
            Span::styled("System Overview", Style::default().fg(self.theme.text_primary).bold()),
            Span::styled("  │  ", Style::default().fg(self.theme.border)),
            Span::styled("[r] Refresh  [b/Esc] Back", Style::default().fg(self.theme.text_muted)),
        ]));
        frame.render_widget(header, chunks[0]);

        // Health indicators
        self.render_health(chunks[1], frame.buffer_mut());

        // Content - split into three columns
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(35), // System Status
                Constraint::Percentage(35), // Trading Summary
                Constraint::Percentage(30), // Training Summary
            ])
            .split(chunks[2]);

        // Left: System Status
        self.system_status.render(frame, content_chunks[0])?;

        // Center: Trading Summary
        self.render_trading_summary(frame, content_chunks[1]);

        // Right: Training Summary
        self.render_training_summary(frame, content_chunks[2]);

        // Sparklines section
        let sparkline_block = Panel::new(&self.theme).title("Trends").block();
        let sparkline_inner = sparkline_block.inner(chunks[3]);
        frame.render_widget(sparkline_block, chunks[3]);

        if sparkline_inner.height >= 3 {
            let spark_rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .split(sparkline_inner);

            self.portfolio_sparkline.render(spark_rows[0], frame.buffer_mut());
            self.pnl_sparkline.render(spark_rows[1], frame.buffer_mut());
            self.positions_sparkline.render(spark_rows[2], frame.buffer_mut());
        }

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        match key {
            crossterm::event::KeyCode::Char('r') => {
                Ok(ViewAction::Continue) // App will call update() separately
            }
            _ => Ok(ViewAction::Continue),
        }
    }
}
