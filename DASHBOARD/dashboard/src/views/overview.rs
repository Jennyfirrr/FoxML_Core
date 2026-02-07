//! Overview view - combined system overview

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use crate::launcher::system_status::SystemStatus;
use crate::api::client::DashboardClient;

/// Overview view - combined system status
pub struct OverviewView {
    system_status: SystemStatus,
    client: DashboardClient,
    trading_metrics: Option<serde_json::Value>,
}

impl OverviewView {
    pub fn new() -> Self {
        Self {
            system_status: SystemStatus::new(),
            client: DashboardClient::new("127.0.0.1:8765"),
            trading_metrics: None,
        }
    }

    /// Update metrics
    pub async fn update(&mut self) {
        self.system_status.refresh();
        if let Ok(metrics) = self.client.get_metrics().await {
            self.trading_metrics = Some(metrics);
        }
    }
}

impl Default for OverviewView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for OverviewView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Header
                Constraint::Min(5),     // Content
            ])
            .split(area);

        // Header
        let header = Block::default()
            .title("System Overview")
            .borders(Borders::ALL);
        let header_text = "[r] Refresh  [b/Esc] Back to Launcher  [Tab] Switch View";
        let header_para = Paragraph::new(header_text).block(header);
        frame.render_widget(header_para, chunks[0]);

        // Content - split horizontally
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),  // System Status
                Constraint::Percentage(50),  // Trading Metrics
            ])
            .split(chunks[1]);

        // Left: System Status
        self.system_status.render(frame, content_chunks[0])?;

        // Right: Trading Metrics
        let metrics_block = Block::default()
            .title("Trading Metrics")
            .borders(Borders::ALL);

        let metrics_text = if let Some(ref metrics) = self.trading_metrics {
            format!(
                "Portfolio: ${:.2}\nDaily P&L: ${:.2}\nCash: ${:.2}\nPositions: {}",
                metrics["portfolio_value"].as_f64().unwrap_or(0.0),
                metrics["daily_pnl"].as_f64().unwrap_or(0.0),
                metrics["cash_balance"].as_f64().unwrap_or(0.0),
                metrics["positions_count"].as_u64().unwrap_or(0) as usize,
            )
        } else {
            "Not connected to trading engine".to_string()
        };

        let metrics_para = Paragraph::new(metrics_text)
            .block(metrics_block)
            .wrap(Wrap { trim: true });
        frame.render_widget(metrics_para, content_chunks[1]);

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<bool> {
        match key {
            crossterm::event::KeyCode::Char('r') => {
                // Refresh overview - use async context from app
                // Note: This will be called from app's async context
                Ok(false) // App will call update() separately
            }
            _ => Ok(false),
        }
    }
}
