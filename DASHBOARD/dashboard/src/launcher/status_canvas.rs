//! Status canvas - system health and connection status display

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::time::Instant;
use crate::api::client::DashboardClient;
use crate::api::alpaca::BridgeHealth;
use crate::ui::borders::Separators;

/// Service connection status
#[derive(Clone, Debug)]
struct ServiceStatus {
    name: &'static str,
    connected: bool,
    details: String,
}

/// Status canvas showing system health
pub struct StatusCanvas {
    client: DashboardClient,
    last_update: Instant,
    // Connection statuses
    bridge_connected: bool,
    alpaca_connected: bool,
    alpaca_paper: bool,
    engine_status: String,
    // Pipeline state
    pipeline_state: String,
    run_id: String,
    // Control state
    kill_switch_active: bool,
    paused: bool,
}

impl StatusCanvas {
    pub fn new() -> Self {
        Self {
            client: DashboardClient::new(&crate::config::bridge_url()),
            last_update: Instant::now(),
            bridge_connected: false,
            alpaca_connected: false,
            alpaca_paper: true,
            engine_status: "Unknown".to_string(),
            pipeline_state: "IDLE".to_string(),
            run_id: "N/A".to_string(),
            kill_switch_active: false,
            paused: false,
        }
    }

    /// Update status data (non-blocking, returns immediately if update not needed)
    pub async fn update(&mut self) {
        // Update every 2 seconds
        if self.last_update.elapsed().as_secs() < 2 {
            return;
        }

        // Get bridge health (includes Alpaca status)
        match self.client.get_health().await {
            Ok(health) => {
                self.bridge_connected = health.status == "ok";
                self.alpaca_connected = health.alpaca_connected;
            }
            Err(_) => {
                self.bridge_connected = false;
                self.alpaca_connected = false;
            }
        }

        // Get Alpaca status for paper/live mode
        if let Ok(status) = self.client.get_alpaca_status().await {
            self.alpaca_paper = status.paper.unwrap_or(true);
        }

        // Get pipeline state
        if let Ok(state) = self.client.get_state().await {
            if let Some(status) = state.get("status").and_then(|s| s.as_str()) {
                self.engine_status = status.to_string();
                self.pipeline_state = status.to_uppercase();
            }
            if let Some(current_stage) = state.get("current_stage").and_then(|s| s.as_str()) {
                if !current_stage.is_empty() && current_stage != "idle" {
                    self.pipeline_state = current_stage.to_uppercase();
                }
            }

            // Try to get run ID
            if let Some(run) = state.get("run_id").and_then(|r| r.as_str()) {
                self.run_id = if run.len() > 8 {
                    format!("{}...", &run[..8])
                } else {
                    run.to_string()
                };
            }
        }

        // Get control status
        if let Ok(control) = self.client.get_control_status().await {
            self.kill_switch_active = control
                .get("kill_switch_active")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            self.paused = control
                .get("paused")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        }

        self.last_update = Instant::now();
    }

    /// Check if bridge is connected
    pub fn is_bridge_connected(&self) -> bool {
        self.bridge_connected
    }

    /// Render status canvas
    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) -> Result<()> {
        // Layout: Services, Pipeline, Controls
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Services section
                Constraint::Length(1), // Separator
                Constraint::Length(3), // Pipeline section
                Constraint::Length(1), // Separator
                Constraint::Min(0),    // Controls/Info section
            ])
            .split(area);

        // Services section
        self.render_services(frame, chunks[0], theme);

        // Separator
        let sep = Paragraph::new("─".repeat(chunks[1].width as usize))
            .style(Style::default().fg(theme.border));
        frame.render_widget(sep, chunks[1]);

        // Pipeline section
        self.render_pipeline(frame, chunks[2], theme);

        // Separator
        let sep = Paragraph::new("─".repeat(chunks[3].width as usize))
            .style(Style::default().fg(theme.border));
        frame.render_widget(sep, chunks[3]);

        // Controls section
        self.render_controls(frame, chunks[4], theme);

        Ok(())
    }

    /// Render services connection status
    fn render_services(&self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Length(1), // Bridge
                Constraint::Length(1), // Alpaca
                Constraint::Length(1), // Engine
                Constraint::Min(0),
            ])
            .split(area);

        // Header
        let header = Line::from(vec![
            Span::styled("Services", Style::default().fg(theme.text_muted)),
        ]);
        frame.render_widget(Paragraph::new(header), rows[0]);

        // Bridge status
        let (bridge_icon, bridge_color) = if self.bridge_connected {
            (Separators::CIRCLE_FILLED, theme.success)
        } else {
            (Separators::CIRCLE_EMPTY, theme.error)
        };
        let bridge_line = Line::from(vec![
            Span::styled(format!(" {} ", bridge_icon), Style::default().fg(bridge_color)),
            Span::styled("Bridge", Style::default().fg(theme.text_secondary)),
            Span::styled(
                if self.bridge_connected { "  Connected" } else { "  Offline" },
                Style::default().fg(bridge_color),
            ),
        ]);
        frame.render_widget(Paragraph::new(bridge_line), rows[1]);

        // Alpaca status
        let (alpaca_icon, alpaca_color) = if self.alpaca_connected {
            (Separators::CIRCLE_FILLED, theme.success)
        } else {
            (Separators::CIRCLE_EMPTY, theme.text_muted)
        };
        let mode = if self.alpaca_paper { "Paper" } else { "Live" };
        let alpaca_line = Line::from(vec![
            Span::styled(format!(" {} ", alpaca_icon), Style::default().fg(alpaca_color)),
            Span::styled("Alpaca", Style::default().fg(theme.text_secondary)),
            Span::styled(
                format!("  {} {}", if self.alpaca_connected { mode } else { "Offline" }, ""),
                Style::default().fg(alpaca_color),
            ),
        ]);
        frame.render_widget(Paragraph::new(alpaca_line), rows[2]);

        // Engine status
        let engine_running = matches!(self.engine_status.to_lowercase().as_str(),
            "running" | "live" | "active");
        let (engine_icon, engine_color) = if engine_running {
            (Separators::CIRCLE_FILLED, theme.success)
        } else {
            (Separators::CIRCLE_EMPTY, theme.text_muted)
        };
        let engine_line = Line::from(vec![
            Span::styled(format!(" {} ", engine_icon), Style::default().fg(engine_color)),
            Span::styled("Engine", Style::default().fg(theme.text_secondary)),
            Span::styled(
                format!("  {}", self.engine_status),
                Style::default().fg(engine_color),
            ),
        ]);
        frame.render_widget(Paragraph::new(engine_line), rows[3]);
    }

    /// Render pipeline status
    fn render_pipeline(&self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Stage
                Constraint::Length(1), // Run ID
                Constraint::Min(0),
            ])
            .split(area);

        // Pipeline stage
        let stage_color = match self.pipeline_state.as_str() {
            "LIVE" | "RUNNING" | "ACTIVE" => theme.success,
            "TRAINING" | "IN_PROGRESS" | "PREDICTION" | "BLENDING" => theme.warning,
            "ERROR" | "FAILED" => theme.error,
            _ => theme.text_muted,
        };
        let stage_line = Line::from(vec![
            Span::styled("Stage: ", Style::default().fg(theme.text_muted)),
            Span::styled(&self.pipeline_state, Style::default().fg(stage_color).bold()),
        ]);
        frame.render_widget(Paragraph::new(stage_line), rows[0]);

        // Run ID
        let run_line = Line::from(vec![
            Span::styled("Run:   ", Style::default().fg(theme.text_muted)),
            Span::styled(&self.run_id, Style::default().fg(theme.text_secondary)),
        ]);
        frame.render_widget(Paragraph::new(run_line), rows[1]);
    }

    /// Render control status
    fn render_controls(&self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Kill switch
                Constraint::Length(1), // Paused
                Constraint::Min(0),
            ])
            .split(area);

        // Kill switch status
        let (ks_text, ks_color) = if self.kill_switch_active {
            ("ACTIVE", theme.error)
        } else {
            ("OFF", theme.success)
        };
        let ks_line = Line::from(vec![
            Span::styled("Kill Switch: ", Style::default().fg(theme.text_muted)),
            Span::styled(ks_text, Style::default().fg(ks_color).bold()),
        ]);
        frame.render_widget(Paragraph::new(ks_line), rows[0]);

        // Paused status
        if self.paused {
            let paused_line = Line::from(vec![
                Span::styled("Status: ", Style::default().fg(theme.text_muted)),
                Span::styled("PAUSED", Style::default().fg(theme.warning).bold()),
            ]);
            frame.render_widget(Paragraph::new(paused_line), rows[1]);
        }
    }
}

impl Default for StatusCanvas {
    fn default() -> Self {
        Self::new()
    }
}
