//! System status view - shows FoxML system component status

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use sysinfo::System;

/// Path to training PID file
fn training_pid_file() -> std::path::PathBuf { crate::config::training_pid_file() }

/// System status
pub struct SystemStatus {
    system: System,
}

impl SystemStatus {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self { system }
    }

    /// Refresh system info
    pub fn refresh(&mut self) {
        self.system.refresh_all();
    }

    /// Check IPC bridge status
    fn check_bridge() -> (bool, String) {
        let output = Command::new("curl")
            .args(["-s", "-m", "1", &format!("http://{}/health", crate::config::bridge_url())])
            .output();

        match output {
            Ok(output) => {
                if output.status.success() {
                    (true, "Running".to_string())
                } else {
                    (false, "Stopped".to_string())
                }
            }
            Err(_) => (false, "Stopped".to_string()),
        }
    }

    /// Check systemd user service status
    fn check_user_service(name: &str) -> (bool, String) {
        let output = Command::new("systemctl")
            .args(["--user", "is-active", name])
            .output();

        match output {
            Ok(output) => {
                let status_str = String::from_utf8_lossy(&output.stdout);
                let status = status_str.trim();
                match status {
                    "active" => (true, "Running".to_string()),
                    "inactive" => (false, "Stopped".to_string()),
                    "failed" => (false, "Failed".to_string()),
                    _ => (false, "Not installed".to_string()),
                }
            }
            Err(_) => (false, "Unknown".to_string()),
        }
    }

    /// Check for running training process via PID file
    fn check_training_process() -> (bool, Option<String>) {
        let content = match fs::read_to_string(&training_pid_file()) {
            Ok(c) => c,
            Err(_) => return (false, None),
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(_) => return (false, None),
        };

        let pid = match json.get("pid").and_then(|v| v.as_u64()) {
            Some(p) => p as u32,
            None => return (false, None),
        };

        let run_id = json.get("run_id").and_then(|v| v.as_str()).map(|s| s.to_string());

        // Check if process is still alive
        let proc_path = PathBuf::from(format!("/proc/{}", pid));
        if proc_path.exists() {
            (true, run_id)
        } else {
            (false, None)
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        self.system.refresh_all();

        let block = Block::default()
            .title("System Status")
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Gather status
        let cpu_usage = self.system.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
            / self.system.cpus().len() as f32;
        let total_memory = self.system.total_memory();
        let used_memory = self.system.used_memory();
        let memory_percent = (used_memory as f64 / total_memory as f64) * 100.0;

        let (bridge_up, bridge_status) = Self::check_bridge();
        let (trading_up, trading_status) = Self::check_user_service("foxml-trading");
        let (training_up, training_run_id) = Self::check_training_process();

        // Build status lines with colored indicators
        let mut lines = vec![
            Line::from(Span::styled(
                "Components",
                Style::default().bold(),
            )),
        ];

        // IPC Bridge
        let bridge_indicator = if bridge_up { "●" } else { "○" };
        let bridge_color = if bridge_up { Color::Green } else { Color::Gray };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(bridge_indicator, Style::default().fg(bridge_color)),
            Span::raw(" IPC Bridge: "),
            Span::styled(&bridge_status, Style::default().fg(bridge_color)),
        ]));

        // Trading Service
        let trading_indicator = if trading_up { "●" } else { "○" };
        let trading_color = if trading_up { Color::Green } else { Color::Gray };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(trading_indicator, Style::default().fg(trading_color)),
            Span::raw(" Trading Service: "),
            Span::styled(&trading_status, Style::default().fg(trading_color)),
        ]));

        // Training Process
        let training_indicator = if training_up { "●" } else { "○" };
        let training_color = if training_up { Color::Green } else { Color::Gray };
        let training_text = if training_up {
            if let Some(run_id) = &training_run_id {
                // Truncate long run_ids
                let display_id = if run_id.len() > 30 {
                    format!("{}...", &run_id[..27])
                } else {
                    run_id.clone()
                };
                format!("Running ({})", display_id)
            } else {
                "Running".to_string()
            }
        } else {
            "Idle".to_string()
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(training_indicator, Style::default().fg(training_color)),
            Span::raw(" Training Pipeline: "),
            Span::styled(&training_text, Style::default().fg(training_color)),
        ]));

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "System Resources",
            Style::default().bold(),
        )));

        // CPU
        let cpu_color = if cpu_usage > 80.0 {
            Color::Red
        } else if cpu_usage > 50.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        lines.push(Line::from(vec![
            Span::raw("  CPU: "),
            Span::styled(format!("{:.1}%", cpu_usage), Style::default().fg(cpu_color)),
        ]));

        // Memory
        let mem_color = if memory_percent > 80.0 {
            Color::Red
        } else if memory_percent > 50.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        lines.push(Line::from(vec![
            Span::raw("  Memory: "),
            Span::styled(
                format!(
                    "{:.1}% ({:.1} GB / {:.1} GB)",
                    memory_percent,
                    used_memory as f64 / 1_073_741_824.0,
                    total_memory as f64 / 1_073_741_824.0
                ),
                Style::default().fg(mem_color),
            ),
        ]));

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);

        Ok(())
    }
}

impl Default for SystemStatus {
    fn default() -> Self {
        Self::new()
    }
}

