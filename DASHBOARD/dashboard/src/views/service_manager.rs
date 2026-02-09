//! Service Manager View - manage systemd services
//!
//! Provides UI for managing trading-related systemd services.
//! Discovers services dynamically from systemd user directory.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Service status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServiceStatus {
    Running,
    Stopped,
    Failed,
    NotInstalled,
    Unknown,
}

impl ServiceStatus {
    fn color(&self, theme: &Theme) -> Color {
        match self {
            ServiceStatus::Running => theme.success,
            ServiceStatus::Stopped => theme.text_muted,
            ServiceStatus::Failed => theme.error,
            ServiceStatus::NotInstalled => theme.text_muted,
            ServiceStatus::Unknown => theme.warning,
        }
    }

    fn label(&self) -> &str {
        match self {
            ServiceStatus::Running => "Running",
            ServiceStatus::Stopped => "Stopped",
            ServiceStatus::Failed => "Failed",
            ServiceStatus::NotInstalled => "Not Installed",
            ServiceStatus::Unknown => "Unknown",
        }
    }

    fn indicator(&self) -> &str {
        match self {
            ServiceStatus::Running => "●",
            ServiceStatus::Stopped => "○",
            ServiceStatus::Failed => "✗",
            ServiceStatus::NotInstalled => "◌",
            ServiceStatus::Unknown => "?",
        }
    }

    fn is_installed(&self) -> bool {
        !matches!(self, ServiceStatus::NotInstalled)
    }
}

/// Service definition
#[derive(Debug, Clone)]
pub struct Service {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub status: ServiceStatus,
    pub status_lines: Vec<String>,
}

impl Service {
    fn new(name: &str, display_name: &str, description: &str) -> Self {
        let mut service = Self {
            name: name.to_string(),
            display_name: display_name.to_string(),
            description: description.to_string(),
            status: ServiceStatus::Unknown,
            status_lines: Vec::new(),
        };
        service.refresh();
        service
    }

    /// Check if the systemd unit file exists
    fn is_installed(&self) -> bool {
        // Check if unit file exists via systemctl
        let output = Command::new("systemctl")
            .args(["--user", "cat", &self.name])
            .output();

        match output {
            Ok(out) => out.status.success(),
            Err(_) => false,
        }
    }

    fn refresh(&mut self) {
        // First check if service is installed
        if !self.is_installed() {
            self.status = ServiceStatus::NotInstalled;
            self.status_lines = vec![
                format!("Service '{}' is not installed.", self.name),
                String::new(),
                "To install, create a unit file at:".to_string(),
                format!("  ~/.config/systemd/user/{}.service", self.name),
            ];
            return;
        }

        // Check if service is active
        let output = Command::new("systemctl")
            .args(["--user", "is-active", &self.name])
            .output();

        self.status = match output {
            Ok(out) => {
                let status_str = String::from_utf8_lossy(&out.stdout);
                match status_str.trim() {
                    "active" => ServiceStatus::Running,
                    "inactive" => ServiceStatus::Stopped,
                    "failed" => ServiceStatus::Failed,
                    _ => ServiceStatus::Unknown,
                }
            }
            Err(_) => ServiceStatus::Unknown,
        };

        // Get detailed status
        let output = Command::new("systemctl")
            .args(["--user", "status", &self.name, "--no-pager", "-l"])
            .output();

        self.status_lines = match output {
            Ok(out) => {
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .take(15)
                    .map(|s| s.to_string())
                    .collect()
            }
            Err(_) => vec!["Unable to get status".to_string()],
        };
    }

    fn start(&mut self) -> Result<String> {
        if self.status == ServiceStatus::NotInstalled {
            return Ok(format!("{} is not installed", self.display_name));
        }

        let output = Command::new("systemctl")
            .args(["--user", "start", &self.name])
            .output()?;

        std::thread::sleep(Duration::from_millis(500));
        self.refresh();

        if output.status.success() {
            Ok(format!("Started {}", self.display_name))
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Ok(format!("Failed to start: {}", err.trim()))
        }
    }

    fn stop(&mut self) -> Result<String> {
        if self.status == ServiceStatus::NotInstalled {
            return Ok(format!("{} is not installed", self.display_name));
        }

        let output = Command::new("systemctl")
            .args(["--user", "stop", &self.name])
            .output()?;

        std::thread::sleep(Duration::from_millis(500));
        self.refresh();

        if output.status.success() {
            Ok(format!("Stopped {}", self.display_name))
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Ok(format!("Failed to stop: {}", err.trim()))
        }
    }

    fn restart(&mut self) -> Result<String> {
        if self.status == ServiceStatus::NotInstalled {
            return Ok(format!("{} is not installed", self.display_name));
        }

        let output = Command::new("systemctl")
            .args(["--user", "restart", &self.name])
            .output()?;

        std::thread::sleep(Duration::from_millis(500));
        self.refresh();

        if output.status.success() {
            Ok(format!("Restarted {}", self.display_name))
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Ok(format!("Failed to restart: {}", err.trim()))
        }
    }
}

/// Service manager view
pub struct ServiceManagerView {
    theme: Theme,
    services: Vec<Service>,
    selected: usize,
    last_refresh: Instant,
    message: Option<(String, bool)>, // (message, is_error)
}

impl ServiceManagerView {
    pub fn new() -> Self {
        let mut view = Self {
            theme: Theme::load(),
            services: Vec::new(),
            selected: 0,
            last_refresh: Instant::now(),
            message: None,
        };
        view.discover_services();
        view
    }

    /// Discover foxml-related services
    fn discover_services(&mut self) {
        // Start with known services
        let mut services = vec![
            Service::new(
                "foxml-trading",
                "Trading Engine",
                "Live trading execution engine",
            ),
            Service::new(
                "foxml-bridge",
                "IPC Bridge",
                "Dashboard IPC bridge (FastAPI)",
            ),
            Service::new(
                "foxml-training",
                "Training Runner",
                "ML training pipeline runner",
            ),
        ];

        // Collect names we already have
        let known_names: Vec<String> = services.iter().map(|s| s.name.clone()).collect();

        // Discover additional foxml-* services from systemd user directory
        if let Some(home) = std::env::var_os("HOME") {
            let systemd_user_dir = PathBuf::from(home).join(".config/systemd/user");
            if systemd_user_dir.exists() {
                if let Ok(entries) = fs::read_dir(&systemd_user_dir) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if let Some(file_name) = path.file_name() {
                            let name_str = file_name.to_string_lossy();
                            // Look for foxml-*.service files we don't already know about
                            if name_str.starts_with("foxml-") && name_str.ends_with(".service") {
                                let service_name = name_str.trim_end_matches(".service");
                                if !known_names.contains(&service_name.to_string()) {
                                    // Create display name from service name
                                    let display_name = service_name
                                        .strip_prefix("foxml-")
                                        .unwrap_or(service_name)
                                        .replace('-', " ")
                                        .split_whitespace()
                                        .map(|word| {
                                            let mut chars = word.chars();
                                            match chars.next() {
                                                None => String::new(),
                                                Some(c) => c.to_uppercase().chain(chars).collect(),
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join(" ");

                                    services.push(Service::new(
                                        service_name,
                                        &display_name,
                                        "User-defined service",
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        self.services = services;
    }

    /// Get the selected service index
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    /// Get the selected service display name (for dialog message)
    pub fn selected_service_name(&self) -> Option<&str> {
        self.services.get(self.selected).map(|s| s.display_name.as_str())
    }

    /// Stop a service by index (public for confirmation dialog)
    pub fn stop_service(&mut self, idx: usize) -> Result<String> {
        if idx < self.services.len() {
            self.services[idx].stop()
        } else {
            Ok("Invalid service index".to_string())
        }
    }

    /// Restart a service by index (public for confirmation dialog)
    pub fn restart_service(&mut self, idx: usize) -> Result<String> {
        if idx < self.services.len() {
            self.services[idx].restart()
        } else {
            Ok("Invalid service index".to_string())
        }
    }

    /// Refresh all services
    fn refresh_all(&mut self) {
        for service in &mut self.services {
            service.refresh();
        }
        self.last_refresh = Instant::now();
    }

    /// Render service list
    fn render_service_list(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Services").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let items: Vec<ListItem> = self
            .services
            .iter()
            .enumerate()
            .map(|(i, service)| {
                let is_selected = i == self.selected;
                let status_color = service.status.color(&self.theme);

                let content = Line::from(vec![
                    Span::styled(
                        format!(" {} ", service.status.indicator()),
                        Style::default().fg(status_color),
                    ),
                    Span::styled(
                        &service.display_name,
                        Style::default().fg(if is_selected {
                            self.theme.accent
                        } else {
                            self.theme.text_primary
                        }),
                    ),
                    Span::styled(
                        format!(" - {}", service.status.label()),
                        Style::default().fg(service.status.color(&self.theme)),
                    ),
                ]);

                let style = if is_selected {
                    Style::default().bg(self.theme.surface)
                } else {
                    Style::default()
                };

                ListItem::new(content).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render service details
    fn render_details(&self, frame: &mut Frame, area: Rect) {
        if self.services.is_empty() {
            return;
        }

        let service = &self.services[self.selected];
        let title = format!("{} Details", service.display_name);
        let block = Panel::new(&self.theme).title(&title).block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Service info + status lines
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Service: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(&service.name, Style::default().fg(self.theme.text_primary)),
            ]),
            Line::from(vec![
                Span::styled("Status: ", Style::default().fg(self.theme.text_muted)),
                Span::styled(
                    format!("{} {}", service.status.indicator(), service.status.label()),
                    Style::default().fg(service.status.color(&self.theme)),
                ),
            ]),
            Line::from(vec![Span::styled(
                &service.description,
                Style::default().fg(self.theme.text_muted).italic(),
            )]),
            Line::from(""),
            Line::from(Span::styled(
                "systemctl status:",
                Style::default().fg(self.theme.text_muted),
            )),
        ];

        for status_line in &service.status_lines {
            lines.push(Line::from(Span::styled(
                status_line,
                Style::default().fg(self.theme.text_secondary),
            )));
        }

        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        frame.render_widget(paragraph, inner);
    }

    /// Render footer with keybinds
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = vec![
            ("[j/k]", "Select"),
            ("[s]", "Start"),
            ("[x]", "Stop"),
            ("[r]", "Restart"),
            ("[R]", "Refresh"),
            ("[q/Esc]", "Back"),
        ];

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  ", Style::default()));
            }
            spans.push(Span::styled(*key, Style::default().fg(self.theme.accent)));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        // Show message if any
        if let Some((msg, is_error)) = &self.message {
            spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            spans.push(Span::styled(
                msg,
                Style::default().fg(if *is_error {
                    self.theme.error
                } else {
                    self.theme.success
                }),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for ServiceManagerView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for ServiceManagerView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Auto-refresh every 5 seconds
        if self.last_refresh.elapsed().as_secs() >= 5 {
            self.refresh_all();
        }

        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3 + self.services.len() as u16), // Service list
                Constraint::Min(0),                                  // Details
                Constraint::Length(1),                               // Footer
            ])
            .margin(1)
            .split(area);

        self.render_service_list(frame, chunks[0]);
        self.render_details(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        // Clear message on any key
        self.message = None;

        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(ViewAction::Back);
            }
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.services.len().saturating_sub(1) {
                    self.selected += 1;
                }
            }
            // Actions
            KeyCode::Char('s') => {
                if !self.services.is_empty() {
                    match self.services[self.selected].start() {
                        Ok(msg) => self.message = Some((msg, false)),
                        Err(e) => self.message = Some((format!("Error: {}", e), true)),
                    }
                }
            }
            KeyCode::Char('x') => {
                if !self.services.is_empty() {
                    match self.services[self.selected].stop() {
                        Ok(msg) => self.message = Some((msg, false)),
                        Err(e) => self.message = Some((format!("Error: {}", e), true)),
                    }
                }
            }
            KeyCode::Char('r') => {
                if !self.services.is_empty() {
                    match self.services[self.selected].restart() {
                        Ok(msg) => self.message = Some((msg, false)),
                        Err(e) => self.message = Some((format!("Error: {}", e), true)),
                    }
                }
            }
            KeyCode::Char('R') => {
                self.refresh_all();
                self.message = Some(("Refreshed all services".to_string(), false));
            }
            _ => {}
        }

        Ok(ViewAction::Continue)
    }
}
