//! Status bar component
//!
//! Bottom status bar showing system status, key metrics, and help hints.

use chrono::Local;
use ratatui::prelude::*;
use ratatui::widgets::{Paragraph, Widget};

use crate::themes::Theme;
use crate::ui::borders::Separators;

/// Service status for display in status bar
#[derive(Clone, Debug)]
pub struct ServiceStatus {
    pub name: String,
    pub active: bool,
}

/// Status bar widget
pub struct StatusBar<'a> {
    theme: &'a Theme,
    services: Vec<ServiceStatus>,
    key_metrics: Vec<(String, String)>,
    show_help_hint: bool,
}

impl<'a> StatusBar<'a> {
    /// Create a new status bar
    pub fn new(theme: &'a Theme) -> Self {
        Self {
            theme,
            services: Vec::new(),
            key_metrics: Vec::new(),
            show_help_hint: true,
        }
    }

    /// Add a service status indicator
    pub fn service(mut self, name: impl Into<String>, active: bool) -> Self {
        self.services.push(ServiceStatus {
            name: name.into(),
            active,
        });
        self
    }

    /// Add a key metric
    pub fn metric(mut self, label: impl Into<String>, value: impl Into<String>) -> Self {
        self.key_metrics.push((label.into(), value.into()));
        self
    }

    /// Hide the help hint
    pub fn hide_help_hint(mut self) -> Self {
        self.show_help_hint = false;
        self
    }

    /// Build the status bar content as a Line
    fn build_content(&self) -> Line<'a> {
        let mut spans = Vec::new();

        // Logo
        spans.push(Span::styled(
            format!("{} FOX ML", Separators::DIAMOND),
            Style::default().fg(self.theme.accent).bold(),
        ));

        // Separator
        spans.push(Span::styled(
            " │ ",
            Style::default().fg(self.theme.border),
        ));

        // Services
        for (i, service) in self.services.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(
                    "  ",
                    Style::default(),
                ));
            }

            let dot = if service.active {
                Separators::CIRCLE_FILLED
            } else {
                Separators::CIRCLE_EMPTY
            };

            let color = if service.active {
                self.theme.success
            } else {
                self.theme.text_muted
            };

            spans.push(Span::styled(
                format!("{}: ", service.name),
                Style::default().fg(self.theme.text_secondary),
            ));
            spans.push(Span::styled(
                dot,
                Style::default().fg(color),
            ));
            spans.push(Span::styled(
                if service.active { " Active" } else { " Stopped" },
                Style::default().fg(color),
            ));
        }

        // Separator if we have services
        if !self.services.is_empty() {
            spans.push(Span::styled(
                " │ ",
                Style::default().fg(self.theme.border),
            ));
        }

        // Key metrics
        for (i, (label, value)) in self.key_metrics.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(
                    " │ ",
                    Style::default().fg(self.theme.border),
                ));
            }

            spans.push(Span::styled(
                format!("{}: ", label),
                Style::default().fg(self.theme.text_secondary),
            ));

            // Color positive/negative values
            let value_color = if value.starts_with('+') {
                self.theme.success
            } else if value.starts_with('-') {
                self.theme.error
            } else {
                self.theme.text_primary
            };

            spans.push(Span::styled(
                value.clone(),
                Style::default().fg(value_color),
            ));
        }

        // Time
        let time_str = Local::now().format("%H:%M:%S").to_string();
        spans.push(Span::styled(
            " │ ",
            Style::default().fg(self.theme.border),
        ));
        spans.push(Span::styled(
            time_str,
            Style::default().fg(self.theme.text_muted),
        ));

        // Help hint (right-aligned, will be handled in render)
        if self.show_help_hint {
            spans.push(Span::styled(
                " │ ",
                Style::default().fg(self.theme.border),
            ));
            spans.push(Span::styled(
                "[?] Help",
                Style::default().fg(self.theme.text_muted),
            ));
        }

        Line::from(spans)
    }
}

impl<'a> Widget for StatusBar<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Background
        let bg_style = Style::default().bg(self.theme.surface);
        for x in area.x..area.x + area.width {
            buf.get_mut(x, area.y).set_style(bg_style);
        }

        // Content
        let content = self.build_content();
        Paragraph::new(content).render(area, buf);
    }
}

/// Compact status indicator for use in headers
pub struct StatusIndicator<'a> {
    label: &'a str,
    active: bool,
    theme: &'a Theme,
}

impl<'a> StatusIndicator<'a> {
    /// Create a new status indicator
    pub fn new(theme: &'a Theme, label: &'a str, active: bool) -> Self {
        Self {
            label,
            active,
            theme,
        }
    }

    /// Render the indicator
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let dot = if self.active {
            Separators::CIRCLE_FILLED
        } else {
            Separators::CIRCLE_EMPTY
        };

        let color = if self.active {
            self.theme.success
        } else {
            self.theme.text_muted
        };

        let text = format!("{} {}", dot, self.label);
        Paragraph::new(text)
            .style(Style::default().fg(color))
            .render(area, buf);
    }
}
