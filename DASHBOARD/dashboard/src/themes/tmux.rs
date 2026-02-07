//! Tmux config parser

use regex::Regex;
use std::fs;
use crate::themes::theme::{LegacyTheme, Theme};

/// Parse colors from tmux config
pub fn parse_tmux_colors(config_path: &str) -> Option<LegacyTheme> {
    let content = fs::read_to_string(config_path).ok()?;

    // Tmux uses: set -g status-bg "#1e1e2e"
    // Or: set -g status-style bg="#1e1e2e"
    let hex_re = Regex::new(r#"#[0-9a-fA-F]{6}"#).ok()?;

    let mut background = None;
    let mut foreground = None;

    for line in content.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.contains("background") || line_lower.contains("bg=") {
            if let Some(cap) = hex_re.find(line) {
                background = Theme::hex_to_color(cap.as_str());
            }
        } else if line_lower.contains("foreground") || line_lower.contains("fg=") {
            if let Some(cap) = hex_re.find(line) {
                foreground = Theme::hex_to_color(cap.as_str());
            }
        }
    }

    if background.is_some() || foreground.is_some() {
        Some(LegacyTheme {
            background: background.unwrap_or(ratatui::style::Color::Rgb(31, 36, 43)),
            foreground: foreground.unwrap_or(ratatui::style::Color::Rgb(245, 245, 247)),
            primary: ratatui::style::Color::Rgb(244, 181, 138),
            secondary: ratatui::style::Color::Rgb(245, 169, 184),
            success: ratatui::style::Color::Rgb(139, 213, 162),
            warning: ratatui::style::Color::Rgb(249, 226, 175),
            error: ratatui::style::Color::Rgb(255, 107, 107),
        })
    } else {
        None
    }
}
