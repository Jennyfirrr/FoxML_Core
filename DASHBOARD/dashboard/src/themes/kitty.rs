//! Kitty config parser

use regex::Regex;
use std::fs;
use crate::themes::theme::{LegacyTheme, Theme};

/// Parse colors from kitty config
pub fn parse_kitty_colors(config_path: &str) -> Option<LegacyTheme> {
    let content = fs::read_to_string(config_path).ok()?;

    // Kitty uses: background #1e1e2e
    // Or: color0 #1e1e2e
    let hex_re = Regex::new(r#"(?:background|foreground|color\d+)\s+(#[0-9a-fA-F]{6})"#).ok()?;

    let mut background = None;
    let mut foreground = None;
    let mut colors = std::collections::HashMap::new();

    for cap in hex_re.captures_iter(&content) {
        let hex = cap.get(1)?.as_str();
        let line = cap.get(0)?.as_str().to_lowercase();

        if line.starts_with("background") {
            background = Theme::hex_to_color(hex);
        } else if line.starts_with("foreground") {
            foreground = Theme::hex_to_color(hex);
        } else if line.starts_with("color") {
            // Extract color index
            let idx_str = line.strip_prefix("color")?.split_whitespace().next()?;
            if let Ok(idx) = idx_str.parse::<usize>() {
                if let Some(color) = Theme::hex_to_color(hex) {
                    colors.insert(idx, color);
                }
            }
        }
    }

    // Use color0/color7 if background/foreground not found
    if background.is_none() {
        background = colors.get(&0).copied();
    }
    if foreground.is_none() {
        foreground = colors.get(&7).or_else(|| colors.get(&15)).copied();
    }

    let primary = colors.get(&4).or_else(|| colors.get(&6)).copied();

    if background.is_some() || foreground.is_some() {
        Some(LegacyTheme {
            background: background.unwrap_or(ratatui::style::Color::Rgb(31, 36, 43)),
            foreground: foreground.unwrap_or(ratatui::style::Color::Rgb(245, 245, 247)),
            primary: primary.unwrap_or(ratatui::style::Color::Rgb(244, 181, 138)),
            secondary: ratatui::style::Color::Rgb(245, 169, 184),
            success: ratatui::style::Color::Rgb(139, 213, 162),
            warning: ratatui::style::Color::Rgb(249, 226, 175),
            error: ratatui::style::Color::Rgb(255, 107, 107),
        })
    } else {
        None
    }
}
