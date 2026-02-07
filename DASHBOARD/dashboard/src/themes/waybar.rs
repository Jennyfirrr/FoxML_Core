//! Waybar config parser

use regex::Regex;
use std::fs;
use crate::themes::theme::{LegacyTheme, Theme};

/// Parse colors from waybar CSS file (style.css)
pub fn parse_waybar_css(css_path: &str) -> Option<LegacyTheme> {
    let content = fs::read_to_string(css_path).ok()?;

    // Parse @define-color directives: @define-color bg #1f242b;
    let color_re = Regex::new(r#"@define-color\s+(\w+)\s+#([0-9a-fA-F]{6})"#).ok()?;

    let mut bg = None;
    let mut fg = None;
    let mut primary = None;
    let mut secondary = None;
    let mut success = None;
    let mut warning = None;
    let mut error = None;

    for cap in color_re.captures_iter(&content) {
        let name = cap.get(1)?.as_str().to_lowercase();
        let hex = format!("#{}", cap.get(2)?.as_str());

        match name.as_str() {
            "bg" | "background" => {
                bg = Theme::hex_to_color(&hex);
            }
            "fg" | "foreground" => {
                fg = Theme::hex_to_color(&hex);
            }
            "peach" => {
                // Peach is the primary accent color
                primary = Theme::hex_to_color(&hex);
            }
            "pink" => {
                // Pink is the secondary accent color
                secondary = Theme::hex_to_color(&hex);
            }
            "lavender" | "primary" | "accent" => {
                // Use lavender as primary if peach not found
                if primary.is_none() {
                    primary = Theme::hex_to_color(&hex);
                }
            }
            "green" | "success" => {
                success = Theme::hex_to_color(&hex);
            }
            "yellow" | "warning" => {
                warning = Theme::hex_to_color(&hex);
            }
            "red" | "error" => {
                error = Theme::hex_to_color(&hex);
            }
            _ => {}
        }
    }

    // If we found colors, create theme
    if bg.is_some() || fg.is_some() {
        Some(LegacyTheme {
            background: bg.unwrap_or(ratatui::style::Color::Rgb(31, 36, 43)),
            foreground: fg.unwrap_or(ratatui::style::Color::Rgb(245, 245, 247)),
            primary: primary.unwrap_or(ratatui::style::Color::Rgb(244, 181, 138)),
            secondary: secondary.unwrap_or(ratatui::style::Color::Rgb(245, 169, 184)),
            success: success.unwrap_or(ratatui::style::Color::Rgb(139, 213, 162)),
            warning: warning.unwrap_or(ratatui::style::Color::Rgb(249, 226, 175)),
            error: error.unwrap_or(ratatui::style::Color::Rgb(255, 107, 107)),
        })
    } else {
        None
    }
}

/// Parse colors from waybar config
pub fn parse_waybar_colors(config_path: &str) -> Option<LegacyTheme> {
    let content = fs::read_to_string(config_path).ok()?;

    // Waybar uses JSON format, look for color definitions
    // Common patterns: "background": "#1e1e2e", "foreground": "#cdd6f4"
    // Use a different raw string delimiter to avoid conflicts with # in the pattern
    let hex_re = Regex::new(r##""([^"]*color[^"]*)":\s*"#([0-9a-fA-F]{6})""##).ok()?;

    let mut background = None;
    let mut foreground = None;
    let mut primary = None;

    for cap in hex_re.captures_iter(&content) {
        let key_match = cap.get(1)?;
        let key = key_match.as_str().to_lowercase();
        let hex = format!("#{}", cap.get(2)?.as_str());

        if key.contains("background") {
            background = Theme::hex_to_color(&hex);
        } else if key.contains("foreground") || key.contains("text") {
            foreground = Theme::hex_to_color(&hex);
        } else if key.contains("primary") || key.contains("accent") {
            primary = Theme::hex_to_color(&hex);
        }
    }

    // If we found at least some colors, create theme
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
