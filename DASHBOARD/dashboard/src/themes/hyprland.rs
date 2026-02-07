//! Hyprland config parser

use regex::Regex;
use std::fs;
use crate::themes::theme::{LegacyTheme, Theme};

/// Parse colors from hyprland config
pub fn parse_hyprland_colors(config_path: &str) -> Option<LegacyTheme> {
    let content = fs::read_to_string(config_path).ok()?;

    // Try to parse theme.conf format first (rgba hex values)
    // Pattern: col.active_border = rgba(f4b58aff) rgba(f5a9b8ff) 45deg
    let rgba_hex_re = Regex::new(r#"rgba\(([0-9a-fA-F]{6})(?:[0-9a-fA-F]{2})?\)"#).ok()?;
    let mut primary_from_rgba = None;
    let mut secondary_from_rgba = None;

    for cap in rgba_hex_re.captures_iter(&content) {
        let hex = cap.get(1)?.as_str();
        if let Some(color) = Theme::hex_to_color(&format!("#{}", hex)) {
            // Use first color found as primary (usually peach)
            if primary_from_rgba.is_none() {
                primary_from_rgba = Some(color);
            } else if secondary_from_rgba.is_none() {
                // Second color as secondary (usually pink)
                secondary_from_rgba = Some(color);
            }
        }
    }

    // Also try standard hyprland color format: $color0 = rgb(30,30,46) or $color0 = 0x1e1e2e
    let rgb_re = Regex::new(r#"\$color(\d+)\s*=\s*rgb\((\d+),(\d+),(\d+)\)"#).ok()?;
    let hex_re = Regex::new(r#"\$color(\d+)\s*=\s*0x([0-9a-fA-F]{6})"#).ok()?;

    let mut colors = std::collections::HashMap::new();

    // Parse rgb() format
    for cap in rgb_re.captures_iter(&content) {
        let idx: usize = cap.get(1)?.as_str().parse().ok()?;
        let r: u8 = cap.get(2)?.as_str().parse().ok()?;
        let g: u8 = cap.get(3)?.as_str().parse().ok()?;
        let b: u8 = cap.get(4)?.as_str().parse().ok()?;
        colors.insert(idx, ratatui::style::Color::Rgb(r, g, b));
    }

    // Parse hex format
    for cap in hex_re.captures_iter(&content) {
        let idx: usize = cap.get(1)?.as_str().parse().ok()?;
        let hex = cap.get(2)?.as_str();
        if let Some(color) = Theme::hex_to_color(&format!("#{}", hex)) {
            colors.insert(idx, color);
        }
    }

    // Try to extract bg/fg from comments or direct hex values in theme.conf
    // Look for hex values in comments: # bg: #1f242b
    let comment_hex_re = Regex::new(r#"#\s*(?:bg|fg|background|foreground)[:\s]+#([0-9a-fA-F]{6})"#).ok()?;
    let mut bg_from_comment = None;
    let mut fg_from_comment = None;

    for cap in comment_hex_re.captures_iter(&content) {
        let hex = format!("#{}", cap.get(1)?.as_str());
        if let Some(color) = Theme::hex_to_color(&hex) {
            if bg_from_comment.is_none() {
                bg_from_comment = Some(color);
            } else if fg_from_comment.is_none() {
                fg_from_comment = Some(color);
            }
        }
    }

    // color0 = background, color7 = foreground (typically)
    let background = colors.get(&0).copied().or(bg_from_comment);
    let foreground = colors.get(&7).or_else(|| colors.get(&15)).copied().or(fg_from_comment);
    let primary = colors.get(&4).or_else(|| colors.get(&6)).copied().or(primary_from_rgba);
    let secondary = secondary_from_rgba;

    // If we found any colors, create theme
    if background.is_some() || foreground.is_some() || primary.is_some() {
        Some(LegacyTheme {
            background: background.unwrap_or(ratatui::style::Color::Rgb(31, 36, 43)),
            foreground: foreground.unwrap_or(ratatui::style::Color::Rgb(245, 245, 247)),
            primary: primary.unwrap_or(ratatui::style::Color::Rgb(244, 181, 138)),
            secondary: secondary.unwrap_or(ratatui::style::Color::Rgb(245, 169, 184)),
            success: ratatui::style::Color::Rgb(139, 213, 162),
            warning: ratatui::style::Color::Rgb(249, 226, 175),
            error: ratatui::style::Color::Rgb(255, 107, 107),
        })
    } else {
        None
    }
}
