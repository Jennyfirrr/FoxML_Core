//! Theme structure and application
//!
//! Semantic color system inspired by modern UI frameworks.
//! Colors are organized by purpose, not by color name.

use ratatui::style::Color;

use crate::themes::{hyprland, kitty, tmux, waybar};

/// Theme colors with semantic organization
#[derive(Clone, Debug)]
pub struct Theme {
    // ─────────────────────────────────────────────────────────────────────────
    // Surfaces - Background layers
    // ─────────────────────────────────────────────────────────────────────────
    /// Main background color
    pub background: Color,
    /// Card/panel background (slightly elevated)
    pub surface: Color,
    /// Modals, dropdowns, tooltips (most elevated)
    pub surface_elevated: Color,

    // ─────────────────────────────────────────────────────────────────────────
    // Text - Foreground colors
    // ─────────────────────────────────────────────────────────────────────────
    /// Primary text - headers, active items, important content
    pub text_primary: Color,
    /// Secondary text - descriptions, labels, less important
    pub text_secondary: Color,
    /// Muted text - hints, disabled states, timestamps
    pub text_muted: Color,

    // ─────────────────────────────────────────────────────────────────────────
    // Accents - Brand/highlight colors
    // ─────────────────────────────────────────────────────────────────────────
    /// Primary accent - main brand color (peach)
    pub accent: Color,
    /// Secondary accent - gradient pair (pink)
    pub accent_secondary: Color,
    /// Tertiary accent - complementary (lavender)
    pub accent_tertiary: Color,
    /// Muted accent - hover states, subtle highlights
    pub accent_muted: Color,

    // ─────────────────────────────────────────────────────────────────────────
    // Semantic - Status colors
    // ─────────────────────────────────────────────────────────────────────────
    /// Success - positive actions, confirmations
    pub success: Color,
    /// Warning - caution, attention needed
    pub warning: Color,
    /// Error - failures, destructive actions
    pub error: Color,
    /// Info - informational, neutral
    pub info: Color,

    // ─────────────────────────────────────────────────────────────────────────
    // Borders - Edge colors
    // ─────────────────────────────────────────────────────────────────────────
    /// Default border color
    pub border: Color,
    /// Focused/active border color
    pub border_focused: Color,

    // ─────────────────────────────────────────────────────────────────────────
    // Legacy - Backwards compatibility
    // ─────────────────────────────────────────────────────────────────────────
    /// Legacy: foreground (alias for text_primary)
    pub foreground: Color,
    /// Legacy: primary (alias for accent)
    pub primary: Color,
    /// Legacy: secondary (alias for accent_secondary)
    pub secondary: Color,
    /// Legacy: primary_text (alias for text_primary)
    pub primary_text: Color,
    /// Legacy: secondary_text (alias for text_secondary)
    pub secondary_text: Color,
}

impl Default for Theme {
    fn default() -> Self {
        // Fox ML theme - peach/pink/lavender gradient aesthetic
        Self::foxml_dark()
    }
}

impl Theme {
    /// Fox ML Dark theme - the default theme
    /// Based on ~/THEME/FoxML color palette
    pub fn foxml_dark() -> Self {
        Self {
            // Surfaces
            background: Color::Rgb(31, 36, 43),         // #1f242b - Dark slate
            surface: Color::Rgb(42, 48, 56),            // Slightly lighter
            surface_elevated: Color::Rgb(58, 65, 75),   // #3a414b - Slate

            // Text
            text_primary: Color::Rgb(245, 245, 247),    // #f5f5f7 - Off-white
            text_secondary: Color::Rgb(180, 180, 185),  // Muted
            text_muted: Color::Rgb(136, 136, 136),      // #888888 - Very muted

            // Accents (peach-pink gradient)
            accent: Color::Rgb(244, 181, 138),          // #f4b58a - Peach
            accent_secondary: Color::Rgb(245, 169, 184),// #f5a9b8 - Pink
            accent_tertiary: Color::Rgb(154, 138, 196), // #9a8ac4 - Lavender
            accent_muted: Color::Rgb(194, 145, 110),    // Muted peach

            // Semantic
            success: Color::Rgb(139, 213, 162),         // #8bd5a2 - Soft green
            warning: Color::Rgb(249, 226, 175),         // #f9e2af - Warm yellow
            error: Color::Rgb(255, 107, 107),           // #ff6b6b - Soft red
            info: Color::Rgb(137, 180, 250),            // #89b4fa - Soft blue

            // Borders
            border: Color::Rgb(58, 65, 75),             // #3a414b - Slate
            border_focused: Color::Rgb(244, 181, 138),  // Peach when focused

            // Legacy aliases
            foreground: Color::Rgb(245, 245, 247),
            primary: Color::Rgb(244, 181, 138),
            secondary: Color::Rgb(245, 169, 184),
            primary_text: Color::Rgb(245, 245, 247),
            secondary_text: Color::Rgb(180, 180, 185),
        }
    }

    /// Load theme from config files
    /// Tries ~/THEME/FoxML -> waybar -> hyprland -> tmux -> kitty -> default
    pub fn load() -> Self {
        if let Some(home) = dirs::home_dir() {
            // First try ~/THEME/FoxML/ theme files
            let theme_dir = home.join("THEME/FoxML");
            let waybar_theme = theme_dir.join("waybar/style.css");
            let hyprland_theme = theme_dir.join("hyprland/theme.conf");

            // Try waybar theme from ~/THEME first
            if waybar_theme.exists() {
                if let Some(t) = waybar::parse_waybar_css(&waybar_theme.to_string_lossy()) {
                    return Self::from_legacy_theme(t);
                }
            }

            // Try hyprland theme from ~/THEME
            if hyprland_theme.exists() {
                if let Some(t) = hyprland::parse_hyprland_colors(&hyprland_theme.to_string_lossy())
                {
                    return Self::from_legacy_theme(t);
                }
            }

            // Fallback to standard config locations
            let waybar_config = home.join(".config/waybar/config");
            if waybar_config.exists() {
                if let Some(t) = waybar::parse_waybar_colors(&waybar_config.to_string_lossy()) {
                    return Self::from_legacy_theme(t);
                }
            }

            let hyprland_config = home.join(".config/hypr/hyprland.conf");
            if hyprland_config.exists() {
                if let Some(t) = hyprland::parse_hyprland_colors(&hyprland_config.to_string_lossy())
                {
                    return Self::from_legacy_theme(t);
                }
            }

            let tmux_config = home.join(".tmux.conf");
            if tmux_config.exists() {
                if let Some(t) = tmux::parse_tmux_colors(&tmux_config.to_string_lossy()) {
                    return Self::from_legacy_theme(t);
                }
            }

            let kitty_config = home.join(".config/kitty/kitty.conf");
            if kitty_config.exists() {
                if let Some(t) = kitty::parse_kitty_colors(&kitty_config.to_string_lossy()) {
                    return Self::from_legacy_theme(t);
                }
            }
        }

        // Fallback to default Fox ML theme
        Self::default()
    }

    /// Convert legacy theme format to new semantic format
    fn from_legacy_theme(legacy: LegacyTheme) -> Self {
        // Extract RGB values from primary color to create accent variants
        let (accent, accent_secondary, accent_tertiary) = Self::derive_accent_colors(legacy.primary);

        Self {
            // Surfaces
            background: legacy.background,
            surface: Self::lighten_color(legacy.background, 0.05),
            surface_elevated: Self::lighten_color(legacy.background, 0.12),

            // Text
            text_primary: legacy.foreground,
            text_secondary: Self::blend_colors(legacy.foreground, legacy.background, 0.7),
            text_muted: Self::blend_colors(legacy.foreground, legacy.background, 0.5),

            // Accents
            accent,
            accent_secondary,
            accent_tertiary,
            accent_muted: Self::blend_colors(accent, legacy.background, 0.7),

            // Semantic
            success: legacy.success,
            warning: legacy.warning,
            error: legacy.error,
            info: Color::Rgb(137, 180, 250), // Default info blue

            // Borders
            border: Self::lighten_color(legacy.background, 0.12),
            border_focused: accent,

            // Legacy aliases
            foreground: legacy.foreground,
            primary: legacy.primary,
            secondary: legacy.secondary,
            primary_text: legacy.foreground,
            secondary_text: Self::blend_colors(legacy.foreground, legacy.background, 0.7),
        }
    }

    /// Derive accent color variants from primary color
    fn derive_accent_colors(primary: Color) -> (Color, Color, Color) {
        if let Color::Rgb(r, g, b) = primary {
            // Shift hue for secondary and tertiary
            let secondary = Color::Rgb(
                r.saturating_add(10),
                g.saturating_sub(10),
                b.saturating_add(30),
            );
            let tertiary = Color::Rgb(
                r.saturating_sub(60),
                g.saturating_sub(30),
                b.saturating_add(50),
            );
            (primary, secondary, tertiary)
        } else {
            // Default Fox ML colors
            (
                Color::Rgb(244, 181, 138), // Peach
                Color::Rgb(245, 169, 184), // Pink
                Color::Rgb(154, 138, 196), // Lavender
            )
        }
    }

    /// Lighten a color by a factor (0.0 - 1.0)
    fn lighten_color(color: Color, factor: f32) -> Color {
        if let Color::Rgb(r, g, b) = color {
            let lighten = |c: u8| -> u8 {
                let c = c as f32;
                (c + (255.0 - c) * factor).min(255.0) as u8
            };
            Color::Rgb(lighten(r), lighten(g), lighten(b))
        } else {
            color
        }
    }

    /// Blend two colors with a ratio (0.0 = color1, 1.0 = color2)
    fn blend_colors(color1: Color, color2: Color, ratio: f32) -> Color {
        match (color1, color2) {
            (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
                let blend = |c1: u8, c2: u8| -> u8 {
                    let c1 = c1 as f32;
                    let c2 = c2 as f32;
                    (c1 * ratio + c2 * (1.0 - ratio)) as u8
                };
                Color::Rgb(blend(r1, r2), blend(g1, g2), blend(b1, b2))
            }
            _ => color1,
        }
    }

    /// Convert hex string to Color
    pub fn hex_to_color(hex: &str) -> Option<Color> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;

        Some(Color::Rgb(r, g, b))
    }

    /// Get gradient colors for borders (peach -> pink)
    pub fn gradient_colors(&self) -> (Color, Color) {
        (self.accent, self.accent_secondary)
    }

    /// Get status indicator color
    pub fn status_color(&self, status: &str) -> Color {
        match status.to_lowercase().as_str() {
            "running" | "active" | "live" | "connected" | "success" | "ok" => self.success,
            "warning" | "degraded" | "slow" => self.warning,
            "error" | "failed" | "disconnected" | "critical" => self.error,
            "info" | "pending" | "starting" => self.info,
            _ => self.text_muted,
        }
    }

    /// Get status dot character
    pub fn status_dot(&self, active: bool) -> &'static str {
        if active {
            "●"
        } else {
            "○"
        }
    }
}

/// Legacy theme format for backwards compatibility with theme parsers
pub struct LegacyTheme {
    pub background: Color,
    pub foreground: Color,
    pub primary: Color,
    pub secondary: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
}

impl Default for LegacyTheme {
    fn default() -> Self {
        Self {
            background: Color::Rgb(31, 36, 43),
            foreground: Color::Rgb(245, 245, 247),
            primary: Color::Rgb(244, 181, 138),
            secondary: Color::Rgb(245, 169, 184),
            success: Color::Rgb(139, 213, 162),
            warning: Color::Rgb(249, 226, 175),
            error: Color::Rgb(255, 107, 107),
        }
    }
}
