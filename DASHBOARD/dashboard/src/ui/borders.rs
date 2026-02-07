//! Custom border sets for the dashboard
//!
//! Provides rounded corner border sets and utilities for consistent styling.

use ratatui::widgets::block::BorderType;
use ratatui::symbols::border;

/// Rounded border set using Unicode box-drawing characters
pub const ROUNDED_BORDER_SET: border::Set = border::Set {
    top_left: "╭",
    top_right: "╮",
    bottom_left: "╰",
    bottom_right: "╯",
    vertical_left: "│",
    vertical_right: "│",
    horizontal_top: "─",
    horizontal_bottom: "─",
};

/// Double-line border set for emphasis
pub const DOUBLE_BORDER_SET: border::Set = border::Set {
    top_left: "╔",
    top_right: "╗",
    bottom_left: "╚",
    bottom_right: "╝",
    vertical_left: "║",
    vertical_right: "║",
    horizontal_top: "═",
    horizontal_bottom: "═",
};

/// Thick border set for high emphasis
pub const THICK_BORDER_SET: border::Set = border::Set {
    top_left: "┏",
    top_right: "┓",
    bottom_left: "┗",
    bottom_right: "┛",
    vertical_left: "┃",
    vertical_right: "┃",
    horizontal_top: "━",
    horizontal_bottom: "━",
};

/// Wrapper for rounded border functionality
pub struct RoundedBorder;

impl RoundedBorder {
    /// Get the rounded border type
    pub fn border_type() -> BorderType {
        BorderType::Rounded
    }

    /// Get the rounded border set
    pub fn border_set() -> border::Set {
        ROUNDED_BORDER_SET
    }

    /// Create a custom border string for top of a panel
    pub fn top_border(width: usize, title: Option<&str>) -> String {
        if let Some(title) = title {
            let title_len = title.chars().count() + 2; // Space on each side
            let remaining = width.saturating_sub(title_len + 2); // -2 for corners
            let left_len = remaining / 2;
            let right_len = remaining - left_len;

            format!(
                "╭{}─ {} {}╮",
                "─".repeat(left_len),
                title,
                "─".repeat(right_len)
            )
        } else {
            format!("╭{}╮", "─".repeat(width.saturating_sub(2)))
        }
    }

    /// Create a custom border string for bottom of a panel
    pub fn bottom_border(width: usize) -> String {
        format!("╰{}╯", "─".repeat(width.saturating_sub(2)))
    }

    /// Create a horizontal separator
    pub fn separator(width: usize) -> String {
        format!("├{}┤", "─".repeat(width.saturating_sub(2)))
    }

    /// Create a rounded separator that connects to sides
    pub fn inner_separator(width: usize) -> String {
        format!("│{}│", "─".repeat(width.saturating_sub(2)))
    }
}

/// Separator characters for visual hierarchy
pub struct Separators;

impl Separators {
    /// Light horizontal line
    pub const LIGHT_HORIZONTAL: &'static str = "─";

    /// Medium horizontal line with dots
    pub const DOTTED: &'static str = "┄";

    /// Heavy horizontal line
    pub const HEAVY_HORIZONTAL: &'static str = "━";

    /// Bullet point
    pub const BULLET: &'static str = "•";

    /// Arrow right
    pub const ARROW_RIGHT: &'static str = "→";

    /// Arrow left
    pub const ARROW_LEFT: &'static str = "←";

    /// Check mark
    pub const CHECK: &'static str = "✓";

    /// Cross mark
    pub const CROSS: &'static str = "✗";

    /// Diamond (for branding)
    pub const DIAMOND: &'static str = "◆";

    /// Empty diamond
    pub const DIAMOND_EMPTY: &'static str = "◇";

    /// Filled circle (status indicator)
    pub const CIRCLE_FILLED: &'static str = "●";

    /// Empty circle
    pub const CIRCLE_EMPTY: &'static str = "○";

    /// Triangle right (for expandable items)
    pub const TRIANGLE_RIGHT: &'static str = "▶";

    /// Triangle down (for expanded items)
    pub const TRIANGLE_DOWN: &'static str = "▼";

    /// Vertical bar (for cursor)
    pub const CURSOR: &'static str = "│";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_border_with_title() {
        let border = RoundedBorder::top_border(20, Some("Test"));
        assert!(border.starts_with("╭"));
        assert!(border.ends_with("╮"));
        assert!(border.contains("Test"));
    }

    #[test]
    fn test_top_border_without_title() {
        let border = RoundedBorder::top_border(10, None);
        assert_eq!(border, "╭────────╮");
    }

    #[test]
    fn test_bottom_border() {
        let border = RoundedBorder::bottom_border(10);
        assert_eq!(border, "╰────────╯");
    }
}
