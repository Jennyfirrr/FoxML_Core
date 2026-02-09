//! Help overlay
//!
//! Keyboard shortcuts reference overlay.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Widget};

use crate::themes::Theme;

/// A keyboard shortcut entry
struct ShortcutEntry {
    key: &'static str,
    description: &'static str,
}

/// Help overlay showing keyboard shortcuts
pub struct HelpOverlay {
    pub visible: bool,
}

impl HelpOverlay {
    /// Create a new help overlay
    pub fn new() -> Self {
        Self { visible: false }
    }

    /// Show the overlay
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide the overlay
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Toggle visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Get navigation shortcuts
    fn navigation_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "1", description: "Trading Monitor" },
            ShortcutEntry { key: "2", description: "Training Monitor" },
            ShortcutEntry { key: "3", description: "System Overview" },
            ShortcutEntry { key: "Tab", description: "Cycle Views" },
            ShortcutEntry { key: "b/Esc", description: "Back to Dashboard" },
        ]
    }

    /// Get general shortcuts
    fn general_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "Ctrl+P", description: "Command Palette" },
            ShortcutEntry { key: "/", description: "Command Palette / Search" },
            ShortcutEntry { key: "?", description: "This Help" },
            ShortcutEntry { key: "r", description: "Refresh" },
            ShortcutEntry { key: "j/k", description: "Navigate Down/Up" },
            ShortcutEntry { key: "Ctrl+Q", description: "Quit" },
        ]
    }

    /// Get sidebar shortcuts
    fn sidebar_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "[", description: "Toggle Sidebar" },
            ShortcutEntry { key: "]", description: "Pin/Unpin Sidebar" },
        ]
    }

    /// Get trading shortcuts
    fn trading_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "Ctrl+K", description: "Toggle Kill Switch" },
            ShortcutEntry { key: "k", description: "Kill Switch (in Trading)" },
            ShortcutEntry { key: "p", description: "Pause/Resume Trading" },
            ShortcutEntry { key: "s", description: "Cycle Position Sort" },
            ShortcutEntry { key: "c", description: "Toggle P&L Chart" },
            ShortcutEntry { key: "w", description: "Connect WebSocket" },
            ShortcutEntry { key: "Enter", description: "Position Detail" },
        ]
    }

    /// Get training shortcuts
    fn training_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "x", description: "Cancel Training Run" },
            ShortcutEntry { key: "c", description: "Clear Events" },
            ShortcutEntry { key: "r", description: "Rescan Runs" },
        ]
    }

    /// Get log viewer shortcuts
    fn log_viewer_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "/", description: "Search" },
            ShortcutEntry { key: "n/N", description: "Next/Prev Match" },
        ]
    }

    /// Get model selector shortcuts
    fn model_selector_shortcuts() -> Vec<ShortcutEntry> {
        vec![
            ShortcutEntry { key: "a", description: "Activate Model" },
            ShortcutEntry { key: "Enter", description: "Toggle Details" },
        ]
    }

    /// Render a section of shortcuts
    fn render_section(
        &self,
        title: &str,
        shortcuts: &[ShortcutEntry],
        area: Rect,
        buf: &mut Buffer,
        theme: &Theme,
    ) {
        // Title
        Paragraph::new(title)
            .style(Style::default().fg(theme.text_secondary).underlined())
            .render(
                Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: 1,
                },
                buf,
            );

        // Shortcuts
        for (i, entry) in shortcuts.iter().enumerate() {
            if i as u16 + 1 >= area.height {
                break;
            }

            let row = Rect {
                x: area.x,
                y: area.y + i as u16 + 1,
                width: area.width,
                height: 1,
            };

            // Key
            Paragraph::new(format!("{:>12}", entry.key))
                .style(Style::default().fg(theme.accent).bold())
                .render(
                    Rect {
                        x: row.x,
                        y: row.y,
                        width: 12,
                        height: 1,
                    },
                    buf,
                );

            // Description
            Paragraph::new(format!("  {}", entry.description))
                .style(Style::default().fg(theme.text_primary))
                .render(
                    Rect {
                        x: row.x + 12,
                        y: row.y,
                        width: row.width.saturating_sub(12),
                        height: 1,
                    },
                    buf,
                );
        }
    }

    /// Render the help overlay
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        if !self.visible {
            return;
        }

        // Calculate overlay size and position (centered, 70% width, 80% height)
        let width = ((area.width as f32 * 0.7) as u16).min(area.width.saturating_sub(4));
        let height = ((area.height as f32 * 0.8) as u16).min(area.height.saturating_sub(4));
        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;

        let overlay_area = Rect {
            x,
            y,
            width,
            height,
        };

        // Clear the area
        Clear.render(overlay_area, buf);

        // Main block
        let block = Block::default()
            .title(" Keyboard Shortcuts ")
            .title_style(Style::default().fg(theme.accent).bold())
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.border))
            .style(Style::default().bg(theme.surface_elevated));

        let inner = block.inner(overlay_area);
        block.render(overlay_area, buf);

        // Split into two columns
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner);

        // Left column: Global shortcuts
        let left_sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7), // Navigation
                Constraint::Length(1), // Spacer
                Constraint::Length(8), // General
                Constraint::Length(1), // Spacer
                Constraint::Min(0),    // Sidebar
            ])
            .split(columns[0]);

        self.render_section(
            "Navigation",
            &Self::navigation_shortcuts(),
            left_sections[0],
            buf,
            theme,
        );
        self.render_section(
            "General",
            &Self::general_shortcuts(),
            left_sections[2],
            buf,
            theme,
        );
        self.render_section(
            "Sidebar",
            &Self::sidebar_shortcuts(),
            left_sections[4],
            buf,
            theme,
        );

        // Right column: View-specific shortcuts
        let right_sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9), // Trading
                Constraint::Length(1), // Spacer
                Constraint::Length(5), // Training
                Constraint::Length(1), // Spacer
                Constraint::Length(4), // Log Viewer
                Constraint::Length(1), // Spacer
                Constraint::Min(0),    // Model Selector
            ])
            .split(columns[1]);

        self.render_section(
            "Trading View",
            &Self::trading_shortcuts(),
            right_sections[0],
            buf,
            theme,
        );
        self.render_section(
            "Training View",
            &Self::training_shortcuts(),
            right_sections[2],
            buf,
            theme,
        );
        self.render_section(
            "Log Viewer",
            &Self::log_viewer_shortcuts(),
            right_sections[4],
            buf,
            theme,
        );
        self.render_section(
            "Model Selector",
            &Self::model_selector_shortcuts(),
            right_sections[6],
            buf,
            theme,
        );

        // Footer
        let footer_area = Rect {
            x: inner.x,
            y: inner.y + inner.height.saturating_sub(2),
            width: inner.width,
            height: 1,
        };
        Paragraph::new("[Press ? or any key to close]")
            .style(Style::default().fg(theme.text_muted))
            .alignment(Alignment::Center)
            .render(footer_area, buf);
    }
}

impl Default for HelpOverlay {
    fn default() -> Self {
        Self::new()
    }
}
