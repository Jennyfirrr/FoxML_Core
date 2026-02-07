//! Launcher view - main entry screen with menu and live dashboard
//!
//! Modern launcher with ASCII art logo, grouped menu, and live status panel
//! showing real-time training and trading data.

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Paragraph};

use crate::launcher::menu::MainMenu;
use crate::launcher::live_dashboard::LiveDashboard;
use crate::themes::Theme;
use crate::ui::borders::Separators;
use crate::ui::panels::Panel;

/// ASCII art logo for Fox ML - simplified block text
const FOXML_LOGO: &[&str] = &[
    "███████╗ ██████╗ ██╗  ██╗    ███╗   ███╗██╗     ",
    "██╔════╝██╔═══██╗╚██╗██╔╝    ████╗ ████║██║     ",
    "█████╗  ██║   ██║ ╚███╔╝     ██╔████╔██║██║     ",
    "██╔══╝  ██║   ██║ ██╔██╗     ██║╚██╔╝██║██║     ",
    "██║     ╚██████╔╝██╔╝ ██╗    ██║ ╚═╝ ██║███████╗",
    "╚═╝      ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝╚══════╝",
    "",
    "Intelligent Trading Research Platform",
];

/// Compact logo for status bar
const FOXML_COMPACT: &str = "◆ FOX ML";

/// Launcher view - main menu with live dashboard
pub struct LauncherView {
    menu: MainMenu,
    live_dashboard: LiveDashboard,
    theme: Theme,
    bridge_connected: bool,
}

impl LauncherView {
    pub fn new() -> Self {
        Self {
            menu: MainMenu::new(),
            live_dashboard: LiveDashboard::new(),
            theme: Theme::load(),
            bridge_connected: false,
        }
    }

    /// Update live dashboard (call periodically)
    pub async fn update_status(&mut self) {
        // Poll training events (synchronous, reads from file)
        self.live_dashboard.poll_training_events();
        // Update bridge data (async)
        self.live_dashboard.update_bridge_data().await;
        // Sync connection status
        self.bridge_connected = self.live_dashboard.is_bridge_connected();
    }

    /// Handle key press for menu navigation (vim keybinds + arrows)
    pub fn handle_key(
        &mut self,
        key: crossterm::event::KeyCode,
    ) -> Option<crate::launcher::menu::MenuAction> {
        match key {
            crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                self.menu.move_up();
                None
            }
            crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                self.menu.move_down();
                None
            }
            crossterm::event::KeyCode::Char('h') => {
                // Move to previous group (vim left)
                self.menu.move_to_previous_group();
                None
            }
            crossterm::event::KeyCode::Char('l') => {
                // Move to next group (vim right)
                self.menu.move_to_next_group();
                None
            }
            crossterm::event::KeyCode::Char('g') => {
                // Go to top (vim gg)
                self.menu.move_to_top();
                None
            }
            crossterm::event::KeyCode::Char('G') => {
                // Go to bottom (vim shift+G)
                self.menu.move_to_bottom();
                None
            }
            crossterm::event::KeyCode::Enter => {
                // Return selected action
                self.menu.get_selected_action()
            }
            _ => None,
        }
    }

    /// Render the header with logo
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let logo_lines: Vec<Line> = FOXML_LOGO
            .iter()
            .map(|&line| {
                if line.contains("Intelligent") {
                    // Tagline - secondary text color
                    Line::from(Span::styled(line, Style::default().fg(self.theme.text_secondary)))
                } else if line.is_empty() {
                    Line::from("")
                } else {
                    // Block text - accent color
                    Line::from(Span::styled(line, Style::default().fg(self.theme.accent)))
                }
            })
            .collect();

        let logo = Paragraph::new(logo_lines).alignment(Alignment::Center);
        frame.render_widget(logo, area);
    }

    /// Render the footer with keybindings
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = vec![
            ("↑↓/jk", "Navigate"),
            ("Enter", "Select"),
            ("Tab", "Switch"),
            ("?", "Help"),
            ("q", "Quit"),
        ];

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            }
            spans.push(Span::styled(
                format!("[{}]", key),
                Style::default().fg(self.theme.accent),
            ));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_secondary),
            ));
        }

        let footer = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
        frame.render_widget(footer, area);
    }

    /// Render the status bar at the bottom
    fn render_status_bar(&self, frame: &mut Frame, area: Rect) {
        let mut spans = Vec::new();

        // Logo
        spans.push(Span::styled(
            FOXML_COMPACT,
            Style::default().fg(self.theme.accent).bold(),
        ));

        // Separator
        spans.push(Span::styled(
            " │ ",
            Style::default().fg(self.theme.border),
        ));

        // Version info
        spans.push(Span::styled(
            "v0.1.0",
            Style::default().fg(self.theme.text_muted),
        ));

        // Separator
        spans.push(Span::styled(
            " │ ",
            Style::default().fg(self.theme.border),
        ));

        // Status indicators - use actual connection status
        let (icon, text, color) = if self.bridge_connected {
            (Separators::CIRCLE_FILLED, "Connected", self.theme.success)
        } else {
            (Separators::CIRCLE_EMPTY, "Offline", self.theme.error)
        };
        spans.push(Span::styled(
            format!("{} Bridge: ", icon),
            Style::default().fg(color),
        ));
        spans.push(Span::styled(
            text,
            Style::default().fg(color),
        ));

        let status_bar =
            Paragraph::new(Line::from(spans)).style(Style::default().bg(self.theme.surface));
        frame.render_widget(status_bar, area);
    }
}

impl Default for LauncherView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for LauncherView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background with theme color
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        // Main layout: Header, Content, Status Bar
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // ASCII art header with info box
                Constraint::Min(0),     // Menu and status
                Constraint::Length(1),  // Status bar
            ])
            .split(area);

        // Render header with logo
        self.render_header(frame, main_chunks[0]);

        // Content area: Menu and Status Canvas
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60), // Menu
                Constraint::Percentage(40), // Status canvas
            ])
            .margin(1)
            .split(main_chunks[1]);

        // Menu area with footer
        let menu_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),    // Menu
                Constraint::Length(2), // Footer
            ])
            .split(content_chunks[0]);

        // Render menu with rounded panel
        let menu_panel = Panel::new(&self.theme).title("Control Center").block();
        let menu_inner = menu_panel.inner(menu_chunks[0]);
        frame.render_widget(menu_panel, menu_chunks[0]);
        self.menu.render(frame, menu_inner, &self.theme)?;

        // Footer with instructions
        self.render_footer(frame, menu_chunks[1]);

        // Live dashboard - shows training and trading status
        self.live_dashboard.render(frame, content_chunks[1], &self.theme)?;

        // Status bar at bottom
        self.render_status_bar(frame, main_chunks[2]);

        Ok(())
    }
}
