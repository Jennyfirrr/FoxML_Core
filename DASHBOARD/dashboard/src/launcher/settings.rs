//! Dashboard settings

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Dashboard settings
pub struct Settings;

impl Settings {
    pub fn new() -> Self {
        Self
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title("Dashboard Settings")
            .borders(Borders::ALL);

        let text = Paragraph::new("Settings - coming soon")
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(text, area);

        Ok(())
    }
}
