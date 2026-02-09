//! Placeholder view for unimplemented features

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Placeholder view - shows a message
pub struct PlaceholderView {
    message: String,
}

impl PlaceholderView {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl super::ViewTrait for PlaceholderView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title("Coming Soon")
            .borders(Borders::ALL);

        let text = format!(
            "{}\n\nThis feature is coming soon!\n\n[b/Esc] Back to Launcher  [Tab] Switch View  [q] Quit",
            self.message
        );

        let paragraph = Paragraph::new(text)
            .block(block)
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Center);

        frame.render_widget(paragraph, area);

        Ok(())
    }

    fn handle_key(&mut self, key: crossterm::event::KeyCode) -> Result<super::ViewAction> {
        // Placeholder view doesn't handle keys - let app handle them
        Ok(super::ViewAction::Continue)
    }
}
