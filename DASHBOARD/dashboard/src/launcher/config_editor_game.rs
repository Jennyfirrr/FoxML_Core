//! Video game-style interactive config editor

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Video game-style config editor
pub struct GameStyleConfigEditor {
    file_path: String,
}

impl GameStyleConfigEditor {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title(format!("Interactive Config Editor: {}", self.file_path))
            .borders(Borders::ALL);

        let text = Paragraph::new("Video game-style config editor - coming soon")
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(text, area);

        Ok(())
    }
}
