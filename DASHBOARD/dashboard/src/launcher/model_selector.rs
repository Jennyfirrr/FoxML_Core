//! Model selector for LIVE_TRADING

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Model selector
pub struct ModelSelector {
    run_id: Option<String>,
}

impl ModelSelector {
    pub fn new() -> Self {
        Self { run_id: None }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title("Model Selector - Choose Models for LIVE_TRADING")
            .borders(Borders::ALL);

        let text = Paragraph::new("Model selector - coming soon")
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(text, area);

        Ok(())
    }
}
