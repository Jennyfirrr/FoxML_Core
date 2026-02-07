//! Chart widget for historical metrics

use ratatui::prelude::*;
use ratatui::widgets::*;

/// Chart widget
pub struct Chart {
    data: Vec<f64>,
}

impl Chart {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Performance Chart")
            .borders(Borders::ALL);

        // TODO: Render ASCII chart
        let text = "Chart - coming soon";
        let paragraph = Paragraph::new(text).block(block);
        paragraph.render(area, buf);
    }
}
