//! Confirmation dialog widget
//!
//! Renders a centered modal overlay for destructive action confirmation.

use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::themes::Theme;

/// User choice in the dialog
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialogChoice {
    Yes,
    No,
}

/// Result of handling a key in the dialog
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DialogResult {
    /// Dialog still active, no decision yet
    Pending,
    /// User confirmed the action
    Confirmed,
    /// User cancelled the action
    Cancelled,
}

/// A centered confirmation dialog overlay
pub struct ConfirmDialog {
    pub title: String,
    pub message: String,
    pub selected: DialogChoice,
}

impl ConfirmDialog {
    /// Create a new confirmation dialog
    pub fn new(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            message: message.into(),
            selected: DialogChoice::No, // Default to No for safety
        }
    }

    /// Handle a key press, returning the dialog result
    pub fn handle_key(&mut self, key: KeyCode) -> DialogResult {
        match key {
            KeyCode::Left | KeyCode::Char('h') => {
                self.selected = DialogChoice::Yes;
                DialogResult::Pending
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.selected = DialogChoice::No;
                DialogResult::Pending
            }
            KeyCode::Char('y') | KeyCode::Char('Y') => DialogResult::Confirmed,
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => DialogResult::Cancelled,
            KeyCode::Enter => match self.selected {
                DialogChoice::Yes => DialogResult::Confirmed,
                DialogChoice::No => DialogResult::Cancelled,
            },
            KeyCode::Tab => {
                self.selected = match self.selected {
                    DialogChoice::Yes => DialogChoice::No,
                    DialogChoice::No => DialogChoice::Yes,
                };
                DialogResult::Pending
            }
            _ => DialogResult::Pending,
        }
    }

    /// Render the dialog as a centered overlay
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        // Calculate centered dialog dimensions (40% width, ~7 lines height)
        let dialog_width = (area.width * 2 / 5).max(30).min(area.width);
        let dialog_height = 7_u16.min(area.height);
        let x = area.x + (area.width.saturating_sub(dialog_width)) / 2;
        let y = area.y + (area.height.saturating_sub(dialog_height)) / 2;
        let dialog_area = Rect::new(x, y, dialog_width, dialog_height);

        // Dim background
        for dy in area.top()..area.bottom() {
            for dx in area.left()..area.right() {
                if let Some(cell) = buf.cell_mut((dx, dy)) {
                    if let Color::Rgb(r, g, b) = cell.fg {
                        cell.fg = Color::Rgb(r / 3, g / 3, b / 3);
                    }
                    if let Color::Rgb(r, g, b) = cell.bg {
                        cell.bg = Color::Rgb(r / 3, g / 3, b / 3);
                    }
                }
            }
        }

        // Clear dialog area
        Clear.render(dialog_area, buf);

        // Draw border
        let block = Block::default()
            .title(format!(" {} ", self.title))
            .title_style(Style::default().fg(theme.warning).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.warning))
            .style(Style::default().bg(theme.surface));
        let inner = block.inner(dialog_area);
        block.render(dialog_area, buf);

        // Render message and buttons inside
        if inner.height >= 3 {
            // Message
            let msg = Paragraph::new(Line::from(Span::styled(
                &self.message,
                Style::default().fg(theme.text_primary),
            )))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
            let msg_area = Rect::new(inner.x, inner.y, inner.width, inner.height.saturating_sub(2));
            msg.render(msg_area, buf);

            // Buttons on last line
            let btn_y = inner.y + inner.height.saturating_sub(1);
            let btn_area = Rect::new(inner.x, btn_y, inner.width, 1);

            let yes_style = if self.selected == DialogChoice::Yes {
                Style::default().fg(theme.background).bg(theme.warning).bold()
            } else {
                Style::default().fg(theme.text_muted)
            };
            let no_style = if self.selected == DialogChoice::No {
                Style::default().fg(theme.background).bg(theme.accent).bold()
            } else {
                Style::default().fg(theme.text_muted)
            };

            let buttons = Line::from(vec![
                Span::styled("  [ Yes ] ", yes_style),
                Span::styled("   ", Style::default()),
                Span::styled("  [ No ] ", no_style),
            ]);
            let btn_paragraph = Paragraph::new(buttons).alignment(Alignment::Center);
            btn_paragraph.render(btn_area, buf);
        }
    }
}
