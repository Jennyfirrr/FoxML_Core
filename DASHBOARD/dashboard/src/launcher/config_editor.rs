//! YAML config editor with full text editing

use anyhow::{Context, Result};
use ratatui::prelude::*;
use ratatui::text::{Line, Span};
use ratatui::widgets::*;
use ropey::Rope;
use serde_yaml;
use std::fs;
use std::path::PathBuf;

/// YAML config editor with ropey-based text editing
pub struct ConfigEditor {
    file_path: PathBuf,
    content: Rope,
    cursor_line: usize,
    cursor_col: usize,
    scroll_offset: usize,
    modified: bool,
}

impl ConfigEditor {
    pub fn new(file_path: String) -> Result<Self> {
        let path = PathBuf::from(&file_path);
        let content = if path.exists() {
            Rope::from(fs::read_to_string(&path)?)
        } else {
            Rope::new()
        };

        Ok(Self {
            file_path: path,
            content,
            cursor_line: 0,
            cursor_col: 0,
            scroll_offset: 0,
            modified: false,
        })
    }

    /// Load file
    pub fn load(&mut self) -> Result<()> {
        if self.file_path.exists() {
            self.content = Rope::from(fs::read_to_string(&self.file_path)?);
            self.modified = false;
            self.cursor_line = 0;
            self.cursor_col = 0;
        }
        Ok(())
    }

    /// Save file with YAML validation and backup
    pub fn save(&mut self) -> Result<()> {
        // Validate YAML before saving
        let content_str = self.content.to_string();
        match serde_yaml::from_str::<serde_yaml::Value>(&content_str) {
            Ok(_) => {
                // Valid YAML - create backup and save
                if self.file_path.exists() {
                    let backup_path = self.file_path.with_extension("yaml.bak");
                    fs::copy(&self.file_path, &backup_path)
                        .context("Failed to create backup")?;
                }

                fs::write(&self.file_path, content_str)?;
                self.modified = false;
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Invalid YAML: {}", e);
            }
        }
    }

    /// Get current line
    fn current_line(&self) -> String {
        if self.cursor_line < self.content.len_lines() {
            self.content.line(self.cursor_line).to_string()
        } else {
            String::new()
        }
    }

    /// Get cursor position in rope
    fn cursor_char_idx(&self) -> usize {
        if self.cursor_line >= self.content.len_lines() {
            return self.content.len_chars();
        }
        
        let line_start = self.content.line_to_char(self.cursor_line);
        let line = self.content.line(self.cursor_line);
        let line_len = line.len_chars();
        let col = self.cursor_col.min(line_len);
        line_start + col
    }

    /// Move cursor up (k or Up)
    pub fn move_up(&mut self) {
        if self.cursor_line > 0 {
            self.cursor_line -= 1;
            // Adjust column to not exceed line length
            let line = self.content.line(self.cursor_line);
            self.cursor_col = self.cursor_col.min(line.len_chars());
        }
    }

    /// Move cursor down (j or Down)
    pub fn move_down(&mut self) {
        if self.cursor_line < self.content.len_lines().saturating_sub(1) {
            self.cursor_line += 1;
            // Adjust column to not exceed line length
            let line = self.content.line(self.cursor_line);
            self.cursor_col = self.cursor_col.min(line.len_chars());
        }
    }

    /// Move cursor left (h or Left)
    pub fn move_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_line > 0 {
            // Move to end of previous line
            self.cursor_line -= 1;
            let line = self.content.line(self.cursor_line);
            self.cursor_col = line.len_chars();
        }
    }

    /// Move cursor right (l or Right)
    pub fn move_right(&mut self) {
        let line = if self.cursor_line < self.content.len_lines() {
            self.content.line(self.cursor_line)
        } else {
            return;
        };
        
        if self.cursor_col < line.len_chars() {
            self.cursor_col += 1;
        } else if self.cursor_line < self.content.len_lines().saturating_sub(1) {
            // Move to start of next line
            self.cursor_line += 1;
            self.cursor_col = 0;
        }
    }

    /// Move to start of line (0)
    pub fn move_to_line_start(&mut self) {
        self.cursor_col = 0;
    }

    /// Move to end of line ($)
    pub fn move_to_line_end(&mut self) {
        if self.cursor_line < self.content.len_lines() {
            let line = self.content.line(self.cursor_line);
            self.cursor_col = line.len_chars();
        }
    }

    /// Move to top of file (gg)
    pub fn move_to_top(&mut self) {
        self.cursor_line = 0;
        self.cursor_col = 0;
    }

    /// Move to bottom of file (G)
    pub fn move_to_bottom(&mut self) {
        if self.content.len_lines() > 0 {
            self.cursor_line = self.content.len_lines() - 1;
            let line = self.content.line(self.cursor_line);
            self.cursor_col = line.len_chars();
        }
    }

    /// Insert character at cursor
    pub fn insert_char(&mut self, c: char) {
        let idx = self.cursor_char_idx();
        self.content.insert_char(idx, c);
        self.cursor_col += 1;
        self.modified = true;
    }

    /// Insert newline at cursor
    pub fn insert_newline(&mut self) {
        let idx = self.cursor_char_idx();
        self.content.insert_char(idx, '\n');
        self.cursor_line += 1;
        self.cursor_col = 0;
        self.modified = true;
    }

    /// Delete character backward (backspace)
    pub fn delete_backward(&mut self) {
        if self.cursor_col > 0 {
            let idx = self.cursor_char_idx();
            self.content.remove(idx - 1..idx);
            self.cursor_col -= 1;
            self.modified = true;
        } else if self.cursor_line > 0 {
            // Join with previous line
            let prev_line_len = self.content.line(self.cursor_line - 1).len_chars();
            let idx = self.cursor_char_idx();
            self.content.remove(idx - 1..idx);
            self.cursor_line -= 1;
            self.cursor_col = prev_line_len;
            self.modified = true;
        }
    }

    /// Delete character forward (delete)
    pub fn delete_forward(&mut self) {
        let idx = self.cursor_char_idx();
        if idx < self.content.len_chars() {
            self.content.remove(idx..idx + 1);
            self.modified = true;
        }
    }

    /// Render editor
    pub fn render(&mut self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) -> Result<()> {
        // Update scroll to keep cursor visible
        let visible_lines = area.height.saturating_sub(2) as usize; // -2 for borders
        if self.cursor_line < self.scroll_offset {
            self.scroll_offset = self.cursor_line;
        } else if self.cursor_line >= self.scroll_offset + visible_lines {
            self.scroll_offset = self.cursor_line.saturating_sub(visible_lines - 1);
        }

        let title = if self.modified {
            format!("Config Editor: {} *", self.file_path.display())
        } else {
            format!("Config Editor: {}", self.file_path.display())
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.secondary_text));

        // Build text with line numbers and cursor
        let mut lines = Vec::new();
        let start_line = self.scroll_offset;
        let end_line = (start_line + visible_lines).min(self.content.len_lines());

        for line_idx in start_line..end_line {
            let line = self.content.line(line_idx);
            let line_str = line.to_string();
            let line_num = format!("{:4} ", line_idx + 1);
            
            // Highlight current line
            let style = if line_idx == self.cursor_line {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            // Add cursor indicator
            let mut display_line = format!("{}{}", line_num, line_str);
            if line_idx == self.cursor_line {
                // Insert cursor at cursor_col position
                let cursor_pos = line_num.len() + self.cursor_col;
                if cursor_pos <= display_line.len() {
                    display_line.insert(cursor_pos, '█');
                } else {
                    display_line.push('█');
                }
            }

            let span = Span::styled(display_line, style);
            lines.push(Line::from(vec![span]));
        }

        let paragraph = Paragraph::new(lines)
            .block(block)
            .scroll((0, self.scroll_offset as u16));

        frame.render_widget(paragraph, area);

        // Footer with status and instructions
        let status = format!(
            "Line: {} Col: {} | [Ctrl+S] Save [q/Esc] Quit [hjkl] Move [i] Insert",
            self.cursor_line + 1,
            self.cursor_col + 1
        );
        let footer = Paragraph::new(status)
            .style(Style::default().fg(theme.secondary_text));
        let footer_area = Rect::new(area.x, area.y + area.height - 1, area.width, 1);
        frame.render_widget(footer, footer_area);

        Ok(())
    }
}
