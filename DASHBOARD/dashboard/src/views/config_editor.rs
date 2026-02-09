//! Config Browser View - browse and preview experiment configs
//!
//! Supports two editing modes:
//! - External editor ($EDITOR): TUI suspends, editor gets full terminal access
//! - Inline editor: built-in YAML editor with cursor/edit/save/validation

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;

use crate::launcher::config_editor::ConfigEditor;
use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Config browser view
pub struct ConfigEditorView {
    theme: Theme,
    // Config list
    configs: Vec<ConfigEntry>,
    selected: usize,
    scroll: usize,
    // Preview
    preview_content: Vec<String>,
    preview_scroll: usize,
    // Inline editor (None = browser mode, Some = editing mode)
    inline_editor: Option<ConfigEditor>,
    // Status
    message: Option<(String, bool)>,
}

/// Config entry
#[derive(Debug, Clone)]
struct ConfigEntry {
    name: String,
    path: PathBuf,
    is_experiment: bool,
}

impl ConfigEditorView {
    pub fn new() -> Self {
        let mut view = Self {
            theme: Theme::load(),
            configs: Vec::new(),
            selected: 0,
            scroll: 0,
            preview_content: Vec::new(),
            preview_scroll: 0,
            inline_editor: None,
            message: None,
        };
        view.scan_configs();
        view.update_preview();
        view
    }

    /// Scan for available configs
    fn scan_configs(&mut self) {
        let mut configs = Vec::new();

        // Scan experiment configs
        let experiments_dir = crate::config::config_dir().join("experiments");
        if experiments_dir.exists() {
            if let Ok(entries) = fs::read_dir(&experiments_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
                        if let Some(stem) = path.file_stem() {
                            configs.push(ConfigEntry {
                                name: stem.to_string_lossy().to_string(),
                                path,
                                is_experiment: true,
                            });
                        }
                    }
                }
            }
        }

        // Sort experiments
        configs.sort_by(|a, b| a.name.cmp(&b.name));

        // Add pipeline configs section
        let pipeline_configs = [
            ("pipeline.yaml", "CONFIG/pipeline/pipeline.yaml"),
            ("training/families.yaml", "CONFIG/pipeline/training/families.yaml"),
            ("intelligent_training_config.yaml", "CONFIG/intelligent_training_config.yaml"),
        ];

        for (name, path_str) in pipeline_configs {
            let path = PathBuf::from(path_str);
            if path.exists() {
                configs.push(ConfigEntry {
                    name: name.to_string(),
                    path,
                    is_experiment: false,
                });
            }
        }

        self.configs = configs;
    }

    /// Update preview for selected config
    pub fn update_preview(&mut self) {
        self.preview_content.clear();
        self.preview_scroll = 0;

        if self.configs.is_empty() {
            return;
        }

        let entry = &self.configs[self.selected];
        match fs::read_to_string(&entry.path) {
            Ok(content) => {
                self.preview_content = content
                    .lines()
                    .take(500)
                    .map(|l| l.to_string())
                    .collect();
            }
            Err(e) => {
                self.preview_content = vec![format!("Error reading file: {}", e)];
            }
        }
    }

    /// Check if the inline editor is currently active
    pub fn has_inline_editor(&self) -> bool {
        self.inline_editor.is_some()
    }

    /// Get path of the currently selected config
    fn selected_config_path(&self) -> Option<PathBuf> {
        self.configs.get(self.selected).map(|e| e.path.clone())
    }

    /// Open file (for compatibility with app.rs)
    pub fn open(&mut self, path: PathBuf) -> Result<()> {
        // Find the config in our list
        for (i, config) in self.configs.iter().enumerate() {
            if config.path == path {
                self.selected = i;
                self.update_preview();
                return Ok(());
            }
        }
        // If not found, just refresh and show first
        self.scan_configs();
        self.update_preview();
        Ok(())
    }

    /// Render config list
    fn render_list(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme)
            .title("Configs [j/k to navigate]")
            .block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.configs.is_empty() {
            let text = Paragraph::new("No configs found")
                .style(Style::default().fg(self.theme.text_muted));
            frame.render_widget(text, inner);
            return;
        }

        let visible_height = inner.height as usize;
        let start = self.scroll;
        let end = (start + visible_height).min(self.configs.len());

        // Track section headers
        let mut last_is_experiment: Option<bool> = None;
        let mut items: Vec<ListItem> = Vec::new();

        for (i, entry) in self.configs[start..end].iter().enumerate() {
            let idx = start + i;
            let is_selected = idx == self.selected;

            // Add section header if type changes
            if last_is_experiment != Some(entry.is_experiment) {
                if !items.is_empty() {
                    items.push(ListItem::new(""));
                }
                let header = if entry.is_experiment {
                    "── Experiments ──"
                } else {
                    "── Pipeline Configs ──"
                };
                items.push(ListItem::new(Line::from(Span::styled(
                    header,
                    Style::default().fg(self.theme.text_muted).bold(),
                ))));
                last_is_experiment = Some(entry.is_experiment);
            }

            let style = if is_selected {
                Style::default().fg(self.theme.accent).bg(self.theme.surface)
            } else {
                Style::default().fg(self.theme.text_primary)
            };

            let prefix = if is_selected { "> " } else { "  " };
            items.push(ListItem::new(format!("{}{}", prefix, entry.name)).style(style));
        }

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render config preview
    fn render_preview(&self, frame: &mut Frame, area: Rect) {
        let title = if self.configs.is_empty() {
            "Preview".to_string()
        } else {
            let entry = &self.configs[self.selected];
            format!("Preview: {} (read-only)", entry.path.display())
        };

        let block = Panel::new(&self.theme).title(&title).block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.preview_content.is_empty() {
            return;
        }

        let visible_height = inner.height as usize;
        let start = self.preview_scroll;
        let end = (start + visible_height).min(self.preview_content.len());

        let lines: Vec<Line> = self.preview_content[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let line_num = start + i + 1;
                // Simple YAML syntax highlighting
                let style = if line.trim().starts_with('#') {
                    Style::default().fg(self.theme.text_muted)
                } else if line.contains(':') && !line.trim().starts_with('-') {
                    Style::default().fg(self.theme.accent)
                } else {
                    Style::default().fg(self.theme.text_secondary)
                };

                Line::from(vec![
                    Span::styled(format!("{:4} ", line_num), Style::default().fg(self.theme.text_muted)),
                    Span::styled(line.as_str(), style),
                ])
            })
            .collect();

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);
    }

    /// Render footer
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = if self.inline_editor.is_some() {
            vec![
                ("[F2]", "Save"),
                ("[Esc]", "Close editor"),
                ("[arrows]", "Move"),
            ]
        } else {
            vec![
                ("[j/k]", "Navigate"),
                ("[PgUp/PgDn]", "Scroll preview"),
                ("[e]", "$EDITOR"),
                ("[i]", "Inline edit"),
                ("[r]", "Refresh"),
                ("[q]", "Back"),
            ]
        };

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  ", Style::default()));
            }
            spans.push(Span::styled(*key, Style::default().fg(self.theme.accent)));
            spans.push(Span::styled(format!(" {}", desc), Style::default().fg(self.theme.text_muted)));
        }

        // Show message
        if let Some((msg, is_error)) = &self.message {
            spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            spans.push(Span::styled(
                msg.as_str(),
                Style::default().fg(if *is_error { self.theme.error } else { self.theme.success }),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for ConfigEditorView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for ConfigEditorView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .margin(1)
            .split(area);

        // If inline editor is active, render it full-screen in the content area
        if let Some(ref mut editor) = self.inline_editor {
            let theme = self.theme.clone();
            let _ = editor.render(frame, chunks[0], &theme);
            self.render_footer(frame, chunks[1]);
            return Ok(());
        }

        // Split content into list and preview
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(30), // Config list
                Constraint::Percentage(70), // Preview
            ])
            .split(chunks[0]);

        self.render_list(frame, content_chunks[0]);
        self.render_preview(frame, content_chunks[1]);
        self.render_footer(frame, chunks[1]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;

        // If inline editor is active, delegate keys to it
        if let Some(ref mut editor) = self.inline_editor {
            match key {
                KeyCode::Esc => {
                    // Exit inline editor
                    self.inline_editor = None;
                    self.update_preview();
                    self.message = Some(("Editor closed".to_string(), false));
                    return Ok(ViewAction::Continue);
                }
                KeyCode::F(2) => {
                    // Save
                    match editor.save() {
                        Ok(_) => {
                            self.message = Some(("Saved".to_string(), false));
                            self.update_preview();
                        }
                        Err(e) => {
                            self.message = Some((format!("Save failed: {}", e), true));
                        }
                    }
                    return Ok(ViewAction::Continue);
                }
                // Movement keys
                KeyCode::Up => { editor.move_up(); }
                KeyCode::Down => { editor.move_down(); }
                KeyCode::Left => { editor.move_left(); }
                KeyCode::Right => { editor.move_right(); }
                KeyCode::Home => { editor.move_to_line_start(); }
                KeyCode::End => { editor.move_to_line_end(); }
                // Editing keys
                KeyCode::Char(c) => { editor.insert_char(c); }
                KeyCode::Enter => { editor.insert_newline(); }
                KeyCode::Backspace => { editor.delete_backward(); }
                KeyCode::Delete => { editor.delete_forward(); }
                _ => {}
            }
            return Ok(ViewAction::Continue);
        }

        // Browser mode
        // Clear message
        self.message = None;

        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(ViewAction::Back);
            }
            // List navigation
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    if self.selected < self.scroll {
                        self.scroll = self.selected;
                    }
                    self.update_preview();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.configs.len().saturating_sub(1) {
                    self.selected += 1;
                    self.update_preview();
                }
            }
            // Preview scrolling
            KeyCode::PageUp => {
                self.preview_scroll = self.preview_scroll.saturating_sub(20);
            }
            KeyCode::PageDown => {
                let max_scroll = self.preview_content.len().saturating_sub(20);
                self.preview_scroll = (self.preview_scroll + 20).min(max_scroll);
            }
            // Open in external editor (TUI suspension)
            KeyCode::Char('e') => {
                if let Some(path) = self.selected_config_path() {
                    return Ok(ViewAction::SpawnEditor(path));
                }
            }
            // Open inline editor
            KeyCode::Char('i') => {
                if let Some(path) = self.selected_config_path() {
                    match ConfigEditor::new(path.to_string_lossy().to_string()) {
                        Ok(editor) => {
                            self.inline_editor = Some(editor);
                            self.message = Some(("Inline editor opened".to_string(), false));
                        }
                        Err(e) => {
                            self.message = Some((format!("Failed to open editor: {}", e), true));
                        }
                    }
                }
            }
            KeyCode::Char('r') => {
                self.scan_configs();
                self.update_preview();
                self.message = Some(("Refreshed".to_string(), false));
            }
            _ => {}
        }

        Ok(ViewAction::Continue)
    }
}
