//! File Browser View - navigate and preview files
//!
//! Browse project directories with file preview and quick navigation.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use chrono::{DateTime, Local};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Quick jump destinations
const QUICK_JUMPS: &[(&str, &str)] = &[
    ("1", "CONFIG"),
    ("2", "RESULTS"),
    ("3", "TRAINING"),
    ("4", "LIVE_TRADING"),
    ("5", "DASHBOARD"),
];

/// File entry with metadata
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
    pub modified: Option<SystemTime>,
}

impl FileEntry {
    fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());

        let metadata = fs::metadata(&path).ok();
        let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
        let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
        let modified = metadata.and_then(|m| m.modified().ok());

        Self {
            name,
            path,
            is_dir,
            size,
            modified,
        }
    }

    fn size_str(&self) -> String {
        if self.is_dir {
            "-".to_string()
        } else if self.size < 1024 {
            format!("{}B", self.size)
        } else if self.size < 1024 * 1024 {
            format!("{:.1}KB", self.size as f64 / 1024.0)
        } else if self.size < 1024 * 1024 * 1024 {
            format!("{:.1}MB", self.size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1}GB", self.size as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    fn modified_str(&self) -> String {
        self.modified
            .map(|t| {
                let dt: DateTime<Local> = t.into();
                dt.format("%Y-%m-%d %H:%M").to_string()
            })
            .unwrap_or_else(|| "-".to_string())
    }
}

/// File browser view
pub struct FileBrowserView {
    theme: Theme,
    current_path: PathBuf,
    entries: Vec<FileEntry>,
    selected: usize,
    scroll: usize,
    preview_lines: Vec<String>,
    preview_scroll: usize,
    show_hidden: bool,
}

impl FileBrowserView {
    pub fn new() -> Self {
        let current_path = PathBuf::from(".");
        let mut view = Self {
            theme: Theme::load(),
            current_path,
            entries: Vec::new(),
            selected: 0,
            scroll: 0,
            preview_lines: Vec::new(),
            preview_scroll: 0,
            show_hidden: false,
        };
        view.refresh_entries();
        view
    }

    /// Refresh directory listing
    fn refresh_entries(&mut self) {
        let mut entries = Vec::new();

        // Add parent directory entry
        if let Some(parent) = self.current_path.parent() {
            entries.push(FileEntry {
                name: "..".to_string(),
                path: parent.to_path_buf(),
                is_dir: true,
                size: 0,
                modified: None,
            });
        }

        // List directory contents
        if let Ok(read_dir) = fs::read_dir(&self.current_path) {
            let mut dirs = Vec::new();
            let mut files = Vec::new();

            for entry in read_dir.filter_map(|e| e.ok()) {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden files unless enabled
                if !self.show_hidden && name.starts_with('.') {
                    continue;
                }

                let file_entry = FileEntry::from_path(path);
                if file_entry.is_dir {
                    dirs.push(file_entry);
                } else {
                    files.push(file_entry);
                }
            }

            // Sort alphabetically
            dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
            files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

            entries.extend(dirs);
            entries.extend(files);
        }

        self.entries = entries;
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        self.update_preview();
    }

    /// Update file preview
    fn update_preview(&mut self) {
        self.preview_lines.clear();
        self.preview_scroll = 0;

        if self.entries.is_empty() {
            return;
        }

        let entry = &self.entries[self.selected];

        if entry.is_dir {
            // For directories, show contents summary
            if let Ok(read_dir) = fs::read_dir(&entry.path) {
                let count = read_dir.count();
                self.preview_lines.push(format!("Directory: {} items", count));
            }
            return;
        }

        // Check if file is readable as text
        let extension = entry
            .path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let is_text = matches!(
            extension.as_str(),
            "txt"
                | "md"
                | "rs"
                | "py"
                | "yaml"
                | "yml"
                | "toml"
                | "json"
                | "sh"
                | "bash"
                | "css"
                | "html"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "sql"
                | "log"
                | "cfg"
                | "conf"
                | "ini"
                | ""
        );

        if !is_text && entry.size > 100_000 {
            self.preview_lines
                .push("Binary or large file - no preview".to_string());
            return;
        }

        // Read file content
        match fs::read_to_string(&entry.path) {
            Ok(content) => {
                self.preview_lines = content
                    .lines()
                    .take(200) // Limit preview lines
                    .map(|l| {
                        if l.len() > 200 {
                            format!("{}...", &l[..200])
                        } else {
                            l.to_string()
                        }
                    })
                    .collect();
            }
            Err(_) => {
                self.preview_lines
                    .push("Unable to read file".to_string());
            }
        }
    }

    /// Navigate to a path
    fn navigate_to(&mut self, path: PathBuf) {
        if path.is_dir() {
            self.current_path = path;
            self.selected = 0;
            self.scroll = 0;
            self.refresh_entries();
        }
    }

    /// Enter selected directory or open file
    fn enter_selected(&mut self) -> super::ViewAction {
        if self.entries.is_empty() {
            return super::ViewAction::Continue;
        }

        let entry = &self.entries[self.selected];
        if entry.is_dir {
            self.navigate_to(entry.path.clone());
            return super::ViewAction::Continue;
        }

        // Check if it's a text file we can open in an editor
        if self.is_text_file(&entry.path) {
            let abs_path = if entry.path.is_absolute() {
                entry.path.clone()
            } else {
                std::env::current_dir()
                    .unwrap_or_default()
                    .join(&entry.path)
            };
            return super::ViewAction::SpawnEditor(abs_path);
        }

        // Binary file — can't open
        super::ViewAction::Continue
    }

    /// Check if a file is likely a text file based on extension
    fn is_text_file(&self, path: &PathBuf) -> bool {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        matches!(
            extension.as_str(),
            "txt"
                | "md"
                | "rs"
                | "py"
                | "yaml"
                | "yml"
                | "toml"
                | "json"
                | "sh"
                | "bash"
                | "css"
                | "html"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "sql"
                | "log"
                | "cfg"
                | "conf"
                | "ini"
                | "lock"
                | "xml"
                | "csv"
                | ""
        )
    }

    /// Render file list
    fn render_list(&self, frame: &mut Frame, area: Rect) {
        let title = format!("Files: {}", self.current_path.display());
        let block = Panel::new(&self.theme).title(&title).block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.entries.is_empty() {
            let text = Paragraph::new("Empty directory")
                .style(Style::default().fg(self.theme.text_muted))
                .alignment(Alignment::Center);
            frame.render_widget(text, inner);
            return;
        }

        let visible_height = inner.height as usize;
        let start = self.scroll;
        let end = (start + visible_height).min(self.entries.len());

        let items: Vec<ListItem> = self.entries[start..end]
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let idx = start + i;
                let is_selected = idx == self.selected;

                let icon = if entry.name == ".." {
                    ".."
                } else if entry.is_dir {
                    "/"
                } else {
                    " "
                };

                let name_color = if entry.is_dir {
                    self.theme.accent
                } else {
                    self.theme.text_primary
                };

                let content = Line::from(vec![
                    Span::styled(
                        if is_selected { ">" } else { " " },
                        Style::default().fg(self.theme.accent),
                    ),
                    Span::styled(
                        format!("{} ", icon),
                        Style::default().fg(self.theme.text_muted),
                    ),
                    Span::styled(
                        format!("{:30}", &entry.name[..30.min(entry.name.len())]),
                        Style::default().fg(name_color),
                    ),
                    Span::styled(
                        format!(" {:>8}", entry.size_str()),
                        Style::default().fg(self.theme.text_muted),
                    ),
                    Span::styled(
                        format!(" {:>10}", entry.modified_str()),
                        Style::default().fg(self.theme.text_secondary),
                    ),
                ]);

                let style = if is_selected {
                    Style::default().bg(self.theme.surface)
                } else {
                    Style::default()
                };

                ListItem::new(content).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render file preview
    fn render_preview(&self, frame: &mut Frame, area: Rect) {
        let title = if self.entries.is_empty() {
            "Preview".to_string()
        } else {
            format!("Preview: {}", self.entries[self.selected].name)
        };

        let block = Panel::new(&self.theme).title(&title).block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.preview_lines.is_empty() {
            let text = Paragraph::new("No preview available")
                .style(Style::default().fg(self.theme.text_muted))
                .alignment(Alignment::Center);
            frame.render_widget(text, inner);
            return;
        }

        let visible_height = inner.height as usize;
        let start = self.preview_scroll;
        let end = (start + visible_height).min(self.preview_lines.len());

        let lines: Vec<Line> = self.preview_lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let line_num = start + i + 1;
                Line::from(vec![
                    Span::styled(
                        format!("{:4} ", line_num),
                        Style::default().fg(self.theme.text_muted),
                    ),
                    Span::styled(line, Style::default().fg(self.theme.text_secondary)),
                ])
            })
            .collect();

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);
    }

    /// Render footer with quick jump keys
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let mut spans = Vec::new();

        // Quick jumps
        for (key, name) in QUICK_JUMPS {
            spans.push(Span::styled(
                format!("[{}]", key),
                Style::default().fg(self.theme.accent),
            ));
            spans.push(Span::styled(
                format!(" {} ", name),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        spans.push(Span::styled("  ", Style::default()));

        // Other keys
        let keybinds = [
            ("[h]", "Hidden"),
            ("[Enter]", "Open"),
            ("[q/Esc]", "Back"),
        ];

        for (key, desc) in keybinds {
            spans.push(Span::styled(key, Style::default().fg(self.theme.accent)));
            spans.push(Span::styled(
                format!(" {} ", desc),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for FileBrowserView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for FileBrowserView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .margin(1)
            .split(area);

        // Split content into list and preview
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50), // File list
                Constraint::Percentage(50), // Preview
            ])
            .split(main_chunks[0]);

        self.render_list(frame, content_chunks[0]);
        self.render_preview(frame, content_chunks[1]);
        self.render_footer(frame, main_chunks[1]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(ViewAction::Back);
            }
            // Navigation
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                    // Update scroll
                    if self.selected < self.scroll {
                        self.scroll = self.selected;
                    }
                    self.update_preview();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.entries.len().saturating_sub(1) {
                    self.selected += 1;
                    self.update_preview();
                }
            }
            KeyCode::PageUp => {
                self.preview_scroll = self.preview_scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                let max_scroll = self.preview_lines.len().saturating_sub(10);
                self.preview_scroll = (self.preview_scroll + 10).min(max_scroll);
            }
            KeyCode::Enter => {
                return Ok(self.enter_selected());
            }
            KeyCode::Backspace => {
                // Go to parent
                if let Some(parent) = self.current_path.parent() {
                    self.navigate_to(parent.to_path_buf());
                }
            }
            KeyCode::Char('h') => {
                // Toggle hidden files
                self.show_hidden = !self.show_hidden;
                self.refresh_entries();
            }
            KeyCode::Char('r') => {
                // Refresh
                self.refresh_entries();
            }
            // Quick jumps
            KeyCode::Char('1') => self.navigate_to(crate::config::config_dir()),
            KeyCode::Char('2') => self.navigate_to(crate::config::results_dir()),
            KeyCode::Char('3') => self.navigate_to(crate::config::project_root().join("TRAINING")),
            KeyCode::Char('4') => self.navigate_to(crate::config::project_root().join("LIVE_TRADING")),
            KeyCode::Char('5') => self.navigate_to(crate::config::project_root().join("DASHBOARD")),
            _ => {}
        }

        Ok(ViewAction::Continue)
    }
}
