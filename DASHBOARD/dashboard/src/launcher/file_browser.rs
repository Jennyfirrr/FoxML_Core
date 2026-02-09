//! File browser

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;

/// File entry
#[derive(Debug, Clone)]
pub struct FileEntry {
    name: String,
    path: PathBuf,
    is_dir: bool,
}

/// File browser
pub struct FileBrowser {
    current_path: PathBuf,
    entries: Vec<FileEntry>,
    selected: usize,
}

impl FileBrowser {
    pub fn new() -> Self {
        let current_path = crate::config::results_dir();
        let entries = Self::list_directory(&current_path);
        
        Self {
            current_path,
            entries,
            selected: 0,
        }
    }

    /// List directory contents
    fn list_directory(path: &PathBuf) -> Vec<FileEntry> {
        let mut entries = Vec::new();
        
        if !path.exists() || !path.is_dir() {
            return entries;
        }

        // Add parent directory
        if let Some(parent) = path.parent() {
            entries.push(FileEntry {
                name: "..".to_string(),
                path: parent.to_path_buf(),
                is_dir: true,
            });
        }

        // List directory contents
        if let Ok(read_dir) = fs::read_dir(path) {
            let mut dirs = Vec::new();
            let mut files = Vec::new();

            for entry in read_dir.filter_map(|e| e.ok()) {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = path.is_dir();

                if is_dir {
                    dirs.push(FileEntry { name, path, is_dir: true });
                } else {
                    files.push(FileEntry { name, path, is_dir: false });
                }
            }

            // Sort: directories first, then files
            dirs.sort_by(|a, b| a.name.cmp(&b.name));
            files.sort_by(|a, b| a.name.cmp(&b.name));
            
            entries.extend(dirs);
            entries.extend(files);
        }

        entries
    }

    /// Navigate to directory
    pub fn navigate(&mut self, path: &PathBuf) {
        self.current_path = path.clone();
        self.entries = Self::list_directory(&self.current_path);
        self.selected = 0;
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title(format!("File Browser: {}", self.current_path.display()))
            .borders(Borders::ALL);

        let items: Vec<ListItem> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let prefix = if i == self.selected { "> " } else { "  " };
                let icon = if entry.is_dir { "📁 " } else { "📄 " };
                ListItem::new(format!("{}{}{}", prefix, icon, entry.name))
            })
            .collect();

        let list = List::new(items)
            .block(block)
            .highlight_style(Style::default().fg(Color::Yellow));

        let mut state = ratatui::widgets::ListState::default();
        state.select(Some(self.selected));
        frame.render_stateful_widget(list, area, &mut state);

        Ok(())
    }
}

