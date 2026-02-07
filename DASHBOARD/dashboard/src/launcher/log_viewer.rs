//! Log viewer

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Log viewer
pub struct LogViewer {
    log_path: Option<PathBuf>,
    lines: Vec<String>,
    scroll: usize,
}

impl LogViewer {
    pub fn new() -> Self {
        Self {
            log_path: None,
            lines: Vec::new(),
            scroll: 0,
        }
    }

    /// Load log file
    pub fn load(&mut self, path: String) -> Result<()> {
        let log_path = PathBuf::from(&path);
        if log_path.exists() {
            let file = fs::File::open(&log_path)?;
            let reader = BufReader::new(file);
            self.lines = reader
                .lines()
                .filter_map(|l| l.ok())
                .collect();
            self.log_path = Some(log_path);
            self.scroll = self.lines.len().saturating_sub(20); // Show last 20 lines
        }
        Ok(())
    }

    /// Load journalctl logs for service
    pub fn load_service_logs(&mut self, service_name: &str) -> Result<()> {
        let output = std::process::Command::new("journalctl")
            .args(&["-u", service_name, "--no-pager", "-n", "100"])
            .output()?;
        
        self.lines = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|s| s.to_string())
            .collect();
        
        self.scroll = self.lines.len().saturating_sub(20);
        Ok(())
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let title = if let Some(ref path) = self.log_path {
            format!("Log Viewer: {}", path.display())
        } else {
            "Log Viewer".to_string()
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL);

        let visible_lines: Vec<String> = self
            .lines
            .iter()
            .skip(self.scroll)
            .take(area.height as usize - 2)
            .map(|s| s.clone())
            .collect();

        let items: Vec<ListItem> = visible_lines
            .iter()
            .map(|line| ListItem::new(line.as_str()))
            .collect();

        let list = List::new(items).block(block);
        frame.render_widget(list, area);

        Ok(())
    }
}

