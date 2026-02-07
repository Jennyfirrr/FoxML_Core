//! Log Viewer View - browse and tail log files
//!
//! Supports training logs, trading logs, and systemd journal.
//! Discovers all .log files and allows selection between them.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Log source options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogSource {
    TrainingLogs,
    TradingLogs,
    SystemJournal,
}

impl LogSource {
    fn label(&self) -> &str {
        match self {
            LogSource::TrainingLogs => "Training Logs",
            LogSource::TradingLogs => "Trading Logs",
            LogSource::SystemJournal => "System Journal",
        }
    }
}

/// Discovered log file entry
#[derive(Debug, Clone)]
struct LogFileEntry {
    path: PathBuf,
    display_name: String,
}

/// Log viewer view
pub struct LogViewerView {
    theme: Theme,
    source: LogSource,
    // Discovered log files for current source
    discovered_logs: Vec<LogFileEntry>,
    selected_log: usize,
    // Current log content
    log_path: Option<PathBuf>,
    lines: Vec<String>,
    scroll: usize,
    follow_mode: bool,
    last_poll: Instant,
    file_pos: u64,
    search_query: String,
    searching: bool,
    // UI state
    show_log_list: bool,
}

impl LogViewerView {
    pub fn new() -> Self {
        let mut view = Self {
            theme: Theme::load(),
            source: LogSource::TrainingLogs,
            discovered_logs: Vec::new(),
            selected_log: 0,
            log_path: None,
            lines: Vec::new(),
            scroll: 0,
            follow_mode: true,
            last_poll: Instant::now(),
            file_pos: 0,
            search_query: String::new(),
            searching: false,
            show_log_list: false,
        };
        view.discover_logs();
        view.load_selected_log();
        view
    }

    /// Discover available logs for current source
    fn discover_logs(&mut self) {
        self.discovered_logs.clear();
        self.selected_log = 0;

        match self.source {
            LogSource::TrainingLogs => {
                self.discover_training_logs();
            }
            LogSource::TradingLogs => {
                self.discover_trading_logs();
            }
            LogSource::SystemJournal => {
                // Journal doesn't have file discovery
            }
        }
    }

    /// Load logs for current source
    fn load_current_source(&mut self) {
        self.discover_logs();
        self.load_selected_log();
    }

    /// Load the currently selected log file
    fn load_selected_log(&mut self) {
        match self.source {
            LogSource::SystemJournal => {
                self.load_journal_logs();
            }
            _ => {
                if let Some(entry) = self.discovered_logs.get(self.selected_log) {
                    self.load_file(&entry.path.clone());
                } else {
                    self.lines = vec![format!("No {} found", self.source.label())];
                    self.log_path = None;
                }
            }
        }
    }

    /// Discover training log files from all runs
    fn discover_training_logs(&mut self) {
        let results_dir = PathBuf::from("RESULTS");
        if !results_dir.exists() {
            return;
        }

        // Collect all run directories recursively
        let mut all_run_paths: Vec<PathBuf> = Vec::new();

        if let Ok(subdirs) = fs::read_dir(&results_dir) {
            for subdir_entry in subdirs.filter_map(|e| e.ok()) {
                let subdir_path = subdir_entry.path();
                if !subdir_path.is_dir() {
                    continue;
                }

                // Check if this directory has manifest.json (it's a run)
                if subdir_path.join("manifest.json").exists() {
                    all_run_paths.push(subdir_path);
                    continue;
                }

                // Otherwise, scan its subdirectories
                if let Ok(entries) = fs::read_dir(&subdir_path) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if path.is_dir() && path.join("manifest.json").exists() {
                            all_run_paths.push(path);
                        }
                    }
                }
            }
        }

        // Sort by name (descending - newest first based on timestamp)
        all_run_paths.sort_by(|a, b| {
            let a_name = a.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let b_name = b.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            b_name.cmp(&a_name)
        });

        // Find log files in each run directory
        for run_path in all_run_paths.iter().take(20) {
            let run_name = run_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            // Check direct log files
            self.find_log_files_in_dir(run_path, &run_name);

            // Check logs/ subdirectory
            let logs_dir = run_path.join("logs");
            if logs_dir.exists() {
                self.find_log_files_in_dir(&logs_dir, &run_name);
            }
        }
    }

    /// Find .log files in a directory and add to discovered_logs
    fn find_log_files_in_dir(&mut self, dir: &PathBuf, run_name: &str) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "log" {
                            let file_name = path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "unknown".to_string());
                            self.discovered_logs.push(LogFileEntry {
                                path,
                                display_name: format!("{}/{}", run_name, file_name),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Discover trading log files
    fn discover_trading_logs(&mut self) {
        // Check common trading log locations
        let locations = [
            PathBuf::from("LIVE_TRADING/logs"),
            PathBuf::from("logs"),
        ];

        for log_dir in locations {
            if log_dir.exists() {
                if let Ok(entries) = fs::read_dir(&log_dir) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if path.is_file() {
                            if let Some(ext) = path.extension() {
                                if ext == "log" {
                                    let file_name = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| "unknown".to_string());
                                    self.discovered_logs.push(LogFileEntry {
                                        path,
                                        display_name: file_name,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Load systemd journal logs
    fn load_journal_logs(&mut self) {
        match std::process::Command::new("journalctl")
            .args(["--user", "-u", "foxml-trading", "--no-pager", "-n", "500"])
            .output()
        {
            Ok(output) => {
                self.lines = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(|s| s.to_string())
                    .collect();
                self.log_path = None;
                if self.follow_mode {
                    self.scroll = self.lines.len().saturating_sub(1);
                }
            }
            Err(e) => {
                self.lines = vec![format!("Failed to read journal: {}", e)];
            }
        }
    }

    /// Load a file
    fn load_file(&mut self, path: &PathBuf) {
        match fs::File::open(path) {
            Ok(file) => {
                let reader = BufReader::new(&file);
                self.lines = reader.lines().filter_map(|l| l.ok()).collect();
                self.log_path = Some(path.clone());
                if self.follow_mode {
                    self.scroll = self.lines.len().saturating_sub(1);
                }
                // Track file position for tailing
                self.file_pos = file.metadata().map(|m| m.len()).unwrap_or(0);
            }
            Err(e) => {
                self.lines = vec![format!("Failed to open {}: {}", path.display(), e)];
                self.log_path = None;
            }
        }
    }

    /// Poll for new log lines (tailing)
    fn poll_new_lines(&mut self) {
        if !self.follow_mode {
            return;
        }

        // Only poll every 500ms
        if self.last_poll.elapsed().as_millis() < 500 {
            return;
        }
        self.last_poll = Instant::now();

        match self.source {
            LogSource::SystemJournal => {
                // Reload journal
                self.load_journal_logs();
            }
            _ => {
                // Tail file
                if let Some(ref path) = self.log_path {
                    if let Ok(file) = fs::File::open(path) {
                        let mut reader = BufReader::new(file);
                        if reader.seek(SeekFrom::Start(self.file_pos)).is_ok() {
                            let mut new_lines = Vec::new();
                            let mut line = String::new();
                            while reader.read_line(&mut line).unwrap_or(0) > 0 {
                                new_lines.push(line.trim_end().to_string());
                                line.clear();
                            }
                            if !new_lines.is_empty() {
                                self.lines.extend(new_lines);
                                if self.follow_mode {
                                    self.scroll = self.lines.len().saturating_sub(1);
                                }
                            }
                            if let Ok(pos) = reader.stream_position() {
                                self.file_pos = pos;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get log level color for a line
    fn line_color(&self, line: &str) -> Color {
        let lower = line.to_lowercase();
        if lower.contains("error") || lower.contains("exception") || lower.contains("fatal") {
            self.theme.error
        } else if lower.contains("warn") {
            self.theme.warning
        } else if lower.contains("debug") || lower.contains("trace") {
            self.theme.text_muted
        } else {
            self.theme.text_secondary
        }
    }

    /// Render source selector and log file info
    fn render_source_selector(&self, frame: &mut Frame, area: Rect) {
        let sources = [
            (LogSource::TrainingLogs, "1"),
            (LogSource::TradingLogs, "2"),
            (LogSource::SystemJournal, "3"),
        ];

        let mut spans = Vec::new();
        for (i, (source, key)) in sources.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            }
            let style = if *source == self.source {
                Style::default().fg(self.theme.accent).bold()
            } else {
                Style::default().fg(self.theme.text_muted)
            };
            spans.push(Span::styled(format!("[{}] {}", key, source.label()), style));
        }

        // Show log file selector if we have discovered logs
        if !self.discovered_logs.is_empty() {
            spans.push(Span::styled("  │  ", Style::default().fg(self.theme.border)));
            spans.push(Span::styled(
                format!(
                    "[Tab] File {}/{}: {}",
                    self.selected_log + 1,
                    self.discovered_logs.len(),
                    self.discovered_logs.get(self.selected_log)
                        .map(|e| e.display_name.as_str())
                        .unwrap_or("?")
                ),
                Style::default().fg(self.theme.text_secondary),
            ));
        }

        let selector = Paragraph::new(Line::from(spans));
        frame.render_widget(selector, area);
    }

    /// Render log content
    fn render_content(&self, frame: &mut Frame, area: Rect) {
        let title = match &self.log_path {
            Some(path) => format!("{} - {}", self.source.label(), path.display()),
            None => self.source.label().to_string(),
        };

        let follow_indicator = if self.follow_mode {
            format!(" {} Follow", "●")
        } else {
            String::new()
        };

        let full_title = format!("{}{}", title, follow_indicator);
        let block = Panel::new(&self.theme)
            .title(&full_title)
            .block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let visible_height = inner.height as usize;
        let start = self.scroll;
        let end = (start + visible_height).min(self.lines.len());

        let items: Vec<ListItem> = self.lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let line_num = start + i + 1;
                let color = self.line_color(line);

                // Truncate long lines
                let display_line = if line.len() > inner.width as usize - 8 {
                    format!("{:5} {}...", line_num, &line[..inner.width as usize - 12])
                } else {
                    format!("{:5} {}", line_num, line)
                };

                ListItem::new(Line::from(Span::styled(display_line, Style::default().fg(color))))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render footer
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let mut keybinds = vec![
            ("[1-3]", "Source"),
            ("[↑↓/jk]", "Scroll"),
            ("[g/G]", "Top/Bottom"),
            ("[f]", "Follow"),
            ("[r]", "Refresh"),
            ("[q]", "Back"),
        ];

        // Add Tab if we have multiple log files
        if self.discovered_logs.len() > 1 {
            keybinds.insert(1, ("[Tab]", "Next file"));
        }

        let mut spans = Vec::new();
        for (i, (key, desc)) in keybinds.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled("  ", Style::default()));
            }
            spans.push(Span::styled(*key, Style::default().fg(self.theme.accent)));
            spans.push(Span::styled(
                format!(" {}", desc),
                Style::default().fg(self.theme.text_muted),
            ));
        }

        // Position info
        spans.push(Span::styled(
            format!("  │  {} / {} lines", self.scroll + 1, self.lines.len()),
            Style::default().fg(self.theme.text_muted),
        ));

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for LogViewerView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for LogViewerView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Poll for new lines
        self.poll_new_lines();

        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Source selector
                Constraint::Min(0),    // Content
                Constraint::Length(1), // Footer
            ])
            .margin(1)
            .split(area);

        self.render_source_selector(frame, chunks[0]);
        self.render_content(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(true); // Go back
            }
            // Source selection
            KeyCode::Char('1') => {
                self.source = LogSource::TrainingLogs;
                self.load_current_source();
            }
            KeyCode::Char('2') => {
                self.source = LogSource::TradingLogs;
                self.load_current_source();
            }
            KeyCode::Char('3') => {
                self.source = LogSource::SystemJournal;
                self.load_current_source();
            }
            // Cycle through discovered log files
            KeyCode::Tab => {
                if !self.discovered_logs.is_empty() {
                    self.selected_log = (self.selected_log + 1) % self.discovered_logs.len();
                    self.load_selected_log();
                }
            }
            KeyCode::BackTab => {
                if !self.discovered_logs.is_empty() {
                    self.selected_log = if self.selected_log == 0 {
                        self.discovered_logs.len() - 1
                    } else {
                        self.selected_log - 1
                    };
                    self.load_selected_log();
                }
            }
            // Scrolling
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll = self.scroll.saturating_sub(1);
                self.follow_mode = false;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.scroll < self.lines.len().saturating_sub(1) {
                    self.scroll += 1;
                }
                // Re-enable follow if at bottom
                if self.scroll >= self.lines.len().saturating_sub(1) {
                    self.follow_mode = true;
                }
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(20);
                self.follow_mode = false;
            }
            KeyCode::PageDown => {
                self.scroll = (self.scroll + 20).min(self.lines.len().saturating_sub(1));
                if self.scroll >= self.lines.len().saturating_sub(1) {
                    self.follow_mode = true;
                }
            }
            KeyCode::Char('g') => {
                self.scroll = 0;
                self.follow_mode = false;
            }
            KeyCode::Char('G') => {
                self.scroll = self.lines.len().saturating_sub(1);
                self.follow_mode = true;
            }
            // Follow mode toggle
            KeyCode::Char('f') => {
                self.follow_mode = !self.follow_mode;
                if self.follow_mode {
                    self.scroll = self.lines.len().saturating_sub(1);
                }
            }
            // Refresh
            KeyCode::Char('r') => {
                self.load_current_source();
            }
            _ => {}
        }
        Ok(false)
    }
}
