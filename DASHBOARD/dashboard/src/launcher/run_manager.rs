//! Run manager for training runs

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::path::PathBuf;
use walkdir::WalkDir;

/// Training run info
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub run_id: String,
    pub path: PathBuf,
    pub status: String,
}

/// Run manager
pub struct RunManager {
    runs: Vec<RunInfo>,
    selected: usize,
    results_dir: PathBuf,
}

impl RunManager {
    pub fn new() -> Result<Self> {
        let results_dir = crate::config::results_dir().join("runs");
        let runs = Self::scan_runs(&results_dir)?;
        
        Ok(Self {
            runs,
            selected: 0,
            results_dir,
        })
    }

    /// Scan for training runs
    fn scan_runs(results_dir: &PathBuf) -> Result<Vec<RunInfo>> {
        let mut runs = Vec::new();
        
        if !results_dir.exists() {
            return Ok(runs);
        }

        for entry in WalkDir::new(results_dir)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_dir() {
                // Check for manifest.json
                let manifest = path.join("manifest.json");
                if manifest.exists() {
                    // Extract run_id from path
                    if let Some(run_id) = path.file_name().and_then(|n| n.to_str()) {
                        runs.push(RunInfo {
                            run_id: run_id.to_string(),
                            path: path.to_path_buf(),
                            status: "completed".to_string(),
                        });
                    }
                }
            }
        }

        // Sort by run_id (newest first)
        runs.sort_by(|a, b| b.run_id.cmp(&a.run_id));
        
        Ok(runs)
    }

    /// Refresh runs list
    pub fn refresh(&mut self) -> Result<()> {
        self.runs = Self::scan_runs(&self.results_dir)?;
        Ok(())
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) -> Result<()> {
        let block = Block::default()
            .title("Run Manager - Training Runs")
            .borders(Borders::ALL);

        if self.runs.is_empty() {
            let text = Paragraph::new("No training runs found in RESULTS/runs/")
                .block(block)
                .wrap(Wrap { trim: true });
            frame.render_widget(text, area);
            return Ok(());
        }

        let items: Vec<ListItem> = self
            .runs
            .iter()
            .enumerate()
            .map(|(i, run)| {
                let prefix = if i == self.selected { "> " } else { "  " };
                ListItem::new(format!("{}{} - {}", prefix, run.run_id, run.status))
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

