//! Training launcher view - start training runs
//!
//! Simple config selection and run launch. Progress monitored in Training view.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;

use crate::launcher::training_executor::TrainingExecutor;
use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Focus state for the launcher form
#[derive(Debug, Clone, Copy, PartialEq)]
enum LauncherFocus {
    ConfigList,
    OutputDir,
    DeterministicToggle,
}

/// Training launcher view with config selection
pub struct TrainingLauncherView {
    executor: TrainingExecutor,
    theme: Theme,
    // Config selection
    configs: Vec<String>,
    selected_config: usize,
    // Output directory
    output_dir: String,
    editing_output: bool,
    // Options
    deterministic: bool,
    // UI state
    focus: LauncherFocus,
    message: Option<(String, bool)>, // (message, is_error)
}

impl TrainingLauncherView {
    pub fn new() -> Self {
        let mut view = Self {
            executor: TrainingExecutor::new(),
            theme: Theme::load(),
            configs: Vec::new(),
            selected_config: 0,
            output_dir: String::new(),
            editing_output: false,
            deterministic: true, // Default to deterministic mode
            focus: LauncherFocus::ConfigList,
            message: None,
        };
        view.scan_configs();
        view.generate_output_dir();
        view
    }

    /// Scan CONFIG/experiments/ for available configs
    fn scan_configs(&mut self) {
        let config_dir = crate::config::config_dir().join("experiments");
        if !config_dir.exists() {
            self.configs = vec!["production_baseline".to_string()];
            return;
        }

        let mut configs = Vec::new();
        if let Ok(entries) = fs::read_dir(&config_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
                    if let Some(stem) = path.file_stem() {
                        configs.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }

        configs.sort();
        if configs.is_empty() {
            configs.push("production_baseline".to_string());
        }
        self.configs = configs;

        // Try to select production_baseline by default
        if let Some(idx) = self.configs.iter().position(|c| c == "production_baseline") {
            self.selected_config = idx;
        }
    }

    /// Generate a default output directory name
    fn generate_output_dir(&mut self) {
        let now = chrono::Local::now();
        let config_name = self.configs.get(self.selected_config)
            .map(|s| s.as_str())
            .unwrap_or("run");
        self.output_dir = format!("{}/{}_{}", crate::config::results_dir().display(), config_name, now.format("%Y%m%d_%H%M"));
    }

    /// Get the command that will be run
    fn get_command_preview(&self) -> Vec<String> {
        let config = self.configs.get(self.selected_config)
            .map(|s| s.as_str())
            .unwrap_or("production_baseline");

        if self.deterministic {
            vec![
                "bin/run_deterministic.sh python -m \\".to_string(),
                "  TRAINING.orchestration.intelligent_trainer \\".to_string(),
                format!("  --experiment-config {} \\", config),
                format!("  --output-dir {}", self.output_dir),
            ]
        } else {
            vec![
                "python -m TRAINING.orchestration.intelligent_trainer \\".to_string(),
                format!("  --experiment-config {} \\", config),
                format!("  --output-dir {}", self.output_dir),
            ]
        }
    }

    /// Start training with current settings
    fn start_training(&mut self) -> Result<()> {
        let config = self.configs.get(self.selected_config)
            .map(|s| s.clone())
            .unwrap_or_else(|| "production_baseline".to_string());

        self.executor.set_experiment_config(config.clone());
        self.executor.set_output_dir(PathBuf::from(&self.output_dir));
        self.executor.set_deterministic(self.deterministic);
        self.executor.start_training()?;

        self.message = Some((
            format!("Started: {} → {} (view progress in Training tab)", config, self.output_dir),
            false
        ));
        Ok(())
    }

    /// Render header with status
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let is_running = self.executor.is_running();

        let (title, style) = if is_running {
            if let Some(run_id) = self.executor.get_running_run_id() {
                (format!("Training Pipeline - Running: {}", run_id), Style::default().fg(self.theme.success))
            } else {
                ("Training Pipeline - Running".to_string(), Style::default().fg(self.theme.success))
            }
        } else {
            ("Training Pipeline Launcher".to_string(), Style::default().fg(self.theme.text_primary))
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(self.theme.border));

        let paragraph = Paragraph::new(if is_running {
            "A training run is in progress. You can start another or view progress in the Training tab."
        } else {
            "Select an experiment config and start a training run."
        }).style(style).block(block);

        frame.render_widget(paragraph, area);
    }

    /// Render config selector
    fn render_config_selector(&self, frame: &mut Frame, area: Rect) {
        let is_focused = self.focus == LauncherFocus::ConfigList;
        let border_color = if is_focused { self.theme.accent } else { self.theme.border };

        let block = Block::default()
            .title("Experiment Config [j/k to select]")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let visible_height = inner.height as usize;
        let start = if self.selected_config >= visible_height {
            self.selected_config - visible_height + 1
        } else {
            0
        };
        let end = (start + visible_height).min(self.configs.len());

        let items: Vec<ListItem> = self.configs[start..end]
            .iter()
            .enumerate()
            .map(|(i, config)| {
                let idx = start + i;
                let is_selected = idx == self.selected_config;
                let style = if is_selected {
                    Style::default().fg(self.theme.accent).bg(self.theme.surface)
                } else {
                    Style::default().fg(self.theme.text_primary)
                };
                let prefix = if is_selected { "> " } else { "  " };
                ListItem::new(format!("{}{}", prefix, config)).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render output directory input
    fn render_output_dir(&self, frame: &mut Frame, area: Rect) {
        let is_focused = self.focus == LauncherFocus::OutputDir;
        let border_color = if is_focused { self.theme.accent } else { self.theme.border };

        let title = if self.editing_output {
            "Output Directory [typing... Enter to confirm]"
        } else {
            "Output Directory [Enter to edit]"
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let display_text = if self.editing_output {
            format!("{}█", self.output_dir)
        } else {
            self.output_dir.clone()
        };

        let style = if self.editing_output {
            Style::default().fg(self.theme.warning)
        } else {
            Style::default().fg(self.theme.text_primary)
        };

        let paragraph = Paragraph::new(display_text).style(style);
        frame.render_widget(paragraph, inner);
    }

    /// Render options (deterministic toggle)
    fn render_options(&self, frame: &mut Frame, area: Rect) {
        let is_focused = self.focus == LauncherFocus::DeterministicToggle;
        let border_color = if is_focused { self.theme.accent } else { self.theme.border };

        let block = Block::default()
            .title("Options [Space to toggle]")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        let checkbox = if self.deterministic { "[x]" } else { "[ ]" };
        let style = if is_focused {
            Style::default().fg(self.theme.accent)
        } else {
            Style::default().fg(self.theme.text_primary)
        };

        let text = format!("{} Use deterministic mode (recommended for reproducibility)", checkbox);
        let paragraph = Paragraph::new(text).style(style);
        frame.render_widget(paragraph, inner);
    }

    /// Render command preview
    fn render_command_preview(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Command Preview").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let preview = self.get_command_preview();
        let lines: Vec<Line> = preview
            .iter()
            .map(|line| Line::from(Span::styled(line.as_str(), Style::default().fg(self.theme.text_secondary))))
            .collect();

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, inner);
    }

    /// Render footer
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let keybinds = if self.editing_output {
            vec![
                ("[Enter]", "Confirm"),
                ("[Esc]", "Cancel"),
            ]
        } else {
            vec![
                ("[Tab]", "Next field"),
                ("[Enter]", "Start training"),
                ("[Space]", "Toggle option"),
                ("[r]", "Refresh configs"),
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

impl Default for TrainingLauncherView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for TrainingLauncherView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Clear background
        let bg = Block::default().style(Style::default().bg(self.theme.background));
        frame.render_widget(bg, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // Header
                Constraint::Length(12), // Config selector
                Constraint::Length(3),  // Output dir
                Constraint::Length(3),  // Options
                Constraint::Length(6),  // Command preview
                Constraint::Min(0),     // Spacing
                Constraint::Length(1),  // Footer
            ])
            .margin(1)
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_config_selector(frame, chunks[1]);
        self.render_output_dir(frame, chunks[2]);
        self.render_options(frame, chunks[3]);
        self.render_command_preview(frame, chunks[4]);
        self.render_footer(frame, chunks[6]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        // Handle output editing mode
        if self.editing_output {
            match key {
                KeyCode::Enter => {
                    self.editing_output = false;
                }
                KeyCode::Esc => {
                    self.editing_output = false;
                    self.generate_output_dir(); // Reset to default
                }
                KeyCode::Backspace => {
                    self.output_dir.pop();
                }
                KeyCode::Char(c) => {
                    self.output_dir.push(c);
                }
                _ => {}
            }
            return Ok(ViewAction::Continue);
        }

        // Clear message on any key
        self.message = None;

        // Normal navigation
        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(ViewAction::Back);
            }
            KeyCode::Tab => {
                // Cycle focus
                self.focus = match self.focus {
                    LauncherFocus::ConfigList => LauncherFocus::OutputDir,
                    LauncherFocus::OutputDir => LauncherFocus::DeterministicToggle,
                    LauncherFocus::DeterministicToggle => LauncherFocus::ConfigList,
                };
            }
            KeyCode::BackTab => {
                // Reverse cycle
                self.focus = match self.focus {
                    LauncherFocus::ConfigList => LauncherFocus::DeterministicToggle,
                    LauncherFocus::OutputDir => LauncherFocus::ConfigList,
                    LauncherFocus::DeterministicToggle => LauncherFocus::OutputDir,
                };
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.focus == LauncherFocus::ConfigList && self.selected_config > 0 {
                    self.selected_config -= 1;
                    self.generate_output_dir();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.focus == LauncherFocus::ConfigList
                    && self.selected_config < self.configs.len().saturating_sub(1)
                {
                    self.selected_config += 1;
                    self.generate_output_dir();
                }
            }
            KeyCode::Char(' ') => {
                // Toggle deterministic
                if self.focus == LauncherFocus::DeterministicToggle {
                    self.deterministic = !self.deterministic;
                }
            }
            KeyCode::Enter => {
                match self.focus {
                    LauncherFocus::OutputDir => {
                        self.editing_output = true;
                    }
                    LauncherFocus::DeterministicToggle => {
                        self.deterministic = !self.deterministic;
                    }
                    LauncherFocus::ConfigList => {
                        // Start training
                        if let Err(e) = self.start_training() {
                            self.message = Some((format!("Error: {}", e), true));
                        }
                    }
                }
            }
            KeyCode::Char('r') => {
                // Refresh configs
                self.scan_configs();
                self.message = Some(("Configs refreshed".to_string(), false));
            }
            KeyCode::Char('s') => {
                // Stop running training
                if self.executor.is_running() {
                    if let Err(e) = self.executor.stop_training() {
                        self.message = Some((format!("Error stopping: {}", e), true));
                    } else {
                        self.message = Some(("Stop signal sent".to_string(), false));
                    }
                }
            }
            _ => {}
        }

        Ok(ViewAction::Continue)
    }
}
