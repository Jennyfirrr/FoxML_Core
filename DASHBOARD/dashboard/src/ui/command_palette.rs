//! Command palette
//!
//! Fuzzy-searchable command interface inspired by VS Code / Claude Code.

use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Widget};

use crate::themes::Theme;

/// Command category for grouping
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommandCategory {
    Navigation,
    Trading,
    Training,
    Config,
    System,
}

impl CommandCategory {
    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            CommandCategory::Navigation => "Navigation",
            CommandCategory::Trading => "Trading",
            CommandCategory::Training => "Training",
            CommandCategory::Config => "Config",
            CommandCategory::System => "System",
        }
    }
}

/// A command that can be executed
#[derive(Clone, Debug)]
pub struct Command {
    /// Command ID for internal use
    pub id: String,
    /// Display name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Category for grouping
    pub category: CommandCategory,
    /// Keyboard shortcut (for display)
    pub shortcut: Option<String>,
}

impl Command {
    /// Create a new command
    pub fn new(id: impl Into<String>, name: impl Into<String>, category: CommandCategory) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            category,
            shortcut: None,
        }
    }

    /// Add a description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a keyboard shortcut
    pub fn shortcut(mut self, shortcut: impl Into<String>) -> Self {
        self.shortcut = Some(shortcut.into());
        self
    }
}

/// Command palette state
pub struct CommandPalette {
    /// All available commands
    commands: Vec<Command>,
    /// Current search query
    query: String,
    /// Filtered commands (indices into commands)
    filtered: Vec<usize>,
    /// Currently selected index (in filtered)
    selected: usize,
    /// List state for rendering
    list_state: ListState,
    /// Fuzzy matcher
    matcher: SkimMatcherV2,
    /// Whether the palette is visible
    pub visible: bool,
}

impl CommandPalette {
    /// Create a new command palette
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            query: String::new(),
            filtered: Vec::new(),
            selected: 0,
            list_state: ListState::default(),
            matcher: SkimMatcherV2::default(),
            visible: false,
        }
    }

    /// Create with default commands
    pub fn with_default_commands() -> Self {
        let mut palette = Self::new();

        // Navigation commands
        palette.add_command(
            Command::new("nav.dashboard", "Go to Dashboard", CommandCategory::Navigation)
                .shortcut("D"),
        );
        palette.add_command(
            Command::new("nav.trading", "Go to Trading Monitor", CommandCategory::Navigation)
                .shortcut("1"),
        );
        palette.add_command(
            Command::new("nav.training", "Go to Training Monitor", CommandCategory::Navigation)
                .shortcut("2"),
        );
        palette.add_command(
            Command::new("nav.models", "Go to Model Manager", CommandCategory::Navigation)
                .shortcut("3"),
        );
        palette.add_command(
            Command::new("nav.config", "Go to Config Editor", CommandCategory::Navigation)
                .shortcut("4"),
        );

        // Trading commands
        palette.add_command(
            Command::new("trading.killswitch", "Toggle Kill Switch", CommandCategory::Trading)
                .description("Emergency stop all trading")
                .shortcut("Ctrl+K"),
        );
        palette.add_command(
            Command::new("trading.pause", "Pause Trading", CommandCategory::Trading)
                .description("Temporarily pause trading pipeline")
                .shortcut("Ctrl+P"),
        );
        palette.add_command(
            Command::new("trading.resume", "Resume Trading", CommandCategory::Trading)
                .description("Resume paused trading"),
        );
        palette.add_command(
            Command::new("trading.refresh", "Refresh Trading Data", CommandCategory::Trading)
                .shortcut("R"),
        );

        // Training commands
        palette.add_command(
            Command::new("training.start", "Start Training Run", CommandCategory::Training)
                .description("Launch a new training run")
                .shortcut("Ctrl+T"),
        );
        palette.add_command(
            Command::new("training.stop", "Stop Training Run", CommandCategory::Training)
                .description("Stop the current training run")
                .shortcut("Ctrl+S"),
        );
        palette.add_command(
            Command::new("training.logs", "View Training Logs", CommandCategory::Training)
                .shortcut("Ctrl+L"),
        );

        // Config commands
        palette.add_command(
            Command::new("config.edit", "Edit Experiment Config", CommandCategory::Config)
                .shortcut("Ctrl+E"),
        );
        palette.add_command(
            Command::new("config.reload", "Reload Configuration", CommandCategory::Config),
        );

        // System commands
        palette.add_command(
            Command::new("system.refresh", "Refresh All", CommandCategory::System)
                .shortcut("Ctrl+R"),
        );
        palette.add_command(
            Command::new("system.help", "Show Help", CommandCategory::System)
                .shortcut("?"),
        );
        palette.add_command(
            Command::new("system.quit", "Quit", CommandCategory::System)
                .shortcut("Q"),
        );

        palette.update_filtered();
        palette
    }

    /// Add a command
    pub fn add_command(&mut self, command: Command) {
        self.commands.push(command);
    }

    /// Show the palette
    pub fn show(&mut self) {
        self.visible = true;
        self.query.clear();
        self.update_filtered();
    }

    /// Hide the palette
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Toggle visibility
    pub fn toggle(&mut self) {
        if self.visible {
            self.hide();
        } else {
            self.show();
        }
    }

    /// Handle a character input
    pub fn input(&mut self, c: char) {
        self.query.push(c);
        self.update_filtered();
    }

    /// Handle backspace
    pub fn backspace(&mut self) {
        self.query.pop();
        self.update_filtered();
    }

    /// Move selection up
    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.list_state.select(Some(self.selected));
        }
    }

    /// Move selection down
    pub fn down(&mut self) {
        if self.selected < self.filtered.len().saturating_sub(1) {
            self.selected += 1;
            self.list_state.select(Some(self.selected));
        }
    }

    /// Get the currently selected command
    pub fn selected_command(&self) -> Option<&Command> {
        self.filtered.get(self.selected).map(|&i| &self.commands[i])
    }

    /// Execute the selected command and return its ID
    pub fn execute(&mut self) -> Option<String> {
        let id = self.selected_command().map(|c| c.id.clone());
        self.hide();
        id
    }

    /// Update filtered list based on query
    fn update_filtered(&mut self) {
        if self.query.is_empty() {
            self.filtered = (0..self.commands.len()).collect();
        } else {
            let mut scored: Vec<(usize, i64)> = self
                .commands
                .iter()
                .enumerate()
                .filter_map(|(i, cmd)| {
                    let score = self.matcher.fuzzy_match(&cmd.name, &self.query);
                    score.map(|s| (i, s))
                })
                .collect();

            scored.sort_by(|a, b| b.1.cmp(&a.1));
            self.filtered = scored.into_iter().map(|(i, _)| i).collect();
        }

        // Reset selection
        self.selected = 0;
        self.list_state.select(if self.filtered.is_empty() {
            None
        } else {
            Some(0)
        });
    }

    /// Render the command palette
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        if !self.visible {
            return;
        }

        // Calculate palette size and position (centered, 60% width, 50% height)
        let width = (area.width * 60 / 100).min(80);
        let height = (area.height * 50 / 100).min(20);
        let x = area.x + (area.width - width) / 2;
        let y = area.y + (area.height - height) / 4; // Slightly above center

        let palette_area = Rect {
            x,
            y,
            width,
            height,
        };

        // Clear the area
        Clear.render(palette_area, buf);

        // Main block
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.accent))
            .style(Style::default().bg(theme.surface_elevated));

        let inner = block.inner(palette_area);
        block.render(palette_area, buf);

        // Split inner area
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Search input
                Constraint::Length(1), // Separator
                Constraint::Min(0),    // Results
            ])
            .split(inner);

        // Search input
        let search_text = format!("> {}_", self.query);
        Paragraph::new(search_text)
            .style(Style::default().fg(theme.text_primary))
            .render(chunks[0], buf);

        // Separator
        let separator = "─".repeat(chunks[1].width as usize);
        Paragraph::new(separator)
            .style(Style::default().fg(theme.border))
            .render(chunks[1], buf);

        // Build list items grouped by category
        let mut items = Vec::new();
        let mut current_category: Option<CommandCategory> = None;

        for &idx in &self.filtered {
            let cmd = &self.commands[idx];

            // Add category header if changed
            if current_category != Some(cmd.category) {
                if current_category.is_some() {
                    items.push(ListItem::new(""));
                }
                items.push(ListItem::new(format!("  {}", cmd.category.name()))
                    .style(Style::default().fg(theme.text_muted)));
                current_category = Some(cmd.category);
            }

            // Build command line
            let shortcut_display = cmd
                .shortcut
                .as_ref()
                .map(|s| format!("  {}", s))
                .unwrap_or_default();

            let line = format!(
                "    {}{}",
                cmd.name,
                shortcut_display,
            );

            items.push(ListItem::new(line));
        }

        // Render list
        let list = List::new(items)
            .highlight_style(
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("  ");

        ratatui::widgets::StatefulWidget::render(list, chunks[2], buf, &mut self.list_state);
    }
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::with_default_commands()
    }
}
