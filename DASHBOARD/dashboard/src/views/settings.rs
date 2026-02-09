//! Settings View - configure dashboard preferences
//!
//! Provides UI for configuring dashboard settings like theme, refresh intervals,
//! default view, and keybindings.

use anyhow::Result;
use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::themes::Theme;
use crate::ui::panels::Panel;

/// Settings schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSettings {
    #[serde(default)]
    pub theme: ThemeSettings,
    #[serde(default)]
    pub refresh: RefreshSettings,
    #[serde(default)]
    pub display: DisplaySettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeSettings {
    #[serde(default = "default_theme_mode")]
    pub mode: String, // auto, light, dark
}

fn default_theme_mode() -> String {
    "auto".to_string()
}

impl Default for ThemeSettings {
    fn default() -> Self {
        Self {
            mode: default_theme_mode(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshSettings {
    #[serde(default = "default_bridge_interval")]
    pub bridge_interval_ms: u64,
    #[serde(default = "default_training_interval")]
    pub training_interval_ms: u64,
}

fn default_bridge_interval() -> u64 {
    2000
}
fn default_training_interval() -> u64 {
    500
}

impl Default for RefreshSettings {
    fn default() -> Self {
        Self {
            bridge_interval_ms: default_bridge_interval(),
            training_interval_ms: default_training_interval(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplaySettings {
    #[serde(default = "default_view")]
    pub default_view: String,
    #[serde(default = "default_show_timestamps")]
    pub show_timestamps: bool,
}

fn default_view() -> String {
    "launcher".to_string()
}
fn default_show_timestamps() -> bool {
    true
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            default_view: default_view(),
            show_timestamps: default_show_timestamps(),
        }
    }
}

impl Default for DashboardSettings {
    fn default() -> Self {
        Self {
            theme: ThemeSettings::default(),
            refresh: RefreshSettings::default(),
            display: DisplaySettings::default(),
        }
    }
}

/// Settings menu item
#[derive(Debug, Clone)]
struct SettingItem {
    key: String,
    label: String,
    value: String,
    options: Vec<String>,
}

/// Settings view
pub struct SettingsView {
    theme: Theme,
    settings: DashboardSettings,
    items: Vec<SettingItem>,
    selected: usize,
    editing: bool,
    edit_index: usize,
    message: Option<(String, bool)>, // (message, is_error)
    config_path: PathBuf,
}

impl SettingsView {
    pub fn new() -> Self {
        let config_path = Self::get_config_path();
        let settings = Self::load_settings(&config_path);
        let items = Self::build_items(&settings);

        Self {
            theme: Theme::load(),
            settings,
            items,
            selected: 0,
            editing: false,
            edit_index: 0,
            message: None,
            config_path,
        }
    }

    /// Get config file path
    fn get_config_path() -> PathBuf {
        // Try XDG config dir first
        if let Some(config_dir) = dirs::config_dir() {
            let app_dir = config_dir.join("foxml-dashboard");
            return app_dir.join("settings.yaml");
        }
        // Fallback to home directory
        PathBuf::from(".foxml-dashboard-settings.yaml")
    }

    /// Load settings from file
    fn load_settings(path: &PathBuf) -> DashboardSettings {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(settings) = serde_yaml::from_str(&content) {
                    return settings;
                }
            }
        }
        DashboardSettings::default()
    }

    /// Save settings to file
    fn save_settings(&self) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_yaml::to_string(&self.settings)?;
        fs::write(&self.config_path, content)?;
        Ok(())
    }

    /// Build menu items from settings
    fn build_items(settings: &DashboardSettings) -> Vec<SettingItem> {
        vec![
            SettingItem {
                key: "theme.mode".to_string(),
                label: "Theme Mode".to_string(),
                value: settings.theme.mode.clone(),
                options: vec!["auto".to_string(), "light".to_string(), "dark".to_string()],
            },
            SettingItem {
                key: "refresh.bridge_interval_ms".to_string(),
                label: "Bridge Refresh (ms)".to_string(),
                value: settings.refresh.bridge_interval_ms.to_string(),
                options: vec![
                    "500".to_string(),
                    "1000".to_string(),
                    "2000".to_string(),
                    "5000".to_string(),
                ],
            },
            SettingItem {
                key: "refresh.training_interval_ms".to_string(),
                label: "Training Refresh (ms)".to_string(),
                value: settings.refresh.training_interval_ms.to_string(),
                options: vec![
                    "100".to_string(),
                    "250".to_string(),
                    "500".to_string(),
                    "1000".to_string(),
                ],
            },
            SettingItem {
                key: "display.default_view".to_string(),
                label: "Default View".to_string(),
                value: settings.display.default_view.clone(),
                options: vec![
                    "launcher".to_string(),
                    "trading".to_string(),
                    "training".to_string(),
                ],
            },
            SettingItem {
                key: "display.show_timestamps".to_string(),
                label: "Show Timestamps".to_string(),
                value: settings.display.show_timestamps.to_string(),
                options: vec!["true".to_string(), "false".to_string()],
            },
        ]
    }

    /// Apply current items to settings
    fn apply_items(&mut self) {
        for item in &self.items {
            match item.key.as_str() {
                "theme.mode" => self.settings.theme.mode = item.value.clone(),
                "refresh.bridge_interval_ms" => {
                    self.settings.refresh.bridge_interval_ms =
                        item.value.parse().unwrap_or(2000);
                }
                "refresh.training_interval_ms" => {
                    self.settings.refresh.training_interval_ms =
                        item.value.parse().unwrap_or(500);
                }
                "display.default_view" => self.settings.display.default_view = item.value.clone(),
                "display.show_timestamps" => {
                    self.settings.display.show_timestamps =
                        item.value.parse().unwrap_or(true);
                }
                _ => {}
            }
        }
    }

    /// Render settings list
    fn render_settings(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Settings").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = i == self.selected;
                let is_editing = self.editing && i == self.selected;

                let value_display = if is_editing {
                    // Show all options with current selection highlighted
                    let idx = self.edit_index;
                    format!("< {} >", item.options.get(idx).unwrap_or(&item.value))
                } else {
                    item.value.clone()
                };

                let content = Line::from(vec![
                    Span::styled(
                        if is_selected { "> " } else { "  " },
                        Style::default().fg(self.theme.accent),
                    ),
                    Span::styled(
                        format!("{:25}", item.label),
                        Style::default().fg(if is_selected {
                            self.theme.accent
                        } else {
                            self.theme.text_primary
                        }),
                    ),
                    Span::styled(
                        value_display,
                        Style::default().fg(if is_editing {
                            self.theme.warning
                        } else {
                            self.theme.text_secondary
                        }),
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

    /// Render help panel
    fn render_help(&self, frame: &mut Frame, area: Rect) {
        let block = Panel::new(&self.theme).title("Help").block();
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let help_text = if self.editing {
            vec![
                Line::from(Span::styled(
                    "Editing mode:",
                    Style::default().fg(self.theme.accent).bold(),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("[</>]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Change value", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[Enter]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Confirm", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[Esc]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Cancel", Style::default().fg(self.theme.text_secondary)),
                ]),
            ]
        } else {
            vec![
                Line::from(Span::styled(
                    "Navigation:",
                    Style::default().fg(self.theme.accent).bold(),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("[j/k]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Navigate", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[Enter]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Edit value", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[s]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Save settings", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[r]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Reset to defaults", Style::default().fg(self.theme.text_secondary)),
                ]),
                Line::from(vec![
                    Span::styled("[q/Esc]", Style::default().fg(self.theme.accent)),
                    Span::styled(" Back", Style::default().fg(self.theme.text_secondary)),
                ]),
            ]
        };

        let paragraph = Paragraph::new(help_text);
        frame.render_widget(paragraph, inner);
    }

    /// Render footer
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![
            Span::styled("Config: ", Style::default().fg(self.theme.text_muted)),
            Span::styled(
                self.config_path.display().to_string(),
                Style::default().fg(self.theme.text_secondary),
            ),
        ];

        if let Some((msg, is_error)) = &self.message {
            spans.push(Span::styled("  |  ", Style::default().fg(self.theme.border)));
            spans.push(Span::styled(
                msg,
                Style::default().fg(if *is_error {
                    self.theme.error
                } else {
                    self.theme.success
                }),
            ));
        }

        let footer = Paragraph::new(Line::from(spans));
        frame.render_widget(footer, area);
    }
}

impl Default for SettingsView {
    fn default() -> Self {
        Self::new()
    }
}

impl super::ViewTrait for SettingsView {
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

        // Split content into settings and help
        let content_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60), // Settings
                Constraint::Percentage(40), // Help
            ])
            .split(main_chunks[0]);

        self.render_settings(frame, content_chunks[0]);
        self.render_help(frame, content_chunks[1]);
        self.render_footer(frame, main_chunks[1]);

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<super::ViewAction> {
        use super::ViewAction;
        // Clear message
        self.message = None;

        if self.editing {
            match key {
                KeyCode::Esc => {
                    self.editing = false;
                }
                KeyCode::Enter => {
                    // Apply the selected option
                    if let Some(item) = self.items.get_mut(self.selected) {
                        if let Some(new_value) = item.options.get(self.edit_index) {
                            item.value = new_value.clone();
                        }
                    }
                    self.editing = false;
                    self.apply_items();
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    if self.edit_index > 0 {
                        self.edit_index -= 1;
                    }
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    if let Some(item) = self.items.get(self.selected) {
                        if self.edit_index < item.options.len().saturating_sub(1) {
                            self.edit_index += 1;
                        }
                    }
                }
                _ => {}
            }
            return Ok(ViewAction::Continue);
        }

        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                return Ok(ViewAction::Back);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.items.len().saturating_sub(1) {
                    self.selected += 1;
                }
            }
            KeyCode::Enter => {
                // Start editing
                if let Some(item) = self.items.get(self.selected) {
                    // Find current value index in options
                    self.edit_index = item
                        .options
                        .iter()
                        .position(|o| o == &item.value)
                        .unwrap_or(0);
                    self.editing = true;
                }
            }
            KeyCode::Char('s') => {
                // Save
                match self.save_settings() {
                    Ok(_) => self.message = Some(("Settings saved".to_string(), false)),
                    Err(e) => self.message = Some((format!("Save failed: {}", e), true)),
                }
            }
            KeyCode::Char('r') => {
                // Reset to defaults
                self.settings = DashboardSettings::default();
                self.items = Self::build_items(&self.settings);
                self.message = Some(("Reset to defaults".to_string(), false));
            }
            _ => {}
        }

        Ok(ViewAction::Continue)
    }
}
