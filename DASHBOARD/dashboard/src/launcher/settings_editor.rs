//! Settings editor for /etc/foxml-trading.conf

use anyhow::{Context, Result};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;

/// Settings editor for trading engine config
pub struct SettingsEditor {
    config_path: PathBuf,
    settings: HashMap<String, String>,
    selected_field: usize,
    fields: Vec<SettingField>,
    modified: bool,
}

#[derive(Clone)]
struct SettingField {
    key: String,
    label: String,
    value: String,
    field_type: FieldType,
}

#[derive(Clone)]
enum FieldType {
    Number { min: i32, max: i32 },
    Text,
    Dropdown { options: Vec<String> },
    Toggle,
}

impl SettingsEditor {
    pub fn new(config_path: Option<String>) -> Result<Self> {
        let path = if let Some(p) = config_path {
            PathBuf::from(p)
        } else {
            PathBuf::from("/etc/foxml-trading.conf")
        };

        let settings = if path.exists() {
            Self::parse_config_file(&path)?
        } else {
            HashMap::new()
        };

        let fields = vec![
            SettingField {
                key: "FOXML_CYCLE_INTERVAL".to_string(),
                label: "Cycle Interval (seconds)".to_string(),
                value: settings.get("FOXML_CYCLE_INTERVAL").cloned().unwrap_or_else(|| "60".to_string()),
                field_type: FieldType::Number { min: 1, max: 3600 },
            },
            SettingField {
                key: "FOXML_RUN_ID".to_string(),
                label: "Run ID".to_string(),
                value: settings.get("FOXML_RUN_ID").cloned().unwrap_or_else(|| "latest".to_string()),
                field_type: FieldType::Text,
            },
            SettingField {
                key: "FOXML_BROKER".to_string(),
                label: "Broker".to_string(),
                value: settings.get("FOXML_BROKER").cloned().unwrap_or_else(|| "paper".to_string()),
                field_type: FieldType::Dropdown {
                    options: vec!["paper".to_string(), "alpaca".to_string(), "ibkr".to_string()],
                },
            },
            SettingField {
                key: "FOXML_MARKET_HOURS_ONLY".to_string(),
                label: "Market Hours Only".to_string(),
                value: settings.get("FOXML_MARKET_HOURS_ONLY").cloned().unwrap_or_else(|| "true".to_string()),
                field_type: FieldType::Toggle,
            },
            SettingField {
                key: "FOXML_LOG_LEVEL".to_string(),
                label: "Log Level".to_string(),
                value: settings.get("FOXML_LOG_LEVEL").cloned().unwrap_or_else(|| "INFO".to_string()),
                field_type: FieldType::Dropdown {
                    options: vec!["DEBUG".to_string(), "INFO".to_string(), "WARNING".to_string(), "ERROR".to_string()],
                },
            },
        ];

        Ok(Self {
            config_path: path,
            settings,
            selected_field: 0,
            fields,
            modified: false,
        })
    }

    /// Parse bash config file (key=value format)
    fn parse_config_file(path: &PathBuf) -> Result<HashMap<String, String>> {
        let content = fs::read_to_string(path)?;
        let mut settings = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse key=value
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().trim_matches('"').trim_matches('\'').to_string();
                settings.insert(key, value);
            }
        }

        Ok(settings)
    }

    /// Save settings to config file
    pub fn save(&mut self) -> Result<()> {
        // Validate settings
        for field in &self.fields {
            match &field.field_type {
                FieldType::Number { min, max } => {
                    if let Ok(num) = field.value.parse::<i32>() {
                        if num < *min || num > *max {
                            anyhow::bail!("{} must be between {} and {}", field.label, min, max);
                        }
                    } else {
                        anyhow::bail!("{} must be a number", field.label);
                    }
                }
                FieldType::Dropdown { options } => {
                    if !options.contains(&field.value) {
                        anyhow::bail!("{} must be one of: {}", field.label, options.join(", "));
                    }
                }
                FieldType::Toggle => {
                    if field.value != "true" && field.value != "false" {
                        anyhow::bail!("{} must be 'true' or 'false'", field.label);
                    }
                }
                FieldType::Text => {
                    // No validation for text fields
                }
            }
        }

        // Create backup
        if self.config_path.exists() {
            let backup_path = self.config_path.with_extension("conf.bak");
            fs::copy(&self.config_path, &backup_path)
                .context("Failed to create backup")?;
        }

        // Write config file
        let mut content = String::from("# FoxML Trading Configuration\n");
        content.push_str("# Auto-generated by dashboard\n\n");

        for field in &self.fields {
            content.push_str(&format!("{}={}\n", field.key, field.value));
        }

        fs::write(&self.config_path, content)?;
        self.modified = false;
        Ok(())
    }

    /// Move selection up
    pub fn move_up(&mut self) {
        if self.selected_field > 0 {
            self.selected_field -= 1;
        }
    }

    /// Move selection down
    pub fn move_down(&mut self) {
        if self.selected_field < self.fields.len().saturating_sub(1) {
            self.selected_field += 1;
        }
    }

    /// Adjust current field value
    pub fn adjust_value(&mut self, delta: i32) {
        if let Some(field) = self.fields.get_mut(self.selected_field) {
            match &field.field_type {
                FieldType::Number { min, max } => {
                    if let Ok(mut num) = field.value.parse::<i32>() {
                        num += delta;
                        num = num.max(*min).min(*max);
                        field.value = num.to_string();
                        self.modified = true;
                    }
                }
                FieldType::Dropdown { options } => {
                    if let Some(current_idx) = options.iter().position(|o| o == &field.value) {
                        let new_idx = (current_idx as i32 + delta)
                            .max(0)
                            .min(options.len() as i32 - 1) as usize;
                        field.value = options[new_idx].clone();
                        self.modified = true;
                    }
                }
                FieldType::Toggle => {
                    field.value = if field.value == "true" { "false".to_string() } else { "true".to_string() };
                    self.modified = true;
                }
                FieldType::Text => {
                    // Text fields need separate editing mode
                }
            }
        }
    }

    /// Render settings editor
    pub fn render(&mut self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) -> Result<()> {
        let title = if self.modified {
            format!("Trading Engine Settings: {} *", self.config_path.display())
        } else {
            format!("Trading Engine Settings: {}", self.config_path.display())
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.secondary_text));

        // Build field list
        let mut items = Vec::new();
        for (idx, field) in self.fields.iter().enumerate() {
            let is_selected = idx == self.selected_field;
            
            let value_display = match &field.field_type {
                FieldType::Toggle => {
                    if field.value == "true" {
                        "[✓] Enabled".to_string()
                    } else {
                        "[ ] Disabled".to_string()
                    }
                }
                FieldType::Dropdown { .. } => {
                    format!("[{} ▼]", field.value)
                }
                _ => field.value.clone(),
            };

            let prefix = if is_selected { "> " } else { "  " };
            let style = if is_selected {
                Style::default().fg(theme.primary_text)
            } else {
                Style::default().fg(theme.secondary_text)
            };

            let line = format!("{}{}: {}", prefix, field.label, value_display);
            items.push(ListItem::new(line).style(style));
        }

        let list = List::new(items)
            .block(block);

        let mut state = ratatui::widgets::ListState::default();
        state.select(Some(self.selected_field));
        frame.render_stateful_widget(list, area, &mut state);

        // Footer
        let footer = Paragraph::new("[↑↓] Navigate [←→] Adjust [s] Save [q] Quit")
            .style(Style::default().fg(theme.secondary_text));
        let footer_area = Rect::new(area.x, area.y + area.height - 1, area.width, 1);
        frame.render_widget(footer, footer_area);

        Ok(())
    }
}
