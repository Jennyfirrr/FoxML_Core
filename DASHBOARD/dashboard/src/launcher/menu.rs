//! Main menu for launcher with grouped structure

use anyhow::Result;
use ratatui::prelude::*;
use ratatui::widgets::*;

/// Menu group
struct MenuGroup {
    title: String,
    items: Vec<MenuItem>,
}

/// Menu item
struct MenuItem {
    title: String,
    description: String,
    action: MenuAction,
}

/// Menu action
#[derive(Clone, Copy, Debug)]
pub enum MenuAction {
    TradingMonitor,
    TrainingMonitor,
    ConfigEditor,
    ModelSelector,
    ServiceManager,
    RunManager,
    SystemStatus,
    LogViewer,
    FileBrowser,
    Settings,
}

/// Main menu with grouped structure
pub struct MainMenu {
    groups: Vec<MenuGroup>,
    active_group: usize,
    active_item: usize,
}

impl MainMenu {
    pub fn new() -> Self {
        let groups = vec![
            MenuGroup {
                title: "Trading".to_string(),
                items: vec![
                    MenuItem {
                        title: "Trading Monitor".to_string(),
                        description: "Real-time trading dashboard".to_string(),
                        action: MenuAction::TradingMonitor,
                    },
                ],
            },
            MenuGroup {
                title: "Training".to_string(),
                items: vec![
                    MenuItem {
                        title: "Training Monitor".to_string(),
                        description: "Training pipeline progress".to_string(),
                        action: MenuAction::TrainingMonitor,
                    },
                    MenuItem {
                        title: "Config Editor".to_string(),
                        description: "Edit YAML configs".to_string(),
                        action: MenuAction::ConfigEditor,
                    },
                    MenuItem {
                        title: "Model Selector".to_string(),
                        description: "Choose models for LIVE_TRADING".to_string(),
                        action: MenuAction::ModelSelector,
                    },
                    MenuItem {
                        title: "Run Manager".to_string(),
                        description: "Manage training runs".to_string(),
                        action: MenuAction::RunManager,
                    },
                ],
            },
            MenuGroup {
                title: "System".to_string(),
                items: vec![
                    MenuItem {
                        title: "Service Manager".to_string(),
                        description: "Manage systemd services".to_string(),
                        action: MenuAction::ServiceManager,
                    },
                    MenuItem {
                        title: "System Status".to_string(),
                        description: "System health check".to_string(),
                        action: MenuAction::SystemStatus,
                    },
                    MenuItem {
                        title: "Log Viewer".to_string(),
                        description: "Browse log files".to_string(),
                        action: MenuAction::LogViewer,
                    },
                    MenuItem {
                        title: "File Browser".to_string(),
                        description: "Browse RESULTS/".to_string(),
                        action: MenuAction::FileBrowser,
                    },
                    MenuItem {
                        title: "Settings".to_string(),
                        description: "Dashboard settings".to_string(),
                        action: MenuAction::Settings,
                    },
                ],
            },
        ];

        Self {
            groups,
            active_group: 0,
            active_item: 0,
        }
    }

    /// Move selection up (k or Up)
    pub fn move_up(&mut self) {
        if self.active_item > 0 {
            self.active_item -= 1;
        } else if self.active_group > 0 {
            self.active_group -= 1;
            self.active_item = self.groups[self.active_group].items.len().saturating_sub(1);
        }
    }

    /// Move selection down (j or Down)
    pub fn move_down(&mut self) {
        if self.active_item < self.groups[self.active_group].items.len().saturating_sub(1) {
            self.active_item += 1;
        } else if self.active_group < self.groups.len().saturating_sub(1) {
            self.active_group += 1;
            self.active_item = 0;
        }
    }

    /// Move to previous group (h or Left)
    pub fn move_to_previous_group(&mut self) {
        if self.active_group > 0 {
            self.active_group -= 1;
            // Keep item index if possible, otherwise use first item
            if self.active_item >= self.groups[self.active_group].items.len() {
                self.active_item = 0;
            }
        }
    }

    /// Move to next group (l or Right)
    pub fn move_to_next_group(&mut self) {
        if self.active_group < self.groups.len().saturating_sub(1) {
            self.active_group += 1;
            // Keep item index if possible, otherwise use first item
            if self.active_item >= self.groups[self.active_group].items.len() {
                self.active_item = 0;
            }
        }
    }

    /// Move to top (gg)
    pub fn move_to_top(&mut self) {
        self.active_group = 0;
        self.active_item = 0;
    }

    /// Move to bottom (G)
    pub fn move_to_bottom(&mut self) {
        if let Some(last_group) = self.groups.last() {
            self.active_group = self.groups.len().saturating_sub(1);
            self.active_item = last_group.items.len().saturating_sub(1);
        }
    }

    /// Get current selected action
    pub fn get_selected_action(&self) -> Option<MenuAction> {
        if let Some(group) = self.groups.get(self.active_group) {
            if let Some(item) = group.items.get(self.active_item) {
                return Some(item.action);
            }
        }
        None
    }

    /// Render menu with visual hierarchy
    pub fn render(&mut self, frame: &mut Frame, area: Rect, theme: &crate::themes::Theme) -> Result<()> {
        // Build list items with groups
        let mut list_items = Vec::new();

        for (g_idx, group) in self.groups.iter().enumerate() {
            // Group header (primary color, brighter)
            let is_active_group = g_idx == self.active_group;
            let header_style = if is_active_group {
                Style::default().fg(theme.primary_text)
            } else {
                Style::default().fg(theme.secondary_text)
            };

            let header_text = format!("▶ {}", group.title);
            list_items.push(ListItem::new(header_text).style(header_style));

            // Group items
            for (i_idx, item) in group.items.iter().enumerate() {
                let is_selected = g_idx == self.active_group && i_idx == self.active_item;

                // Use arrow indicator for selected item, space for others
                let indicator = if is_selected { "▸" } else { " " };

                let item_style = if is_selected {
                    // Highlight selected item with accent color and bold
                    Style::default().fg(theme.accent).bold()
                } else {
                    Style::default().fg(theme.secondary_text)
                };

                let item_text = format!("  {} {}", indicator, item.title);
                list_items.push(ListItem::new(item_text).style(item_style));
            }

            // Blank line between groups (except last)
            if g_idx < self.groups.len() - 1 {
                list_items.push(ListItem::new(""));
            }
        }

        let list = List::new(list_items);

        frame.render_widget(list, area);

        Ok(())
    }
}

impl Default for MainMenu {
    fn default() -> Self {
        Self::new()
    }
}
