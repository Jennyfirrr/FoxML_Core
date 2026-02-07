//! Auto-hide sidebar
//!
//! Navigation sidebar that can be shown/hidden.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Widget};
use std::time::{Duration, Instant};

use crate::themes::Theme;
use crate::ui::borders::Separators;

/// Sidebar navigation item
#[derive(Clone, Debug)]
pub struct SidebarItem {
    pub id: String,
    pub label: String,
    pub shortcut: Option<char>,
    pub active: bool,
}

impl SidebarItem {
    /// Create a new sidebar item
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            shortcut: None,
            active: false,
        }
    }

    /// Set the keyboard shortcut
    pub fn shortcut(mut self, shortcut: char) -> Self {
        self.shortcut = Some(shortcut);
        self
    }

    /// Set active state
    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

/// Service status for sidebar display
#[derive(Clone, Debug)]
pub struct SidebarService {
    pub name: String,
    pub status: String,
    pub running: bool,
}

impl SidebarService {
    /// Create a new service
    pub fn new(name: impl Into<String>, status: impl Into<String>, running: bool) -> Self {
        Self {
            name: name.into(),
            status: status.into(),
            running,
        }
    }
}

/// Quick action for sidebar
#[derive(Clone, Debug)]
pub struct QuickAction {
    pub shortcut: char,
    pub label: String,
    pub value: String,
}

impl QuickAction {
    /// Create a new quick action
    pub fn new(shortcut: char, label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            shortcut,
            label: label.into(),
            value: value.into(),
        }
    }
}

/// Auto-hide sidebar state
pub struct Sidebar {
    /// Navigation items
    items: Vec<SidebarItem>,
    /// Service statuses
    services: Vec<SidebarService>,
    /// Quick actions
    quick_actions: Vec<QuickAction>,
    /// Currently selected item
    selected: usize,
    /// List state
    list_state: ListState,
    /// Whether the sidebar is visible
    pub visible: bool,
    /// Whether the sidebar is pinned (won't auto-hide)
    pub pinned: bool,
    /// Width of the sidebar
    pub width: u16,
    /// Hover zone width (pixels from left edge to trigger show)
    pub hover_zone: u16,
    /// Time when mouse left the sidebar (for hide delay)
    last_mouse_leave: Option<Instant>,
    /// Hide delay duration
    hide_delay: Duration,
}

impl Sidebar {
    /// Create a new sidebar
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            services: Vec::new(),
            quick_actions: Vec::new(),
            selected: 0,
            list_state: ListState::default().with_selected(Some(0)),
            visible: false,
            pinned: false,
            width: 30,
            hover_zone: 3,
            last_mouse_leave: None,
            hide_delay: Duration::from_millis(500),
        }
    }

    /// Create with default items
    pub fn with_default_items() -> Self {
        let mut sidebar = Self::new();

        sidebar.add_item(SidebarItem::new("dashboard", "Dashboard").shortcut('D'));
        sidebar.add_item(SidebarItem::new("trading", "Trading").shortcut('1'));
        sidebar.add_item(SidebarItem::new("training", "Training").shortcut('2'));
        sidebar.add_item(SidebarItem::new("models", "Models").shortcut('3'));
        sidebar.add_item(SidebarItem::new("config", "Config").shortcut('4'));
        sidebar.add_item(SidebarItem::new("logs", "Logs").shortcut('5'));

        sidebar.add_service(SidebarService::new("Trading", "Running", true));
        sidebar.add_service(SidebarService::new("Training", "Stopped", false));
        sidebar.add_service(SidebarService::new("Bridge", "Connected", true));

        sidebar.add_quick_action(QuickAction::new('K', "Kill Switch", "OFF"));
        sidebar.add_quick_action(QuickAction::new('P', "Trading", "Active"));

        sidebar
    }

    /// Add a navigation item
    pub fn add_item(&mut self, item: SidebarItem) {
        self.items.push(item);
    }

    /// Add a service
    pub fn add_service(&mut self, service: SidebarService) {
        self.services.push(service);
    }

    /// Add a quick action
    pub fn add_quick_action(&mut self, action: QuickAction) {
        self.quick_actions.push(action);
    }

    /// Set the active item by ID
    pub fn set_active(&mut self, id: &str) {
        for (i, item) in self.items.iter_mut().enumerate() {
            item.active = item.id == id;
            if item.active {
                self.selected = i;
                self.list_state.select(Some(i));
            }
        }
    }

    /// Show the sidebar
    pub fn show(&mut self) {
        self.visible = true;
        self.last_mouse_leave = None;
    }

    /// Hide the sidebar (if not pinned)
    pub fn hide(&mut self) {
        if !self.pinned {
            self.visible = false;
        }
    }

    /// Toggle pin state
    pub fn toggle_pin(&mut self) {
        self.pinned = !self.pinned;
        if self.pinned {
            self.visible = true;
        }
    }

    /// Handle mouse position (for auto-show/hide)
    pub fn handle_mouse(&mut self, x: u16, _y: u16) {
        if x <= self.hover_zone {
            self.show();
        } else if self.visible && !self.pinned && x > self.width {
            // Start hide delay
            if self.last_mouse_leave.is_none() {
                self.last_mouse_leave = Some(Instant::now());
            }
        }
    }

    /// Update hide delay (call every frame)
    pub fn update(&mut self) {
        if let Some(leave_time) = self.last_mouse_leave {
            if leave_time.elapsed() > self.hide_delay {
                self.hide();
                self.last_mouse_leave = None;
            }
        }
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
        if self.selected < self.items.len().saturating_sub(1) {
            self.selected += 1;
            self.list_state.select(Some(self.selected));
        }
    }

    /// Get the selected item
    pub fn selected_item(&self) -> Option<&SidebarItem> {
        self.items.get(self.selected)
    }

    /// Render the sidebar
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        // If not visible, just render a thin indicator line
        if !self.visible {
            // Thin peach line on the left edge
            for y in area.y..area.y + area.height {
                buf.get_mut(area.x, y)
                    .set_char('│')
                    .set_fg(theme.accent_muted);
            }
            return;
        }

        let sidebar_area = Rect {
            x: area.x,
            y: area.y,
            width: self.width.min(area.width),
            height: area.height,
        };

        // Clear area
        Clear.render(sidebar_area, buf);

        // Main block
        let pin_indicator = if self.pinned { " 📌" } else { "" };
        let block = Block::default()
            .title(format!("{} FOX ML{}", Separators::DIAMOND, pin_indicator))
            .title_style(Style::default().fg(theme.accent).bold())
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(theme.border))
            .style(Style::default().bg(theme.surface));

        let inner = block.inner(sidebar_area);
        block.render(sidebar_area, buf);

        // Split inner area
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(8),    // Navigation items
                Constraint::Length(1), // Separator
                Constraint::Length(self.services.len() as u16 + 2), // Services
                Constraint::Length(1), // Separator
                Constraint::Min(0),    // Quick actions
            ])
            .split(inner);

        // Navigation items
        self.render_navigation(sections[0], buf, theme);

        // Separator
        Paragraph::new("─".repeat(sections[1].width as usize))
            .style(Style::default().fg(theme.border))
            .render(sections[1], buf);

        // Services
        self.render_services(sections[2], buf, theme);

        // Separator
        Paragraph::new("─".repeat(sections[3].width as usize))
            .style(Style::default().fg(theme.border))
            .render(sections[3], buf);

        // Quick actions
        self.render_quick_actions(sections[4], buf, theme);
    }

    /// Render navigation items
    fn render_navigation(&mut self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let items: Vec<ListItem> = self
            .items
            .iter()
            .map(|item| {
                let marker = if item.active {
                    Separators::DIAMOND
                } else {
                    Separators::DIAMOND_EMPTY
                };

                let shortcut = item
                    .shortcut
                    .map(|s| format!(" [{}]", s))
                    .unwrap_or_default();

                let style = if item.active {
                    Style::default().fg(theme.accent)
                } else {
                    Style::default().fg(theme.text_secondary)
                };

                ListItem::new(format!("  {} {}{}", marker, item.label, shortcut)).style(style)
            })
            .collect();

        let list = List::new(items)
            .highlight_style(Style::default().fg(theme.text_primary).bold())
            .highlight_symbol("▸ ");

        ratatui::widgets::StatefulWidget::render(list, area, buf, &mut self.list_state);
    }

    /// Render services section
    fn render_services(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        // Header
        Paragraph::new("  Services")
            .style(Style::default().fg(theme.text_muted))
            .render(
                Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: 1,
                },
                buf,
            );

        // Services
        for (i, service) in self.services.iter().enumerate() {
            if i as u16 + 1 >= area.height {
                break;
            }

            let dot = if service.running {
                Separators::CIRCLE_FILLED
            } else {
                Separators::CIRCLE_EMPTY
            };

            let color = if service.running {
                theme.success
            } else {
                theme.text_muted
            };

            let line = format!("  {} {}  {}", dot, service.name, service.status);
            Paragraph::new(line)
                .style(Style::default().fg(color))
                .render(
                    Rect {
                        x: area.x,
                        y: area.y + i as u16 + 1,
                        width: area.width,
                        height: 1,
                    },
                    buf,
                );
        }
    }

    /// Render quick actions section
    fn render_quick_actions(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        // Header
        Paragraph::new("  Quick Actions")
            .style(Style::default().fg(theme.text_muted))
            .render(
                Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: 1,
                },
                buf,
            );

        // Actions
        for (i, action) in self.quick_actions.iter().enumerate() {
            if i as u16 + 1 >= area.height {
                break;
            }

            let line = format!("  [{}] {}: {}", action.shortcut, action.label, action.value);
            Paragraph::new(line)
                .style(Style::default().fg(theme.text_secondary))
                .render(
                    Rect {
                        x: area.x,
                        y: area.y + i as u16 + 1,
                        width: area.width,
                        height: 1,
                    },
                    buf,
                );
        }
    }

    /// Get the main content area (after sidebar)
    pub fn content_area(&self, area: Rect) -> Rect {
        if self.visible {
            Rect {
                x: area.x + self.width,
                y: area.y,
                width: area.width.saturating_sub(self.width),
                height: area.height,
            }
        } else {
            Rect {
                x: area.x + 1, // Account for indicator line
                y: area.y,
                width: area.width.saturating_sub(1),
                height: area.height,
            }
        }
    }
}

impl Default for Sidebar {
    fn default() -> Self {
        Self::with_default_items()
    }
}
