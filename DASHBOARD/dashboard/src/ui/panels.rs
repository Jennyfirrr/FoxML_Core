//! Reusable panel components
//!
//! Provides styled panels/cards with consistent theming.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::themes::Theme;

/// Panel style variants
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PanelStyle {
    /// Default panel style
    #[default]
    Default,
    /// Focused panel (highlighted border)
    Focused,
    /// Elevated panel (for modals, dropdowns)
    Elevated,
    /// Success state
    Success,
    /// Warning state
    Warning,
    /// Error state
    Error,
    /// Muted/disabled state
    Muted,
}

/// A styled panel component
#[derive(Clone)]
pub struct Panel<'a> {
    title: Option<&'a str>,
    style: PanelStyle,
    theme: &'a Theme,
    padding: Padding,
    borders: Borders,
}

impl<'a> Panel<'a> {
    /// Create a new panel with the given theme
    pub fn new(theme: &'a Theme) -> Self {
        Self {
            title: None,
            style: PanelStyle::Default,
            theme,
            padding: Padding::uniform(1),
            borders: Borders::ALL,
        }
    }

    /// Set the panel title
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = Some(title);
        self
    }

    /// Set the panel style
    pub fn style(mut self, style: PanelStyle) -> Self {
        self.style = style;
        self
    }

    /// Set focused style
    pub fn focused(mut self, focused: bool) -> Self {
        if focused {
            self.style = PanelStyle::Focused;
        }
        self
    }

    /// Set the padding
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Set no padding
    pub fn no_padding(mut self) -> Self {
        self.padding = Padding::zero();
        self
    }

    /// Set borders
    pub fn borders(mut self, borders: Borders) -> Self {
        self.borders = borders;
        self
    }

    /// Build the Block widget
    pub fn block(&self) -> Block<'a> {
        let border_color = match self.style {
            PanelStyle::Default => self.theme.border,
            PanelStyle::Focused => self.theme.border_focused,
            PanelStyle::Elevated => self.theme.accent,
            PanelStyle::Success => self.theme.success,
            PanelStyle::Warning => self.theme.warning,
            PanelStyle::Error => self.theme.error,
            PanelStyle::Muted => self.theme.text_muted,
        };

        let title_color = match self.style {
            PanelStyle::Default => self.theme.text_secondary,
            PanelStyle::Focused => self.theme.accent,
            PanelStyle::Elevated => self.theme.accent,
            PanelStyle::Success => self.theme.success,
            PanelStyle::Warning => self.theme.warning,
            PanelStyle::Error => self.theme.error,
            PanelStyle::Muted => self.theme.text_muted,
        };

        let mut block = Block::default()
            .borders(self.borders)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .padding(self.padding);

        if let Some(title) = self.title {
            block = block
                .title(title)
                .title_style(Style::default().fg(title_color).bold());
        }

        block
    }

    /// Get the inner area after accounting for borders and padding
    pub fn inner(&self, area: Rect) -> Rect {
        self.block().inner(area)
    }
}

/// A card component - elevated panel with shadow effect (simulated)
pub struct Card<'a> {
    panel: Panel<'a>,
}

impl<'a> Card<'a> {
    /// Create a new card with the given theme
    pub fn new(theme: &'a Theme) -> Self {
        Self {
            panel: Panel::new(theme).style(PanelStyle::Elevated),
        }
    }

    /// Set the card title
    pub fn title(mut self, title: &'a str) -> Self {
        self.panel = self.panel.title(title);
        self
    }

    /// Build the Block widget
    pub fn block(&self) -> Block<'a> {
        self.panel.block()
    }

    /// Get the inner area
    pub fn inner(&self, area: Rect) -> Rect {
        self.panel.inner(area)
    }
}

/// Simple info box for displaying key-value pairs
pub struct InfoBox<'a> {
    items: Vec<(&'a str, String)>,
    theme: &'a Theme,
    title: Option<&'a str>,
}

impl<'a> InfoBox<'a> {
    /// Create a new info box
    pub fn new(theme: &'a Theme) -> Self {
        Self {
            items: Vec::new(),
            theme,
            title: None,
        }
    }

    /// Set the title
    pub fn title(mut self, title: &'a str) -> Self {
        self.title = Some(title);
        self
    }

    /// Add an item
    pub fn item(mut self, label: &'a str, value: impl Into<String>) -> Self {
        self.items.push((label, value.into()));
        self
    }

    /// Render the info box
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let panel = Panel::new(self.theme);
        let panel = if let Some(title) = self.title {
            panel.title(title)
        } else {
            panel
        };

        let block = panel.block();
        let inner = block.inner(area);
        block.render(area, buf);

        // Render items
        for (i, (label, value)) in self.items.iter().enumerate() {
            if i as u16 >= inner.height {
                break;
            }

            let row = Rect {
                x: inner.x,
                y: inner.y + i as u16,
                width: inner.width,
                height: 1,
            };

            // Split into label and value
            let label_width = (inner.width / 2).min(20);
            let value_width = inner.width.saturating_sub(label_width);

            // Render label
            let label_area = Rect {
                x: row.x,
                y: row.y,
                width: label_width,
                height: 1,
            };
            Paragraph::new(format!("{}:", label))
                .style(Style::default().fg(self.theme.text_secondary))
                .render(label_area, buf);

            // Render value
            let value_area = Rect {
                x: row.x + label_width,
                y: row.y,
                width: value_width,
                height: 1,
            };
            Paragraph::new(value.as_str())
                .style(Style::default().fg(self.theme.text_primary))
                .render(value_area, buf);
        }
    }
}

/// Progress bar component
pub struct ProgressBar<'a> {
    progress: f64,
    label: Option<&'a str>,
    theme: &'a Theme,
    show_percentage: bool,
}

impl<'a> ProgressBar<'a> {
    /// Create a new progress bar
    pub fn new(theme: &'a Theme, progress: f64) -> Self {
        Self {
            progress: progress.clamp(0.0, 1.0),
            label: None,
            theme,
            show_percentage: true,
        }
    }

    /// Set the label
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Hide percentage
    pub fn hide_percentage(mut self) -> Self {
        self.show_percentage = false;
        self
    }

    /// Render the progress bar
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.height < 1 {
            return;
        }

        // Calculate filled width
        let bar_width = if self.show_percentage {
            area.width.saturating_sub(6) // Leave room for " XX%"
        } else {
            area.width
        };

        let filled = ((bar_width as f64) * self.progress) as u16;
        let empty = bar_width.saturating_sub(filled);

        // Build the bar string
        let filled_str = "█".repeat(filled as usize);
        let empty_str = "░".repeat(empty as usize);

        // Choose color based on progress
        let bar_color = if self.progress >= 1.0 {
            self.theme.success
        } else if self.progress >= 0.5 {
            self.theme.accent
        } else if self.progress >= 0.25 {
            self.theme.warning
        } else {
            self.theme.error
        };

        // Render filled portion
        let filled_area = Rect {
            x: area.x,
            y: area.y,
            width: filled,
            height: 1,
        };
        Paragraph::new(filled_str)
            .style(Style::default().fg(bar_color))
            .render(filled_area, buf);

        // Render empty portion
        let empty_area = Rect {
            x: area.x + filled,
            y: area.y,
            width: empty,
            height: 1,
        };
        Paragraph::new(empty_str)
            .style(Style::default().fg(self.theme.text_muted))
            .render(empty_area, buf);

        // Render percentage
        if self.show_percentage {
            let pct = format!(" {:>3.0}%", self.progress * 100.0);
            let pct_area = Rect {
                x: area.x + bar_width,
                y: area.y,
                width: 5,
                height: 1,
            };
            Paragraph::new(pct)
                .style(Style::default().fg(self.theme.text_secondary))
                .render(pct_area, buf);
        }
    }
}
