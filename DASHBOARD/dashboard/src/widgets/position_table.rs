//! Position table widget - displays current positions with P&L

use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::api::events::Position;
use crate::themes::Theme;

/// Sort mode for position table
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSort {
    Symbol,
    PnlDesc,
    SizeDesc,
    WeightDesc,
}

impl PositionSort {
    /// Cycle to next sort mode
    pub fn next(self) -> Self {
        match self {
            Self::Symbol => Self::PnlDesc,
            Self::PnlDesc => Self::SizeDesc,
            Self::SizeDesc => Self::WeightDesc,
            Self::WeightDesc => Self::Symbol,
        }
    }

    /// Display label for sort mode
    pub fn label(self) -> &'static str {
        match self {
            Self::Symbol => "Symbol",
            Self::PnlDesc => "P&L",
            Self::SizeDesc => "Size",
            Self::WeightDesc => "Weight",
        }
    }
}

/// Position table widget
pub struct PositionTable {
    positions: Vec<Position>,
    selected: usize,
    sort_mode: PositionSort,
}

impl PositionTable {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            selected: 0,
            sort_mode: PositionSort::Symbol,
        }
    }

    /// Update positions from API
    pub fn update(&mut self, positions: Vec<Position>) {
        self.positions = positions;
        self.apply_sort();
        // Keep selection in bounds
        if self.selected >= self.positions.len() && !self.positions.is_empty() {
            self.selected = self.positions.len() - 1;
        }
    }

    /// Cycle to next sort mode
    pub fn cycle_sort(&mut self) {
        self.sort_mode = self.sort_mode.next();
        self.apply_sort();
    }

    /// Get current sort mode label
    pub fn sort_label(&self) -> &'static str {
        self.sort_mode.label()
    }

    /// Apply current sort to positions
    fn apply_sort(&mut self) {
        match self.sort_mode {
            PositionSort::Symbol => {
                self.positions.sort_by(|a, b| a.symbol.cmp(&b.symbol));
            }
            PositionSort::PnlDesc => {
                self.positions.sort_by(|a, b| {
                    b.unrealized_pnl.partial_cmp(&a.unrealized_pnl).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            PositionSort::SizeDesc => {
                self.positions.sort_by(|a, b| {
                    b.shares.abs().cmp(&a.shares.abs())
                });
            }
            PositionSort::WeightDesc => {
                self.positions.sort_by(|a, b| {
                    b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    /// Get number of positions
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Move selection up
    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    /// Move selection down
    pub fn down(&mut self) {
        if self.selected < self.positions.len().saturating_sub(1) {
            self.selected += 1;
        }
    }

    /// Get selected position
    pub fn selected_position(&self) -> Option<&Position> {
        self.positions.get(self.selected)
    }

    /// Render with theme support
    pub fn render_themed(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        if self.positions.is_empty() {
            let text = "No positions";
            buf.set_string(
                area.x + (area.width.saturating_sub(text.len() as u16)) / 2,
                area.y + area.height / 2,
                text,
                Style::default().fg(theme.text_muted),
            );
            return;
        }

        // Header row
        let header = vec!["Symbol", "Shares", "Entry", "Current", "P&L", "P&L%", "Wgt%"];
        let header_style = Style::default().fg(theme.text_primary).bold();

        // Calculate column widths
        let widths = [8u16, 8, 9, 9, 11, 8, 6]; // Total ~59
        let total_width: u16 = widths.iter().sum();

        // Render header
        let mut x = area.x;
        for (i, col) in header.iter().enumerate() {
            buf.set_string(x, area.y, *col, header_style);
            x += widths[i] + 1;
        }

        // Separator line
        let sep_y = area.y + 1;
        let sep = "─".repeat(total_width.min(area.width) as usize);
        buf.set_string(area.x, sep_y, &sep, Style::default().fg(theme.border));

        // Data rows
        let data_start_y = area.y + 2;
        let visible_rows = (area.height as usize).saturating_sub(2);

        // Calculate scroll offset to keep selected item visible
        let scroll_offset = if self.selected >= visible_rows {
            self.selected - visible_rows + 1
        } else {
            0
        };

        for (i, pos) in self.positions.iter().skip(scroll_offset).take(visible_rows).enumerate() {
            let y = data_start_y + i as u16;
            if y >= area.bottom() {
                break;
            }

            let is_selected = (i + scroll_offset) == self.selected;
            let row_bg = if is_selected {
                theme.surface
            } else {
                theme.background
            };

            let pnl_color = if pos.unrealized_pnl >= 0.0 {
                theme.success
            } else {
                theme.error
            };

            // Format values
            let shares_str = format!("{}", pos.shares);
            let entry_str = format!("${:.2}", pos.entry_price);
            let current_str = format!("${:.2}", pos.current_price);
            let pnl_str = if pos.unrealized_pnl >= 0.0 {
                format!("+${:.2}", pos.unrealized_pnl)
            } else {
                format!("-${:.2}", pos.unrealized_pnl.abs())
            };
            let pnl_pct_str = if pos.unrealized_pnl_pct >= 0.0 {
                format!("+{:.1}%", pos.unrealized_pnl_pct)
            } else {
                format!("{:.1}%", pos.unrealized_pnl_pct)
            };
            let weight_str = format!("{:.1}%", pos.weight * 100.0);

            // Render row
            let mut x = area.x;
            let row_style = Style::default().bg(row_bg);

            // Symbol
            buf.set_string(x, y, &pos.symbol, row_style.fg(theme.text_primary));
            x += widths[0] + 1;

            // Shares
            buf.set_string(x, y, &shares_str, row_style.fg(theme.text_secondary));
            x += widths[1] + 1;

            // Entry
            buf.set_string(x, y, &entry_str, row_style.fg(theme.text_muted));
            x += widths[2] + 1;

            // Current
            buf.set_string(x, y, &current_str, row_style.fg(theme.text_secondary));
            x += widths[3] + 1;

            // P&L
            buf.set_string(x, y, &pnl_str, row_style.fg(pnl_color));
            x += widths[4] + 1;

            // P&L %
            buf.set_string(x, y, &pnl_pct_str, row_style.fg(pnl_color));
            x += widths[5] + 1;

            // Weight
            buf.set_string(x, y, &weight_str, row_style.fg(theme.accent));
        }
    }

    /// Render without theme (fallback)
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Positions")
            .borders(Borders::ALL);

        if self.positions.is_empty() {
            let text = "No positions";
            let paragraph = Paragraph::new(text).block(block);
            paragraph.render(area, buf);
            return;
        }

        // Build table
        let header_cells = ["Symbol", "Shares", "Entry", "Current", "P&L", "P&L %", "Wgt"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::White).bold()));
        let header = Row::new(header_cells).height(1);

        let rows = self.positions.iter().enumerate().map(|(i, pos)| {
            let pnl_color = if pos.unrealized_pnl >= 0.0 {
                Color::Green
            } else {
                Color::Red
            };

            let selected_style = if i == self.selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(pos.symbol.clone()),
                Cell::from(format!("{}", pos.shares)),
                Cell::from(format!("${:.2}", pos.entry_price)),
                Cell::from(format!("${:.2}", pos.current_price)),
                Cell::from(format!("${:.2}", pos.unrealized_pnl))
                    .style(Style::default().fg(pnl_color)),
                Cell::from(format!("{:.1}%", pos.unrealized_pnl_pct))
                    .style(Style::default().fg(pnl_color)),
                Cell::from(format!("{:.1}%", pos.weight * 100.0)),
            ])
            .style(selected_style)
        });

        let widths = [
            Constraint::Length(8),  // Symbol
            Constraint::Length(8),  // Shares
            Constraint::Length(10), // Entry
            Constraint::Length(10), // Current
            Constraint::Length(12), // P&L
            Constraint::Length(8),  // P&L %
            Constraint::Length(6),  // Weight
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(block);

        use ratatui::widgets::Widget;
        Widget::render(table, area, buf);
    }
}

impl Default for PositionTable {
    fn default() -> Self {
        Self::new()
    }
}
