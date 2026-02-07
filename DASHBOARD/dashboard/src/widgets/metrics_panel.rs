//! Metrics panel widget

use ratatui::prelude::*;
use ratatui::widgets::*;

/// Metrics panel widget
pub struct MetricsPanel {
    portfolio_value: f64,
    daily_pnl: f64,
    cash_balance: f64,
    positions_count: usize,
}

impl MetricsPanel {
    pub fn new() -> Self {
        Self {
            portfolio_value: 0.0,
            daily_pnl: 0.0,
            cash_balance: 0.0,
            positions_count: 0,
        }
    }

    pub fn update(&mut self, portfolio: f64, pnl: f64, cash: f64, positions: usize) {
        self.portfolio_value = portfolio;
        self.daily_pnl = pnl;
        self.cash_balance = cash;
        self.positions_count = positions;
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title("Metrics")
            .borders(Borders::ALL);

        let pnl_color = if self.daily_pnl >= 0.0 {
            Color::Green
        } else {
            Color::Red
        };

        let text = vec![
            Line::from(vec![
                Span::styled("Portfolio Value: ", Style::default()),
                Span::styled(
                    format!("${:.2}", self.portfolio_value),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("Daily P&L: ", Style::default()),
                Span::styled(
                    format!("${:.2}", self.daily_pnl),
                    Style::default().fg(pnl_color),
                ),
            ]),
            Line::from(vec![
                Span::styled("Cash Balance: ", Style::default()),
                Span::styled(
                    format!("${:.2}", self.cash_balance),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::styled("Positions: ", Style::default()),
                Span::styled(
                    format!("{}", self.positions_count),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
        ];

        let paragraph = Paragraph::new(text).block(block);
        paragraph.render(area, buf);
    }
}
