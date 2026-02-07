//! Metrics data structures

use serde::{Deserialize, Serialize};

/// Trading metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub portfolio_value: f64,
    pub daily_pnl: f64,
    pub cash_balance: f64,
    pub positions_count: usize,
    pub sharpe_ratio: Option<f64>,
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            portfolio_value: 0.0,
            daily_pnl: 0.0,
            cash_balance: 0.0,
            positions_count: 0,
            sharpe_ratio: None,
        }
    }
}
