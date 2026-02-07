//! Alpaca API types and WebSocket event structures
//!
//! Types for deserializing Alpaca trade updates and account events
//! from the IPC bridge.

use serde::{Deserialize, Serialize};

/// Alpaca connection status from the bridge
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlpacaStatus {
    pub available: bool,
    pub connected: bool,
    pub paper: Option<bool>,
    pub has_credentials: Option<bool>,
    pub error: Option<String>,
}

impl Default for AlpacaStatus {
    fn default() -> Self {
        Self {
            available: false,
            connected: false,
            paper: None,
            has_credentials: None,
            error: Some("Not connected".to_string()),
        }
    }
}

/// Event types from Alpaca streams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlpacaEventType {
    TradeUpdate,
    AccountUpdate,
    Connected,
    Disconnected,
    Authenticated,
    Error,
}

impl std::fmt::Display for AlpacaEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TradeUpdate => write!(f, "Trade"),
            Self::AccountUpdate => write!(f, "Account"),
            Self::Connected => write!(f, "Connected"),
            Self::Disconnected => write!(f, "Disconnected"),
            Self::Authenticated => write!(f, "Authenticated"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Trade update event types (order lifecycle)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeEvent {
    New,
    Fill,
    PartialFill,
    Canceled,
    Expired,
    DoneForDay,
    Replaced,
    Rejected,
    PendingNew,
    Stopped,
    PendingCancel,
    PendingReplace,
    Calculated,
    Suspended,
    OrderReplaceRejected,
    OrderCancelRejected,
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for TradeEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::New => write!(f, "NEW"),
            Self::Fill => write!(f, "FILL"),
            Self::PartialFill => write!(f, "PARTIAL"),
            Self::Canceled => write!(f, "CANCELED"),
            Self::Expired => write!(f, "EXPIRED"),
            Self::DoneForDay => write!(f, "DONE"),
            Self::Replaced => write!(f, "REPLACED"),
            Self::Rejected => write!(f, "REJECTED"),
            Self::PendingNew => write!(f, "PENDING"),
            Self::Stopped => write!(f, "STOPPED"),
            Self::PendingCancel => write!(f, "CANCELING"),
            Self::PendingReplace => write!(f, "REPLACING"),
            Self::Calculated => write!(f, "CALC"),
            Self::Suspended => write!(f, "SUSPENDED"),
            Self::OrderReplaceRejected => write!(f, "REPLACE_REJ"),
            Self::OrderCancelRejected => write!(f, "CANCEL_REJ"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
            Self::Unknown => write!(f, "???"),
        }
    }
}

/// Trade update data from Alpaca
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TradeUpdateData {
    pub event: String,
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub qty: f64,
    pub filled_qty: f64,
    pub filled_avg_price: Option<f64>,
    pub order_type: String,
    pub status: String,
    pub timestamp: String,
}

impl TradeUpdateData {
    /// Get a formatted summary of the trade
    pub fn summary(&self) -> String {
        let price_str = self
            .filled_avg_price
            .map(|p| format!("@${:.2}", p))
            .unwrap_or_default();

        format!(
            "{} {} {} {}/{}{}",
            self.event.to_uppercase(),
            self.side.to_uppercase(),
            self.symbol,
            self.filled_qty,
            self.qty,
            price_str
        )
    }

    /// Check if this is a significant event (fill, rejection, etc.)
    pub fn is_significant(&self) -> bool {
        matches!(
            self.event.as_str(),
            "fill" | "partial_fill" | "canceled" | "rejected" | "expired"
        )
    }
}

/// Alpaca event from the bridge
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlpacaEvent {
    pub source: String,
    pub event_type: String,
    pub timestamp: String,
    pub data: serde_json::Value,
}

impl AlpacaEvent {
    /// Try to parse as a trade update
    pub fn as_trade_update(&self) -> Option<TradeUpdateData> {
        if self.event_type == "trade_update" {
            serde_json::from_value(self.data.clone()).ok()
        } else {
            None
        }
    }

    /// Get a short summary for display
    pub fn summary(&self) -> String {
        if let Some(trade) = self.as_trade_update() {
            trade.summary()
        } else {
            format!("{}: {:?}", self.event_type, self.data)
        }
    }
}

/// Health status including Alpaca info
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BridgeHealth {
    pub status: String,
    pub observability_available: bool,
    pub alpaca_available: bool,
    pub alpaca_connected: bool,
    pub active_connections: i32,
}

impl Default for BridgeHealth {
    fn default() -> Self {
        Self {
            status: "unknown".to_string(),
            observability_available: false,
            alpaca_available: false,
            alpaca_connected: false,
            active_connections: 0,
        }
    }
}
