//! UI components module
//!
//! Reusable UI components for the dashboard including panels, borders,
//! command palette, notifications, status bar, and animations.

pub mod animation;
pub mod borders;
pub mod dialog;
pub mod panels;
pub mod status_bar;
pub mod notification;
pub mod command_palette;
pub mod help_overlay;
pub mod sidebar;

pub use borders::{RoundedBorder, ROUNDED_BORDER_SET};
pub use panels::{Panel, PanelStyle};
pub use status_bar::StatusBar;
pub use notification::{Notification, NotificationLevel, NotificationManager};
pub use command_palette::{CommandPalette, Command, CommandCategory};
pub use help_overlay::HelpOverlay;
pub use sidebar::Sidebar;
