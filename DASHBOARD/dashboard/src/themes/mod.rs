//! Theme system for color management
//!
//! Provides semantic color system with auto-detection from system configs.
//! See theme.rs for the full Theme struct and color utilities.

pub mod theme;
pub mod waybar;
pub mod hyprland;
pub mod tmux;
pub mod kitty;

pub use theme::{Theme, LegacyTheme};
