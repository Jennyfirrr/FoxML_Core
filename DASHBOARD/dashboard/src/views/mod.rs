//! View modules for different dashboard screens

use anyhow::Result;
use ratatui::prelude::*;
use std::path::PathBuf;

pub mod launcher;
pub mod trading;
pub mod training;
pub mod overview;
pub mod placeholder;
pub mod training_launcher;
pub mod config_editor;
pub mod log_viewer;
pub mod service_manager;
pub mod model_selector;
pub mod file_browser;
pub mod settings;

/// Available views
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    Launcher,
    Trading,
    Training,
    Overview,
    Placeholder,
    TrainingLauncher,
    ConfigEditor,
    LogViewer,
    ServiceManager,
    ModelSelector,
    FileBrowser,
    Settings,
}

/// Action returned by a view's key handler
pub enum ViewAction {
    /// Key handled, stay in current view
    Continue,
    /// Request to navigate back (to launcher)
    Back,
    /// Request TUI suspension to spawn an external editor on a file
    SpawnEditor(PathBuf),
}

/// Trait for views that can be rendered
pub trait ViewTrait {
    /// Render the view
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()>;

    /// Handle key input
    fn handle_key(&mut self, _key: crossterm::event::KeyCode) -> Result<ViewAction> {
        // Default: don't handle
        Ok(ViewAction::Continue)
    }
}

// View rendering is now handled in App struct
