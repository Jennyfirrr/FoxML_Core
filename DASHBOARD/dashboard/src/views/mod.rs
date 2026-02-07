//! View modules for different dashboard screens

use anyhow::Result;
use ratatui::prelude::*;

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

/// Trait for views that can be rendered
pub trait ViewTrait {
    /// Render the view
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()>;

    /// Handle key input
    fn handle_key(&mut self, _key: crossterm::event::KeyCode) -> Result<bool> {
        // Default: don't handle
        Ok(false)
    }
}

// View rendering is now handled in App struct
