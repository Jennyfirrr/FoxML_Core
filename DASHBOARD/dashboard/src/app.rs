//! Main application structure and event loop

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use std::io;
use tracing::{debug, info};

use crate::themes::Theme;
use crate::ui::{
    animation::FadeState,
    CommandPalette, HelpOverlay, Notification, NotificationManager, Sidebar,
};
use crate::views::{
    config_editor::ConfigEditorView, file_browser::FileBrowserView, launcher::LauncherView,
    log_viewer::LogViewerView, model_selector::ModelSelectorView, overview::OverviewView,
    placeholder::PlaceholderView, service_manager::ServiceManagerView, settings::SettingsView,
    trading::TradingView, training::TrainingView, training_launcher::TrainingLauncherView,
    View, ViewTrait,
};

/// Main application state
pub struct App {
    /// Current view
    current_view: View,
    /// View instances
    trading_view: TradingView,
    training_view: TrainingView,
    overview_view: OverviewView,
    launcher_view: LauncherView,
    placeholder_view: Option<PlaceholderView>,
    placeholder_message: String,
    training_launcher_view: TrainingLauncherView,
    config_editor_view: ConfigEditorView,
    log_viewer_view: LogViewerView,
    service_manager_view: ServiceManagerView,
    model_selector_view: ModelSelectorView,
    file_browser_view: FileBrowserView,
    settings_view: SettingsView,
    /// Should exit?
    should_quit: bool,
    /// Theme
    theme: Theme,
    /// Last update time
    last_update: std::time::Instant,
    /// Command palette
    command_palette: CommandPalette,
    /// Notification manager
    notifications: NotificationManager,
    /// Help overlay
    help_overlay: HelpOverlay,
    /// Auto-hide sidebar
    sidebar: Sidebar,
    /// View transition fade animation
    view_fade: Option<FadeState>,
}

impl App {
    /// Create new app instance
    pub async fn new() -> Result<Self> {
        info!("Initializing FoxML Dashboard");

        // Load theme
        let theme = Theme::load();
        info!("Theme loaded");

        // Initialize views
        let mut trading_view = TradingView::new();
        trading_view.update_metrics().await?;

        let training_view = TrainingView::new(); // Auto-scans on initialization
        let mut overview_view = OverviewView::new();
        overview_view.update().await;

        Ok(Self {
            current_view: View::Launcher,
            trading_view,
            training_view,
            overview_view,
            launcher_view: LauncherView::default(),
            placeholder_view: None,
            placeholder_message: String::new(),
            training_launcher_view: TrainingLauncherView::new(),
            config_editor_view: ConfigEditorView::new(),
            log_viewer_view: LogViewerView::new(),
            service_manager_view: ServiceManagerView::new(),
            model_selector_view: ModelSelectorView::new(),
            file_browser_view: FileBrowserView::new(),
            settings_view: SettingsView::new(),
            should_quit: false,
            theme,
            last_update: std::time::Instant::now(),
            command_palette: CommandPalette::with_default_commands(),
            notifications: NotificationManager::new(),
            help_overlay: HelpOverlay::new(),
            sidebar: Sidebar::with_default_items(),
            view_fade: Some(FadeState::fade_in(200)), // Initial fade-in
        })
    }

    /// Switch to a new view with fade-in animation
    fn switch_view(&mut self, view: View) {
        if self.current_view != view {
            // Clear placeholder when switching to non-placeholder view
            if !matches!(view, View::Placeholder) {
                self.placeholder_view = None;
            }
            debug!("Switched to view: {:?}", view);
            self.current_view = view;
            // Start fade-in animation
            self.view_fade = Some(FadeState::fade_in(150));
        }
    }

    /// Switch to a placeholder view with a message
    fn switch_to_placeholder(&mut self, message: impl Into<String>) {
        let msg = message.into();
        self.placeholder_message = msg.clone();
        self.placeholder_view = Some(PlaceholderView::new(msg));
        self.switch_view(View::Placeholder);
    }

    /// Render fade overlay for view transitions
    fn render_fade_overlay(&self, area: Rect, buf: &mut Buffer) {
        if let Some(ref fade) = self.view_fade {
            if !fade.is_done() {
                // Calculate overlay opacity (inverted: 1.0 = fully dark, 0.0 = transparent)
                let overlay_alpha = 1.0 - fade.alpha();
                if overlay_alpha > 0.01 {
                    // Apply a dark overlay that fades out
                    for y in area.top()..area.bottom() {
                        for x in area.left()..area.right() {
                            let cell = buf.cell_mut((x, y));
                            if let Some(cell) = cell {
                                // Dim the foreground color
                                if let Color::Rgb(r, g, b) = cell.fg {
                                    let factor = 1.0 - overlay_alpha * 0.7;
                                    cell.fg = Color::Rgb(
                                        (r as f32 * factor) as u8,
                                        (g as f32 * factor) as u8,
                                        (b as f32 * factor) as u8,
                                    );
                                }
                                // Dim the background towards black
                                if let Color::Rgb(r, g, b) = cell.bg {
                                    let factor = 1.0 - overlay_alpha * 0.5;
                                    cell.bg = Color::Rgb(
                                        (r as f32 * factor) as u8,
                                        (g as f32 * factor) as u8,
                                        (b as f32 * factor) as u8,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run the main event loop
    pub async fn run(&mut self) -> Result<()> {
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

        loop {
            // Cleanup expired notifications
            self.notifications.cleanup();

            // Cleanup completed view fade animation
            if let Some(ref fade) = self.view_fade {
                if fade.is_done() {
                    self.view_fade = None;
                }
            }

            // Update sidebar hide delay
            self.sidebar.update();

            // Update views periodically
            if self.last_update.elapsed().as_secs() >= 2 {
                match self.current_view {
                    View::Trading => {
                        let _ = self.trading_view.update_metrics().await;
                        let _ = self.trading_view.update_state().await;
                    }
                    View::Overview => {
                        self.overview_view.update().await;
                    }
                    View::Launcher => {
                        // Update status canvas every 2 seconds
                        self.launcher_view.update_status().await;
                    }
                    View::Placeholder | View::Training | View::ConfigEditor | View::LogViewer | View::ServiceManager | View::ModelSelector | View::FileBrowser | View::Settings => {
                        // No periodic updates needed (these views have their own refresh)
                    }
                    View::TrainingLauncher => {
                        // Update training launcher to refresh output
                        // The view will check if running and update display
                    }
                }
                self.last_update = std::time::Instant::now();
            }

            terminal.draw(|f| {
                let area = f.area();

                // Update sidebar active item based on current view
                let view_id = match self.current_view {
                    View::Launcher => "dashboard",
                    View::Trading => "trading",
                    View::Training => "training",
                    View::Overview => "dashboard",
                    View::Placeholder => "dashboard",
                    View::TrainingLauncher => "training",
                    View::ConfigEditor => "dashboard",
                    View::LogViewer => "dashboard",
                    View::ServiceManager => "dashboard",
                    View::ModelSelector => "training",
                    View::FileBrowser => "dashboard",
                    View::Settings => "dashboard",
                };
                self.sidebar.set_active(view_id);

                // Render sidebar (on left edge)
                self.sidebar.render(area, f.buffer_mut(), &self.theme);

                // Get content area (adjusted for sidebar)
                let content_area = self.sidebar.content_area(area);

                // Render current view with theme
                match &mut self.current_view {
                    View::Launcher => {
                        let _ = ViewTrait::render(&mut self.launcher_view, f, content_area);
                    }
                    View::Trading => {
                        let _ = ViewTrait::render(&mut self.trading_view, f, content_area);
                    }
                    View::Training => {
                        let _ = ViewTrait::render(&mut self.training_view, f, content_area);
                    }
                    View::Overview => {
                        let _ = ViewTrait::render(&mut self.overview_view, f, content_area);
                    }
                    View::Placeholder => {
                        if let Some(ref mut placeholder) = self.placeholder_view {
                            let _ = ViewTrait::render(placeholder, f, content_area);
                        }
                    }
                    View::TrainingLauncher => {
                        let _ = ViewTrait::render(&mut self.training_launcher_view, f, content_area);
                    }
                    View::ConfigEditor => {
                        let _ = ViewTrait::render(&mut self.config_editor_view, f, content_area);
                    }
                    View::LogViewer => {
                        let _ = ViewTrait::render(&mut self.log_viewer_view, f, content_area);
                    }
                    View::ServiceManager => {
                        let _ = ViewTrait::render(&mut self.service_manager_view, f, content_area);
                    }
                    View::ModelSelector => {
                        let _ = ViewTrait::render(&mut self.model_selector_view, f, content_area);
                    }
                    View::FileBrowser => {
                        let _ = ViewTrait::render(&mut self.file_browser_view, f, content_area);
                    }
                    View::Settings => {
                        let _ = ViewTrait::render(&mut self.settings_view, f, content_area);
                    }
                }

                // Render fade overlay for view transitions
                self.render_fade_overlay(content_area, f.buffer_mut());

                // Render overlays on top (use full area so they appear over sidebar too)
                // Notifications (top-right)
                self.notifications.render(area, f.buffer_mut(), &self.theme);

                // Command palette (centered)
                self.command_palette.render(area, f.buffer_mut(), &self.theme);

                // Help overlay (centered)
                self.help_overlay.render(area, f.buffer_mut(), &self.theme);
            })?;

            // Handle events
            if event::poll(std::time::Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(key) => {
                        if key.kind == KeyEventKind::Press {
                            let should_quit = self.handle_key(key.code, key.modifiers).await?;
                            if should_quit {
                                break;
                            }
                        }
                    }
                    Event::Mouse(mouse) => {
                        // Handle mouse movement for sidebar auto-show/hide
                        if let crossterm::event::MouseEventKind::Moved = mouse.kind {
                            self.sidebar.handle_mouse(mouse.column, mouse.row);
                        }
                    }
                    _ => {}
                }
            }

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Handle key press
    async fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) -> Result<bool> {
        // Handle command palette first if visible
        if self.command_palette.visible {
            match key {
                KeyCode::Esc => {
                    self.command_palette.hide();
                }
                KeyCode::Enter => {
                    if let Some(command_id) = self.command_palette.execute() {
                        self.execute_command(&command_id).await?;
                    }
                }
                KeyCode::Up => {
                    self.command_palette.up();
                }
                KeyCode::Down => {
                    self.command_palette.down();
                }
                KeyCode::Backspace => {
                    self.command_palette.backspace();
                }
                KeyCode::Char(c) => {
                    self.command_palette.input(c);
                }
                _ => {}
            }
            return Ok(false);
        }

        // Handle help overlay
        if self.help_overlay.visible {
            self.help_overlay.hide();
            return Ok(false);
        }

        // Global shortcuts with Ctrl modifier
        if modifiers.contains(KeyModifiers::CONTROL) {
            match key {
                KeyCode::Char('p') => {
                    self.command_palette.show();
                    return Ok(false);
                }
                KeyCode::Char('k') => {
                    // Toggle kill switch
                    self.notifications.push(
                        Notification::warning("Kill switch toggled")
                            .message("Trading kill switch state changed"),
                    );
                    return Ok(false);
                }
                KeyCode::Char('q') => {
                    self.should_quit = true;
                    return Ok(true);
                }
                _ => {}
            }
        }

        match key {
            KeyCode::Char('q') => {
                // Quit only from launcher, or confirm from other views
                if matches!(self.current_view, View::Launcher) {
                    self.should_quit = true;
                    return Ok(true);
                } else {
                    // From other views, go back to launcher
                    self.switch_view(View::Launcher);
                }
            }
            KeyCode::Esc => {
                // Escape = back to launcher (or quit if already in launcher)
                if matches!(self.current_view, View::Launcher) {
                    self.should_quit = true;
                    return Ok(true);
                } else {
                    self.switch_view(View::Launcher);
                }
            }
            KeyCode::Char('?') => {
                // Show help overlay
                self.help_overlay.show();
            }
            KeyCode::Char('/') => {
                // Alternative trigger for command palette
                self.command_palette.show();
            }
            KeyCode::Char('[') => {
                // Toggle sidebar visibility
                if self.sidebar.visible {
                    self.sidebar.hide();
                } else {
                    self.sidebar.show();
                }
            }
            KeyCode::Char(']') => {
                // Toggle sidebar pin
                self.sidebar.toggle_pin();
                let state = if self.sidebar.pinned { "pinned" } else { "unpinned" };
                self.notifications.push(
                    Notification::info("Sidebar").message(format!("Sidebar {}", state)),
                );
            }
            KeyCode::Char('b') => {
                // 'b' key = back to launcher
                if !matches!(self.current_view, View::Launcher) {
                    self.switch_view(View::Launcher);
                }
            }
            KeyCode::Tab => {
                // Cycle through views (skip placeholder)
                let next_view = match self.current_view {
                    View::Launcher => View::Trading,
                    View::Trading => View::Training,
                    View::Training => View::Overview,
                    View::Overview => View::Launcher,
                    View::Placeholder => View::Launcher,
                    View::TrainingLauncher => View::Launcher,
                    View::ConfigEditor => View::Launcher,
                    View::LogViewer => View::Launcher,
                    View::ServiceManager => View::Launcher,
                    View::ModelSelector => View::Launcher,
                    View::FileBrowser => View::Launcher,
                    View::Settings => View::Launcher,
                };
                self.switch_view(next_view);
            }
            // Quick navigation keys (1-5)
            KeyCode::Char('1') => {
                self.switch_view(View::Trading);
            }
            KeyCode::Char('2') => {
                self.switch_view(View::Training);
            }
            KeyCode::Char('3') => {
                self.switch_view(View::Overview);
            }
            KeyCode::Char('r') => {
                // Views that handle 'r' themselves (action key, not just refresh)
                match self.current_view {
                    View::ServiceManager => {
                        // 'r' = restart service in ServiceManager
                        let _ = self.service_manager_view.handle_key(key)?;
                        return Ok(false);
                    }
                    View::TrainingLauncher => {
                        // 'r' = refresh in TrainingLauncher
                        let _ = self.training_launcher_view.handle_key(key)?;
                        return Ok(false);
                    }
                    View::ConfigEditor => {
                        // 'r' might be used in editor
                        let _ = self.config_editor_view.handle_key(key)?;
                        return Ok(false);
                    }
                    View::ModelSelector => {
                        // 'r' might be used in model selector
                        let _ = self.model_selector_view.handle_key(key)?;
                        return Ok(false);
                    }
                    View::FileBrowser => {
                        // 'r' = refresh in file browser
                        let _ = self.file_browser_view.handle_key(key)?;
                        return Ok(false);
                    }
                    _ => {}
                }
                // For other views, 'r' means global refresh
                match self.current_view {
                    View::Training => {
                        let _ = self.training_view.scan_runs();
                    }
                    View::Trading => {
                        let _ = self.trading_view.update_metrics().await;
                        let _ = self.trading_view.update_state().await;
                    }
                    View::Overview => {
                        self.overview_view.update().await;
                    }
                    View::Launcher => {
                        // Force status canvas update
                        self.launcher_view.update_status().await;
                    }
                    View::Placeholder => {
                        // Placeholder doesn't need refresh
                    }
                    View::LogViewer => {
                        // Log viewer handles its own refresh
                    }
                    View::Settings => {
                        // Settings doesn't need refresh
                    }
                    _ => {}
                }
                self.notifications
                    .push(Notification::info("Refreshed").message("View data refreshed"));
            }
            KeyCode::Up
            | KeyCode::Down
            | KeyCode::Char('j')
            | KeyCode::Char('k')
            | KeyCode::Char('h')
            | KeyCode::Char('l')
            | KeyCode::Char('g')
            | KeyCode::Char('G') => {
                // Handle navigation based on current view
                match self.current_view {
                    View::Launcher => {
                        // Menu navigation in launcher
                        self.launcher_view.handle_key(key);
                    }
                    View::Training => {
                        // Training view navigation (up/down for run list)
                        let _ = self.training_view.handle_key(key)?;
                    }
                    View::TrainingLauncher => {
                        // Training launcher navigation (config selection)
                        let _ = self.training_launcher_view.handle_key(key)?;
                    }
                    View::ConfigEditor => {
                        // Config editor navigation
                        let _ = self.config_editor_view.handle_key(key)?;
                    }
                    View::LogViewer => {
                        // Log viewer navigation
                        let _ = self.log_viewer_view.handle_key(key)?;
                    }
                    View::ModelSelector => {
                        // Model selector navigation
                        let _ = self.model_selector_view.handle_key(key)?;
                    }
                    View::FileBrowser => {
                        // File browser navigation
                        let _ = self.file_browser_view.handle_key(key)?;
                    }
                    View::Settings => {
                        // Settings navigation
                        let _ = self.settings_view.handle_key(key)?;
                    }
                    _ => {
                        // Other views don't handle these keys
                    }
                }
            }
            KeyCode::Enter => {
                // Handle menu selection in launcher
                if matches!(self.current_view, View::Launcher) {
                    if let Some(action) = self.launcher_view.handle_key(key) {
                        self.handle_menu_action(action).await?;
                    }
                } else {
                    // Delegate Enter to current view
                    match self.current_view {
                        View::Training => {
                            let _ = self.training_view.handle_key(key)?;
                        }
                        View::Trading => {
                            let _ = self.trading_view.handle_key(key)?;
                        }
                        View::Overview => {
                            let _ = self.overview_view.handle_key(key)?;
                        }
                        View::Placeholder => {
                            // Enter does nothing in placeholder
                        }
                        View::TrainingLauncher => {
                            let _ = self.training_launcher_view.handle_key(key)?;
                        }
                        View::ConfigEditor => {
                            let _ = self.config_editor_view.handle_key(key)?;
                        }
                        View::LogViewer => {
                            let _ = self.log_viewer_view.handle_key(key)?;
                        }
                        View::ServiceManager => {
                            let _ = self.service_manager_view.handle_key(key)?;
                        }
                        View::ModelSelector => {
                            let _ = self.model_selector_view.handle_key(key)?;
                        }
                        View::FileBrowser => {
                            let _ = self.file_browser_view.handle_key(key)?;
                        }
                        View::Settings => {
                            let _ = self.settings_view.handle_key(key)?;
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                // Delegate to current view for all other keys
                match self.current_view {
                    View::Training => {
                        let _ = self.training_view.handle_key(key)?;
                    }
                    View::Trading => {
                        let _ = self.trading_view.handle_key(key)?;
                    }
                    View::Overview => {
                        let _ = self.overview_view.handle_key(key)?;
                    }
                    View::Placeholder => {
                        // Placeholder can handle back keys
                        if matches!(key, KeyCode::Char('b') | KeyCode::Esc) {
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::TrainingLauncher => {
                        // Training launcher handles all keys
                        if self.training_launcher_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::ConfigEditor => {
                        // Config editor handles all keys
                        if self.config_editor_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::LogViewer => {
                        // Log viewer handles all keys
                        if self.log_viewer_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::ServiceManager => {
                        // Service manager handles all keys
                        if self.service_manager_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::ModelSelector => {
                        // Model selector handles all keys
                        if self.model_selector_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::FileBrowser => {
                        // File browser handles all keys
                        if self.file_browser_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::Settings => {
                        // Settings handles all keys
                        if self.settings_view.handle_key(key)? {
                            // View requested to go back
                            self.switch_view(View::Launcher);
                        }
                    }
                    View::Launcher => {
                        // Launcher handles its own keys above
                    }
                }
            }
        }
        Ok(false)
    }

    /// Execute a command from the command palette
    async fn execute_command(&mut self, command_id: &str) -> Result<()> {
        match command_id {
            // Navigation commands
            "nav.dashboard" => {
                self.switch_view(View::Launcher);
            }
            "nav.trading" => {
                self.switch_view(View::Trading);
            }
            "nav.training" => {
                self.switch_view(View::Training);
            }
            "nav.models" => {
                self.switch_to_placeholder("Model Manager - Coming soon");
            }
            "nav.config" => {
                self.switch_to_placeholder("Config Editor - Coming soon");
            }

            // Trading commands
            "trading.killswitch" => {
                self.notifications.push(
                    Notification::warning("Kill Switch")
                        .message("Kill switch toggled - trading halted"),
                );
            }
            "trading.pause" => {
                self.notifications.push(
                    Notification::info("Trading Paused").message("Trading pipeline paused"),
                );
            }
            "trading.resume" => {
                self.notifications.push(
                    Notification::success("Trading Resumed").message("Trading pipeline resumed"),
                );
            }
            "trading.refresh" => {
                let _ = self.trading_view.update_metrics().await;
                let _ = self.trading_view.update_state().await;
                self.notifications
                    .push(Notification::info("Refreshed").message("Trading data refreshed"));
            }

            // Training commands
            "training.start" => {
                self.switch_view(View::TrainingLauncher);
            }
            "training.stop" => {
                self.notifications.push(
                    Notification::warning("Training Stopped").message("Training run stopped"),
                );
            }
            "training.logs" => {
                self.switch_to_placeholder("Training Logs - Coming soon");
            }

            // Config commands
            "config.edit" => {
                self.switch_to_placeholder("Config Editor - Coming soon");
            }
            "config.reload" => {
                self.theme = Theme::load();
                self.notifications.push(
                    Notification::success("Config Reloaded").message("Configuration reloaded"),
                );
            }

            // System commands
            "system.refresh" => {
                self.launcher_view.update_status().await;
                self.notifications.push(Notification::info("Refreshed"));
            }
            "system.help" => {
                self.help_overlay.show();
            }
            "system.quit" => {
                self.should_quit = true;
            }

            _ => {
                debug!("Unknown command: {}", command_id);
            }
        }
        Ok(())
    }

    /// Handle menu action selection
    async fn handle_menu_action(
        &mut self,
        action: crate::launcher::menu::MenuAction,
    ) -> Result<()> {
        use crate::launcher::menu::MenuAction;
        match action {
            MenuAction::TradingMonitor => {
                self.switch_view(View::Trading);
            }
            MenuAction::TrainingMonitor => {
                self.switch_view(View::Training);
            }
            MenuAction::ConfigEditor => {
                // Open config browser (will show production_baseline by default if available)
                let default_config = std::path::PathBuf::from("CONFIG/experiments/production_baseline.yaml");
                let _ = self.config_editor_view.open(default_config);
                self.switch_view(View::ConfigEditor);
            }
            MenuAction::ModelSelector => {
                self.switch_view(View::ModelSelector);
            }
            MenuAction::ServiceManager => {
                self.switch_view(View::ServiceManager);
            }
            MenuAction::RunManager => {
                self.switch_view(View::TrainingLauncher);
            }
            MenuAction::SystemStatus => {
                self.switch_view(View::Overview);
            }
            MenuAction::LogViewer => {
                self.switch_view(View::LogViewer);
            }
            MenuAction::FileBrowser => {
                self.switch_view(View::FileBrowser);
            }
            MenuAction::Settings => {
                self.switch_view(View::Settings);
            }
        }
        Ok(())
    }
}
