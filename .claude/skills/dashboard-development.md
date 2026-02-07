# Dashboard Development

## Adding a New View

Views are full-screen components that represent different dashboard screens.

### Step 1: Create the View Module

Create a new file in `DASHBOARD/dashboard/src/views/`:

```rust
// src/views/my_view.rs
use anyhow::Result;
use ratatui::prelude::*;
use crossterm::event::KeyCode;

use super::ViewTrait;

pub struct MyView {
    // View state
}

impl MyView {
    pub fn new() -> Self {
        Self {
            // Initialize state
        }
    }
}

impl ViewTrait for MyView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        // Render UI here using ratatui widgets
        let block = ratatui::widgets::Block::default()
            .title("My View")
            .borders(ratatui::widgets::Borders::ALL);
        frame.render_widget(block, area);
        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Enter => {
                // Handle enter
                Ok(true) // Return true if handled
            }
            _ => Ok(false)
        }
    }
}
```

### Step 2: Register the View

1. Add to `src/views/mod.rs`:
```rust
pub mod my_view;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum View {
    Launcher,
    Trading,
    Training,
    Overview,
    MyView,  // Add new variant
    // ...
}
```

2. Add to `src/app.rs`:
```rust
use crate::views::my_view::MyView;

pub struct App {
    // ... existing fields
    my_view: MyView,
}

impl App {
    pub async fn new() -> Result<Self> {
        // ... in initialization
        let my_view = MyView::new();
        // ...
    }
}
```

3. Add rendering in `App::run()`:
```rust
View::MyView => {
    let _ = ViewTrait::render(&mut self.my_view, f, f.size());
}
```

4. Add navigation in `App::handle_key()`:
```rust
View::MyView => {
    let _ = self.my_view.handle_key(key)?;
}
```

### Step 3: Add Menu Action (Optional)

In `src/launcher/menu.rs`:
```rust
pub enum MenuAction {
    // ... existing actions
    MyView,
}
```

In `src/app.rs` `handle_menu_action()`:
```rust
MenuAction::MyView => {
    self.current_view = View::MyView;
    debug!("Switched to MyView");
}
```

## Adding a New Widget

Widgets are reusable UI components used across views.

### Create the Widget

```rust
// src/widgets/my_widget.rs
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

pub struct MyWidget {
    pub data: Vec<String>,
}

impl MyWidget {
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title("My Widget")
            .borders(Borders::ALL);

        let text = self.data.join("\n");
        let paragraph = Paragraph::new(text).block(block);
        frame.render_widget(paragraph, area);
    }
}
```

### Register in mod.rs

```rust
// src/widgets/mod.rs
pub mod my_widget;
pub use my_widget::MyWidget;
```

### Use in a View

```rust
use crate::widgets::MyWidget;

pub struct MyView {
    widget: MyWidget,
}

impl ViewTrait for MyView {
    fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
        self.widget.render(frame, area);
        Ok(())
    }
}
```

## Adding a New Launcher Feature

Launcher features appear in the main menu and have their own modules.

### Create the Feature Module

```rust
// src/launcher/my_feature.rs
use anyhow::Result;
use ratatui::prelude::*;
use crossterm::event::KeyCode;

pub struct MyFeature {
    // State
}

impl MyFeature {
    pub fn new() -> Self {
        Self {}
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        // Render feature UI
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        // Handle input
        Ok(false)
    }
}
```

### Register in mod.rs

```rust
// src/launcher/mod.rs
pub mod my_feature;
pub use my_feature::MyFeature;
```

## Theme System

### Using Theme Colors

```rust
use crate::themes::Theme;

// Load theme once (in App::new)
let theme = Theme::load();

// Use colors
let style = Style::default()
    .fg(theme.text)
    .bg(theme.background);
```

### Theme Structure

```rust
pub struct Theme {
    pub background: Color,
    pub foreground: Color,
    pub text: Color,
    pub accent: Color,
    pub highlight: Color,
    pub error: Color,
    pub warning: Color,
    pub success: Color,
}
```

### Adding a New Config Parser

Create in `src/themes/`:

```rust
// src/themes/alacritty.rs
use std::fs;
use std::path::PathBuf;
use regex::Regex;

pub fn parse_alacritty() -> Option<Vec<(String, String)>> {
    let config_path = dirs::config_dir()?.join("alacritty/alacritty.toml");
    let content = fs::read_to_string(config_path).ok()?;

    // Parse colors from config
    let mut colors = Vec::new();
    // ... parsing logic
    Some(colors)
}
```

Add to theme detection order in `src/themes/theme.rs`:
```rust
impl Theme {
    pub fn load() -> Self {
        // Try parsers in order
        if let Some(colors) = waybar::parse_waybar() {
            return Self::from_colors(colors);
        }
        if let Some(colors) = alacritty::parse_alacritty() {
            return Self::from_colors(colors);
        }
        // ... fallback to default
    }
}
```

## API Client

### Making HTTP Requests

```rust
use crate::api::client::ApiClient;

let client = ApiClient::new("http://127.0.0.1:8765");

// Get metrics
let metrics = client.get_metrics().await?;

// Get state
let state = client.get_state().await?;
```

### WebSocket Events

```rust
use crate::api::events::EventStream;

let mut stream = EventStream::connect("ws://127.0.0.1:8765/ws").await?;

while let Some(event) = stream.next().await {
    match event {
        Event::Trade(trade) => { /* handle */ }
        Event::Error(err) => { /* handle */ }
    }
}
```

## Testing

```bash
# Build and run
cd DASHBOARD/dashboard
cargo build
cargo run

# Release build
cargo build --release
cargo run --release

# Check for errors
cargo check
cargo clippy
```

## Common Patterns

### Async Updates

Views that need periodic updates:
```rust
impl App {
    pub async fn run(&mut self) -> Result<()> {
        // In event loop
        if self.last_update.elapsed().as_secs() >= 2 {
            match self.current_view {
                View::MyView => {
                    self.my_view.update().await?;
                }
                // ...
            }
            self.last_update = std::time::Instant::now();
        }
    }
}
```

### Layout with Constraints

```rust
use ratatui::layout::{Layout, Constraint, Direction};

fn render(&mut self, frame: &mut Frame, area: Rect) -> Result<()> {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Main content
            Constraint::Length(1),  // Footer
        ])
        .split(area);

    // Render to each chunk
    self.render_header(frame, chunks[0]);
    self.render_main(frame, chunks[1]);
    self.render_footer(frame, chunks[2]);
    Ok(())
}
```

### Scrollable Lists

```rust
use ratatui::widgets::{List, ListItem, ListState};

pub struct MyView {
    items: Vec<String>,
    state: ListState,
}

impl MyView {
    fn render_list(&mut self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.items
            .iter()
            .map(|s| ListItem::new(s.as_str()))
            .collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL))
            .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        frame.render_stateful_widget(list, area, &mut self.state);
    }

    fn next(&mut self) {
        let i = match self.state.selected() {
            Some(i) => (i + 1).min(self.items.len() - 1),
            None => 0,
        };
        self.state.select(Some(i));
    }
}
```

## Dependencies

Key dependencies in `Cargo.toml`:
- `ratatui` - TUI framework
- `crossterm` - Terminal manipulation
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `tokio-tungstenite` - WebSocket client
- `serde` / `serde_json` - JSON serialization
- `sysinfo` - System information
- `walkdir` - Directory traversal
- `regex` - Config parsing
- `tracing` - Logging
