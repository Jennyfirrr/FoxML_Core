# FoxML Dashboard

Rust TUI dashboard for monitoring autonomous trading and training pipeline systems.

## Overview

High-performance terminal UI built with `ratatui` that provides:
- Real-time trading monitoring
- Training pipeline progress
- Config editing (YAML + interactive)
- Model selection for LIVE_TRADING
- Service management (systemd)
- Run management
- System status and diagnostics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Trading/Training Services (systemd)             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ foxml-trading    │         │ Training Pipeline│         │
│  │ (systemd service)│         │ (runs separately)│         │
│  └──────────────────┘         └──────────────────┘         │
│         │                              │                    │
│         └──────────────┬───────────────┘                    │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │   IPC Bridge      │                          │
│              │  (Python/FastAPI) │                          │
│              │  (optional, for   │                          │
│              │   monitoring)    │                          │
│              └─────────┬─────────┘                          │
│                        │                                     │
│              ┌─────────▼─────────┐                          │
│              │  Rust TUI         │                          │
│              │  (Viewer/Control) │                          │
│              └───────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust (latest stable)
- Python 3.10+
- systemd (for service management)

### Building

```bash
cd DASHBOARD/dashboard
cargo build --release
```

### Running

```bash
# From project root
bin/foxml

# Or directly
cd DASHBOARD/dashboard
cargo run --release
```

The `bin/foxml` script will:
1. Check if IPC bridge is running (auto-starts if not)
2. Build dashboard if needed
3. Launch the dashboard

## Project Structure

```
DASHBOARD/
├── dashboard/              # Rust TUI project
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs        # Entry point
│       ├── app.rs         # Main app structure
│       ├── views/         # View modules
│       │   ├── launcher.rs
│       │   ├── trading.rs
│       │   ├── training.rs
│       │   └── overview.rs
│       ├── widgets/       # UI widgets
│       ├── api/           # IPC bridge client
│       ├── themes/        # Color theme system
│       └── launcher/      # Launcher features
│           ├── menu.rs
│           ├── config_editor.rs
│           ├── config_editor_game.rs
│           ├── model_selector.rs
│           ├── service_manager.rs
│           └── ...
├── bridge/                # Python IPC bridge
│   ├── server.py          # FastAPI server
│   └── requirements.txt
└── README.md
```

## Features

### Launcher
- Main menu with grid navigation
- Config editor (YAML + interactive)
- Model selector
- Service manager
- Run manager
- System status
- Log viewer
- File browser

### Trading Monitor
- Real-time metrics (portfolio, P&L, positions)
- Event log
- Pipeline status
- Position table
- Performance charts
- Kill switch controls

### Training Monitor
- Pipeline stage progress
- Model grid
- Log viewer
- Run selector

## Development

### Adding a New View

1. Create view module in `src/views/`
2. Implement `ViewTrait`
3. Add to `View` enum in `src/views/mod.rs`
4. Add navigation in `app.rs`

### Adding a New Widget

1. Create widget module in `src/widgets/`
2. Implement render method
3. Use in views

## IPC Bridge

The Python IPC bridge exposes EventBus and Metrics via HTTP/WebSocket.

### Starting Bridge Manually

```bash
cd DASHBOARD/bridge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Bridge runs on `http://127.0.0.1:8765` by default.

## Configuration

Dashboard automatically detects colors from:
- `~/.config/waybar/config`
- `~/.config/hypr/hyprland.conf`
- `~/.tmux.conf`
- `~/.config/kitty/kitty.conf`

Falls back to dark theme if configs not found.

## License

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
Copyright (c) 2025-2026 Fox ML Infrastructure LLC
