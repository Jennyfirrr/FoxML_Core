# Dashboard Overview

## What is the Dashboard?

FoxML Dashboard (`foxml` command) is a Rust TUI built with `ratatui` that provides a unified interface for:
- Real-time trading monitoring
- Training pipeline progress
- Config editing
- Model selection for LIVE_TRADING
- Service management (systemd)
- Run management
- Log viewing and file browsing

**Key principle**: Trading runs independently via systemd. The TUI is an optional viewer/controller - closing it doesn't stop trading.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Trading Service (systemd: foxml-trading.service)          │
│  - Runs 24/7 independently                                  │
│  - Configured via /etc/foxml-trading.conf                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ (EventBus/Metrics)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  IPC Bridge (Python/FastAPI)                                │
│  - Optional, only for monitoring                            │
│  - Auto-started by TUI if needed                            │
│  - Exposes EventBus/Metrics via WebSocket/HTTP             │
│  - Runs on http://127.0.0.1:8765                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ (WebSocket/HTTP)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Rust TUI Dashboard (foxml command)                         │
│  - Launcher (entry point with grid menu)                    │
│  - Trading Monitor (real-time metrics, events, pipeline)    │
│  - Training Monitor (run progress, model grid)              │
│  - Overview (system health, resources)                      │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
DASHBOARD/
├── dashboard/              # Rust TUI project
│   ├── Cargo.toml         # Dependencies
│   └── src/
│       ├── main.rs        # Entry point
│       ├── app.rs         # Main app structure, event loop
│       ├── views/         # View modules (screens)
│       │   ├── mod.rs     # View enum and ViewTrait
│       │   ├── launcher.rs
│       │   ├── trading.rs
│       │   ├── training.rs
│       │   └── overview.rs
│       ├── widgets/       # Reusable UI components
│       │   ├── metrics_panel.rs
│       │   ├── event_log.rs
│       │   ├── pipeline_status.rs
│       │   ├── position_table.rs
│       │   └── chart.rs
│       ├── api/           # IPC bridge client
│       │   ├── client.rs  # HTTP/WebSocket client
│       │   ├── events.rs
│       │   └── metrics.rs
│       ├── themes/        # Color theme system
│       │   ├── theme.rs   # Theme struct
│       │   ├── waybar.rs  # Waybar config parser
│       │   ├── hyprland.rs
│       │   ├── tmux.rs
│       │   └── kitty.rs
│       └── launcher/      # Launcher features
│           ├── menu.rs
│           ├── config_editor.rs
│           ├── service_manager.rs
│           ├── run_manager.rs
│           ├── system_status.rs
│           ├── log_viewer.rs
│           └── file_browser.rs
├── bridge/                # Python IPC bridge
│   ├── server.py          # FastAPI server
│   └── requirements.txt
└── README.md
```

## Running the Dashboard

```bash
# Recommended: use the launcher script
bin/foxml

# Or directly (for development)
cd DASHBOARD/dashboard
cargo run --release
```

The `bin/foxml` script will:
1. Check if IPC bridge is running (auto-starts if not)
2. Build dashboard if needed
3. Launch the dashboard

## Views

| View | Description | Menu Key |
|------|-------------|----------|
| Launcher | Main menu with grid navigation | - |
| Trading | Real-time trading metrics, events, pipeline status | 1 |
| Training | Training run list, progress, model grid | 2 |
| Overview | System health, resources, quick metrics | 7 |

## Keyboard Shortcuts

### Global
- `q` - Quit (from launcher) or return to launcher (from views)
- `Esc` - Return to launcher or quit
- `Tab` - Cycle through views
- `r` - Context-sensitive: restart in ServiceManager, refresh in other views
- `b` - Back to launcher

### Navigation (vim-style)
- `j/↓` - Move down
- `k/↑` - Move up
- `h/←` - Move left
- `l/→` - Move right
- `g` - Jump to top
- `G` - Jump to bottom

### Trading View
- `k` - Toggle kill switch
- `p` - Pause/resume trading

## Launcher Features

| Feature | Description | Status |
|---------|-------------|--------|
| Trading Monitor | Real-time metrics, events, pipeline | Working |
| Training Monitor | Run discovery, progress tracking, stage tags (TR/FS/TRN) | Working |
| Config Editor | YAML editing with syntax highlighting | Working |
| Model Selector | Choose models for LIVE_TRADING | Working |
| Service Manager | systemd service control (start/stop/restart) | Working |
| Run Manager | Start/stop training runs | Working |
| System Status | Health checks, resource usage | Working |
| Log Viewer | Browse and search logs | Working |
| File Browser | Navigate directories | Working |

## Color Theme System

Dashboard auto-detects colors from (in order):
1. `~/.config/waybar/config`
2. `~/.config/hypr/hyprland.conf`
3. `~/.tmux.conf`
4. `~/.config/kitty/kitty.conf`
5. Falls back to default dark theme

## IPC Bridge

The Python IPC bridge connects the Rust TUI to the Python trading engine:

- **REST API**: `/api/metrics`, `/api/state`, `/api/events/recent`, `/health`
- **WebSocket**: Real-time event streaming
- **Graceful fallback**: Runs in mock mode if trading engine not available

See `dashboard-ipc-bridge.md` for bridge development.

## Key Files

| File | Purpose |
|------|---------|
| `DASHBOARD/dashboard/src/app.rs` | Main app state, event loop, view switching |
| `DASHBOARD/dashboard/src/views/mod.rs` | View enum and ViewTrait definition |
| `DASHBOARD/dashboard/src/launcher/menu.rs` | Menu actions and grid navigation |
| `DASHBOARD/bridge/server.py` | Python FastAPI IPC bridge |
| `bin/foxml` | Launch script |
