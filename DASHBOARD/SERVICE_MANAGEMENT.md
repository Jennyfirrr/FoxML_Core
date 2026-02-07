# Service Management Features

## Overview

The TUI dashboard provides complete systemd service management for the trading engine, allowing you to configure and control the autonomous trading system without terminal commands.

## Key Principle

**Services Run Independently - TUI is Optional Viewer/Control Panel**

- Trading runs 24/7 via systemd (`foxml-trading.service`)
- TUI is optional - just for viewing/managing
- If TUI closes, trading continues running
- Services don't depend on TUI

## Service Manager Features

### 1. Service Status View

**What you see:**
- Service state (running/stopped/failed)
- Service uptime
- Last restart time
- Process ID
- Resource usage (CPU, memory)

**Example:**
```
┌─────────────────────────────────────────────────────────────┐
│ Service Manager - foxml-trading.service                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Status:        🟢 Running                                    │
│ Uptime:        2h 15m 32s                                    │
│ PID:           12345                                         │
│ CPU:           2.5%                                          │
│ Memory:        512 MB                                         │
│                                                               │
│ [Start] [Stop] [Restart] [Edit Settings] [View Logs]        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Trading Engine Settings Editor

**What you can configure:**

**Cycle/Heartbeat Settings:**
- `FOXML_CYCLE_INTERVAL` - Seconds between trading cycles (default: 60)
  - This is the "heartbeat" - how often the engine runs a cycle
  - Lower = more frequent trading decisions
  - Higher = less frequent, saves resources

**Trading Configuration:**
- `FOXML_RUN_ID` - Training run ID to use (or "latest" for auto-detect)
- `FOXML_BROKER` - Broker selection (paper/alpaca/ibkr)
- `FOXML_MARKET_HOURS_ONLY` - Only trade during market hours (true/false)
- `FOXML_LOG_LEVEL` - Logging verbosity (DEBUG/INFO/WARNING/ERROR)

**Systemd Service Settings:**
- Restart policy (on-failure, always, never)
- Restart delay (seconds)
- Resource limits (CPU quota, memory max)
- User/group
- Working directory

**Example Settings View:**
```
┌─────────────────────────────────────────────────────────────┐
│ Trading Engine Settings - /etc/foxml-trading.conf            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Cycle Interval (Heartbeat):  [60] seconds                    │
│                               ↑                              │
│                               Adjust with ↑↓                 │
│                                                               │
│ Market Hours Only:            [✓] Enabled                    │
│ Run ID:                       [latest ▼]                     │
│ Broker:                       [paper ▼]                       │
│ Log Level:                    [INFO ▼]                       │
│                                                               │
│ Systemd Settings:                                            │
│   Restart Policy:            [on-failure ▼]                  │
│   Restart Delay:              [60] seconds                    │
│   CPU Quota:                  [200%]                          │
│   Memory Max:                 [8G]                            │
│                                                               │
│ [Save] [Cancel] [Restart Service to Apply]                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3. Service Control Actions

**From TUI, you can:**
- **Start** service: `systemctl start foxml-trading`
- **Stop** service: `systemctl stop foxml-trading`
- **Restart** service: `systemctl restart foxml-trading`
- **Enable on boot**: `systemctl enable foxml-trading`
- **Disable on boot**: `systemctl disable foxml-trading`
- **Reload config**: `systemctl daemon-reload` (after editing config)

**All via keyboard shortcuts:**
- `s` - Start service
- `x` - Stop service
- `r` - Restart service
- `e` - Edit settings
- `l` - View logs

### 4. Log Viewing (journalctl Integration)

**What you can do:**
- View systemd journal logs: `journalctl -u foxml-trading`
- Real-time log tailing: `journalctl -u foxml-trading -f`
- Filter by time: `--since "1 hour ago"`
- Search logs: Filter by keyword
- Export logs: Save to file

**Example Log Viewer:**
```
┌─────────────────────────────────────────────────────────────┐
│ Service Logs - foxml-trading.service                         │
├─────────────────────────────────────────────────────────────┤
│ [10:45:23] TradingEngine initialized                        │
│ [10:45:23] Broker: paper | Cash: $100,000.00                │
│ [10:45:23] Cycle #1 started                                  │
│ [10:45:23] Prediction: 5 horizons processed (45ms)          │
│ [10:45:23] Trade filled: AAPL 100 @ $150.00                │
│ [10:45:24] Cycle #1 completed (142ms)                       │
│ [10:46:24] Cycle #2 started                                  │
│                                                               │
│ [Filter: All] [Auto-scroll: ON] [Export]                    │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Files

### Trading Engine Config
- **Location**: `/etc/foxml-trading.conf`
- **Format**: Shell script (sourced by systemd service)
- **Editable from**: TUI Service Manager
- **Applied**: After restarting service

### Systemd Service File
- **Location**: `/etc/systemd/system/foxml-trading.service`
- **Format**: systemd unit file
- **Editable from**: TUI Service Manager (advanced)
- **Applied**: After `systemctl daemon-reload`

## Usage Examples

### Example 1: Change Cycle Interval (Heartbeat)

```bash
foxml
# Press '3' to open service manager
# Press 'e' to edit settings
# Change cycle interval from 60 to 30 seconds
# Save settings
# Press 'r' to restart service
# New heartbeat: 30 seconds (cycles run twice as often)
```

### Example 2: Switch Training Run

```bash
foxml
# Press '3' to open service manager
# Press 'e' to edit settings
# Change RUN_ID from "latest" to "prod_run_20250118"
# Save settings
# Press 'r' to restart service
# Trading now uses models from prod_run_20250118
```

### Example 3: Enable Market Hours Only

```bash
foxml
# Press '3' to open service manager
# Press 'e' to edit settings
# Toggle "Market Hours Only" to enabled
# Save settings
# Press 'r' to restart service
# Trading will only run during market hours (9:30 AM - 4:00 PM ET)
```

## Architecture: How Services Run

```
┌─────────────────────────────────────────────────────────────┐
│  systemd (Service Manager)                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  foxml-trading.service                                │  │
│  │  - Runs 24/7                                          │  │
│  │  - Auto-restarts on failure                           │  │
│  │  - Starts on boot (if enabled)                        │  │
│  │  - Configured via /etc/foxml-trading.conf            │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ (ExecStart)                                        │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Trading Engine (Python)                              │  │
│  │  - Runs trading cycles every N seconds                │  │
│  │  - Reads config from /etc/foxml-trading.conf          │  │
│  │  - Emits events to EventBus                           │  │
│  │  - Independent of TUI                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ (EventBus/Metrics)                                │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IPC Bridge (Optional, for monitoring)                │  │
│  │  - Exposes EventBus via WebSocket/HTTP                │  │
│  │  - Auto-started by TUI if needed                      │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ (WebSocket/HTTP)                                  │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Rust TUI Dashboard (foxml command)                  │  │
│  │  - Views real-time metrics                            │  │
│  │  - Manages services                                   │  │
│  │  - Edits settings                                      │  │
│  │  - Optional - services run without it                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

✅ **No Terminal Commands**: Manage services from TUI
✅ **Visual Feedback**: See service status at a glance
✅ **Easy Configuration**: Edit settings with form-based UI
✅ **Safe Operations**: Confirmation dialogs for destructive actions
✅ **Real-Time Monitoring**: View logs and metrics while managing
✅ **Independent Operation**: Services run even if TUI is closed

## Security Considerations

- Service management requires appropriate permissions (may need sudo)
- Config file editing requires write access to `/etc/foxml-trading.conf`
- Systemd operations require systemd D-Bus access
- TUI will prompt for credentials if needed (via polkit or sudo)

## Future Enhancements

- Multiple service management (trading, training, data processing)
- Service scheduling (start/stop at specific times)
- Service health monitoring (auto-alerts on failures)
- Service dependency management
- Backup/restore service configurations
