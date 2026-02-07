# FoxML TUI Dashboard - Complete Overview

## The `foxml` Command

**Single command to rule them all:**

```bash
foxml
```

That's it. No need to:
- Start IPC bridge server manually
- Run `python bin/run_live_trading.py` separately
- Edit configs in vim/nano
- Check logs with `tail -f`
- Manage processes with `ps` and `kill`

## What Happens When You Run `foxml`

1. **Auto-detection & Setup**:
   - Checks if IPC bridge server is running (starts it if not)
   - Loads color theme from your waybar/hyprland configs
   - Connects to trading engine (if running)
   - Scans for active training runs

2. **Launcher Appears**:
   - Beautiful entry screen with grid menu
   - Quick status of all systems
   - All features accessible from keyboard

3. **Everything Works**:
   - Monitor trading in real-time
   - Edit configs with built-in YAML editor
   - Start/stop runs
   - View logs
   - Browse files
   - All from one TUI

## What This Plan Enables

### 1. Complete System Management (No Terminal Commands)

**Before (Current State):**
```bash
# Start trading
python bin/run_live_trading.py --run-id my_run --broker paper

# Edit config (separate terminal)
vim CONFIG/pipeline/training/intelligent.yaml

# Check logs (another terminal)
tail -f LIVE_TRADING/logs/trading.log

# Monitor processes
ps aux | grep python
```

**After (With `foxml`):**
```bash
foxml
# Then from launcher:
# - Press '1' to monitor trading (real-time dashboard)
# - Press '2' to edit configs (built-in YAML editor)
# - Press '3' to manage runs (start/stop from UI)
# - Press '4' to view logs (browse and search)
# All in one TUI, no terminal commands needed
```

### 2. Real-Time Trading Monitoring

**What you can see:**
- Portfolio value, daily P&L, cash balance (updates in real-time)
- Current positions with unrealized P&L
- Pipeline status (exactly where each cycle is: Prediction → Blending → Arbitration → Gating → Sizing → Risk)
- Event log (trades, errors, warnings streaming live)
- Performance chart (historical P&L over time)
- Risk metrics (exposure, drawdown, kill switch status)

**What you can do:**
- Toggle kill switch (keyboard shortcut `k`)
- Pause/resume trading
- View detailed position information
- Filter events by type/severity
- Export metrics to file

### 3. Training Pipeline Monitoring

**What you can see:**
- Pipeline stages (Target Ranking → Feature Selection → Training Plan → Training)
- Model training status grid (all 20 models, per-target progress)
- Target ranking results
- Feature importance metrics
- Training logs with filtering

**What you can do:**
- Switch between training runs
- View model performance metrics
- Monitor training progress in real-time
- Browse training artifacts (JSON, parquet files)

### 4. Config Management

**What you can do:**
- Browse entire CONFIG/ directory tree
- Edit any YAML config file with syntax highlighting
- Validate configs before saving
- Search across configs
- View config history/backups

**No more:**
- Opening vim/nano in terminal
- Remembering config file paths
- Manual YAML syntax checking

### 5. Run Management

**What you can do:**
- Start new training runs (select config, set options)
- Start new trading runs (select run-id, broker, symbols)
- Stop running processes (with confirmation)
- View all active runs (training + trading)
- See run status and progress

**No more:**
- Running `python bin/run_live_trading.py` manually
- Finding process IDs to kill
- Managing multiple terminal windows

### 6. Log Viewing

**What you can do:**
- Browse log directories (LIVE_TRADING/logs/, training logs)
- Search and filter logs
- Real-time log tailing
- View formatted logs (syntax highlighting for errors)

**No more:**
- `tail -f` commands
- `grep` through log files
- Opening multiple terminals for different logs

### 7. File Browser

**What you can do:**
- Navigate RESULTS/ directory
- View artifacts (JSON, parquet, training plans)
- Export files
- Browse training run outputs

### 8. System Status

**What you can see:**
- Component health (trading engine, training, IPC bridge)
- Resource usage (CPU, GPU, memory)
- Quick diagnostics
- Connection status

## Architecture: How It All Works

**Key Principle: Services Run Independently, TUI is Optional Viewer**

```
┌─────────────────────────────────────────────────────────────┐
│  Trading Service (systemd)                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  foxml-trading.service                               │  │
│  │  - Runs 24/7 via systemd                             │  │
│  │  - Independent of TUI                                │  │
│  │  - Configured via /etc/foxml-trading.conf            │  │
│  │  - Cycle interval, heartbeat, market hours           │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ (EventBus/Metrics)                                │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  IPC Bridge (Python/FastAPI)                         │  │
│  │  - Optional, only for monitoring                      │  │
│  │  - Auto-started by TUI if needed                      │  │
│  │  - Exposes EventBus/Metrics via WebSocket/HTTP       │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         │ (WebSocket/HTTP)                                  │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Rust TUI Dashboard (foxml command)                  │  │
│  │  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │  Launcher    │  │  Trading     │                  │  │
│  │  │  (Entry)     │  │  Monitor     │                  │  │
│  │  └──────────────┘  └──────────────┘                  │  │
│  │         │                │                            │  │
│  │         └────────────────┘                            │  │
│  │                │                                       │  │
│  │  - View real-time metrics                            │  │
│  │  - Manage systemd services                           │  │
│  │  - Edit trading settings                             │  │
│  │  - Configure cycle interval/heartbeat                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

**Key Points:**
- Trading runs independently via systemd (foxml-trading.service)
- TUI is optional - just for viewing/managing
- If TUI closes, trading continues running
- IPC bridge is optional - only needed for real-time monitoring
- TUI can manage systemd services (start/stop/restart)
- TUI can edit service configs (cycle interval, heartbeat, etc.)
- Services don't depend on TUI - they run autonomously

## Usage Examples

### Example 1: Monitor Autonomous Trading

```bash
# Trading is already running via systemd (foxml-trading.service)
# It runs 24/7, independent of TUI

# Launch dashboard to monitor
foxml
# Press '1' to open trading monitor
# See real-time portfolio, positions, P&L, pipeline status
# Press 'k' to toggle kill switch if needed
# Close TUI - trading continues running
```

### Example 2: Configure Trading Engine Settings

```bash
foxml
# Press '3' to open service manager
# View foxml-trading service status
# Press 'e' to edit trading settings
# Configure:
#   - Cycle interval: 60 seconds (heartbeat)
#   - Market hours only: true
#   - Log level: INFO
#   - Run ID: latest
# Save settings (updates /etc/foxml-trading.conf)
# Restart service to apply changes
# Press '1' to monitor trading
```

### Example 3: Edit Config and Start Training

```bash
foxml
# Press '2' to open config editor
# Navigate to CONFIG/experiments/production_baseline.yaml
# Edit settings (syntax highlighting, validation)
# Save config
# Press '4' to open run manager
# Start new training run with edited config
# Press '5' to monitor training progress
```

### Example 4: Quick System Check

```bash
foxml
# Launcher shows quick status:
# - Trading Service: 🟢 Active (systemd: running)
# - Trading Engine: 🟢 Active (Portfolio: $102,450)
# - Training: 🟡 Running (60% complete)
# - IPC Bridge: 🟢 Connected
# Press '7' for detailed system status
```

## Keyboard Shortcuts (Global)

- `q` - Quit dashboard
- `Esc` - Return to launcher
- `Tab` - Switch between views
- `?` - Show help
- `1-9` - Quick access to launcher menu items

## Keyboard Shortcuts (Trading View)

- `k` - Toggle kill switch
- `p` - Pause/resume trading
- `r` - Refresh metrics
- `↑↓` - Navigate position table
- `f` - Filter events

## Benefits Summary

✅ **Single Command**: Just `foxml` - no complex setup
✅ **No Terminal Commands**: Everything accessible from TUI
✅ **Auto-Management**: IPC bridge auto-starts, processes auto-detected
✅ **Real-Time**: 100+ updates/sec, smooth performance
✅ **Beautiful**: Matches your waybar/hyprland colors
✅ **Complete**: Config editing, run management, monitoring, logs - all in one
✅ **Keyboard-First**: No mouse needed, works in any terminal
✅ **Autonomous Trading Ready**: Perfect for monitoring systems that run 24/7

## What You Can Do Without Terminal Commands

- ✅ **Manage systemd services** (start/stop/restart foxml-trading)
- ✅ **Configure trading engine** (cycle interval, heartbeat, market hours)
- ✅ **Edit service configs** (`/etc/foxml-trading.conf` from TUI)
- ✅ **Start/stop training runs**
- ✅ **Edit any config file** (YAML editor)
- ✅ **View and search logs** (including journalctl)
- ✅ **Monitor real-time metrics** (portfolio, P&L, positions)
- ✅ **Browse training artifacts**
- ✅ **Check system health** (service status, resource usage)
- ✅ **Toggle kill switches**
- ✅ **Export data**

**Everything you need to operate the system is in the TUI.**

**Important**: Trading runs independently via systemd. TUI is just for viewing/managing. If you close TUI, trading continues running 24/7.
