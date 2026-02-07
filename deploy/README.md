# FoxML Deployment

Deployment configurations for running FoxML in production.

## systemd Service

The systemd service runs the live trading engine autonomously during market hours.

### Quick Start

```bash
# 1. Install the service
sudo ./deploy/systemd/install.sh

# 2. (Optional) Customize configuration
sudo cp deploy/systemd/foxml-trading.conf /etc/foxml-trading.conf
sudo vim /etc/foxml-trading.conf

# 3. Start trading
sudo systemctl start foxml-trading

# 4. Check status
sudo systemctl status foxml-trading

# 5. Follow logs
journalctl -u foxml-trading -f
```

### Files

| File | Purpose |
|------|---------|
| `systemd/foxml-trading.service` | systemd unit file |
| `systemd/foxml-trading-launcher.sh` | Wrapper script (conda, market hours) |
| `systemd/foxml-trading.conf` | Configuration file |
| `systemd/install.sh` | Installation script |

### Configuration

Edit `/etc/foxml-trading.conf` (or the service file) to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `FOXML_RUN_ID` | `latest` | Training run ID (or "latest" for auto-detect) |
| `FOXML_BROKER` | `paper` | Broker: paper, alpaca, ibkr |
| `FOXML_CYCLE_INTERVAL` | `60` | Seconds between trading cycles |
| `FOXML_MARKET_HOURS_ONLY` | `true` | Only trade during US market hours |
| `FOXML_LOG_LEVEL` | `INFO` | Logging verbosity |

### Market Hours

The service respects US stock market hours (9:30 AM - 4:00 PM Eastern Time):
- Automatically waits for market open if started early
- Handles weekends (no trading Saturday/Sunday)
- **Note**: Holiday calendar not yet implemented

### Commands

```bash
# Start/stop
sudo systemctl start foxml-trading
sudo systemctl stop foxml-trading
sudo systemctl restart foxml-trading

# Enable on boot
sudo systemctl enable foxml-trading

# Disable on boot
sudo systemctl disable foxml-trading

# View status
sudo systemctl status foxml-trading

# Follow logs
journalctl -u foxml-trading -f

# View recent logs
journalctl -u foxml-trading --since "1 hour ago"
```

### Troubleshooting

**Service won't start:**
```bash
# Check logs
journalctl -u foxml-trading -e

# Test launcher manually
./deploy/systemd/foxml-trading-launcher.sh
```

**No models found:**
```bash
# Run training first
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config production_baseline \
    --output-dir TRAINING/results/prod_run
```

**Permission errors:**
```bash
# Fix permissions
sudo chown -R Jennifer:Jennifer /home/Jennifer/trader/LIVE_TRADING/state
sudo chown -R Jennifer:Jennifer /home/Jennifer/trader/LIVE_TRADING/logs
```

## Future: Docker Deployment

Docker deployment files will be added in a future update.

## Future: Kubernetes Deployment

Kubernetes manifests will be added in a future update.
