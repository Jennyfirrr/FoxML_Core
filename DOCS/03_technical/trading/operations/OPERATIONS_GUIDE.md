# Live Trading Operations Guide

Day-to-day operations for running the FoxML live trading system.

## Daily Checklist

### Pre-Market (Before 8:30 AM CT)

- [ ] Check system is running: `sudo systemctl status foxml-trading`
- [ ] Review overnight logs: `journalctl -u foxml-trading --since "yesterday"`
- [ ] Verify no kill switches triggered
- [ ] Check broker connection status
- [ ] Review any pending model updates

### During Market Hours (8:30 AM - 3:00 PM CT)

- [ ] Monitor decision rate (should see ~1 decision log per cycle)
- [ ] Watch for error patterns in logs
- [ ] Check portfolio value trend
- [ ] Verify trades executing as expected

### Post-Market (After 3:00 PM CT)

- [ ] Review daily P&L
- [ ] Check decision statistics
- [ ] Archive logs if needed
- [ ] Plan any system updates for overnight

## Common Operations

### Viewing Current Status

```bash
# Service status
sudo systemctl status foxml-trading

# Current engine state
cat LIVE_TRADING/state/engine_state.json | jq .

# Recent decisions
tail -20 LIVE_TRADING/logs/decisions.jsonl | jq .

# Today's logs
journalctl -u foxml-trading --since today
```

### Restarting the Service

```bash
# Graceful restart (waits for cycle to complete)
sudo systemctl restart foxml-trading

# Check it's running
sudo systemctl status foxml-trading
```

### Changing Configuration

```bash
# Edit config
sudo vim /etc/foxml-trading.conf

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart foxml-trading
```

### Updating Models (New Training Run)

```bash
# Stop trading
sudo systemctl stop foxml-trading

# Update run ID in config
sudo vim /etc/foxml-trading.conf
# Change: FOXML_RUN_ID=new_run_id

# Or use "latest" to auto-detect
# FOXML_RUN_ID=latest

# Restart
sudo systemctl start foxml-trading
```

### Switching Brokers

```bash
# Stop trading
sudo systemctl stop foxml-trading

# Update broker config
sudo vim /etc/foxml-trading.conf
# Change: FOXML_BROKER=alpaca
# Add API keys as needed

# Restart
sudo systemctl start foxml-trading
```

## Log Management

### Log Locations

```
LIVE_TRADING/
├── logs/
│   ├── live_trading_20260119_083000.log  # Session logs
│   ├── live_trading_20260119_143000.log
│   └── decisions.jsonl                    # Decision audit trail
└── state/
    ├── engine_state.json                  # Current state
    └── cils/                              # Online learning state
```

### Analyzing Decisions

```bash
# Count decisions today
cat LIVE_TRADING/logs/decisions.jsonl | wc -l

# Filter by symbol
cat LIVE_TRADING/logs/decisions.jsonl | jq 'select(.symbol == "AAPL")'

# Filter by decision type
cat LIVE_TRADING/logs/decisions.jsonl | jq 'select(.decision == "TRADE")'

# Get trade statistics
cat LIVE_TRADING/logs/decisions.jsonl | jq -s '
  group_by(.decision) |
  map({decision: .[0].decision, count: length})
'
```

### Log Rotation

Logs accumulate over time. To manage:

```bash
# Archive old logs (keep last 7 days)
find LIVE_TRADING/logs/ -name "*.log" -mtime +7 -exec gzip {} \;

# Move to archive
mkdir -p LIVE_TRADING/logs/archive
mv LIVE_TRADING/logs/*.gz LIVE_TRADING/logs/archive/

# Or set up logrotate (recommended)
sudo vim /etc/logrotate.d/foxml-trading
```

Example logrotate config:
```
/home/Jennifer/trader/LIVE_TRADING/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
}
```

## State Management

### Backing Up State

```bash
# Create backup
cp -r LIVE_TRADING/state LIVE_TRADING/state.backup.$(date +%Y%m%d)

# Or with timestamp
tar czf state_backup_$(date +%Y%m%d_%H%M%S).tar.gz LIVE_TRADING/state/
```

### Resetting State

**Warning**: This clears all position tracking and P&L history.

```bash
# Stop trading first
sudo systemctl stop foxml-trading

# Backup existing state
mv LIVE_TRADING/state LIVE_TRADING/state.old.$(date +%Y%m%d)

# Create fresh state directory
mkdir -p LIVE_TRADING/state

# Restart
sudo systemctl start foxml-trading
```

### Recovering from Crash

If the service crashed mid-cycle:

```bash
# Check state file
cat LIVE_TRADING/state/engine_state.json | jq .

# If state looks corrupt, restore from backup
cp LIVE_TRADING/state.backup.20260118/engine_state.json LIVE_TRADING/state/

# Restart
sudo systemctl start foxml-trading
```

## Performance Monitoring

### Cycle Timing

```bash
# Check cycle duration from logs
journalctl -u foxml-trading | grep "Cycle.*completed in"

# Average cycle time
journalctl -u foxml-trading --since today | \
  grep -oP 'completed in \K[\d.]+' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count, "seconds"}'
```

### Memory Usage

```bash
# Current memory
systemctl status foxml-trading | grep Memory

# Memory over time (requires cgroups v2)
cat /sys/fs/cgroup/system.slice/foxml-trading.service/memory.current
```

### CPU Usage

```bash
# Current CPU
top -p $(pgrep -f run_live_trading)

# Or via systemd
systemd-cgtop -1
```

## Emergency Procedures

### Kill Switch Triggered

If trading is halted due to kill switch:

```bash
# Check what triggered it
cat LIVE_TRADING/state/engine_state.json | jq '.kill_switch'

# Review recent logs
journalctl -u foxml-trading --since "1 hour ago" | grep -i "kill\|halt\|stop"

# Common triggers:
# - Drawdown limit exceeded
# - Daily loss limit hit
# - Position limit exceeded
# - Manual kill switch
```

To reset (after investigating):

```bash
# Stop service
sudo systemctl stop foxml-trading

# Edit state to clear kill switch (use with caution!)
# Only do this if you understand why it triggered
vim LIVE_TRADING/state/engine_state.json
# Set "is_trading_allowed": true

# Restart
sudo systemctl start foxml-trading
```

### Broker Connection Lost

```bash
# Check logs for connection errors
journalctl -u foxml-trading | grep -i "connection\|timeout\|refused"

# Verify broker API is accessible
curl -I https://paper-api.alpaca.markets/v2/account

# Restart to reconnect
sudo systemctl restart foxml-trading
```

### Runaway Process

If the process is consuming too many resources:

```bash
# Check resource usage
top -p $(pgrep -f run_live_trading)

# Graceful stop
sudo systemctl stop foxml-trading

# If that fails, force kill
sudo systemctl kill foxml-trading

# Investigate logs
journalctl -u foxml-trading --since "1 hour ago"
```

## Scheduled Maintenance

### Weekly

- Archive old logs
- Review cumulative P&L
- Check disk space: `df -h /home/Jennifer/trader`
- Review error rate trends

### Monthly

- Evaluate model performance
- Consider retraining with fresh data
- Update dependencies if needed
- Review and tune risk parameters

### Quarterly

- Full system backup
- Performance review
- Strategy evaluation
- Infrastructure updates

## Contacts and Escalation

| Issue | Action |
|-------|--------|
| Service won't start | Check logs, verify config |
| Kill switch triggered | Review state, investigate cause |
| Broker API down | Check broker status page, wait or switch |
| Unexpected losses | Stop trading, investigate models |
| System unresponsive | Force restart, check resources |
