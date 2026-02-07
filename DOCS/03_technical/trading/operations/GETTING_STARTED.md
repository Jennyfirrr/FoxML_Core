# Getting Started with Live Trading

This guide walks you through starting the FoxML live trading system after training is complete.

## Prerequisites

Before starting live trading, ensure you have:

1. **Completed a training run** with models in `TRAINING/results/`
2. **Configured your broker** (paper trading works out of the box)
3. **Reviewed the trained models** to ensure quality metrics are acceptable

## Quick Start (Paper Trading)

### Step 1: Verify Training Completed

```bash
# Check for completed training runs
ls -la TRAINING/results/

# Verify models exist
ls TRAINING/results/prod_run/targets/*/models/
```

You should see directories for each trained target with model files inside.

### Step 2: Start Paper Trading (Manual)

For testing or one-off runs:

```bash
# Start paper trading with the latest run
python bin/run_live_trading.py \
    --run-id prod_run \
    --broker paper \
    --interval 60 \
    --log-level INFO
```

Press `Ctrl+C` to stop gracefully.

### Step 3: Start Paper Trading (Systemd - Recommended)

For autonomous operation:

```bash
# Install the systemd service (one-time)
sudo ./deploy/systemd/install.sh

# Start trading
sudo systemctl start foxml-trading

# Check status
sudo systemctl status foxml-trading

# Follow logs
journalctl -u foxml-trading -f
```

The service will:
- Automatically detect the latest training run
- Wait for market hours (9:30 AM - 4:00 PM ET)
- Run continuously until stopped
- Restart automatically on failure

## Configuration

### Option A: Environment Variables

Set before starting:

```bash
export FOXML_RUN_ID=prod_run        # Training run to use
export FOXML_BROKER=paper           # Broker: paper, alpaca, ibkr
export FOXML_CYCLE_INTERVAL=60      # Seconds between cycles
export FOXML_MARKET_HOURS_ONLY=true # Only trade during market hours
```

### Option B: Configuration File

Edit `/etc/foxml-trading.conf` (for systemd) or `deploy/systemd/foxml-trading.conf`:

```bash
FOXML_RUN_ID=latest
FOXML_BROKER=paper
FOXML_CYCLE_INTERVAL=60
FOXML_LOG_LEVEL=INFO
FOXML_MARKET_HOURS_ONLY=true
```

### Option C: Command Line Arguments

```bash
python bin/run_live_trading.py \
    --run-id prod_run \
    --broker paper \
    --interval 60 \
    --symbols AAPL MSFT GOOGL \
    --log-level DEBUG
```

## Broker Configuration

### Paper Trading (Default)

No configuration needed. Uses simulated execution with realistic fills.

### Alpaca

1. Create an account at [alpaca.markets](https://alpaca.markets)
2. Get API keys from the dashboard
3. Configure:

```bash
# In /etc/foxml-trading.conf or environment
FOXML_BROKER=alpaca
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
```

For live trading, change the URL to `https://api.alpaca.markets`.

### Interactive Brokers

1. Install IB Gateway or TWS
2. Enable API connections in settings
3. Configure:

```bash
FOXML_BROKER=ibkr
IBKR_HOST=127.0.0.1
IBKR_PORT=7497        # Paper: 7497, Live: 7496
IBKR_CLIENT_ID=1
```

## Monitoring

### Log Locations

| Location | Contents |
|----------|----------|
| `journalctl -u foxml-trading` | Systemd service logs |
| `LIVE_TRADING/logs/` | Session log files |
| `LIVE_TRADING/logs/decisions.jsonl` | Trade decision audit trail |

### Real-time Monitoring

```bash
# Follow systemd logs
journalctl -u foxml-trading -f

# Watch decision log
tail -f LIVE_TRADING/logs/decisions.jsonl | jq .

# Check current state
cat LIVE_TRADING/state/engine_state.json | jq .
```

### Key Metrics to Watch

- **Cycle duration**: Should be < cycle interval (default 60s)
- **Decisions per cycle**: Number of trading decisions made
- **Kill switch status**: Should be `false` (trading allowed)
- **Portfolio value**: Track for drawdown monitoring

## Stopping the System

### Graceful Shutdown

```bash
# Systemd
sudo systemctl stop foxml-trading

# Manual (press Ctrl+C or send SIGTERM)
kill -SIGTERM <pid>
```

The system will:
1. Complete the current cycle
2. Save state to disk
3. Exit cleanly

### Emergency Stop

If graceful shutdown fails:

```bash
# Force stop
sudo systemctl kill foxml-trading

# Or
kill -9 <pid>
```

**Warning**: Force stop may leave state inconsistent. Check `LIVE_TRADING/state/` on restart.

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u foxml-trading -e --no-pager

# Common issues:
# - No training run found: Run training first
# - Conda not found: Check CONDA_BASE path in launcher
# - Permission denied: Run install.sh as root
```

### No Trades Being Made

Check:
1. **Market hours**: Is market open? (9:30 AM - 4:00 PM ET)
2. **Kill switch**: Check `engine_state.json` for `is_trading_allowed`
3. **Model loading**: Check logs for model load errors
4. **Spread gate**: May be filtering out trades due to wide spreads

### High Memory Usage

```bash
# Check memory
systemctl status foxml-trading

# Reduce if needed (edit service file)
MemoryMax=4G
```

### Cycle Taking Too Long

If cycles exceed the interval:
1. Reduce number of symbols
2. Increase cycle interval
3. Check for slow broker API responses

## Next Steps

- [Operations Guide](./OPERATIONS_GUIDE.md) - Day-to-day operations
- [Monitoring Setup](./MONITORING.md) - Alerts and dashboards
- [Risk Management](./RISK_MANAGEMENT.md) - Kill switches and limits
