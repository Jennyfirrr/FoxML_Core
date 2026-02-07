# Live Trading Quick Reference

## Essential Commands

```bash
# Start/Stop
sudo systemctl start foxml-trading
sudo systemctl stop foxml-trading
sudo systemctl restart foxml-trading

# Status
sudo systemctl status foxml-trading

# Logs
journalctl -u foxml-trading -f           # Follow live
journalctl -u foxml-trading --since today # Today's logs
journalctl -u foxml-trading -e            # Recent errors
```

## Key Files

| File | Purpose |
|------|---------|
| `/etc/foxml-trading.conf` | Configuration |
| `LIVE_TRADING/state/engine_state.json` | Current state |
| `LIVE_TRADING/logs/decisions.jsonl` | Trade decisions |
| `LIVE_TRADING/logs/*.log` | Session logs |

## Configuration Variables

| Variable | Default | Options |
|----------|---------|---------|
| `FOXML_RUN_ID` | `latest` | Run ID or "latest" |
| `FOXML_BROKER` | `paper` | paper, alpaca, ibkr |
| `FOXML_CYCLE_INTERVAL` | `60` | Seconds |
| `FOXML_MARKET_HOURS_ONLY` | `true` | true/false |
| `FOXML_LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR |

## Market Hours

| Timezone | Open | Close |
|----------|------|-------|
| Eastern (ET) | 9:30 AM | 4:00 PM |
| Central (CT) | 8:30 AM | 3:00 PM |
| UTC | 14:30 | 21:00 |

*No trading on weekends. Holiday calendar not implemented.*

## Quick Diagnostics

```bash
# Is it running?
pgrep -f run_live_trading

# Current state
cat LIVE_TRADING/state/engine_state.json | jq '.is_trading_allowed'

# Recent decisions
tail -5 LIVE_TRADING/logs/decisions.jsonl | jq .

# Errors in last hour
journalctl -u foxml-trading --since "1 hour ago" -p err
```

## Emergency Actions

```bash
# Graceful stop
sudo systemctl stop foxml-trading

# Force stop (if hung)
sudo systemctl kill foxml-trading

# Check state after crash
cat LIVE_TRADING/state/engine_state.json | jq .

# Restart fresh (clears state!)
sudo systemctl stop foxml-trading
rm -rf LIVE_TRADING/state/*
sudo systemctl start foxml-trading
```

## Updating Models

```bash
# 1. Run new training
bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py \
    --experiment-config production_baseline \
    --output-dir TRAINING/results/new_run

# 2. Update config
sudo vim /etc/foxml-trading.conf
# Set: FOXML_RUN_ID=new_run

# 3. Restart
sudo systemctl restart foxml-trading
```

## Alpaca Setup

```bash
# In /etc/foxml-trading.conf
FOXML_BROKER=alpaca
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

## IBKR Setup

```bash
# In /etc/foxml-trading.conf
FOXML_BROKER=ibkr
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper: 7497, Live: 7496
IBKR_CLIENT_ID=1
```

## Log Analysis One-Liners

```bash
# Decision count by type
jq -s 'group_by(.decision) | map({d: .[0].decision, n: length})' \
    LIVE_TRADING/logs/decisions.jsonl

# Trades for symbol
jq 'select(.symbol == "AAPL")' LIVE_TRADING/logs/decisions.jsonl

# Errors only
journalctl -u foxml-trading -p err --no-pager

# Cycle timing
journalctl -u foxml-trading | grep "completed in" | tail -10
```

## Support

- Logs: `journalctl -u foxml-trading`
- State: `LIVE_TRADING/state/`
- Config: `/etc/foxml-trading.conf`
- Docs: `DOCS/03_technical/trading/operations/`
