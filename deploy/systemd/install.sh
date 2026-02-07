#!/bin/bash
#
# Install FoxML Trading systemd service
#
# Usage:
#   sudo ./deploy/systemd/install.sh
#
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="foxml-trading"

echo "Installing FoxML Trading systemd service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Make launcher executable
chmod +x "$SCRIPT_DIR/foxml-trading-launcher.sh"

# Copy service file
cp "$SCRIPT_DIR/foxml-trading.service" /etc/systemd/system/

# Create required directories with correct permissions
TRADER_USER="Jennifer"
TRADER_ROOT="/home/Jennifer/trader"

mkdir -p "$TRADER_ROOT/LIVE_TRADING/state"
mkdir -p "$TRADER_ROOT/LIVE_TRADING/logs"
chown -R "$TRADER_USER:$TRADER_USER" "$TRADER_ROOT/LIVE_TRADING/state"
chown -R "$TRADER_USER:$TRADER_USER" "$TRADER_ROOT/LIVE_TRADING/logs"

# Reload systemd
systemctl daemon-reload

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start $SERVICE_NAME     # Start trading"
echo "  sudo systemctl stop $SERVICE_NAME      # Stop trading"
echo "  sudo systemctl status $SERVICE_NAME    # Check status"
echo "  sudo systemctl enable $SERVICE_NAME    # Enable on boot"
echo "  journalctl -u $SERVICE_NAME -f         # Follow logs"
echo ""
echo "Configuration (environment variables in service file):"
echo "  FOXML_RUN_ID            - Training run ID (default: latest)"
echo "  FOXML_BROKER            - Broker: paper, alpaca, ibkr (default: paper)"
echo "  FOXML_CYCLE_INTERVAL    - Seconds between cycles (default: 60)"
echo "  FOXML_MARKET_HOURS_ONLY - Only trade during market hours (default: true)"
echo ""
echo "To customize, edit: /etc/systemd/system/foxml-trading.service"
echo "Then run: sudo systemctl daemon-reload"
