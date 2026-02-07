#!/bin/bash
#
# FoxML Live Trading Launcher
#
# This script is called by systemd to start the trading engine.
# It handles:
# - Conda environment activation
# - Market hours checking (optional - can be handled by engine)
# - Logging setup
# - Graceful shutdown propagation
#
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

TRADER_ROOT="/home/Jennifer/trader"
CONDA_ENV="trader_env"
CONDA_BASE="/home/Jennifer/miniconda3"

# Source config file if it exists
if [[ -f "/etc/foxml-trading.conf" ]]; then
    source "/etc/foxml-trading.conf"
elif [[ -f "$TRADER_ROOT/deploy/systemd/foxml-trading.conf" ]]; then
    source "$TRADER_ROOT/deploy/systemd/foxml-trading.conf"
fi

# Trading configuration (with defaults)
RUN_ID="${FOXML_RUN_ID:-latest}"  # Use 'latest' to auto-detect most recent training run
BROKER="${FOXML_BROKER:-paper}"   # paper, alpaca, ibkr
CYCLE_INTERVAL="${FOXML_CYCLE_INTERVAL:-60}"  # seconds between cycles
LOG_LEVEL="${FOXML_LOG_LEVEL:-INFO}"

# Market hours (Central Time - Birmingham, AL)
# Set to "true" to only trade during market hours
MARKET_HOURS_ONLY="${FOXML_MARKET_HOURS_ONLY:-true}"

# ============================================================================
# CONDA ACTIVATION
# ============================================================================

# Source conda
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "ERROR: Conda not found at $CONDA_BASE" >&2
    exit 1
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

is_market_open() {
    # Check if US stock market is currently open
    # Market hours: 9:30 AM - 4:00 PM Eastern Time
    # In Central Time: 8:30 AM - 3:00 PM

    local now_et
    now_et=$(TZ="America/New_York" date +%H%M)
    local day_of_week
    day_of_week=$(date +%u)  # 1=Monday, 7=Sunday

    # Weekend check
    if [[ $day_of_week -ge 6 ]]; then
        return 1
    fi

    # Market hours check (0930 to 1600 ET)
    if [[ $now_et -ge 0930 && $now_et -lt 1600 ]]; then
        return 0
    fi

    return 1
}

wait_for_market_open() {
    echo "Waiting for market to open..."
    while ! is_market_open; do
        # Check every 5 minutes
        sleep 300
    done
    echo "Market is now open!"
}

find_latest_run() {
    # Find the most recent training run with trained models
    local runs_dir="$TRADER_ROOT/TRAINING/results"

    if [[ ! -d "$runs_dir" ]]; then
        echo ""
        return
    fi

    # Find directories with manifest.json, sorted by modification time
    local latest
    latest=$(find "$runs_dir" -maxdepth 2 -name "manifest.json" -printf '%T@ %h\n' 2>/dev/null | \
             sort -rn | head -1 | awk '{print $2}')

    if [[ -n "$latest" ]]; then
        basename "$latest"
    else
        echo ""
    fi
}

# ============================================================================
# MAIN
# ============================================================================

cd "$TRADER_ROOT"

echo "=============================================="
echo "FoxML Live Trading Engine"
echo "=============================================="
echo "Time: $(date)"
echo "Broker: $BROKER"
echo "Cycle Interval: ${CYCLE_INTERVAL}s"
echo "Market Hours Only: $MARKET_HOURS_ONLY"
echo "=============================================="

# Resolve run ID
if [[ "$RUN_ID" == "latest" ]]; then
    RUN_ID=$(find_latest_run)
    if [[ -z "$RUN_ID" ]]; then
        echo "ERROR: No training runs found in TRAINING/results/" >&2
        echo "Run the training pipeline first to generate models." >&2
        exit 1
    fi
    echo "Auto-detected run: $RUN_ID"
fi

# Wait for market if configured
if [[ "$MARKET_HOURS_ONLY" == "true" ]]; then
    if ! is_market_open; then
        wait_for_market_open
    fi
fi

# Start the trading engine
echo "Starting trading engine with run: $RUN_ID"
exec python "$TRADER_ROOT/bin/run_live_trading.py" \
    --run-id "$RUN_ID" \
    --broker "$BROKER" \
    --interval "$CYCLE_INTERVAL" \
    --log-level "$LOG_LEVEL"
