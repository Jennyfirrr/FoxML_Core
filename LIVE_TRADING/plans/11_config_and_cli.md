# Plan 11: Configuration and CLI

## Overview

Configuration files and CLI entry points for running the live trading system.

## Files to Create

### 1. `CONFIG/live_trading/live_trading.yaml`
**Purpose:** Main configuration file for live trading

```yaml
# Live Trading Configuration
# ==========================

live_trading:
  # Supported horizons
  horizons: ["5m", "10m", "15m", "30m", "60m", "1d"]

  # Blending configuration
  blending:
    ridge_lambda: 0.15
    temperature:
      "5m": 0.75
      "10m": 0.85
      "15m": 0.90
      "30m": 1.0
      "60m": 1.0
      "1d": 1.0

  # Cost model
  cost_model:
    k1: 1.0      # Spread penalty
    k2: 0.15     # Volatility timing
    k3: 1.0      # Market impact

  # Arbitration
  arbitration:
    entry_threshold_bps: 2.0
    reserve_bps: 0.5

  # Barrier gating
  barrier_gate:
    g_min: 0.2
    gamma: 1.0
    delta: 0.5
    peak_threshold: 0.6
    valley_threshold: 0.55

  # Standardization
  standardization:
    window_size: 10

  # Position sizing
  sizing:
    z_max: 3.0
    max_weight: 0.05
    gross_target: 0.5
    no_trade_band: 0.008

  # Risk management
  risk:
    max_daily_loss_pct: 2.0
    max_drawdown_pct: 10.0
    max_position_pct: 20.0
    max_gross_exposure: 1.0
    spread_max_bps: 12.0
    quote_age_max_ms: 200.0

  # Paper broker
  paper:
    slippage_bps: 5.0
    fee_bps: 1.0
    initial_cash: 100000.0

  # Data provider
  data:
    backend: "yfinance"
    cache_ttl_seconds: 30.0

  # Logging
  logging:
    level: "INFO"
    trade_log_dir: "logs/trades"
    state_path: "state/engine_state.json"
```

### 2. `CONFIG/live_trading/symbols.yaml`
**Purpose:** Symbol universe configuration

```yaml
# Trading Universe
# ================

symbols:
  # Core universe
  universe:
    - SPY
    - QQQ
    - IWM
    - DIA

  # Sector ETFs
  sectors:
    - XLF   # Financials
    - XLK   # Technology
    - XLE   # Energy
    - XLV   # Healthcare

  # Individual stocks (optional)
  stocks:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - NVDA

  # Default list for paper trading
  default: ["SPY", "QQQ", "AAPL", "MSFT"]
```

### 3. `bin/run_live_trading.py`
**Purpose:** CLI entry point for live trading

```python
#!/usr/bin/env python
"""
Live Trading CLI
================

Entry point for running the live trading engine.

Usage:
    python -m bin.run_live_trading --run-id <run_id> [options]
"""

# MUST import repro_bootstrap FIRST for determinism
import TRAINING.common.repro_bootstrap  # noqa: F401

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from CONFIG.config_loader import get_cfg
from LIVE_TRADING.brokers.interface import get_broker
from LIVE_TRADING.brokers.data_provider import get_data_provider
from LIVE_TRADING.engine.trading_engine import TradingEngine
from LIVE_TRADING.common.constants import DECISION_TRADE

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"live_trading_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    logger.info(f"Logging to {log_file}")


def load_symbols(symbols_file: str | None = None) -> list[str]:
    """Load symbol list from config or file."""
    if symbols_file:
        with open(symbols_file) as f:
            config = yaml.safe_load(f)
            return config.get("symbols", {}).get("default", ["SPY"])

    return get_cfg("live_trading.symbols", default=["SPY", "QQQ"])


def main():
    parser = argparse.ArgumentParser(description="Live Trading Engine")

    parser.add_argument(
        "--run-id",
        required=True,
        help="TRAINING run ID to use for models",
    )
    parser.add_argument(
        "--run-root",
        help="Full path to run directory (overrides run-id)",
    )
    parser.add_argument(
        "--broker",
        default="paper",
        choices=["paper", "ibkr"],
        help="Broker to use (default: paper)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to trade (space-separated)",
    )
    parser.add_argument(
        "--symbols-file",
        help="YAML file with symbol list",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Cycle interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximum cycles (0 = unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (no actual trades)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 60)
    logger.info("LIVE TRADING ENGINE")
    logger.info("=" * 60)

    # Determine run root
    if args.run_root:
        run_root = Path(args.run_root)
    else:
        # Find latest timestamp directory in run
        results_dir = Path("RESULTS/runs") / args.run_id
        if not results_dir.exists():
            logger.error(f"Run directory not found: {results_dir}")
            sys.exit(1)

        # Get latest timestamp directory
        ts_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        if not ts_dirs:
            logger.error(f"No timestamp directories in {results_dir}")
            sys.exit(1)

        run_root = ts_dirs[-1]

    logger.info(f"Using run: {run_root}")

    # Load symbols
    symbols = args.symbols or load_symbols(args.symbols_file)
    logger.info(f"Trading symbols: {symbols}")

    # Initialize broker
    broker = get_broker(args.broker)
    logger.info(f"Broker: {args.broker} | Cash: ${broker.get_cash():,.2f}")

    # Initialize data provider
    data_provider = get_data_provider("yfinance")

    # Initialize engine
    engine = TradingEngine(
        broker=broker,
        data_provider=data_provider,
        run_root=str(run_root),
    )

    # Main loop
    cycle = 0
    try:
        while True:
            cycle += 1

            if args.max_cycles > 0 and cycle > args.max_cycles:
                logger.info(f"Max cycles ({args.max_cycles}) reached")
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"CYCLE {cycle} | {datetime.now()}")
            logger.info(f"{'='*60}")

            # Run trading cycle
            decisions = engine.run_cycle(symbols)

            # Log decisions
            for decision in decisions:
                if decision.decision == DECISION_TRADE:
                    side = "BUY" if decision.shares > 0 else "SELL"
                    logger.info(
                        f"  {decision.symbol}: {side} {abs(decision.shares)} shares | "
                        f"α={decision.alpha:.4f} | {decision.horizon}"
                    )
                else:
                    logger.info(
                        f"  {decision.symbol}: {decision.decision} | {decision.reason}"
                    )

            # Log portfolio state
            cash = broker.get_cash()
            positions = broker.get_positions()
            logger.info(f"\nPortfolio: Cash=${cash:,.2f} | Positions={len(positions)}")

            # Wait for next cycle
            if args.max_cycles == 0 or cycle < args.max_cycles:
                logger.info(f"Sleeping {args.interval}s until next cycle...")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Live trading engine stopped")


if __name__ == "__main__":
    main()
```

### 4. `LIVE_TRADING/README.md`
**Purpose:** Module documentation

```markdown
# LIVE_TRADING Module

Execution engine for deploying models trained by the TRAINING pipeline.

## Quick Start

```bash
# Paper trading with models from a run
python -m bin.run_live_trading --run-id my_run --broker paper

# Specific symbols
python -m bin.run_live_trading --run-id my_run --symbols SPY QQQ AAPL

# Dry run (no trades)
python -m bin.run_live_trading --run-id my_run --dry-run
```

## Pipeline

1. **Prediction** - Multi-horizon predictions (5m to 1d)
2. **Blending** - Ridge risk-parity per horizon
3. **Arbitration** - Cost-aware horizon selection
4. **Gating** - Barrier probability gates
5. **Sizing** - Volatility-scaled position sizing
6. **Risk** - Kill switches and guardrails

## Configuration

See `CONFIG/live_trading/live_trading.yaml` for all settings.

## Architecture

```
LIVE_TRADING/
├── common/          # Exceptions, constants
├── brokers/         # Broker interfaces
├── models/          # Model loading and inference
├── prediction/      # Multi-horizon prediction
├── blending/        # Ridge risk-parity
├── arbitration/     # Cost-aware selection
├── gating/          # Barrier gates
├── sizing/          # Position sizing
├── risk/            # Risk management
├── engine/          # Main orchestrator
└── tests/           # Unit tests
```

## Mathematical Foundations

See `DOCS/03_technical/trading/architecture/MATHEMATICAL_FOUNDATIONS.md`
```

### 5. `LIVE_TRADING/tests/conftest.py`
**Purpose:** Pytest fixtures for all tests

```python
"""
Pytest Fixtures for LIVE_TRADING Tests
======================================
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
import tempfile

import numpy as np
import pandas as pd


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = Mock()
    broker.get_cash.return_value = 100_000.0
    broker.get_positions.return_value = {}
    broker.submit_order.return_value = {
        "order_id": "test_123",
        "status": "filled",
        "fill_price": 150.0,
        "timestamp": datetime.now(),
    }
    broker.now.return_value = datetime.now()
    return broker


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider."""
    provider = Mock()

    # Mock quote
    provider.get_quote.return_value = {
        "symbol": "AAPL",
        "bid": 149.90,
        "ask": 150.10,
        "bid_size": 100,
        "ask_size": 100,
        "timestamp": datetime.now(),
        "spread_bps": 5.0,
    }

    # Mock historical data
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    provider.get_historical.return_value = pd.DataFrame({
        "Open": np.random.randn(30).cumsum() + 150,
        "High": np.random.randn(30).cumsum() + 151,
        "Low": np.random.randn(30).cumsum() + 149,
        "Close": np.random.randn(30).cumsum() + 150,
        "Volume": np.random.randint(1000000, 10000000, 30),
    }, index=dates)

    return provider


@pytest.fixture
def temp_run_dir():
    """Create a temporary run directory with mock artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create directory structure
        target_dir = run_dir / "targets" / "ret_5m" / "models" / "view=CROSS_SECTIONAL" / "family=LightGBM"
        target_dir.mkdir(parents=True)

        # Create mock model
        import pickle
        class MockModel:
            def predict(self, X):
                return np.array([0.001])

        with open(target_dir / "model.pkl", "wb") as f:
            pickle.dump(MockModel(), f)

        # Create mock metadata
        import json
        with open(target_dir / "model_meta.json", "w") as f:
            json.dump({
                "feature_list": ["ret_1d", "vol_10d"],
                "metrics": {"auc": 0.65},
            }, f)

        yield run_dir


@pytest.fixture
def sample_prices():
    """Create sample price DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
    return pd.DataFrame({
        "Open": np.random.randn(50).cumsum() + 150,
        "High": np.random.randn(50).cumsum() + 151,
        "Low": np.random.randn(50).cumsum() + 149,
        "Close": np.random.randn(50).cumsum() + 150,
        "Volume": np.random.randint(1000000, 10000000, 50),
    }, index=dates)
```

## Tests

### Running All Tests

```bash
# Run all LIVE_TRADING tests
pytest LIVE_TRADING/tests/ -v

# With coverage
pytest LIVE_TRADING/tests/ --cov=LIVE_TRADING --cov-report=html

# Specific module
pytest LIVE_TRADING/tests/test_broker_interface.py -v
```

## SST Compliance Checklist

- [ ] All config uses `get_cfg()` with defaults
- [ ] YAML files follow existing config patterns
- [ ] CLI uses argparse with sensible defaults
- [ ] README documents usage and architecture
- [ ] Pytest fixtures enable easy testing

## Estimated Lines of Code

| File | Lines |
|------|-------|
| `live_trading.yaml` | 80 |
| `symbols.yaml` | 30 |
| `run_live_trading.py` | 200 |
| `README.md` | 60 |
| `conftest.py` | 100 |
| **Total** | ~470 |

## Total Module Estimate

| Plan | Est. Lines |
|------|------------|
| 01 Common Infrastructure | 380 |
| 02 Broker Layer | 760 |
| 03 Model Integration | 920 |
| 04 Prediction Pipeline | 855 |
| 05 Blending | 735 |
| 06 Arbitration | 540 |
| 07 Gating | 390 |
| 08 Sizing | 460 |
| 09 Risk | 520 |
| 10 Engine | 590 |
| 11 Config & CLI | 470 |
| **TOTAL** | ~6,620 |

This is comparable to the reference implementation's ~9,200 lines, leaving room for additional features, tests, and documentation.
