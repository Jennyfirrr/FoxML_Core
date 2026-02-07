#!/usr/bin/env python
"""
Live Trading CLI
================

Entry point for running the live trading engine.

Usage:
    # Paper trading with models from a training run
    python -m bin.run_live_trading --run-id my_run --broker paper

    # Specific symbols
    python -m bin.run_live_trading --run-id my_run --symbols SPY QQQ AAPL

    # With symbols file
    python -m bin.run_live_trading --run-id my_run --symbols-file CONFIG/live_trading/symbols.yaml

    # Dry run (no trades, simulated data)
    python -m bin.run_live_trading --dry-run --symbols SPY

    # Limited cycles
    python -m bin.run_live_trading --run-id my_run --max-cycles 10

SST Compliance:
- Imports repro_bootstrap FIRST for determinism
- Uses get_cfg() for configuration
- Uses atomic writes for state persistence
"""

# MUST import repro_bootstrap FIRST for determinism
import TRAINING.common.repro_bootstrap  # noqa: F401 - MUST be first

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.file_utils import write_atomic_json

from LIVE_TRADING.brokers import get_broker
from LIVE_TRADING.cli.config import (
    CLIConfig,
    create_directories,
    resolve_run_root,
    validate_config,
)
from LIVE_TRADING.common.constants import DECISION_TRADE, DECISION_HOLD, DECISION_BLOCKED
from LIVE_TRADING.common.exceptions import LiveTradingError, KillSwitchTriggered
from LIVE_TRADING.engine import TradingEngine, EngineConfig
from LIVE_TRADING.engine.data_provider import get_data_provider

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info(f"Shutdown signal received ({signum})")


def setup_logging(level: str, log_dir: Path) -> Path:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, etc.)
        log_dir: Directory for log files

    Returns:
        Path to log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"live_trading_{timestamp}.log"

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logger.info(f"Logging to {log_file}")
    return log_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading with models from a training run
  python -m bin.run_live_trading --run-id my_run

  # Specific symbols
  python -m bin.run_live_trading --run-id my_run --symbols SPY QQQ AAPL

  # Dry run (simulated data, no run-id needed)
  python -m bin.run_live_trading --dry-run --symbols SPY QQQ

  # Verbose logging
  python -m bin.run_live_trading --run-id my_run --log-level DEBUG
        """,
    )

    # Run identification
    run_group = parser.add_argument_group("Run Configuration")
    run_group.add_argument(
        "--run-id",
        help="TRAINING run ID to use for models",
    )
    run_group.add_argument(
        "--run-root",
        help="Full path to run directory (overrides --run-id)",
    )

    # Broker configuration
    broker_group = parser.add_argument_group("Broker Configuration")
    broker_group.add_argument(
        "--broker",
        default="paper",
        choices=["paper", "ibkr", "alpaca"],
        help="Broker to use (default: paper)",
    )
    broker_group.add_argument(
        "--initial-cash",
        type=float,
        help="Initial cash for paper broker",
    )

    # Symbol configuration
    symbol_group = parser.add_argument_group("Symbol Configuration")
    symbol_group.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to trade (space-separated)",
    )
    symbol_group.add_argument(
        "--symbols-file",
        help="YAML file with symbol list",
    )

    # Cycle configuration
    cycle_group = parser.add_argument_group("Cycle Configuration")
    cycle_group.add_argument(
        "--interval",
        type=int,
        help="Cycle interval in seconds (default: from config)",
    )
    cycle_group.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximum cycles (0 = unlimited, default: 0)",
    )

    # Mode configuration
    mode_group = parser.add_argument_group("Mode Configuration")
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (simulated data, no real orders)",
    )
    mode_group.add_argument(
        "--no-state",
        action="store_true",
        help="Don't save state between cycles",
    )

    # Logging configuration
    log_group = parser.add_argument_group("Logging Configuration")
    log_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    log_group.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files (default: logs)",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> CLIConfig:
    """
    Create CLIConfig from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        CLIConfig instance
    """
    return CLIConfig.from_args_and_config(
        run_id=args.run_id,
        run_root=args.run_root,
        broker=args.broker,
        symbols=args.symbols,
        symbols_file=args.symbols_file,
        interval=args.interval,
        max_cycles=args.max_cycles,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )


def print_banner(config: CLIConfig) -> None:
    """Print startup banner."""
    logger.info("=" * 70)
    logger.info("  LIVE TRADING ENGINE")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"  Run ID:      {config.run_id or 'N/A'}")
    logger.info(f"  Run Root:    {config.run_root or 'N/A'}")
    logger.info(f"  Broker:      {config.broker}")
    logger.info(f"  Symbols:     {', '.join(config.symbols[:5])}{'...' if len(config.symbols) > 5 else ''}")
    logger.info(f"  Dry Run:     {config.dry_run}")
    logger.info(f"  Max Cycles:  {config.max_cycles if config.max_cycles > 0 else 'Unlimited'}")
    logger.info(f"  Interval:    {config.cycle_interval_seconds}s")
    logger.info("")
    logger.info("=" * 70)


def run_trading_loop(
    engine: TradingEngine,
    config: CLIConfig,
    log_dir: Path,
) -> int:
    """
    Run the main trading loop.

    Args:
        engine: TradingEngine instance
        config: CLI configuration
        log_dir: Directory for decision logs

    Returns:
        Exit code (0 = success, 1 = error)
    """
    global _shutdown_requested

    cycle = 0
    decision_log_path = log_dir / "decisions.jsonl"

    try:
        while not _shutdown_requested:
            cycle += 1

            # Check cycle limit
            if config.max_cycles > 0 and cycle > config.max_cycles:
                logger.info(f"Max cycles ({config.max_cycles}) reached")
                break

            # Log cycle start
            cycle_start = datetime.now(timezone.utc)
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"CYCLE {cycle} | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"{'='*60}")

            # Run trading cycle
            try:
                result = engine.run_cycle(config.symbols, current_time=cycle_start)
            except KillSwitchTriggered as e:
                logger.error(f"Kill switch triggered: {e}")
                _save_decision_log(decision_log_path, {
                    "cycle": cycle,
                    "timestamp": cycle_start.isoformat(),
                    "event": "kill_switch",
                    "reason": str(e),
                })
                return 1

            # Log decisions
            trades = 0
            holds = 0
            blocked = 0

            for decision in result.decisions:
                if decision.decision == DECISION_TRADE:
                    trades += 1
                    side = "BUY" if decision.shares > 0 else "SELL"
                    logger.info(
                        f"  {decision.symbol}: {side} {abs(decision.shares)} shares | "
                        f"alpha={decision.alpha*10000:.1f}bps | {decision.horizon}"
                    )
                elif decision.decision == DECISION_HOLD:
                    holds += 1
                    logger.debug(f"  {decision.symbol}: HOLD | {decision.reason}")
                else:  # BLOCKED
                    blocked += 1
                    logger.warning(f"  {decision.symbol}: BLOCKED | {decision.reason}")

            # Log summary
            logger.info("")
            logger.info(f"Summary: {trades} trades, {holds} holds, {blocked} blocked")
            logger.info(
                f"Portfolio: ${result.portfolio_value:,.2f} | "
                f"Cash: ${result.cash:,.2f}"
            )

            # Save decision log
            _save_decision_log(decision_log_path, {
                "cycle": cycle,
                "timestamp": cycle_start.isoformat(),
                "decisions": [d.to_dict() for d in result.decisions],
                "portfolio_value": result.portfolio_value,
                "cash": result.cash,
                "num_trades": trades,
                "num_holds": holds,
                "num_blocked": blocked,
                "kill_switch_reason": result.kill_switch_reason,
            })

            # Check if we should continue
            if not result.is_trading_allowed:
                logger.error(f"Trading halted: {result.kill_switch_reason}")
                return 1

            # Wait for next cycle (unless this is the last cycle)
            if config.max_cycles == 0 or cycle < config.max_cycles:
                # Calculate time to sleep
                cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                sleep_time = max(0, config.cycle_interval_seconds - cycle_duration)

                if sleep_time > 0:
                    logger.info(f"Next cycle in {sleep_time:.0f}s...")

                    # Sleep in chunks to check for shutdown
                    sleep_chunk = 1.0
                    remaining = sleep_time
                    while remaining > 0 and not _shutdown_requested:
                        time.sleep(min(sleep_chunk, remaining))
                        remaining -= sleep_chunk

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SHUTDOWN")
    logger.info("=" * 60)
    logger.info(f"  Cycles completed: {cycle}")

    summary = engine.get_state_summary()
    logger.info(f"  Final portfolio:  ${summary['portfolio_value']:,.2f}")
    logger.info(f"  Final cash:       ${summary['cash']:,.2f}")
    logger.info(f"  Positions:        {summary['num_positions']}")
    logger.info(f"  Daily P&L:        ${summary['daily_pnl']:,.2f}")

    return 0


def _save_decision_log(path: Path, record: dict) -> None:
    """
    Append record to decision log (JSONL format).

    Args:
        path: Path to log file
        record: Decision record to save
    """
    import json

    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Parse arguments
    args = parse_args()

    # Setup logging early
    log_dir = Path(args.log_dir)
    log_file = setup_logging(args.log_level, log_dir)

    # Create config
    config = create_config_from_args(args)

    # Validate config
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        return 1

    # Create required directories
    create_directories(config)

    # Print banner
    print_banner(config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Resolve run root
    run_root: Optional[Path] = None
    if not config.dry_run:
        run_root = resolve_run_root(config)
        if not run_root:
            logger.error("Could not resolve run root")
            return 1
        logger.info(f"Using run: {run_root}")

    # Initialize broker
    if config.broker == "paper":
        broker = get_broker(
            "paper",
            initial_cash=config.initial_cash,
            slippage_bps=config.slippage_bps,
            fee_bps=config.fee_bps,
        )
    else:
        logger.error(f"Broker '{config.broker}' not yet implemented")
        return 1

    logger.info(f"Broker: {config.broker} | Cash: ${broker.get_cash():,.2f}")

    # Initialize data provider
    data_provider = get_data_provider("simulated")
    logger.info("Data provider: simulated")

    # Initialize engine
    engine_config = EngineConfig(
        state_path=config.state_dir / "engine_state.json",
        history_path=config.state_dir / "history.json",
        save_state=not args.no_state,
        save_history=not args.no_state,
        horizons=config.horizons,
        families=config.families if config.families else None,
    )

    engine = TradingEngine(
        broker=broker,
        data_provider=data_provider,
        run_root=str(run_root) if run_root else None,
        config=engine_config,
    )

    logger.info(f"Engine initialized: {len(config.symbols)} symbols")

    # Run trading loop
    return run_trading_loop(engine, config, log_dir)


if __name__ == "__main__":
    sys.exit(main())
