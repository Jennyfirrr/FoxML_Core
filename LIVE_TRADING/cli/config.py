"""
CLI Configuration
=================

Configuration loading and validation for the live trading CLI.

SST Compliance:
- Uses get_cfg() for all config access
- Validates config against expected schema
- No hardcoded values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.constants import (
    HORIZONS,
    FAMILIES,
    DEFAULT_CONFIG,
)
from LIVE_TRADING.common.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """
    Configuration for CLI invocation.

    Combines CLI arguments with config file values.
    CLI args take precedence over config file values.
    """

    # Run identification
    run_id: Optional[str] = None
    run_root: Optional[Path] = None

    # Broker configuration
    broker: str = "paper"
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0
    fee_bps: float = 1.0

    # Symbol configuration
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    symbols_file: Optional[Path] = None

    # Cycle configuration
    cycle_interval_seconds: int = 60
    max_cycles: int = 0  # 0 = unlimited

    # Mode configuration
    dry_run: bool = False
    simulation_mode: bool = True

    # Logging configuration
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    state_dir: Path = field(default_factory=lambda: Path("state"))

    # Pipeline configuration
    horizons: List[str] = field(default_factory=lambda: HORIZONS.copy())
    families: List[str] = field(default_factory=list)  # Empty = all

    # Target configuration
    default_target: str = "ret_5m"

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.run_root, str):
            self.run_root = Path(self.run_root)
        if isinstance(self.symbols_file, str):
            self.symbols_file = Path(self.symbols_file)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        if isinstance(self.state_dir, str):
            self.state_dir = Path(self.state_dir)

    @classmethod
    def from_args_and_config(
        cls,
        run_id: Optional[str] = None,
        run_root: Optional[str] = None,
        broker: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        symbols_file: Optional[str] = None,
        interval: Optional[int] = None,
        max_cycles: Optional[int] = None,
        dry_run: bool = False,
        log_level: Optional[str] = None,
    ) -> "CLIConfig":
        """
        Create config from CLI args and config files.

        CLI args override config file values.

        Args:
            run_id: Training run ID
            run_root: Full path to run directory
            broker: Broker type
            symbols: Symbols to trade
            symbols_file: YAML file with symbols
            interval: Cycle interval in seconds
            max_cycles: Maximum cycles
            dry_run: Dry run mode
            log_level: Logging level

        Returns:
            CLIConfig instance
        """
        # Load base config from files
        config = cls(
            run_id=run_id,
            run_root=Path(run_root) if run_root else None,
            broker=broker or get_cfg("live_trading.broker", default="paper"),
            initial_cash=get_cfg("live_trading.paper.initial_cash", default=100_000.0),
            slippage_bps=get_cfg("live_trading.paper.slippage_bps", default=5.0),
            fee_bps=get_cfg("live_trading.paper.fee_bps", default=1.0),
            cycle_interval_seconds=interval or get_cfg(
                "live_trading.engine.cycle_interval_seconds", default=60
            ),
            max_cycles=max_cycles if max_cycles is not None else 0,
            dry_run=dry_run,
            simulation_mode=get_cfg("live_trading.engine.simulation_mode", default=True),
            log_level=log_level or get_cfg("live_trading.logging.level", default="INFO"),
            log_dir=Path(get_cfg("live_trading.logging.log_dir", default="logs")),
            state_dir=Path(get_cfg("live_trading.logging.state_dir", default="state")),
            horizons=get_cfg("live_trading.horizons", default=HORIZONS),
            families=get_cfg("live_trading.models.families", default=[]),
            default_target=get_cfg("live_trading.engine.default_target", default="ret_5m"),
        )

        # Load symbols (CLI args override config file override symbols_file)
        if symbols:
            config.symbols = symbols
        elif symbols_file:
            config.symbols_file = Path(symbols_file)
            config.symbols = load_symbols_from_file(config.symbols_file)
        else:
            config.symbols = get_cfg("live_trading.symbols.default", default=["SPY", "QQQ"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging/serialization."""
        return {
            "broker": self.broker,
            "cycle_interval_seconds": self.cycle_interval_seconds,
            "default_target": self.default_target,
            "dry_run": self.dry_run,
            "families": self.families,
            "fee_bps": self.fee_bps,
            "horizons": self.horizons,
            "initial_cash": self.initial_cash,
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "max_cycles": self.max_cycles,
            "run_id": self.run_id,
            "run_root": str(self.run_root) if self.run_root else None,
            "simulation_mode": self.simulation_mode,
            "slippage_bps": self.slippage_bps,
            "state_dir": str(self.state_dir),
            "symbols": self.symbols,
            "symbols_file": str(self.symbols_file) if self.symbols_file else None,
        }


def load_config() -> CLIConfig:
    """
    Load CLI configuration from config files.

    Uses get_cfg() for all values with appropriate defaults.

    Returns:
        CLIConfig instance
    """
    return CLIConfig.from_args_and_config()


def load_symbols_from_file(path: Path) -> List[str]:
    """
    Load symbols from YAML file.

    Expected format:
        symbols:
          default: [SPY, QQQ, AAPL]

    Or:
        symbols:
          universe: [SPY, QQQ, IWM]
          stocks: [AAPL, MSFT]
          default: [SPY]

    Args:
        path: Path to YAML file

    Returns:
        List of symbols

    Raises:
        ConfigValidationError: If file is invalid
    """
    if not path.exists():
        raise ConfigValidationError(f"Symbols file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML in {path}: {e}")

    if not data or "symbols" not in data:
        raise ConfigValidationError(f"No 'symbols' key in {path}")

    symbols_config = data["symbols"]

    # Check for default list
    if "default" in symbols_config:
        return symbols_config["default"]

    # Otherwise combine all lists
    all_symbols = []
    for key in ["universe", "sectors", "stocks"]:
        if key in symbols_config:
            all_symbols.extend(symbols_config[key])

    if not all_symbols:
        raise ConfigValidationError(f"No symbols found in {path}")

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in all_symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def validate_config(config: CLIConfig) -> List[str]:
    """
    Validate CLI configuration.

    Checks:
    1. Required fields are present
    2. Values are in valid ranges
    3. Paths exist where required

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check run identification
    if not config.run_id and not config.run_root and not config.dry_run:
        errors.append("Either --run-id or --run-root is required (unless --dry-run)")

    # Check run_root exists if provided
    if config.run_root and not config.run_root.exists():
        errors.append(f"Run root does not exist: {config.run_root}")

    # Resolve run_root from run_id if needed
    if config.run_id and not config.run_root:
        results_dir = Path("RESULTS/runs") / config.run_id
        if not results_dir.exists():
            errors.append(f"Run directory not found: {results_dir}")

    # Check broker type
    valid_brokers = ["paper", "ibkr", "alpaca"]
    if config.broker not in valid_brokers:
        errors.append(f"Invalid broker: {config.broker}. Must be one of: {valid_brokers}")

    # Check symbols
    if not config.symbols:
        errors.append("No symbols specified")

    # Check symbols file if specified
    if config.symbols_file and not config.symbols_file.exists():
        errors.append(f"Symbols file not found: {config.symbols_file}")

    # Check cycle interval
    if config.cycle_interval_seconds < 1:
        errors.append(f"Cycle interval must be >= 1 second: {config.cycle_interval_seconds}")

    # Check max_cycles
    if config.max_cycles < 0:
        errors.append(f"Max cycles must be >= 0: {config.max_cycles}")

    # Check initial cash
    if config.initial_cash <= 0:
        errors.append(f"Initial cash must be > 0: {config.initial_cash}")

    # Check horizons
    for h in config.horizons:
        if h not in HORIZONS:
            errors.append(f"Invalid horizon: {h}. Valid: {HORIZONS}")

    # Check families
    for f in config.families:
        if f not in FAMILIES:
            errors.append(f"Invalid family: {f}. Valid: {FAMILIES}")

    # Check log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config.log_level.upper() not in valid_levels:
        errors.append(f"Invalid log level: {config.log_level}. Must be one of: {valid_levels}")

    return errors


def resolve_run_root(config: CLIConfig) -> Optional[Path]:
    """
    Resolve run_root from run_id if needed.

    Finds the latest timestamp directory within the run.

    Args:
        config: Configuration with run_id

    Returns:
        Path to run root, or None if not found
    """
    if config.run_root:
        return config.run_root

    if not config.run_id:
        return None

    # Find run directory
    results_dir = Path("RESULTS/runs") / config.run_id
    if not results_dir.exists():
        logger.error(f"Run directory not found: {results_dir}")
        return None

    # Get latest timestamp directory
    from TRAINING.common.utils.determinism_ordering import iterdir_sorted

    ts_dirs = [d for d in iterdir_sorted(results_dir) if d.is_dir()]
    if not ts_dirs:
        logger.error(f"No timestamp directories in {results_dir}")
        return None

    # Return latest (sorted alphabetically, timestamps sort correctly)
    return ts_dirs[-1]


def create_directories(config: CLIConfig) -> None:
    """
    Create required directories.

    Args:
        config: Configuration with directory paths
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
