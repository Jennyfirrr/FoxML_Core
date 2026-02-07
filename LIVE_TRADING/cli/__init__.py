"""
CLI Module
==========

Command-line interface for the live trading engine.

Provides:
- Configuration validation
- Symbol loading
- Logging setup
- Engine orchestration
"""

from .config import (
    CLIConfig,
    load_config,
    validate_config,
)

__all__ = [
    "CLIConfig",
    "load_config",
    "validate_config",
]
