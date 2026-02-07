"""
Engine Module
=============

Main trading engine orchestrator for LIVE_TRADING.

Provides:
- TradingEngine: Main pipeline coordinator
- EngineState: State management and persistence
- CycleResult: Trading cycle results
- DataProvider: Market data interface
- EngineConfig: Engine configuration
- get_engine: Access global engine instance
- get_engine_state: Access engine state

Example:
    >>> from LIVE_TRADING.engine import TradingEngine, EngineConfig
    >>> from LIVE_TRADING.brokers import get_broker
    >>> from LIVE_TRADING.engine import get_data_provider
    >>>
    >>> broker = get_broker("paper", initial_cash=100_000)
    >>> data_provider = get_data_provider("simulated")
    >>> engine = TradingEngine(
    ...     broker=broker,
    ...     data_provider=data_provider,
    ...     run_root="/path/to/run",
    ... )
    >>> result = engine.run_cycle(["AAPL", "MSFT"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .trading_engine import TradingEngine, EngineConfig
from .state import EngineState, CycleResult
from .data_provider import (
    DataProvider,
    SimulatedDataProvider,
    CachedDataProvider,
    get_data_provider,
)

if TYPE_CHECKING:
    pass

# Global engine instance for dashboard access
_engine_instance: Optional[TradingEngine] = None


def set_engine(engine: Optional[TradingEngine]) -> None:
    """
    Set the global engine instance.

    Call this when creating/starting the trading engine to make it
    accessible to the dashboard IPC bridge.

    Args:
        engine: TradingEngine instance or None to clear
    """
    global _engine_instance
    _engine_instance = engine


def get_engine() -> Optional[TradingEngine]:
    """
    Get the global engine instance.

    Returns:
        TradingEngine instance if set, None otherwise
    """
    return _engine_instance


def get_engine_state() -> Optional[EngineState]:
    """
    Get the engine's state object.

    Returns:
        EngineState if engine is running, None otherwise
    """
    if _engine_instance is not None:
        return _engine_instance.state
    return None


__all__ = [
    "TradingEngine",
    "EngineConfig",
    "EngineState",
    "CycleResult",
    "DataProvider",
    "SimulatedDataProvider",
    "CachedDataProvider",
    "get_data_provider",
    "get_engine",
    "get_engine_state",
    "set_engine",
]
