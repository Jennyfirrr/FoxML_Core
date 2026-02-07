"""
Broker layer for LIVE_TRADING.

Provides:
- Broker Protocol for broker adapters
- PaperBroker for simulation
- AlpacaBroker for Alpaca Markets
- Factory function for broker instantiation
"""

from .interface import Broker, get_broker, normalize_order
from .paper import PaperBroker


def _get_alpaca_broker():
    """Lazy import of AlpacaBroker to handle missing dependency."""
    from .alpaca import AlpacaBroker

    return AlpacaBroker


def _get_ibkr_broker():
    """Lazy import of IBKRBroker to handle missing dependency."""
    from .ibkr import IBKRBroker

    return IBKRBroker


# Re-export for convenience (lazy to avoid import errors)
def __getattr__(name: str):
    """Lazy attribute access for optional broker classes."""
    if name == "AlpacaBroker":
        return _get_alpaca_broker()
    if name == "IBKRBroker":
        return _get_ibkr_broker()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Broker",
    "PaperBroker",
    "AlpacaBroker",
    "IBKRBroker",
    "get_broker",
    "normalize_order",
]
