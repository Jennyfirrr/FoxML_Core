"""
Data Providers Module
=====================

Market data providers for live trading.

Providers:
- SimulatedDataProvider: Synthetic data for testing
- CachedDataProvider: Caching wrapper for any provider
- AlpacaDataProvider: Alpaca Markets data (requires alpaca-py)
- PolygonDataProvider: Polygon.io data (requires requests)

Usage:
    >>> from LIVE_TRADING.data import get_data_provider
    >>> provider = get_data_provider("simulated")
    >>> quote = provider.get_quote("AAPL")
"""

from LIVE_TRADING.data.interface import DataProvider
from LIVE_TRADING.data.simulated import SimulatedDataProvider
from LIVE_TRADING.data.cached import CachedDataProvider

# Lazy imports for providers with external dependencies
_alpaca_provider = None
_polygon_provider = None


def _get_alpaca_class():
    """Lazy import AlpacaDataProvider."""
    global _alpaca_provider
    if _alpaca_provider is None:
        from LIVE_TRADING.data.alpaca import AlpacaDataProvider
        _alpaca_provider = AlpacaDataProvider
    return _alpaca_provider


def _get_polygon_class():
    """Lazy import PolygonDataProvider."""
    global _polygon_provider
    if _polygon_provider is None:
        from LIVE_TRADING.data.polygon import PolygonDataProvider
        _polygon_provider = PolygonDataProvider
    return _polygon_provider


def get_data_provider(provider_type: str = "simulated", **kwargs) -> DataProvider:
    """
    Factory function to get data provider.

    Args:
        provider_type: Provider type:
            - "simulated": Simulated data for testing
            - "alpaca": Alpaca Markets data
            - "polygon": Polygon.io data
        **kwargs: Provider-specific configuration

    Returns:
        DataProvider instance

    Raises:
        ValueError: If provider type is unknown
        DataError: If provider fails to initialize

    Example:
        >>> provider = get_data_provider("simulated", volatility=0.01)
        >>> provider = get_data_provider("alpaca", feed="sip")
    """
    if provider_type == "simulated":
        return SimulatedDataProvider(**kwargs)
    elif provider_type == "alpaca":
        AlpacaDataProvider = _get_alpaca_class()
        return AlpacaDataProvider(**kwargs)
    elif provider_type == "polygon":
        PolygonDataProvider = _get_polygon_class()
        return PolygonDataProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown data provider type: {provider_type}. "
            "Valid options: simulated, alpaca, polygon"
        )


# For direct imports of provider classes
def __getattr__(name: str):
    """Lazy attribute access for provider classes."""
    if name == "AlpacaDataProvider":
        return _get_alpaca_class()
    elif name == "PolygonDataProvider":
        return _get_polygon_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Protocol
    "DataProvider",
    # Providers (always available)
    "SimulatedDataProvider",
    "CachedDataProvider",
    # Providers (require external packages)
    "AlpacaDataProvider",
    "PolygonDataProvider",
    # Factory
    "get_data_provider",
]
