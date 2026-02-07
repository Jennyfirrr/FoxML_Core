"""
Data Provider Interface (Backward Compatibility)
================================================

This module re-exports from LIVE_TRADING.data for backward compatibility.

New code should import from LIVE_TRADING.data directly:
    from LIVE_TRADING.data import DataProvider, SimulatedDataProvider

Deprecated usage:
    from LIVE_TRADING.engine.data_provider import DataProvider  # Still works
"""

import warnings

# Re-export everything from new location
from LIVE_TRADING.data.interface import DataProvider
from LIVE_TRADING.data.simulated import SimulatedDataProvider
from LIVE_TRADING.data.cached import CachedDataProvider
from LIVE_TRADING.data import get_data_provider

__all__ = [
    "DataProvider",
    "SimulatedDataProvider",
    "CachedDataProvider",
    "get_data_provider",
]

# Note: We don't emit deprecation warnings by default to avoid
# breaking existing code. The docstring serves as documentation.
