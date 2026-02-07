"""
Data Provider Interface
=======================

Protocol for market data providers.

All data providers must implement this protocol for interchangeability.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from LIVE_TRADING.common.types import Quote


@runtime_checkable
class DataProvider(Protocol):
    """
    Protocol for market data providers.

    All data provider implementations must provide these methods.
    This enables dependency injection and testability.

    Example:
        >>> def process_data(provider: DataProvider, symbol: str):
        ...     quote = provider.get_quote(symbol)
        ...     hist = provider.get_historical(symbol)
        ...     return quote, hist
    """

    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote with bid/ask data

        Raises:
            DataError: If quote unavailable
        """
        ...

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol
            period: Lookback period (e.g., "1mo", "3mo", "1y")
            interval: Bar interval (e.g., "1d", "1h", "5m")

        Returns:
            DataFrame with OHLCV columns (Open, High, Low, Close, Volume)

        Raises:
            DataError: If data unavailable
        """
        ...

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Get average daily volume in dollars.

        Args:
            symbol: Trading symbol
            lookback_days: Days for average

        Returns:
            ADV in dollars
        """
        ...
