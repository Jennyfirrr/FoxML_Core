"""
Cached Data Provider
====================

Caching wrapper for data providers.

Reduces API calls by caching quotes and historical data
with configurable TTLs.

SST Compliance:
- Uses Clock abstraction for timestamps
- Deterministic cache key generation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.types import Quote
from LIVE_TRADING.data.interface import DataProvider

logger = logging.getLogger(__name__)


class CachedDataProvider:
    """
    Data provider with caching layer.

    Wraps another data provider and caches results
    to reduce API calls.

    Example:
        >>> underlying = AlpacaDataProvider()
        >>> provider = CachedDataProvider(underlying, quote_ttl_seconds=1.0)
        >>> quote = provider.get_quote("AAPL")  # Fetches from API
        >>> quote = provider.get_quote("AAPL")  # Returns cached
    """

    def __init__(
        self,
        provider: DataProvider,
        quote_ttl_seconds: float = 1.0,
        historical_ttl_seconds: float = 60.0,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize cached provider.

        Args:
            provider: Underlying data provider
            quote_ttl_seconds: Quote cache TTL in seconds
            historical_ttl_seconds: Historical data cache TTL in seconds
            clock: Clock instance for time (default: system clock)
        """
        self._clock = clock or get_clock()
        self.provider = provider
        self.quote_ttl = quote_ttl_seconds
        self.historical_ttl = historical_ttl_seconds

        self._quote_cache: Dict[str, Tuple[Quote, datetime]] = {}
        self._historical_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}

        logger.debug(
            f"CachedDataProvider: quote_ttl={quote_ttl_seconds}s, "
            f"hist_ttl={historical_ttl_seconds}s"
        )

    def get_quote(self, symbol: str) -> Quote:
        """
        Get quote with caching.

        Args:
            symbol: Trading symbol

        Returns:
            Quote (potentially cached)
        """
        now = self._clock.now()

        if symbol in self._quote_cache:
            quote, cache_time = self._quote_cache[symbol]
            age = (now - cache_time).total_seconds()
            if age < self.quote_ttl:
                logger.debug(f"Quote cache hit for {symbol} (age={age:.2f}s)")
                return quote
            # H6 FIX: Log cache expiry
            logger.debug(
                f"Quote cache expired for {symbol}: age={age:.1f}s > ttl={self.quote_ttl}s"
            )

        quote = self.provider.get_quote(symbol)
        self._quote_cache[symbol] = (quote, now)
        logger.debug(f"Quote cache miss for {symbol}, fetched new")
        return quote

    def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical data with caching.

        Args:
            symbol: Trading symbol
            period: Lookback period
            interval: Bar interval

        Returns:
            DataFrame with OHLCV (potentially cached)
        """
        now = self._clock.now()
        cache_key = f"{symbol}_{period}_{interval}"

        if cache_key in self._historical_cache:
            data, cache_time = self._historical_cache[cache_key]
            age = (now - cache_time).total_seconds()
            if age < self.historical_ttl:
                logger.debug(f"Historical cache hit for {cache_key} (age={age:.2f}s)")
                return data.copy()
            # H6 FIX: Log cache expiry
            logger.debug(
                f"Historical cache expired for {cache_key}: age={age:.1f}s > ttl={self.historical_ttl}s"
            )

        data = self.provider.get_historical(symbol, period, interval)
        self._historical_cache[cache_key] = (data, now)
        logger.debug(f"Historical cache miss for {cache_key}, fetched new ({len(data)} rows)")
        return data

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Get ADV (uses underlying provider).

        Args:
            symbol: Trading symbol
            lookback_days: Days for average

        Returns:
            ADV in dollars
        """
        return self.provider.get_adv(symbol, lookback_days)

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            symbol: Symbol to clear (None = all)
        """
        if symbol is None:
            self._quote_cache.clear()
            self._historical_cache.clear()
        else:
            self._quote_cache.pop(symbol, None)
            for key in list(self._historical_cache.keys()):
                if key.startswith(f"{symbol}_"):
                    self._historical_cache.pop(key)

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "quote_entries": len(self._quote_cache),
            "historical_entries": len(self._historical_cache),
        }
