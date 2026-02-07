"""
Alpaca Data Provider
====================

Real-time and historical data from Alpaca Markets.

Features:
- Real-time quotes via REST
- Historical bars (minute, hour, day)
- ADV calculation
- Quote caching with TTL

SST Compliance:
- Credentials from environment
- Configuration via get_cfg()

Requires:
- alpaca-py package (pip install alpaca-py)
- ALPACA_API_KEY and ALPACA_API_SECRET environment variables
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.exceptions import LiveTradingError
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class DataError(LiveTradingError):
    """Error fetching market data."""

    pass


class AlpacaDataProvider:
    """
    Alpaca Markets data provider.

    Provides real-time quotes and historical data.

    Example:
        >>> provider = AlpacaDataProvider()
        >>> quote = provider.get_quote("AAPL")
        >>> hist = provider.get_historical("AAPL", period="1mo")

    Note:
        Requires alpaca-py package and valid API credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: Optional[str] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: API key (default: from ALPACA_API_KEY env)
            api_secret: API secret (default: from ALPACA_API_SECRET env)
            feed: Data feed (iex or sip, default from config)
            clock: Clock for timestamps

        Raises:
            DataError: If credentials not found or alpaca package not installed
        """
        self._clock = clock or get_clock()

        # Get credentials
        key_env = get_cfg(
            "live_trading.brokers.alpaca.api_key_env", default="ALPACA_API_KEY"
        )
        secret_env = get_cfg(
            "live_trading.brokers.alpaca.api_secret_env", default="ALPACA_API_SECRET"
        )

        self._api_key = api_key or os.environ.get(key_env)
        self._api_secret = api_secret or os.environ.get(secret_env)
        self._feed = feed or get_cfg("live_trading.data.alpaca.feed", default="iex")

        if not self._api_key or not self._api_secret:
            raise DataError(
                f"Alpaca credentials not found. Set {key_env} and {secret_env} "
                "environment variables."
            )

        # Try to import alpaca
        try:
            from alpaca.data import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
        except ImportError:
            raise DataError(
                "alpaca-py package not installed. "
                "Install with: pip install alpaca-py"
            )

        # Quote cache
        self._quote_cache: Dict[str, tuple[Quote, datetime]] = {}
        self._cache_ttl_seconds = get_cfg(
            "live_trading.data.cache_ttl_seconds", default=1.0
        )

        logger.info(f"AlpacaDataProvider initialized (feed={self._feed})")

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
        from alpaca.data import StockLatestQuoteRequest

        # Check cache
        now = self._clock.now()
        if symbol in self._quote_cache:
            cached_quote, cache_time = self._quote_cache[symbol]
            if (now - cache_time).total_seconds() < self._cache_ttl_seconds:
                return cached_quote

        try:
            request = StockLatestQuoteRequest(
                symbol_or_symbols=symbol, feed=self._feed
            )
            quotes = self._client.get_stock_latest_quote(request)

            if symbol not in quotes:
                raise DataError(f"No quote available for {symbol}")

            q = quotes[symbol]

            quote = Quote(
                symbol=symbol,
                bid=float(q.bid_price),
                ask=float(q.ask_price),
                bid_size=float(q.bid_size),
                ask_size=float(q.ask_size),
                timestamp=q.timestamp.replace(tzinfo=timezone.utc),
            )

            # Cache
            self._quote_cache[symbol] = (quote, now)

            return quote

        except Exception as e:
            if "DataError" in type(e).__name__:
                raise
            raise DataError(f"Failed to get quote for {symbol}: {e}")

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
            period: Lookback period (1d, 5d, 1mo, 3mo, 1y)
            interval: Bar interval (1m, 5m, 15m, 1h, 1d)

        Returns:
            DataFrame with OHLCV columns

        Raises:
            DataError: If data unavailable
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            # Parse period to start date
            end = self._clock.now()
            start = self._parse_period(period, end)

            # Parse interval to TimeFrame
            timeframe = self._parse_interval(interval)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=self._feed,
            )

            bars = self._client.get_stock_bars(request)

            if symbol not in bars.data:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append(
                    {
                        "Open": float(bar.open),
                        "High": float(bar.high),
                        "Low": float(bar.low),
                        "Close": float(bar.close),
                        "Volume": int(bar.volume),
                    }
                )

            df = pd.DataFrame(data)
            if not df.empty:
                # Set index from timestamps
                df.index = pd.DatetimeIndex(
                    [bar.timestamp for bar in bars.data[symbol]]
                )

            return df

        except Exception as e:
            if "DataError" in type(e).__name__:
                raise
            raise DataError(f"Failed to get historical data for {symbol}: {e}")

    def get_adv(self, symbol: str, lookback_days: int = 20) -> float:
        """
        Get average daily volume in dollars.

        Args:
            symbol: Trading symbol
            lookback_days: Days for average

        Returns:
            ADV in dollars
        """
        try:
            hist = self.get_historical(
                symbol, period=f"{lookback_days + 5}d", interval="1d"
            )

            if hist.empty or len(hist) < 5:
                # Fallback estimate
                quote = self.get_quote(symbol)
                return quote.mid * 5_000_000  # Assume 5M shares

            recent = hist.tail(lookback_days)
            return float((recent["Volume"] * recent["Close"]).mean())

        except Exception as e:
            logger.warning(f"Failed to calculate ADV for {symbol}: {e}")
            return 0.0

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear quote cache.

        Args:
            symbol: Symbol to clear (None = all)
        """
        if symbol:
            self._quote_cache.pop(symbol, None)
        else:
            self._quote_cache.clear()

    def _parse_period(self, period: str, end: datetime) -> datetime:
        """Parse period string to start datetime."""
        if period.endswith("d"):
            days = int(period[:-1])
            return end - timedelta(days=days)
        elif period.endswith("mo"):
            months = int(period[:-2])
            return end - timedelta(days=months * 30)
        elif period.endswith("y"):
            years = int(period[:-1])
            return end - timedelta(days=years * 365)
        else:
            return end - timedelta(days=30)  # Default 1 month

    def _parse_interval(self, interval: str) -> Any:
        """Parse interval string to TimeFrame."""
        from alpaca.data.timeframe import TimeFrame

        if interval == "1m":
            return TimeFrame.Minute
        elif interval == "5m":
            return TimeFrame(5, "Min")
        elif interval == "15m":
            return TimeFrame(15, "Min")
        elif interval == "1h":
            return TimeFrame.Hour
        elif interval == "1d":
            return TimeFrame.Day
        else:
            return TimeFrame.Day
