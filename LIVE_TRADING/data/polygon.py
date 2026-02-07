"""
Polygon.io Data Provider
========================

Professional market data from Polygon.io.

Features:
- Real-time quotes
- Historical bars
- Extended hours data

SST Compliance:
- API key from environment
- Configuration via get_cfg()

Requires:
- requests package
- POLYGON_API_KEY environment variable
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.exceptions import LiveTradingError
from LIVE_TRADING.common.types import Quote

logger = logging.getLogger(__name__)


class DataError(LiveTradingError):
    """Error fetching market data."""

    pass


class PolygonDataProvider:
    """
    Polygon.io data provider.

    Provides real-time quotes and historical data.

    Example:
        >>> provider = PolygonDataProvider()
        >>> quote = provider.get_quote("AAPL")
        >>> hist = provider.get_historical("AAPL", period="1mo")

    Note:
        Requires valid POLYGON_API_KEY environment variable.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        clock: Optional[Clock] = None,
    ):
        """
        Initialize Polygon data provider.

        Args:
            api_key: API key (default: from POLYGON_API_KEY env)
            clock: Clock for timestamps

        Raises:
            DataError: If API key not found
        """
        self._clock = clock or get_clock()

        key_env = get_cfg(
            "live_trading.data.polygon.api_key_env", default="POLYGON_API_KEY"
        )
        self._api_key = api_key or os.environ.get(key_env)

        if not self._api_key:
            raise DataError(
                f"Polygon API key not found. Set {key_env} environment variable."
            )

        # Quote cache
        self._quote_cache: Dict[str, Tuple[Quote, datetime]] = {}
        self._cache_ttl_seconds = get_cfg(
            "live_trading.data.cache_ttl_seconds", default=1.0
        )

        logger.info("PolygonDataProvider initialized")

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make API request.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response

        Raises:
            DataError: If request fails
        """
        params = params or {}
        params["apiKey"] = self._api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise DataError(f"Polygon API request failed: {e}")

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
        # Check cache
        now = self._clock.now()
        if symbol in self._quote_cache:
            cached_quote, cache_time = self._quote_cache[symbol]
            if (now - cache_time).total_seconds() < self._cache_ttl_seconds:
                return cached_quote

        try:
            data = self._request(f"/v2/last/nbbo/{symbol}")

            if data.get("status") != "OK":
                raise DataError(f"Failed to get quote for {symbol}: {data}")

            result = data.get("results", {})

            # Polygon NBBO response format:
            # P = bid price, p = ask price (case matters!)
            # S = bid size, s = ask size
            # t = timestamp in nanoseconds
            quote = Quote(
                symbol=symbol,
                bid=float(result.get("P", 0)),
                ask=float(result.get("p", 0)),
                bid_size=float(result.get("S", 0)),
                ask_size=float(result.get("s", 0)),
                timestamp=datetime.fromtimestamp(
                    result.get("t", 0) / 1e9, tz=timezone.utc
                ),
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
        try:
            end = self._clock.now()
            start = self._parse_period(period, end)

            # Map interval to Polygon timespan
            timespan, multiplier = self._parse_interval(interval)

            endpoint = (
                f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
                f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
            )

            data = self._request(endpoint, {"limit": 5000})

            if data.get("status") != "OK":
                return pd.DataFrame()

            results = data.get("results", [])
            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(results)
            df = df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                    "t": "timestamp",
                }
            )

            # Convert timestamp to datetime index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")

            return df[["Open", "High", "Low", "Close", "Volume"]]

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
                quote = self.get_quote(symbol)
                return quote.mid * 5_000_000

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
            return end - timedelta(days=30)

    def _parse_interval(self, interval: str) -> Tuple[str, int]:
        """Parse interval to Polygon timespan and multiplier."""
        if interval == "1m":
            return "minute", 1
        elif interval == "5m":
            return "minute", 5
        elif interval == "15m":
            return "minute", 15
        elif interval == "1h":
            return "hour", 1
        elif interval == "1d":
            return "day", 1
        else:
            return "day", 1
